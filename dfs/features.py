"""Feature engineering module for DFS models.

This module contains feature extraction functions that apply across all player types.
Starting with injury features, but designed to be extended with other cross-cutting
features like:
- Weather impact features
- Vegas line features
- Matchup/defensive features
- Rest/fatigue features
- Primetime/slate features
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

from db_manager import get_db_connection

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Centralized feature engineering for all DFS models."""

    def __init__(self, db_path: str = "data/nfl_dfs.db"):
        self.db_path = db_path

    def get_injury_features(
        self,
        player_id: int,
        season: int,
        week: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract injury-related features for a player.

        Features include:
        - Current injury status and severity
        - Practice participation
        - Recent injury history
        - Games missed
        - Injury risk score

        Args:
            player_id: Player ID
            season: NFL season
            week: NFL week
            lookback_weeks: How many weeks to look back for history

        Returns:
            Dictionary of injury features
        """
        features = {
            'has_injury': 0.0,
            'injury_severity': 0.0,  # 0=healthy, 0.25=Q, 0.5=D, 1=OUT
            'weeks_injured_recent': 0.0,
            'games_missed_recent': 0.0,
            'practice_status_limited': 0.0,
            'practice_status_dnp': 0.0,
            'injury_risk_score': 0.0,
            'consecutive_weeks_injured': 0.0,
        }

        conn = get_db_connection(self.db_path)

        try:
            # Get current week injury status
            current_injury = conn.execute(
                """SELECT injury_status, injury_designation, injury_body_part, practice_status
                   FROM injury_reports
                   WHERE player_id = ? AND season = ? AND week = ?
                   ORDER BY report_date DESC
                   LIMIT 1""",
                (player_id, season, week)
            ).fetchone()

            if current_injury:
                status, designation, body_part, practice = current_injury

                # Current injury status
                if status:
                    features['has_injury'] = 1.0

                    # Map status to severity score
                    severity_map = {
                        'Q': 0.25,        # Questionable - 75% likely to play
                        'D': 0.5,         # Doubtful - 25% likely to play
                        'OUT': 1.0,       # Out - will not play
                        'IR': 1.0,        # Injured Reserve
                        'PUP': 1.0,       # Physically Unable to Perform
                    }
                    features['injury_severity'] = severity_map.get(status, 0.0)

                # Practice participation
                if practice:
                    if 'Limited' in practice:
                        features['practice_status_limited'] = 1.0
                    elif 'Did Not' in practice or 'DNP' in practice:
                        features['practice_status_dnp'] = 1.0

            # Get recent injury history
            start_week = max(1, week - lookback_weeks)

            history = conn.execute(
                """SELECT
                       COUNT(DISTINCT week) as weeks_injured,
                       COUNT(DISTINCT CASE WHEN injury_status = 'OUT' THEN week END) as games_missed,
                       COUNT(DISTINCT CASE WHEN injury_status IN ('Q', 'D') THEN week END) as weeks_questionable
                   FROM injury_reports
                   WHERE player_id = ? AND season = ?
                   AND week >= ? AND week < ?
                   AND injury_status IS NOT NULL""",
                (player_id, season, start_week, week)
            ).fetchone()

            if history:
                features['weeks_injured_recent'] = float(history[0] or 0)
                features['games_missed_recent'] = float(history[1] or 0)

                # Calculate injury risk score based on recent history
                # Higher score = higher risk of re-injury or reduced performance
                weeks_questionable = float(history[2] or 0)

                risk_score = 0.0
                if features['games_missed_recent'] > 0:
                    risk_score += features['games_missed_recent'] * 0.3
                if weeks_questionable > 0:
                    risk_score += weeks_questionable * 0.1
                if features['has_injury']:
                    risk_score += features['injury_severity'] * 0.4

                features['injury_risk_score'] = min(1.0, risk_score)

            # Check for consecutive weeks injured
            consecutive = conn.execute(
                """SELECT week, COUNT(*) as injury_count
                   FROM injury_reports
                   WHERE player_id = ? AND season = ?
                   AND week >= ? AND week <= ?
                   AND injury_status IS NOT NULL
                   GROUP BY week
                   ORDER BY week DESC""",
                (player_id, season, max(1, week - 3), week)
            ).fetchall()

            if consecutive:
                # Count consecutive weeks from most recent
                consecutive_count = 0
                expected_week = week
                for week_num, count in consecutive:
                    if week_num == expected_week and count > 0:
                        consecutive_count += 1
                        expected_week -= 1
                    else:
                        break
                features['consecutive_weeks_injured'] = float(consecutive_count)
            else:
                features['consecutive_weeks_injured'] = 0.0

        except Exception as e:
            logger.warning(f"Error getting injury features for player {player_id}: {e}")
        finally:
            conn.close()

        return features

    def get_injury_adjusted_projections(
        self,
        base_projection: float,
        floor: float,
        ceiling: float,
        injury_features: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """Adjust projections based on injury status.

        Args:
            base_projection: Base projected points
            floor: Base floor projection
            ceiling: Base ceiling projection
            injury_features: Dictionary of injury features

        Returns:
            Tuple of (adjusted_projection, adjusted_floor, adjusted_ceiling)
        """
        # No adjustment if healthy
        if injury_features.get('has_injury', 0) == 0:
            return base_projection, floor, ceiling

        severity = injury_features.get('injury_severity', 0)
        risk_score = injury_features.get('injury_risk_score', 0)

        # Calculate multipliers based on injury severity and risk
        # These are empirical adjustments based on historical performance
        if severity >= 1.0:  # OUT
            return 0.0, 0.0, 0.0

        # Projection multiplier
        proj_mult = 1.0
        floor_mult = 1.0
        ceil_mult = 1.0

        if severity > 0:  # Has injury designation
            # Questionable (0.25): 85-95% of projection
            # Doubtful (0.5): 60-75% of projection
            proj_mult = 1.0 - (severity * 0.4)  # Max 40% reduction
            floor_mult = 1.0 - (severity * 0.5)  # Floor impacted more
            ceil_mult = 1.0 - (severity * 0.3)   # Ceiling impacted less (upside still there)

            # Further adjust based on practice status
            if injury_features.get('practice_status_dnp', 0) > 0:
                proj_mult *= 0.85
                floor_mult *= 0.8
            elif injury_features.get('practice_status_limited', 0) > 0:
                proj_mult *= 0.92
                floor_mult *= 0.9

        # Apply risk adjustment for re-injury potential
        if risk_score > 0:
            risk_adjustment = 1.0 - (risk_score * 0.15)  # Up to 15% additional reduction
            proj_mult *= risk_adjustment
            floor_mult *= (risk_adjustment * 0.95)  # Floor more affected by risk

        adjusted_projection = base_projection * proj_mult
        adjusted_floor = floor * floor_mult
        adjusted_ceiling = ceiling * ceil_mult

        # Ensure floor <= projection <= ceiling
        adjusted_floor = min(adjusted_floor, adjusted_projection)
        adjusted_ceiling = max(adjusted_ceiling, adjusted_projection)

        return adjusted_projection, adjusted_floor, adjusted_ceiling

    def get_all_features(
        self,
        player_id: int,
        season: int,
        week: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Get all engineered features for a player.

        This is the main entry point that combines all feature types.
        Currently includes injury features, but will be extended with
        weather, vegas, matchup, etc.

        Args:
            player_id: Player ID
            season: NFL season
            week: NFL week
            lookback_weeks: How many weeks to look back

        Returns:
            Dictionary of all features
        """
        all_features = {}

        # Get injury features
        injury_features = self.get_injury_features(
            player_id, season, week, lookback_weeks
        )
        all_features.update(injury_features)

        # Future: Add weather features
        # weather_features = self.get_weather_features(...)
        # all_features.update(weather_features)

        # Future: Add vegas features
        # vegas_features = self.get_vegas_features(...)
        # all_features.update(vegas_features)

        # Future: Add matchup features
        # matchup_features = self.get_matchup_features(...)
        # all_features.update(matchup_features)

        return all_features
