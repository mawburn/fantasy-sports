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

    def get_correlation_features(
        self,
        team: str,
        opponent: str,
        position: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Extract correlation features for stacking strategies.

        Features include:
        - QB-WR correlation strength
        - RB-DEF game script correlations
        - Team pace and play volume
        - Opponent defensive tendencies

        Args:
            team: Team abbreviation
            opponent: Opponent team abbreviation
            position: Player position
            season: NFL season
            week: NFL week

        Returns:
            Dictionary of correlation features
        """
        features = {
            'team_pass_rate': 0.0,
            'team_pace': 0.0,
            'team_plays_per_game': 0.0,
            'opp_pass_defense_rank': 0.0,
            'opp_rush_defense_rank': 0.0,
            'game_script_correlation': 0.0,
            'stack_correlation_strength': 0.0,
            'target_share_concentration': 0.0,
            'red_zone_opportunities': 0.0,
        }

        conn = get_db_connection(self.db_path)

        try:
            # Get team offensive tendencies
            team_stats = conn.execute(
                """SELECT
                       AVG(pass_attempts * 1.0 / (pass_attempts + rush_attempts)) as pass_rate,
                       AVG(total_plays) as plays_per_game,
                       AVG(time_of_possession) as avg_top
                   FROM team_stats
                   WHERE team = ? AND season = ? AND week < ?
                   AND week >= MAX(1, ? - 4)""",
                (team, season, week, week)
            ).fetchone()

            if team_stats and team_stats[0] is not None:
                features['team_pass_rate'] = float(team_stats[0] or 0.5)
                features['team_plays_per_game'] = float(team_stats[1] or 65)

                # Calculate pace (plays per minute of possession)
                if team_stats[2] and team_stats[2] > 0:
                    features['team_pace'] = features['team_plays_per_game'] / (team_stats[2] / 60.0)

            # Get opponent defensive rankings
            opp_defense = conn.execute(
                """SELECT
                       AVG(passing_yards_allowed) as pass_yds_allowed,
                       AVG(rushing_yards_allowed) as rush_yds_allowed,
                       AVG(points_allowed) as pts_allowed
                   FROM team_defense
                   WHERE team = ? AND season = ? AND week < ?
                   AND week >= MAX(1, ? - 4)""",
                (opponent, season, week, week)
            ).fetchone()

            if opp_defense and opp_defense[0] is not None:
                # Convert to rankings (lower is better defense)
                # These would ideally be percentile ranks across all teams
                features['opp_pass_defense_rank'] = min(1.0, float(opp_defense[0] or 250) / 300)
                features['opp_rush_defense_rank'] = min(1.0, float(opp_defense[1] or 120) / 150)

            # Position-specific correlations
            if position == 'QB':
                # QB benefits from high pass rate and bad pass defense
                features['game_script_correlation'] = (
                    features['team_pass_rate'] * features['opp_pass_defense_rank']
                )

                # QB-WR stack correlation (how concentrated are targets)
                target_concentration = conn.execute(
                    """SELECT
                           MAX(targets * 1.0 / team_targets) as max_target_share,
                           COUNT(DISTINCT player_id) as num_receivers
                       FROM (
                           SELECT p.player_id, p.targets,
                                  SUM(p.targets) OVER (PARTITION BY p.team) as team_targets
                           FROM player_stats p
                           WHERE p.team = ? AND p.season = ?
                           AND p.week >= MAX(1, ? - 4) AND p.week < ?
                           AND p.position IN ('WR', 'TE')
                       )""",
                    (team, season, week, week)
                ).fetchone()

                if target_concentration and target_concentration[0]:
                    features['target_share_concentration'] = float(target_concentration[0] or 0.25)
                    features['stack_correlation_strength'] = features['target_share_concentration'] * 2.0

            elif position == 'WR':
                # WR correlates with QB and passing game script
                features['game_script_correlation'] = (
                    features['team_pass_rate'] * features['opp_pass_defense_rank'] * 0.8
                )
                features['stack_correlation_strength'] = features['team_pass_rate']

                # Get WR-specific target share
                wr_targets = conn.execute(
                    """SELECT
                           AVG(targets) as avg_targets,
                           AVG(red_zone_targets) as rz_targets
                       FROM player_stats
                       WHERE player_id = ? AND season = ?
                       AND week >= MAX(1, ? - 4) AND week < ?""",
                    (player_id, season, week, week)  # Note: player_id would need to be passed
                ).fetchone()

                if wr_targets and wr_targets[0]:
                    features['red_zone_opportunities'] = float(wr_targets[1] or 0) / max(float(wr_targets[0] or 1), 1)

            elif position == 'RB':
                # RB benefits from positive game script (winning)
                features['game_script_correlation'] = (
                    (1 - features['team_pass_rate']) * features['opp_rush_defense_rank']
                )

                # RB-DEF negative correlation (good DEF = more rushing late)
                features['stack_correlation_strength'] = 1 - features['team_pass_rate']

            elif position == 'TE':
                # TE benefits from red zone and moderate pass rate
                features['game_script_correlation'] = (
                    features['team_pass_rate'] * 0.7 + 0.3
                )

                # TEs often benefit in games with injured WRs
                features['stack_correlation_strength'] = 0.5  # Moderate correlation

            elif position in ['DST', 'DEF']:
                # DEF benefits from negative game script for opponent
                features['game_script_correlation'] = 1 - features['opp_pass_defense_rank']
                features['stack_correlation_strength'] = 0.3  # Weak correlation with offense

        except Exception as e:
            logger.warning(f"Error getting correlation features: {e}")
        finally:
            conn.close()

        return features

    def get_stacking_features(
        self,
        qb_id: int,
        wr_id: int,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Get features for QB-WR stacking correlation.

        Args:
            qb_id: Quarterback player ID
            wr_id: Wide receiver player ID
            season: NFL season
            week: NFL week

        Returns:
            Dictionary of stacking features
        """
        features = {
            'historical_correlation': 0.0,
            'target_share_to_qb': 0.0,
            'td_share_to_qb': 0.0,
            'yards_per_target': 0.0,
            'stack_ceiling_boost': 0.0,
        }

        conn = get_db_connection(self.db_path)

        try:
            # Get historical correlation between QB and WR
            correlation_data = conn.execute(
                """SELECT
                       CORR(q.fantasy_points, w.fantasy_points) as correlation,
                       AVG(w.targets * 1.0 / q.pass_attempts) as target_share,
                       AVG(w.receiving_tds * 1.0 / NULLIF(q.passing_tds, 0)) as td_share,
                       AVG(w.receiving_yards * 1.0 / NULLIF(w.targets, 0)) as yds_per_target
                   FROM player_stats q
                   JOIN player_stats w ON q.game_id = w.game_id
                   WHERE q.player_id = ? AND w.player_id = ?
                   AND q.season = ? AND q.week < ?
                   AND q.week >= MAX(1, ? - 8)""",
                (qb_id, wr_id, season, week, week)
            ).fetchone()

            if correlation_data and correlation_data[0] is not None:
                features['historical_correlation'] = float(correlation_data[0] or 0)
                features['target_share_to_qb'] = float(correlation_data[1] or 0)
                features['td_share_to_qb'] = float(correlation_data[2] or 0)
                features['yards_per_target'] = float(correlation_data[3] or 0)

                # Calculate ceiling boost from stacking
                # Higher correlation + high TD share = bigger ceiling boost
                features['stack_ceiling_boost'] = (
                    features['historical_correlation'] * 0.5 +
                    features['td_share_to_qb'] * 0.3 +
                    min(features['target_share_to_qb'], 0.3) * 0.2
                )

        except Exception as e:
            logger.warning(f"Error getting stacking features: {e}")
        finally:
            conn.close()

        return features

    def get_game_environment_features(
        self,
        game_id: int,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Get game environment features that affect all players.

        Args:
            game_id: Game ID
            season: NFL season
            week: NFL week

        Returns:
            Dictionary of game environment features
        """
        features = {
            'total_line': 0.0,
            'spread': 0.0,
            'implied_team_total': 0.0,
            'is_primetime': 0.0,
            'is_division_game': 0.0,
            'playoff_implications': 0.0,
            'weather_impact': 0.0,
        }

        conn = get_db_connection(self.db_path)

        try:
            # Get game betting lines and environment
            game_info = conn.execute(
                """SELECT
                       g.total_line, g.spread,
                       g.home_team, g.away_team,
                       g.game_time, g.is_dome,
                       t1.division as home_div,
                       t2.division as away_div,
                       w.wind_speed, w.temperature
                   FROM games g
                   LEFT JOIN teams t1 ON g.home_team = t1.team_abbr
                   LEFT JOIN teams t2 ON g.away_team = t2.team_abbr
                   LEFT JOIN weather w ON g.game_id = w.game_id
                   WHERE g.game_id = ?""",
                (game_id,)
            ).fetchone()

            if game_info:
                features['total_line'] = float(game_info[0] or 45.0)
                features['spread'] = abs(float(game_info[1] or 0))

                # Calculate implied team totals
                features['implied_team_total'] = (features['total_line'] - features['spread']) / 2

                # Check if primetime game (Monday, Thursday night, Sunday night)
                game_time = game_info[4]
                if game_time:
                    hour = datetime.fromisoformat(game_time).hour
                    if hour >= 20:  # 8 PM or later
                        features['is_primetime'] = 1.0

                # Division game
                if game_info[6] and game_info[7] and game_info[6] == game_info[7]:
                    features['is_division_game'] = 1.0

                # Weather impact (for outdoor games)
                if not game_info[5]:  # Not a dome
                    wind = float(game_info[8] or 0)
                    temp = float(game_info[9] or 60)

                    # High wind or extreme temps affect scoring
                    if wind > 20:
                        features['weather_impact'] = min(1.0, (wind - 20) / 20)
                    if temp < 32 or temp > 90:
                        features['weather_impact'] = max(
                            features['weather_impact'],
                            abs(temp - 60) / 60
                        )

        except Exception as e:
            logger.warning(f"Error getting game environment features: {e}")
        finally:
            conn.close()

        return features
