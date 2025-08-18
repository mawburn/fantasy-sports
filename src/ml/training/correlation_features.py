"""Extract correlation features that capture player interactions.

This module enhances model training by capturing how players, defenses,
and coaching styles interact to affect fantasy production.

Key Correlations Modeled:
1. Teammate synergies (QB-WR stacks, offensive line impact)
2. Defensive matchups (scheme vulnerabilities, pace factors)
3. Coaching tendencies (play-calling, aggressiveness)
4. Game environment (weather, stadium, game script)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import get_db

logger = logging.getLogger(__name__)


class CorrelationFeatureExtractor:
    """Extract features that capture player interactions and correlations."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize with database connection.

        Args:
            db_session: Database session for queries
        """
        self.db = db_session or next(get_db())

    def extract_qb_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract QB features including teammate and opponent correlations.

        Args:
            player_id: QB player ID
            game_id: Target game ID
            lookback_weeks: Weeks of history to analyze

        Returns:
            Dictionary of correlation features
        """
        features = {}

        # Get game info
        game_info = self._get_game_info(game_id)
        if not game_info:
            return features

        team_id = self._get_player_team(player_id, game_info['game_date'])
        opp_team_id = self._get_opponent_team(team_id, game_id)

        # 1. TEAMMATE CORRELATIONS
        features.update(self._get_qb_teammate_features(
            team_id, game_info['game_date'], lookback_weeks
        ))

        # 2. DEFENSIVE MATCHUP FEATURES
        features.update(self._get_defensive_matchup_features(
            opp_team_id, 'QB', game_info['season'], game_info['week']
        ))

        # 3. COACHING STYLE FEATURES
        features.update(self._get_coaching_tendency_features(
            team_id, opp_team_id, game_info['season']
        ))

        # 4. GAME ENVIRONMENT CORRELATIONS
        features.update(self._get_game_environment_features(game_id))

        # 5. STACKING POTENTIAL
        features.update(self._get_stacking_features(
            player_id, team_id, game_info['game_date'], lookback_weeks
        ))

        return features

    def _get_qb_teammate_features(
        self,
        team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get features about QB's teammates."""
        features = {}

        # Get recent team performance
        query = text("""
            SELECT
                AVG(ps.rushing_yards) as team_rush_ypg,
                AVG(ps.receiving_yards) as team_rec_ypg,
                COUNT(DISTINCT CASE WHEN ps.receiving_yards > 75 THEN ps.player_id END) as num_viable_receivers,
                MAX(ps.targets) as max_target_share,
                AVG(CASE WHEN p.position = 'RB' THEN ps.targets ELSE 0 END) as rb_target_rate,
                AVG(CASE WHEN p.position = 'TE' THEN ps.targets ELSE 0 END) as te_target_rate
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE p.team_id = :team_id
            AND g.game_date >= :start_date
            AND g.game_date < :game_date
            AND g.game_finished = true
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        result = self.db.execute(query, {
            'team_id': team_id,
            'start_date': start_date,
            'game_date': game_date
        }).first()

        if result:
            features['team_rush_ypg'] = result.team_rush_ypg or 0
            features['team_rec_ypg'] = result.team_rec_ypg or 0
            features['num_viable_receivers'] = result.num_viable_receivers or 0
            features['max_target_concentration'] = result.max_target_share or 0
            features['rb_involvement_passing'] = result.rb_target_rate or 0
            features['te_involvement'] = result.te_target_rate or 0

        # Get offensive line strength (affects QB time in pocket)
        features['oline_ranking'] = self._get_oline_ranking(team_id)

        return features

    def _get_defensive_matchup_features(
        self,
        opp_team_id: int,
        position: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Get opponent defensive features specific to position."""
        features = {}

        # Get defensive stats vs position
        query = text("""
            SELECT
                qb_fantasy_points_allowed,
                rb_fantasy_points_allowed,
                wr_fantasy_points_allowed,
                te_fantasy_points_allowed,
                sacks,
                pressure_rate,
                completion_pct_allowed,
                yards_per_pass_allowed,
                third_down_pct_allowed,
                red_zone_pct_allowed,
                qb_def_rank,
                pass_def_rank,
                rush_def_rank
            FROM team_defensive_stats
            WHERE team_id = :team_id
            AND season = :season
            AND week < :week
            ORDER BY week DESC
            LIMIT 4
        """)

        results = self.db.execute(query, {
            'team_id': opp_team_id,
            'season': season,
            'week': week
        }).fetchall()

        if results:
            # Average recent defensive performance
            avg_stats = pd.DataFrame(results).mean()

            if position == 'QB':
                features['opp_qb_points_allowed'] = avg_stats['qb_fantasy_points_allowed']
                features['opp_pressure_rate'] = avg_stats['pressure_rate']
                features['opp_sacks_per_game'] = avg_stats['sacks']
                features['opp_completion_pct_allowed'] = avg_stats['completion_pct_allowed']
            elif position == 'RB':
                features['opp_rb_points_allowed'] = avg_stats['rb_fantasy_points_allowed']
                features['opp_rush_def_rank'] = avg_stats['rush_def_rank']
            elif position == 'WR':
                features['opp_wr_points_allowed'] = avg_stats['wr_fantasy_points_allowed']
                features['opp_pass_yards_allowed'] = avg_stats['yards_per_pass_allowed']
            elif position == 'TE':
                features['opp_te_points_allowed'] = avg_stats['te_fantasy_points_allowed']

            # General defensive strength
            features['opp_third_down_defense'] = avg_stats['third_down_pct_allowed']
            features['opp_red_zone_defense'] = avg_stats['red_zone_pct_allowed']
            features['opp_overall_def_rank'] = (avg_stats['qb_def_rank'] + avg_stats['pass_def_rank'] + avg_stats['rush_def_rank']) / 3

        return features

    def _get_coaching_tendency_features(
        self,
        team_id: int,
        opp_team_id: int,
        season: int
    ) -> Dict[str, float]:
        """Get coaching style and tendency features."""
        features = {}

        # Get offensive coaching tendencies
        query = text("""
            SELECT
                pass_rate_overall,
                pass_rate_red_zone,
                pass_rate_first_down,
                pace_of_play,
                fourth_down_aggressiveness
            FROM coaching_staff
            WHERE team_id = :team_id
            AND season = :season
        """)

        result = self.db.execute(query, {
            'team_id': team_id,
            'season': season
        }).first()

        if result:
            features['team_pass_rate'] = result.pass_rate_overall or 0.5
            features['team_rz_pass_rate'] = result.pass_rate_red_zone or 0.5
            features['team_first_down_pass_rate'] = result.pass_rate_first_down or 0.5
            features['team_pace'] = result.pace_of_play or 30
            features['team_aggressiveness'] = result.fourth_down_aggressiveness or 0.2

        # Get defensive coaching tendencies
        opp_result = self.db.execute(query, {
            'team_id': opp_team_id,
            'season': season
        }).first()

        if opp_result:
            features['opp_def_pace'] = opp_result.pace_of_play or 30
            # Defensive coaches affect offensive production
            features['pace_differential'] = features.get('team_pace', 30) - features.get('opp_def_pace', 30)

        return features

    def _get_game_environment_features(self, game_id: int) -> Dict[str, float]:
        """Get game environment correlation features."""
        features = {}

        query = text("""
            SELECT
                g.weather_temperature,
                g.weather_wind_speed,
                g.roof_type,
                vl.total as vegas_total,
                vl.spread,
                vl.home_team_total,
                vl.away_team_total
            FROM games g
            LEFT JOIN vegas_lines vl ON g.id = vl.game_id
            WHERE g.id = :game_id
        """)

        result = self.db.execute(query, {'game_id': game_id}).first()

        if result:
            # Weather impacts
            features['temperature'] = result.weather_temperature or 70
            features['wind_speed'] = result.weather_wind_speed or 0
            features['is_dome'] = 1 if result.roof_type == 'dome' else 0

            # Vegas correlations (high totals = more fantasy points)
            features['vegas_total'] = result.vegas_total or 45
            features['vegas_spread'] = abs(result.spread) if result.spread else 0
            features['team_implied_total'] = result.home_team_total or 22.5

            # Game script indicators
            features['expected_shootout'] = 1 if features['vegas_total'] > 50 else 0
            features['expected_blowout'] = 1 if features['vegas_spread'] > 10 else 0

        return features

    def _get_stacking_features(
        self,
        qb_id: int,
        team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get QB-receiver stacking correlation features."""
        features = {}

        # Find QB's favorite targets
        query = text("""
            SELECT
                p.position,
                p.display_name,
                AVG(ps.targets) as avg_targets,
                AVG(ps.target_share) as avg_target_share,
                AVG(ps.receiving_yards) as avg_rec_yards,
                COUNT(*) as games_together
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE p.team_id = :team_id
            AND p.position IN ('WR', 'TE')
            AND g.game_date >= :start_date
            AND g.game_date < :game_date
            AND EXISTS (
                SELECT 1 FROM player_stats qb_ps
                WHERE qb_ps.game_id = ps.game_id
                AND qb_ps.player_id = :qb_id
            )
            GROUP BY p.id, p.position, p.display_name
            ORDER BY avg_targets DESC
            LIMIT 3
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        results = self.db.execute(query, {
            'team_id': team_id,
            'qb_id': qb_id,
            'start_date': start_date,
            'game_date': game_date
        }).fetchall()

        if results:
            # WR1/WR2/TE1 target distribution
            for idx, receiver in enumerate(results):
                prefix = f"top_target_{idx+1}"
                features[f"{prefix}_share"] = receiver.avg_target_share or 0
                features[f"{prefix}_yards"] = receiver.avg_rec_yards or 0
                features[f"{prefix}_consistency"] = min(receiver.games_together / lookback_weeks, 1.0)

        return features

    def _get_game_info(self, game_id: int) -> Optional[Dict]:
        """Get basic game information."""
        query = text("""
            SELECT game_date, season, week, home_team_id, away_team_id
            FROM games
            WHERE id = :game_id
        """)

        result = self.db.execute(query, {'game_id': game_id}).first()
        if result:
            # Ensure game_date is a datetime object
            game_date = result.game_date
            if isinstance(game_date, str):
                try:
                    game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                except ValueError:
                    try:
                        game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
                    except ValueError:
                        logger.warning(f"Could not parse game_date: {game_date}")
                        return None

            return {
                'game_date': game_date,
                'season': result.season,
                'week': result.week,
                'home_team_id': result.home_team_id,
                'away_team_id': result.away_team_id
            }
        return None

    def _get_player_team(self, player_id: int, game_date: datetime) -> Optional[int]:
        """Get player's team at a specific date."""
        query = text("""
            SELECT team_id FROM players WHERE id = :player_id
        """)
        result = self.db.execute(query, {'player_id': player_id}).first()
        return result.team_id if result else None

    def _get_opponent_team(self, team_id: int, game_id: int) -> Optional[int]:
        """Get opponent team for a game."""
        query = text("""
            SELECT
                CASE
                    WHEN home_team_id = :team_id THEN away_team_id
                    ELSE home_team_id
                END as opponent_id
            FROM games
            WHERE id = :game_id
        """)
        result = self.db.execute(query, {'team_id': team_id, 'game_id': game_id}).first()
        return result.opponent_id if result else None

    def _get_oline_ranking(self, team_id: int) -> float:
        """Get offensive line ranking (simplified for now)."""
        # TODO: Implement proper O-line rankings from PFF or similar
        # For now, use sacks allowed as proxy
        query = text("""
            SELECT AVG(sacks_allowed) as avg_sacks
            FROM (
                SELECT COUNT(*) as sacks_allowed
                FROM play_by_play
                WHERE posteam = (SELECT team_abbr FROM teams WHERE id = :team_id)
                AND sack = true
                GROUP BY game_id
                ORDER BY game_id DESC
                LIMIT 4
            ) recent_games
        """)

        result = self.db.execute(query, {'team_id': team_id}).first()
        if result and result.avg_sacks:
            # Convert to ranking (lower sacks = better ranking)
            return max(1, min(32, 16 - (result.avg_sacks - 2.5) * 4))
        return 16  # Default to average

    def extract_all_correlation_features(
        self,
        player_id: int,
        game_id: int,
        position: str
    ) -> Dict[str, float]:
        """Extract correlation features for any position.

        Args:
            player_id: Player ID
            game_id: Game ID
            position: Player position

        Returns:
            Dictionary of all correlation features
        """
        if position == 'QB':
            return self.extract_qb_correlation_features(player_id, game_id)
        elif position == 'RB':
            return self.extract_rb_correlation_features(player_id, game_id)
        elif position == 'WR':
            return self.extract_wr_correlation_features(player_id, game_id)
        elif position == 'TE':
            return self.extract_te_correlation_features(player_id, game_id)
        elif position == 'DEF':
            return self.extract_def_correlation_features(player_id, game_id)
        else:
            logger.warning(f"Unknown position: {position}")
            return {}

    def extract_rb_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract RB-specific correlation features."""
        features = {}

        game_info = self._get_game_info(game_id)
        if not game_info:
            return features

        team_id = self._get_player_team(player_id, game_info['game_date'])
        opp_team_id = self._get_opponent_team(team_id, game_id)

        # RB-specific correlations
        features.update(self._get_rb_workload_features(
            player_id, team_id, game_info['game_date'], lookback_weeks
        ))

        # Defensive matchup
        features.update(self._get_defensive_matchup_features(
            opp_team_id, 'RB', game_info['season'], game_info['week']
        ))

        # Game script expectations (RBs benefit from leads)
        features.update(self._get_game_script_features(team_id, game_id))

        # Coaching tendencies
        features.update(self._get_coaching_tendency_features(
            team_id, opp_team_id, game_info['season']
        ))

        return features

    def _get_rb_workload_features(
        self,
        player_id: int,
        team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get RB workload and usage features."""
        features = {}

        query = text("""
            SELECT
                AVG(ps.rushing_attempts) as avg_carries,
                AVG(ps.targets) as avg_targets,
                AVG(ps.rushing_attempts + ps.targets) as avg_touches,
                AVG(CASE WHEN ps.rushing_attempts > 15 THEN 1 ELSE 0 END) as workhorse_rate,
                AVG(ps.rushing_yards) as avg_rush_yards,
                SUM(ps.rushing_tds) as recent_rush_tds
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.id
            WHERE ps.player_id = :player_id
            AND g.game_date >= :start_date
            AND g.game_date < :game_date
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        result = self.db.execute(query, {
            'player_id': player_id,
            'start_date': start_date,
            'game_date': game_date
        }).first()

        if result:
            features['rb_avg_touches'] = result.avg_touches or 0
            features['rb_workhorse_rate'] = result.workhorse_rate or 0
            features['rb_pass_involvement'] = result.avg_targets or 0
            features['rb_td_regression'] = result.recent_rush_tds or 0

        return features

    def _get_game_script_features(self, team_id: int, game_id: int) -> Dict[str, float]:
        """Get expected game script features."""
        features = {}

        query = text("""
            SELECT
                vl.spread,
                vl.home_team_total,
                vl.away_team_total,
                g.home_team_id
            FROM games g
            LEFT JOIN vegas_lines vl ON g.id = vl.game_id
            WHERE g.id = :game_id
        """)

        result = self.db.execute(query, {'game_id': game_id}).first()

        if result:
            is_home = result.home_team_id == team_id
            team_total = result.home_team_total if is_home else result.away_team_total
            spread = result.spread if is_home else -result.spread

            features['expected_team_total'] = team_total or 22.5
            features['expected_point_differential'] = spread or 0
            features['expected_positive_script'] = 1 if spread < -3 else 0
            features['expected_negative_script'] = 1 if spread > 3 else 0

        return features

    def extract_wr_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract WR-specific correlation features."""
        features = {}

        game_info = self._get_game_info(game_id)
        if not game_info:
            return features

        team_id = self._get_player_team(player_id, game_info['game_date'])
        opp_team_id = self._get_opponent_team(team_id, game_id)

        # WR target competition
        features.update(self._get_target_competition_features(
            player_id, team_id, game_info['game_date'], lookback_weeks
        ))

        # QB connection strength
        features.update(self._get_qb_connection_features(
            player_id, team_id, game_info['game_date'], lookback_weeks
        ))

        # Defensive matchup
        features.update(self._get_defensive_matchup_features(
            opp_team_id, 'WR', game_info['season'], game_info['week']
        ))

        return features

    def _get_target_competition_features(
        self,
        player_id: int,
        team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get target competition features for WR/TE."""
        features = {}

        query = text("""
            SELECT
                p.id,
                p.display_name,
                AVG(ps.targets) as avg_targets,
                AVG(ps.target_share) as avg_share
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE p.team_id = :team_id
            AND p.position IN ('WR', 'TE')
            AND g.game_date >= :start_date
            AND g.game_date < :game_date
            GROUP BY p.id, p.display_name
            ORDER BY avg_targets DESC
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        results = self.db.execute(query, {
            'team_id': team_id,
            'start_date': start_date,
            'game_date': game_date
        }).fetchall()

        if results:
            total_targets = sum(r.avg_targets for r in results)
            player_share = next((r.avg_share for r in results if r.id == player_id), 0)

            features['wr_target_share'] = player_share
            features['wr_target_competition'] = len(results) - 1
            features['wr_target_concentration'] = max(r.avg_share for r in results) if results else 0

        return features

    def _get_qb_connection_features(
        self,
        player_id: int,
        team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get QB-receiver connection features."""
        features = {}

        query = text("""
            SELECT
                qb.display_name,
                COUNT(*) as games_together,
                AVG(wr_ps.targets) as avg_targets_from_qb,
                AVG(wr_ps.receiving_yards) as avg_yards_from_qb,
                AVG(qb_ps.passing_yards) as qb_avg_pass_yards
            FROM player_stats wr_ps
            JOIN games g ON wr_ps.game_id = g.id
            JOIN player_stats qb_ps ON qb_ps.game_id = g.id
            JOIN players qb ON qb_ps.player_id = qb.id
            WHERE wr_ps.player_id = :player_id
            AND qb.team_id = :team_id
            AND qb.position = 'QB'
            AND g.game_date >= :start_date
            AND g.game_date < :game_date
            GROUP BY qb.id, qb.display_name
            ORDER BY games_together DESC
            LIMIT 1
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        result = self.db.execute(query, {
            'player_id': player_id,
            'team_id': team_id,
            'start_date': start_date,
            'game_date': game_date
        }).first()

        if result:
            features['qb_connection_games'] = result.games_together or 0
            features['qb_connection_targets'] = result.avg_targets_from_qb or 0
            features['qb_passing_volume'] = result.qb_avg_pass_yards or 0
            features['qb_wr_rapport'] = (result.avg_yards_from_qb or 0) / max(result.qb_avg_pass_yards, 1)

        return features

    def extract_te_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract TE-specific correlation features."""
        # TEs are similar to WRs but with some unique aspects
        features = self.extract_wr_correlation_features(player_id, game_id, lookback_weeks)

        # TE-specific: Red zone usage
        game_info = self._get_game_info(game_id)
        if game_info:
            features.update(self._get_red_zone_features(
                player_id, game_info['game_date'], lookback_weeks
            ))

        return features

    def _get_red_zone_features(
        self,
        player_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get red zone usage features."""
        features = {}

        query = text("""
            SELECT
                COUNT(*) as red_zone_targets,
                SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END) as red_zone_tds
            FROM play_by_play pbp
            WHERE receiver_player_id = :player_id
            AND yardline_100 <= 20
            AND game_date >= :start_date
            AND game_date < :game_date
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        result = self.db.execute(query, {
            'player_id': player_id,
            'start_date': start_date,
            'game_date': game_date
        }).first()

        if result:
            features['red_zone_involvement'] = result.red_zone_targets or 0
            features['red_zone_efficiency'] = (result.red_zone_tds or 0) / max(result.red_zone_targets, 1)

        return features

    def extract_def_correlation_features(
        self,
        team_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract DEF-specific correlation features."""
        features = {}

        game_info = self._get_game_info(game_id)
        if not game_info:
            return features

        opp_team_id = self._get_opponent_team(team_id, game_id)

        # Opponent offensive strength
        features.update(self._get_opponent_offensive_features(
            opp_team_id, game_info['game_date'], lookback_weeks
        ))

        # Game environment (affects defensive scoring)
        features.update(self._get_game_environment_features(game_id))

        return features

    def _get_opponent_offensive_features(
        self,
        opp_team_id: int,
        game_date: datetime,
        lookback_weeks: int
    ) -> Dict[str, float]:
        """Get opponent offensive features for defense."""
        features = {}

        query = text("""
            SELECT
                AVG(home_score + away_score) as avg_total_points,
                AVG(CASE
                    WHEN home_team_id = :team_id THEN home_score
                    ELSE away_score
                END) as avg_team_points,
                COUNT(CASE WHEN turnovers > 2 THEN 1 END) as high_turnover_games
            FROM games g
            WHERE (home_team_id = :team_id OR away_team_id = :team_id)
            AND game_date >= :start_date
            AND game_date < :game_date
            AND game_finished = true
        """)

        # Ensure game_date is a date object for timedelta operations
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse game_date: {game_date}")
                return {}
        start_date = game_date - timedelta(weeks=lookback_weeks)
        result = self.db.execute(query, {
            'team_id': opp_team_id,
            'start_date': start_date,
            'game_date': game_date
        }).first()

        if result:
            features['opp_scoring_rate'] = result.avg_team_points or 20
            features['opp_turnover_prone'] = result.high_turnover_games or 0
            features['expected_game_flow'] = result.avg_total_points or 45

        return features
