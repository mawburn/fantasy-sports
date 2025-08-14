"""Feature extraction pipeline for ML models."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Game, Player, PlayerStats, Team

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features for ML models from NFL data."""

    def __init__(self, db_session: Session | None = None):
        """Initialize feature extractor."""
        self.db = db_session or next(get_db())

    def extract_player_features(
        self,
        player_id: int,
        target_game_date: datetime,
        lookback_games: int = 5,
        season_stats: bool = True,
    ) -> dict:
        """Extract features for a specific player for a target game.

        Args:
            player_id: Database ID of the player
            target_game_date: Date of the game to predict for
            lookback_games: Number of recent games to include in rolling stats
            season_stats: Whether to include season-to-date statistics

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Get player info
        player = self.db.query(Player).filter(Player.id == player_id).first()
        if not player:
            msg = f"Player with ID {player_id} not found"
            raise ValueError(msg)

        # Basic player features
        features.update(self._get_basic_player_features(player))

        # Recent performance features
        recent_stats = self._get_recent_stats(player_id, target_game_date, lookback_games)
        features.update(self._calculate_rolling_stats(recent_stats, "recent"))

        # Season-to-date features
        if season_stats:
            season_year = target_game_date.year
            season_stats_data = self._get_season_stats(player_id, season_year, target_game_date)
            features.update(self._calculate_season_stats(season_stats_data))

        # Opponent-based features
        opponent_features = self._get_opponent_features(player_id, target_game_date)
        features.update(opponent_features)

        # Situational features
        situational_features = self._get_situational_features(target_game_date)
        features.update(situational_features)

        return features

    def _get_basic_player_features(self, player: Player) -> dict:
        """Extract basic player demographic and physical features."""
        return {
            "position_QB": 1 if player.position == "QB" else 0,
            "position_RB": 1 if player.position == "RB" else 0,
            "position_WR": 1 if player.position == "WR" else 0,
            "position_TE": 1 if player.position == "TE" else 0,
            "position_K": 1 if player.position == "K" else 0,
            "position_DEF": 1 if player.position == "DEF" else 0,
            "height_inches": player.height or 0,
            "weight_lbs": player.weight or 0,
            "age": player.age or 0,
            "years_exp": player.years_exp or 0,
            "is_rookie": 1 if (player.years_exp or 0) == 0 else 0,
        }

    def _get_recent_stats(
        self, player_id: int, target_date: datetime, num_games: int
    ) -> pd.DataFrame:
        """Get recent game statistics for a player."""
        stats_query = (
            self.db.query(PlayerStats)
            .join(Game)
            .filter(
                PlayerStats.player_id == player_id,
                Game.game_date < target_date,
                Game.game_finished,
            )
            .order_by(Game.game_date.desc())
            .limit(num_games)
        )

        stats = stats_query.all()

        if not stats:
            return pd.DataFrame()

        # Convert to DataFrame for easier manipulation
        data = []
        for stat in stats:
            data.append(
                {
                    "fantasy_points": stat.fantasy_points or 0,
                    "fantasy_points_ppr": stat.fantasy_points_ppr or 0,
                    "passing_yards": stat.passing_yards,
                    "passing_tds": stat.passing_tds,
                    "passing_interceptions": stat.passing_interceptions,
                    "passing_attempts": stat.passing_attempts,
                    "passing_completions": stat.passing_completions,
                    "rushing_yards": stat.rushing_yards,
                    "rushing_tds": stat.rushing_tds,
                    "rushing_attempts": stat.rushing_attempts,
                    "receiving_yards": stat.receiving_yards,
                    "receiving_tds": stat.receiving_tds,
                    "receptions": stat.receptions,
                    "targets": stat.targets,
                    "fumbles_lost": stat.fumbles_lost,
                    "two_point_conversions": stat.two_point_conversions,
                }
            )

        return pd.DataFrame(data)

    def _calculate_rolling_stats(self, df: pd.DataFrame, prefix: str) -> dict:
        """Calculate rolling statistics from recent games."""
        if df.empty:
            return {f"{prefix}_games_played": 0}

        features = {f"{prefix}_games_played": len(df)}

        # Core fantasy metrics
        numeric_cols = [
            "fantasy_points",
            "fantasy_points_ppr",
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receiving_yards",
            "receiving_tds",
            "receptions",
            "targets",
        ]

        for col in numeric_cols:
            if col in df.columns:
                values = df[col].fillna(0)
                features[f"{prefix}_{col}_avg"] = float(values.mean())
                features[f"{prefix}_{col}_sum"] = float(values.sum())
                features[f"{prefix}_{col}_max"] = float(values.max()) if len(values) > 0 else 0
                features[f"{prefix}_{col}_std"] = float(values.std()) if len(values) > 1 else 0

        # Derived metrics
        if "passing_attempts" in df.columns and "passing_completions" in df.columns:
            completion_pct = df["passing_completions"] / df["passing_attempts"].replace(0, np.nan)
            features[f"{prefix}_completion_pct"] = (
                float(completion_pct.mean()) if not completion_pct.isna().all() else 0
            )

        if "targets" in df.columns and "receptions" in df.columns:
            catch_rate = df["receptions"] / df["targets"].replace(0, np.nan)
            features[f"{prefix}_catch_rate"] = (
                float(catch_rate.mean()) if not catch_rate.isna().all() else 0
            )

        # Consistency metrics (coefficient of variation)
        if len(df) > 1:
            fp_cv = (
                df["fantasy_points"].std() / df["fantasy_points"].mean()
                if df["fantasy_points"].mean() != 0
                else 0
            )
            features[f"{prefix}_fantasy_points_consistency"] = 1.0 / (
                1.0 + fp_cv
            )  # Higher is more consistent

        return features

    def _get_season_stats(self, player_id: int, season: int, before_date: datetime) -> pd.DataFrame:
        """Get season-to-date statistics."""
        stats_query = (
            self.db.query(PlayerStats)
            .join(Game)
            .filter(
                PlayerStats.player_id == player_id,
                Game.season == season,
                Game.game_date < before_date,
                Game.game_finished,
            )
        )

        stats = stats_query.all()

        if not stats:
            return pd.DataFrame()

        data = []
        for stat in stats:
            data.append(
                {
                    "fantasy_points": stat.fantasy_points or 0,
                    "fantasy_points_ppr": stat.fantasy_points_ppr or 0,
                    "passing_yards": stat.passing_yards,
                    "passing_tds": stat.passing_tds,
                    "passing_interceptions": stat.passing_interceptions,
                    "rushing_yards": stat.rushing_yards,
                    "rushing_tds": stat.rushing_tds,
                    "receiving_yards": stat.receiving_yards,
                    "receiving_tds": stat.receiving_tds,
                    "receptions": stat.receptions,
                    "targets": stat.targets,
                }
            )

        return pd.DataFrame(data)

    def _calculate_season_stats(self, df: pd.DataFrame) -> dict:
        """Calculate season-to-date statistics."""
        if df.empty:
            return {"season_games_played": 0}

        return self._calculate_rolling_stats(df, "season")

    def _get_opponent_features(self, player_id: int, target_date: datetime) -> dict:
        """Extract opponent-specific features."""
        # Get the target game
        target_game = (
            self.db.query(Game)
            .join(Team, (Team.id == Game.home_team_id) | (Team.id == Game.away_team_id))
            .join(Player, Player.team_id == Team.id)
            .filter(
                Player.id == player_id,
                Game.game_date >= target_date - timedelta(days=1),
                Game.game_date <= target_date + timedelta(days=1),
            )
            .first()
        )

        if not target_game:
            return {"has_opponent_data": 0}

        # Determine opponent team
        player = self.db.query(Player).filter(Player.id == player_id).first()
        if not player or not player.team_id:
            return {"has_opponent_data": 0}

        if player.team_id == target_game.home_team_id:
            opponent_team_id = target_game.away_team_id
            is_home_game = 0
        else:
            opponent_team_id = target_game.home_team_id
            is_home_game = 1

        # Basic game context
        features = {
            "has_opponent_data": 1,
            "is_home_game": is_home_game,
            "game_week": target_game.week,
            "is_divisional_game": self._is_divisional_matchup(player.team_id, opponent_team_id),
        }

        # Opponent defensive stats (simplified - would need defensive stats in real implementation)
        # For now, just add placeholders
        features.update(
            {
                "opponent_def_rank": 15,  # Placeholder - would calculate from actual data
                "opponent_points_allowed_avg": 22.0,  # Placeholder
            }
        )

        return features

    def _is_divisional_matchup(self, team1_id: int, team2_id: int) -> int:
        """Check if two teams are in the same division."""
        team1 = self.db.query(Team).filter(Team.id == team1_id).first()
        team2 = self.db.query(Team).filter(Team.id == team2_id).first()

        if not team1 or not team2:
            return 0

        return (
            1 if (team1.conference == team2.conference and team1.division == team2.division) else 0
        )

    def _get_situational_features(self, target_date: datetime) -> dict:
        """Extract situational features like weather, time of year, etc."""
        # Get week of season (approximate)
        september_start = datetime(target_date.year, 9, 1)
        days_since_season_start = (target_date - september_start).days
        week_of_season = max(1, min(18, (days_since_season_start // 7) + 1))

        features = {
            "week_of_season": week_of_season,
            "is_early_season": 1 if week_of_season <= 4 else 0,
            "is_mid_season": 1 if 5 <= week_of_season <= 12 else 0,
            "is_late_season": 1 if week_of_season >= 13 else 0,
            "is_playoff_push": 1 if week_of_season >= 15 else 0,
        }

        # Day of week features
        day_of_week = target_date.weekday()  # 0=Monday, 6=Sunday
        features.update(
            {
                "is_sunday_game": 1 if day_of_week == 6 else 0,
                "is_monday_game": 1 if day_of_week == 0 else 0,
                "is_thursday_game": 1 if day_of_week == 3 else 0,
            }
        )

        return features

    def extract_team_features(self, team_id: int) -> dict:
        """Extract team-level features."""
        team = self.db.query(Team).filter(Team.id == team_id).first()
        if not team:
            return {}

        # Basic team features
        features = {
            "team_conference_AFC": 1 if team.conference == "AFC" else 0,
            "team_conference_NFC": 1 if team.conference == "NFC" else 0,
            "team_division_North": 1 if team.division == "North" else 0,
            "team_division_South": 1 if team.division == "South" else 0,
            "team_division_East": 1 if team.division == "East" else 0,
            "team_division_West": 1 if team.division == "West" else 0,
        }

        # Team performance metrics (would need team stats table in real implementation)
        # For now, add placeholders
        features.update(
            {
                "team_offensive_rank": 15,  # Placeholder
                "team_defensive_rank": 15,  # Placeholder
                "team_wins": 5,  # Placeholder
                "team_losses": 5,  # Placeholder
            }
        )

        return features

    def extract_slate_features(
        self, game_date: datetime, positions: list[str] | None = None
    ) -> pd.DataFrame:
        """Extract features for all players in a game slate.

        Args:
            game_date: Date of the games
            positions: List of positions to include (default: all)

        Returns:
            DataFrame with features for all eligible players
        """
        if positions is None:
            positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

        # Get all games for the target date
        games = (
            self.db.query(Game)
            .filter(Game.game_date >= game_date, Game.game_date < game_date + timedelta(days=1))
            .all()
        )

        if not games:
            logger.warning(f"No games found for date {game_date}")
            return pd.DataFrame()

        # Get all players from teams playing on this date
        team_ids = []
        for game in games:
            team_ids.extend([game.home_team_id, game.away_team_id])

        players = (
            self.db.query(Player)
            .filter(
                Player.team_id.in_(team_ids),
                Player.position.in_(positions),
                Player.status == "Active",
            )
            .all()
        )

        logger.info(f"Extracting features for {len(players)} players")

        # Extract features for each player
        features_list = []
        for player in players:
            try:
                player_features = self.extract_player_features(
                    player.id, game_date, lookback_games=5
                )
                player_features["player_id"] = player.id
                player_features["player_name"] = player.display_name
                player_features["position"] = player.position
                features_list.append(player_features)
            except Exception as e:
                logger.warning(f"Failed to extract features for player {player.display_name}: {e}")

        return pd.DataFrame(features_list)


def calculate_fantasy_points(stats: dict, scoring: str = "standard") -> float:
    """Calculate fantasy points from raw statistics.

    Args:
        stats: Dictionary of player statistics
        scoring: Scoring system ("standard", "ppr", "half_ppr")

    Returns:
        Fantasy points total
    """
    points = 0.0

    # Passing
    points += stats.get("passing_yards", 0) * 0.04  # 1 point per 25 yards
    points += stats.get("passing_tds", 0) * 4  # 4 points per TD
    points -= stats.get("passing_interceptions", 0) * 2  # -2 points per INT

    # Rushing
    points += stats.get("rushing_yards", 0) * 0.1  # 1 point per 10 yards
    points += stats.get("rushing_tds", 0) * 6  # 6 points per TD

    # Receiving
    points += stats.get("receiving_yards", 0) * 0.1  # 1 point per 10 yards
    points += stats.get("receiving_tds", 0) * 6  # 6 points per TD

    # Reception bonus
    if scoring == "ppr":
        points += stats.get("receptions", 0) * 1  # 1 point per reception
    elif scoring == "half_ppr":
        points += stats.get("receptions", 0) * 0.5  # 0.5 points per reception

    # Other
    points -= stats.get("fumbles_lost", 0) * 2  # -2 points per fumble lost
    points += stats.get("two_point_conversions", 0) * 2  # 2 points per 2PT conversion

    return round(points, 2)
