"""Feature extraction pipeline for ML models.

This file transforms raw NFL player/game statistics into ML-ready features.
Feature engineering is one of the most critical steps in machine learning,
often determining model success more than algorithm choice.

Feature Categories Created:

1. Basic Player Features: Demographics, physical attributes, position encoding
2. Recent Performance: Rolling statistics from last N games (form/momentum)
3. Season Statistics: Cumulative season performance up to prediction date
4. Opponent Features: Matchup difficulty, home/away, divisional games
5. Situational Features: Week of season, day of week, playoff implications
6. Derived Metrics: Efficiency stats (completion %, catch rate, consistency)

Key ML Concepts:

Feature Engineering: Creating informative input variables from raw data.
Better features often matter more than better algorithms.

Temporal Features: Time-based patterns (recent form, seasonal trends)
are crucial for sports prediction.

Categorical Encoding: Converting text categories (position, team) to
numbers using one-hot encoding (0/1 indicators).

Data Leakage Prevention: Only using information available before the
prediction target (no future data).

Feature Scaling: Will be handled downstream in data preparation.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Game, Player, PlayerStats, Team

logger = logging.getLogger(__name__)


def _ensure_datetime(date_input: datetime | str) -> datetime:
    """Convert string dates to datetime objects safely.

    This function handles the case where dates come from SQLite as strings
    instead of datetime objects. It ensures consistent datetime handling
    throughout the feature extraction pipeline.

    Args:
        date_input: Either a datetime object or string representation

    Returns:
        datetime object

    Raises:
        ValueError: If string cannot be parsed as a date
    """
    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, str):
        try:
            # Handle common SQLite datetime formats
            # Try parsing with microseconds first, then without
            try:
                return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            try:
                # Try date-only format
                return datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Unable to parse date string: {date_input}") from e
    else:
        raise TypeError(f"Expected datetime or string, got {type(date_input)}: {date_input}")


class FeatureExtractor:
    """Extract features for ML models from NFL data.

    This class is responsible for converting raw database records into
    the numerical feature vectors that ML models can understand.

    Core Principles:

    1. Temporal Consistency: Features only use data available before
       the prediction target (prevents data leakage)

    2. Position Agnostic: Same feature extraction process works for
       all positions, with position-specific adjustments handled downstream

    3. Missing Data Handling: Gracefully handles incomplete player histories
       and missing statistics

    4. Feature Versioning: Supports different feature sets for model
       experimentation and A/B testing

    5. Performance Optimization: Efficient database queries and pandas
       operations for production speed

    The class maintains a database session for querying and caches
    expensive computations where possible.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize feature extractor."""
        self.db = db_session or next(get_db())

    def extract_player_features(
        self,
        player_id: int,
        target_game_date: datetime | str,
        lookback_games: int = 5,
        season_stats: bool = True,
    ) -> dict:
        """Extract features for a specific player for a target game.

        This is the main feature extraction method that combines all feature
        categories into a single feature vector for a player-game combination.

        Feature Extraction Pipeline:
        1. Validate player exists and get basic demographics
        2. Extract recent performance (rolling averages, trends)
        3. Calculate season-to-date statistics (cumulative performance)
        4. Analyze opponent matchup difficulty and context
        5. Add situational features (week, weather, etc.)
        6. Combine all features into consistent dictionary

        Temporal Safety:
        All features use only data from BEFORE target_game_date to prevent
        data leakage. This ensures the model can work in production.

        Lookback Strategy:
        Recent games (default 5) capture current form and role changes.
        Season stats provide broader performance context.
        Balance between recency bias and sample size.

        Error Handling:
        - Missing player raises clear error
        - Missing stats default to reasonable values (0 or position averages)
        - Logs warnings for debugging without failing extraction

        Args:
            player_id: Database ID of the player
            target_game_date: Date of the game to predict for (prediction target)
            lookback_games: Number of recent games for rolling stats (5-10 typical)
            season_stats: Whether to include season-to-date statistics (usually True)

        Returns:
            Dictionary of extracted features with consistent naming convention
            Example keys: 'recent_fantasy_points_avg', 'season_targets_sum', 'is_home_game'
        """
        # Convert target_game_date to datetime if it's a string (handles SQLite string dates)
        target_game_date = _ensure_datetime(target_game_date)

        # Initialize feature dictionary that will be returned
        features = {}

        # Step 1: Get player info and validate existence
        player = self.db.query(Player).filter(Player.id == player_id).first()
        if not player:
            msg = f"Player with ID {player_id} not found in database"
            raise ValueError(msg)

        # Step 2: Extract basic player features (demographics, physical attributes)
        # These are static features that don't change game-to-game
        features.update(self._get_basic_player_features(player))

        # Step 3: Calculate recent performance features (rolling statistics)
        # Captures current form, role changes, and momentum
        recent_stats = self._get_recent_stats(player_id, target_game_date, lookback_games)
        features.update(self._calculate_rolling_stats(recent_stats, "recent"))

        # Step 4: Calculate season-to-date features (cumulative statistics)
        # Provides broader performance context and larger sample size
        if season_stats:
            season_year = target_game_date.year  # Current season
            season_stats_data = self._get_season_stats(player_id, season_year, target_game_date)
            features.update(self._calculate_season_stats(season_stats_data))

        # Step 5: Extract opponent-based features (matchup analysis)
        # Captures matchup difficulty, home/away advantage, divisional rivalry
        opponent_features = self._get_opponent_features(player_id, target_game_date)
        features.update(opponent_features)

        # Step 6: Add situational features (temporal and contextual)
        # Week of season, day of week, playoff implications, weather (future)
        situational_features = self._get_situational_features(target_game_date)
        features.update(situational_features)

        return features

    def _get_basic_player_features(self, player: Player) -> dict:
        """Extract basic player demographic and physical features.

        These features provide context about the player's physical profile
        and experience level. They're mostly static but important for:

        Position Encoding (One-Hot):
        - ML models need numerical inputs, not text categories
        - One-hot encoding creates binary features for each position
        - Allows models to learn position-specific patterns

        Physical Attributes:
        - Height/weight matter for certain positions (tall WRs, heavy RBs)
        - Age captures decline curves and experience effects
        - Years of experience vs rookie status (different skill development)

        Why Binary Encoding?
        - 'is_rookie' is clearer than 'years_exp == 0' for model interpretation
        - Binary features are easier for tree-based models to split on
        - More interpretable than continuous encoding for categorical concepts

        Missing Data Strategy:
        - Use 'or 0' to handle None values gracefully
        - Could use position averages instead of 0 for better defaults
        """
        return {
            # Position one-hot encoding (exactly one will be 1, others 0)
            "position_QB": 1 if player.position == "QB" else 0,
            "position_RB": 1 if player.position == "RB" else 0,
            "position_WR": 1 if player.position == "WR" else 0,
            "position_TE": 1 if player.position == "TE" else 0,
            "position_K": 1 if player.position == "K" else 0,  # Kicker
            "position_DEF": 1 if player.position == "DEF" else 0,  # Defense/ST
            # Physical attributes (handle missing data with defaults)
            "height_inches": player.height or 0,  # Could use position average instead
            "weight_lbs": player.weight or 0,  # Could use position average instead
            "age": player.age or 0,  # Could estimate from draft year
            # Experience features
            "years_exp": player.years_exp or 0,  # Years in NFL
            "is_rookie": 1 if (player.years_exp or 0) == 0 else 0,  # First year player
        }

    def _get_recent_stats(
        self, player_id: int, target_date: datetime | str, num_games: int
    ) -> pd.DataFrame:
        """Get recent game statistics for a player."""
        target_date = _ensure_datetime(target_date)
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
        """Calculate rolling statistics from recent games.

        Rolling statistics capture recent performance trends and current form.
        This is crucial for fantasy sports where recent performance often
        predicts future performance better than season averages.

        Statistical Measures Created:
        - avg: Central tendency (most important for predictions)
        - sum: Total production (useful for cumulative stats like yards)
        - max: Peak performance ceiling (upside potential)
        - std: Variability/consistency (lower std = more predictable)

        Why Multiple Statistics?
        - avg: Best single predictor of future performance
        - std: Helps assess risk/consistency
        - max: Shows ceiling potential for GPP lineups
        - sum: Shows total opportunity/usage

        Missing Data Handling:
        - fillna(0): Treat missing stats as 0 (conservative approach)
        - Could use position averages or forward-fill for better imputation

        Feature Naming Convention:
        {prefix}_{stat}_{measure} (e.g., 'recent_fantasy_points_avg')
        This creates consistent, descriptive feature names.

        CRITICAL FIX: Always return the same feature set regardless of data availability
        to ensure consistent feature vectors for ML model training.
        """
        # Start with games played count (important context)
        features = {f"{prefix}_games_played": len(df) if not df.empty else 0}

        # Core fantasy statistics to extract
        # These cover all major fantasy scoring categories across positions
        numeric_cols = [
            "fantasy_points",  # Primary target variable
            "fantasy_points_ppr",  # PPR scoring variant
            "passing_yards",  # QB primary stat
            "passing_tds",  # QB touchdown production
            "passing_interceptions",  # QB negative points
            "rushing_yards",  # RB/QB secondary stat
            "rushing_tds",  # RB/QB touchdown production
            "receiving_yards",  # WR/TE/RB primary stat
            "receiving_tds",  # WR/TE/RB touchdown production
            "receptions",  # PPR scoring component
            "targets",  # Opportunity metric
        ]

        # Calculate comprehensive statistics for each metric
        # ALWAYS generate all features, even if data is empty (use 0 as default)
        for col in numeric_cols:
            if not df.empty and col in df.columns:
                # Handle missing values by replacing with 0 (conservative)
                values = df[col].fillna(0)

                # Core statistical measures
                features[f"{prefix}_{col}_avg"] = float(values.mean())  # Average performance
                features[f"{prefix}_{col}_sum"] = float(values.sum())  # Total production
                features[f"{prefix}_{col}_max"] = (
                    float(values.max()) if len(values) > 0 else 0
                )  # Peak performance
                features[f"{prefix}_{col}_std"] = (
                    float(values.std()) if len(values) > 1 else 0
                )  # Consistency/volatility
            else:
                # No data available - use zeros for all statistics
                features[f"{prefix}_{col}_avg"] = 0.0
                features[f"{prefix}_{col}_sum"] = 0.0
                features[f"{prefix}_{col}_max"] = 0.0
                features[f"{prefix}_{col}_std"] = 0.0

        # Calculate derived efficiency metrics (often more predictive than raw totals)
        # ALWAYS add these features for consistency

        # QB Completion Percentage: Accuracy/efficiency metric
        if (
            not df.empty
            and "passing_attempts" in df.columns
            and "passing_completions" in df.columns
        ):
            # Use replace(0, np.nan) to avoid division by zero
            completion_pct = df["passing_completions"] / df["passing_attempts"].replace(0, np.nan)
            features[f"{prefix}_completion_pct"] = (
                float(completion_pct.mean()) if not completion_pct.isna().all() else 0
            )
        else:
            features[f"{prefix}_completion_pct"] = 0.0

        # Catch Rate: WR/TE/RB efficiency on targets
        if not df.empty and "targets" in df.columns and "receptions" in df.columns:
            # Higher catch rate = more reliable target for QB
            catch_rate = df["receptions"] / df["targets"].replace(0, np.nan)
            features[f"{prefix}_catch_rate"] = (
                float(catch_rate.mean()) if not catch_rate.isna().all() else 0
            )
        else:
            features[f"{prefix}_catch_rate"] = 0.0

        # Consistency metrics using Coefficient of Variation (CV)
        if not df.empty and len(df) > 1:  # Need at least 2 games for standard deviation
            # CV = std / mean (measures relative variability)
            fp_mean = df["fantasy_points"].mean()
            if fp_mean != 0:
                fp_cv = df["fantasy_points"].std() / fp_mean
                # Transform CV to consistency score (higher = more consistent)
                # Uses 1/(1+CV) so CV=0 gives consistency=1, high CV gives consistency near 0
                features[f"{prefix}_fantasy_points_consistency"] = 1.0 / (1.0 + fp_cv)
            else:
                features[f"{prefix}_fantasy_points_consistency"] = 0.0
        else:
            features[f"{prefix}_fantasy_points_consistency"] = 0.0

        return features

    def _get_season_stats(
        self, player_id: int, season: int, before_date: datetime | str
    ) -> pd.DataFrame:
        """Get season-to-date statistics."""
        before_date = _ensure_datetime(before_date)
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
        """Calculate season-to-date statistics.

        This method simply delegates to _calculate_rolling_stats with 'season' prefix
        to ensure consistent feature generation regardless of data availability.
        """
        return self._calculate_rolling_stats(df, "season")

    def _get_opponent_features(self, player_id: int, target_date: datetime | str) -> dict:
        """Extract opponent-specific features.

        Opponent analysis is crucial for fantasy predictions because:
        - Some defenses are much better/worse vs certain positions
        - Home field advantage affects performance (crowd, travel, familiarity)
        - Divisional games have different dynamics (familiarity, intensity)
        - Game script expectations (pace, passing vs rushing emphasis)

        Current Implementation:
        - Basic matchup identification (home/away, opponent team)
        - Divisional game detection (higher intensity, more familiarity)
        - Placeholder defensive rankings (would be calculated from real data)

        Future Enhancements:
        - Opponent defensive rankings by position
        - Weather conditions and stadium factors
        - Historical head-to-head performance
        - Vegas odds and game totals
        - Pace of play and game script predictions
        """
        target_date = _ensure_datetime(target_date)
        # Find the target game using complex join to connect player -> team -> game
        # The join conditions handle both home and away games
        target_game = (
            self.db.query(Game)
            .join(Team, (Team.id == Game.home_team_id) | (Team.id == Game.away_team_id))
            .join(Player, Player.team_id == Team.id)
            .filter(
                Player.id == player_id,
                # Use date range to handle timezone/scheduling variations
                Game.game_date >= target_date - timedelta(days=1),
                Game.game_date <= target_date + timedelta(days=1),
            )
            .first()
        )

        # Get player info first (needed for both success and failure cases)
        player = self.db.query(Player).filter(Player.id == player_id).first()

        # Handle cases where game isn't found or player data is missing
        # ALWAYS return the same feature set for consistency
        if not target_game or not player or not player.team_id:
            # Return default values for all opponent features to maintain consistency
            return {
                "has_opponent_data": 0,  # Flag indicates no matchup data available
                "is_home_game": 0,  # Default to away game
                "game_week": 1,  # Default to week 1
                "is_divisional_game": 0,  # Default to non-divisional
                "opponent_def_rank": 15,  # Default to middle ranking
                "opponent_points_allowed_avg": 22.0,  # Default to league average
            }

        # Determine opponent team and home/away status
        # Logic: if player's team is home team, they're playing at home vs away team
        if player.team_id == target_game.home_team_id:
            opponent_team_id = target_game.away_team_id  # Playing vs away team
            is_home_game = 1  # Playing at home
        else:
            opponent_team_id = target_game.home_team_id  # Playing vs home team
            is_home_game = 0  # Playing away

        # Build basic game context features
        features = {
            "has_opponent_data": 1,  # Indicates successful matchup data extraction
            "is_home_game": is_home_game,  # Home field advantage (1=home, 0=away)
            "game_week": target_game.week,  # Week of season (1-18)
            "is_divisional_game": self._is_divisional_matchup(player.team_id, opponent_team_id),
        }

        # Opponent defensive analysis (currently placeholder - would be calculated from real data)
        # In production, these would come from defensive stats aggregation:
        # - Points allowed by position
        # - Yards allowed by position
        # - Fantasy points allowed rankings
        # - Defensive pressure rates, coverage schemes, etc.
        features.update(
            {
                "opponent_def_rank": 15,  # 1-32 ranking (1=best defense, 32=worst)
                "opponent_points_allowed_avg": 22.0,  # Average points allowed per game
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

    def _get_situational_features(self, target_date: datetime | str) -> dict:
        """Extract situational features like weather, time of year, etc.

        Situational context affects fantasy performance in predictable ways:

        Seasonal Patterns:
        - Early season: Limited data, role uncertainty, rust
        - Mid season: Established patterns, optimal performance
        - Late season: Fatigue, injuries, playoff implications
        - Playoff push: Increased motivation vs rest decisions

        Day of Week Effects:
        - Sunday: Standard preparation time
        - Monday: Short week preparation (Thursday -> Monday)
        - Thursday: Short week preparation (Sunday -> Thursday)
        - Different TV audiences and prime time pressure

        Future Enhancements:
        - Weather conditions (temperature, wind, precipitation)
        - Stadium factors (dome vs outdoor, altitude, surface)
        - Rest days between games
        - Playoff implications and motivation
        """
        target_date = _ensure_datetime(target_date)
        # Calculate approximate week of season based on calendar
        # NFL season typically starts first Sunday in September
        september_start = datetime(target_date.year, 9, 1)
        days_since_season_start = (target_date - september_start).days
        week_of_season = max(1, min(18, (days_since_season_start // 7) + 1))

        # Seasonal progression features
        features = {
            "week_of_season": week_of_season,  # Raw week number (1-18)
            # Season phase indicators (mutually exclusive categories)
            "is_early_season": 1 if week_of_season <= 4 else 0,  # Weeks 1-4: Rust, role uncertainty
            "is_mid_season": 1 if 5 <= week_of_season <= 12 else 0,  # Weeks 5-12: Peak performance
            "is_late_season": 1 if week_of_season >= 13 else 0,  # Weeks 13+: Fatigue, injuries
            "is_playoff_push": 1 if week_of_season >= 15 else 0,  # Weeks 15+: Playoff implications
        }

        # Day of week effects (preparation time and primetime factors)
        # Python weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
        day_of_week = target_date.weekday()
        features.update(
            {
                "is_sunday_game": 1 if day_of_week == 6 else 0,  # Standard week preparation
                "is_monday_game": 1 if day_of_week == 0 else 0,  # Monday Night Football (primetime)
                "is_thursday_game": (
                    1 if day_of_week == 3 else 0
                ),  # Thursday Night Football (short week)
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
        self, game_date: datetime | str, positions: list[str] | None = None
    ) -> pd.DataFrame:
        """Extract features for all players in a game slate.

        Args:
            game_date: Date of the games
            positions: List of positions to include (default: all)

        Returns:
            DataFrame with features for all eligible players
        """
        game_date = _ensure_datetime(game_date)
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

    This function implements the standard fantasy football scoring system
    used by most platforms like ESPN, Yahoo, and DraftKings.

    Scoring Systems:
    - standard: No points for receptions (traditional scoring)
    - ppr: Point Per Reception (1 point per catch)
    - half_ppr: Half Point Per Reception (0.5 points per catch)

    Scoring Rules (Standard DFS):

    Passing:
    - 1 point per 25 yards (0.04 per yard)
    - 4 points per touchdown
    - -2 points per interception

    Rushing/Receiving:
    - 1 point per 10 yards (0.1 per yard)
    - 6 points per touchdown
    - Reception bonus varies by scoring system

    Other:
    - -2 points per fumble lost
    - +2 points per two-point conversion

    Usage:
    This function is used for:
    - Calculating historical fantasy points for training data
    - Validating database fantasy point calculations
    - Converting predictions back to fantasy points

    Args:
        stats: Dictionary of player statistics (keys match database columns)
        scoring: Scoring system ("standard", "ppr", "half_ppr")

    Returns:
        Total fantasy points as float, rounded to 2 decimal places
    """
    # Initialize point total
    points = 0.0

    # Passing scoring (primarily for QBs)
    points += stats.get("passing_yards", 0) * 0.04  # 1 point per 25 yards (25 * 0.04 = 1)
    points += stats.get("passing_tds", 0) * 4  # 4 points per passing touchdown
    points -= stats.get("passing_interceptions", 0) * 2  # -2 points per interception

    # Rushing scoring (RBs, some QBs/WRs)
    points += stats.get("rushing_yards", 0) * 0.1  # 1 point per 10 yards (10 * 0.1 = 1)
    points += stats.get("rushing_tds", 0) * 6  # 6 points per rushing touchdown

    # Receiving scoring (WRs, TEs, RBs)
    points += stats.get("receiving_yards", 0) * 0.1  # 1 point per 10 yards
    points += stats.get("receiving_tds", 0) * 6  # 6 points per receiving touchdown

    # Reception bonus (varies by scoring system)
    if scoring == "ppr":
        points += stats.get("receptions", 0) * 1.0  # Full point per reception
    elif scoring == "half_ppr":
        points += stats.get("receptions", 0) * 0.5  # Half point per reception
    # "standard" scoring gives 0 points per reception

    # Penalty/bonus scoring
    points -= stats.get("fumbles_lost", 0) * 2  # -2 points per fumble lost
    points += stats.get("two_point_conversions", 0) * 2  # 2 points per two-point conversion

    # Round to 2 decimal places for consistency with DFS platforms
    return round(points, 2)
