"""Data preparation utilities for ML model training."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sqlalchemy.orm import Session

from src.data.processing.feature_extractor import FeatureExtractor
from src.database.connection import get_db

logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare training data for ML models."""

    def __init__(self, db_session: Session | None = None):
        """Initialize data preparator.

        Args:
            db_session: Optional database session
        """
        self.db = db_session or next(get_db())
        self.feature_extractor = FeatureExtractor(self.db)
        self.scaler = None
        self.target_scaler = None

    def prepare_training_data(
        self,
        position: str,
        start_date: datetime,
        end_date: datetime,
        min_games: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.2,
        feature_version: str = "v1.0",
    ) -> dict:
        """Prepare complete training dataset for a position.

        Args:
            position: Player position (QB, RB, WR, TE, DEF)
            start_date: Start date for data collection
            end_date: End date for data collection
            min_games: Minimum games played for player inclusion
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            feature_version: Version string for feature extraction

        Returns:
            Dictionary containing train/val/test splits and metadata
        """
        logger.info(f"Preparing training data for {position} from {start_date} to {end_date}")

        # Extract raw training data
        raw_data = self._extract_position_data(position, start_date, end_date, min_games)

        if raw_data.empty:
            raise ValueError(f"No data found for position {position} in date range") from None

        logger.info(f"Extracted {len(raw_data)} samples for {position}")

        # Split data temporally (important for time series)
        train_data, val_data, test_data = self._temporal_split(raw_data, test_size, val_size)

        # Extract features for each split
        X_train, y_train = self._extract_features_and_targets(train_data, position)
        X_val, y_val = self._extract_features_and_targets(val_data, position)
        X_test, y_test = self._extract_features_and_targets(test_data, position)

        # Handle missing values and outliers
        X_train, y_train = self._clean_data(X_train, y_train)
        X_val, y_val = self._clean_data(X_val, y_val, fit_scaler=False)
        X_test, y_test = self._clean_data(X_test, y_test, fit_scaler=False)

        # Generate feature names
        feature_names = self._generate_feature_names(X_train.shape[1], position)

        # Prepare metadata
        metadata = {
            "position": position,
            "feature_version": feature_version,
            "start_date": start_date,
            "end_date": end_date,
            "total_samples": len(raw_data),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_count": X_train.shape[1],
            "feature_names": feature_names,
            "min_games": min_games,
            "target_mean": np.mean(y_train),
            "target_std": np.std(y_train),
        }

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "metadata": metadata,
            "scaler": self.scaler,
            "target_scaler": self.target_scaler,
        }

    def _extract_position_data(
        self, position: str, start_date: datetime, end_date: datetime, min_games: int
    ) -> pd.DataFrame:
        """Extract raw data for a specific position.

        Args:
            position: Player position
            start_date: Start date for data
            end_date: End date for data
            min_games: Minimum games for inclusion

        Returns:
            DataFrame with player game data
        """
        # Query for players in position with sufficient game history
        query = """
        SELECT
            p.id as player_id,
            p.display_name,
            p.position,
            ps.fantasy_points,
            ps.fantasy_points_ppr,
            g.id as game_id,
            g.game_date,
            g.season,
            g.week,
            ps.passing_yards,
            ps.passing_tds,
            ps.passing_interceptions,
            ps.rushing_yards,
            ps.rushing_tds,
            ps.receiving_yards,
            ps.receiving_tds,
            ps.receptions,
            ps.targets
        FROM players p
        JOIN player_stats ps ON p.id = ps.player_id
        JOIN games g ON ps.game_id = g.id
        WHERE p.position = :position
        AND g.game_date >= :start_date
        AND g.game_date <= :end_date
        AND g.game_finished = true
        AND ps.fantasy_points IS NOT NULL
        ORDER BY p.id, g.game_date
        """

        result = self.db.execute(
            query, {"position": position, "start_date": start_date, "end_date": end_date}
        )

        df = pd.DataFrame(result.fetchall())

        if df.empty:
            return df

        # Filter players with minimum games
        player_game_counts = df.groupby("player_id").size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df["player_id"].isin(valid_players)]

        return df

    def _temporal_split(
        self, data: pd.DataFrame, test_size: float, val_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally to avoid data leakage.

        Args:
            data: Complete dataset
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Sort by game date
        data_sorted = data.sort_values("game_date")

        n_total = len(data_sorted)
        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * val_size)

        # Split chronologically
        test_data = data_sorted.tail(n_test)
        remaining_data = data_sorted.head(n_total - n_test)

        val_data = remaining_data.tail(n_val)
        train_data = remaining_data.head(len(remaining_data) - n_val)

        logger.info(
            f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        return train_data, val_data, test_data

    def _extract_features_and_targets(
        self, data: pd.DataFrame, position: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from raw data.

        Args:
            data: Raw player game data
            position: Player position

        Returns:
            Tuple of (features, targets)
        """
        features_list = []
        targets = []

        for _, row in data.iterrows():
            try:
                # Extract features for this player-game combination
                player_features = self.feature_extractor.extract_player_features(
                    player_id=row["player_id"], target_game_date=row["game_date"], lookback_games=5
                )

                # Convert features dict to array
                feature_values = list(player_features.values())
                features_list.append(feature_values)

                # Use fantasy points (or PPR) as target
                target = (
                    row["fantasy_points_ppr"]
                    if position in ["WR", "RB", "TE"]
                    else row["fantasy_points"]
                )
                targets.append(target)

            except Exception as e:
                logger.warning(
                    f"Failed to extract features for player {row['player_id']}, game {row['game_id']}: {e}"
                )
                continue

        if not features_list:
            raise ValueError("No valid features extracted") from None

        X = np.array(features_list)
        y = np.array(targets)

        return X, y

    def _clean_data(
        self, X: np.ndarray, y: np.ndarray, fit_scaler: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clean data by handling missing values and outliers.

        Args:
            X: Feature matrix
            y: Target vector
            fit_scaler: Whether to fit scalers (True for training data)

        Returns:
            Tuple of cleaned (X, y)
        """
        # Handle missing values in features
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove outliers from targets (beyond 3 standard deviations)
        if len(y) > 0:
            y_mean = np.mean(y)
            y_std = np.std(y)
            outlier_mask = np.abs(y - y_mean) <= 3 * y_std

            X_clean = X_clean[outlier_mask]
            y_clean = y[outlier_mask]

            logger.info(
                f"Removed {np.sum(~outlier_mask)} outliers ({100 * np.mean(~outlier_mask):.1f}%)"
            )
        else:
            y_clean = y

        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_scaled = self.scaler.transform(X_clean) if self.scaler is not None else X_clean

        return X_scaled, y_clean

    def _generate_feature_names(self, n_features: int, position: str) -> list[str]:
        """Generate descriptive feature names.

        Args:
            n_features: Number of features
            position: Player position

        Returns:
            List of feature names
        """
        # This would ideally use the actual feature extraction logic
        # For now, generate generic names
        base_features = [
            "recent_fantasy_points_avg",
            "recent_fantasy_points_std",
            "recent_games_played",
            "season_fantasy_points_avg",
            "season_games_played",
            "is_home_game",
            "opponent_def_rank",
            "week_of_season",
            "position_QB",
            "position_RB",
            "position_WR",
            "position_TE",
            "height_inches",
            "weight_lbs",
            "age",
        ]

        # Add position-specific features
        if position == "QB":
            position_features = [
                "recent_passing_yards_avg",
                "recent_passing_tds_avg",
                "recent_completion_pct",
                "recent_rushing_yards_avg",
                "season_passing_yards_avg",
            ]
        elif position in ["RB"]:
            position_features = [
                "recent_rushing_yards_avg",
                "recent_rushing_tds_avg",
                "recent_receiving_yards_avg",
                "recent_targets_avg",
                "season_rushing_yards_avg",
            ]
        elif position in ["WR", "TE"]:
            position_features = [
                "recent_receiving_yards_avg",
                "recent_receptions_avg",
                "recent_targets_avg",
                "recent_catch_rate",
                "season_receiving_yards_avg",
            ]
        else:
            position_features = ["team_offensive_rank", "team_defensive_rank"]

        all_features = base_features + position_features

        # Pad or truncate to match actual feature count
        if len(all_features) < n_features:
            all_features.extend([f"feature_{i}" for i in range(len(all_features), n_features)])
        else:
            all_features = all_features[:n_features]

        return all_features

    def prepare_prediction_data(
        self, player_ids: list[int], game_date: datetime, _position: str
    ) -> tuple[np.ndarray, list[int]]:
        """Prepare data for making predictions on new games.

        Args:
            player_ids: List of player IDs to predict for
            game_date: Date of games to predict
            position: Player position

        Returns:
            Tuple of (features, valid_player_ids)
        """
        features_list = []
        valid_player_ids = []

        for player_id in player_ids:
            try:
                # Extract features for prediction
                player_features = self.feature_extractor.extract_player_features(
                    player_id=player_id, target_game_date=game_date, lookback_games=5
                )

                feature_values = list(player_features.values())
                features_list.append(feature_values)
                valid_player_ids.append(player_id)

            except Exception as e:
                logger.warning(f"Failed to extract features for player {player_id}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid features extracted for prediction") from None

        X = np.array(features_list)

        # Clean and scale features
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.transform(X_clean) if self.scaler is not None else X_clean

        return X_scaled, valid_player_ids

    def get_feature_statistics(self, X: np.ndarray, feature_names: list[str]) -> dict:
        """Calculate feature statistics for data quality assessment.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary of feature statistics
        """
        stats = {}

        for i, name in enumerate(feature_names):
            feature_data = X[:, i]
            stats[name] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "missing_pct": np.mean(np.isnan(feature_data)) * 100,
                "zero_pct": np.mean(feature_data == 0) * 100,
            }

        return stats
