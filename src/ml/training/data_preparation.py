"""Data preparation utilities for ML model training.

This file handles all the data preprocessing steps required before training ML models:

1. Data Extraction: Query database for player statistics and game data
2. Temporal Splitting: Split data chronologically to prevent data leakage
3. Feature Engineering: Convert raw stats into ML-ready features
4. Data Cleaning: Handle missing values, outliers, and scaling
5. Validation: Ensure data quality and consistency

Key Concepts for Beginners:

Temporal Splitting: Unlike random train/test splits, we split by time to simulate
real-world conditions where we predict future games based on past data.

Data Leakage: Using future information to predict past events. This creates
artificially high accuracy that doesn't work in production.

Feature Scaling: ML algorithms perform better when features are on similar scales.
RobustScaler is used instead of StandardScaler because it's less sensitive to outliers.

Outlier Removal: Fantasy football has extreme outliers (50+ point games) that can
skew training. We remove games beyond 3 standard deviations from the mean.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.data.processing.feature_extractor import FeatureExtractor
from src.database.connection import get_db

logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare training data for ML models.

    This class coordinates all data preparation steps for training position-specific
    fantasy football prediction models. It bridges the gap between raw database
    records and ML-ready numpy arrays.

    Responsibilities:
    - Query database for relevant player/game data
    - Apply temporal splitting to prevent data leakage
    - Extract and engineer features for each position
    - Clean data (handle missing values, outliers)
    - Scale features for optimal ML performance
    - Generate metadata for model tracking and debugging

    The class maintains scalers (feature and target) that are fitted on training
    data and applied consistently to validation, test, and prediction data.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize data preparator with database connection and feature extractor.

        The data preparator needs:
        - Database session for querying player/game data
        - Feature extractor for converting raw stats to ML features
        - Scalers for normalizing data (fitted during training, applied during inference)

        Args:
            db_session: Optional database session (creates new one if None)
        """
        # Get database session (create new one if not provided)
        self.db = db_session or next(get_db())

        # Feature extractor handles converting raw stats to ML-ready features
        self.feature_extractor = FeatureExtractor(self.db)

        # Scalers for normalizing features and targets (fitted during training)
        self.scaler = None  # RobustScaler for features (fitted on training data)
        self.target_scaler = None  # Currently unused but reserved for target scaling

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

        This is the main entry point for data preparation. It orchestrates
        all the steps needed to convert raw database records into ML-ready
        training datasets with proper train/validation/test splits.

        Process Overview:
        1. Extract raw player/game data from database
        2. Filter players with sufficient game history (min_games)
        3. Split data temporally to prevent data leakage
        4. Extract ML features for each split
        5. Clean and scale the data
        6. Generate comprehensive metadata for tracking

        Temporal Splitting Strategy:
        - Test set: Most recent 20% of games (what we want to predict)
        - Validation set: 20% of remaining games (for model tuning)
        - Training set: Oldest 64% of games (for learning patterns)

        This mimics real-world usage where we predict future games based on
        historical data, preventing the model from "cheating" by seeing future info.

        Args:
            position: Player position (QB, RB, WR, TE, DEF)
            start_date: Start date for data collection
            end_date: End date for data collection
            min_games: Minimum games played for player inclusion (filters out inconsistent players)
            test_size: Proportion of data for testing (0.2 = 20%)
            val_size: Proportion of remaining data for validation (0.2 = 20% of 80% = 16%)
            feature_version: Version string for feature extraction (for tracking/reproducibility)

        Returns:
            Dictionary containing:
            - X_train, y_train: Training features and targets
            - X_val, y_val: Validation features and targets
            - X_test, y_test: Test features and targets
            - metadata: Comprehensive information about the dataset
            - scaler: Fitted feature scaler for consistent preprocessing
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
            text(query), {"position": position, "start_date": start_date, "end_date": end_date}
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

        Critical Concept: Data Leakage Prevention

        In time series prediction (like fantasy sports), we must split data
        chronologically, not randomly. Random splits would let the model
        "cheat" by learning from future games to predict past games.

        Why This Matters:
        - Random split: Model sees Week 10 data to predict Week 5 (unrealistic)
        - Temporal split: Model sees Weeks 1-5 data to predict Week 10 (realistic)

        Splitting Strategy:
        1. Sort all games by date (oldest to newest)
        2. Take last 20% of games as test set (most recent)
        3. Take last 20% of remaining games as validation set
        4. Use earliest games as training set

        This simulates real-world deployment where we predict upcoming games
        based on historical performance.

        Args:
            data: Complete dataset sorted by game date
            test_size: Proportion for test set (usually 0.2 = 20%)
            val_size: Proportion of remaining data for validation

        Returns:
            Tuple of (train_data, val_data, test_data) in chronological order
        """
        # Sort all games chronologically (oldest first)
        data_sorted = data.sort_values("game_date")

        # Calculate split sizes
        n_total = len(data_sorted)
        n_test = int(n_total * test_size)  # Last N games for testing
        n_val = int((n_total - n_test) * val_size)  # Validation from remaining

        # Perform chronological split (no shuffling!)
        # Test set: Most recent games (what we want to predict)
        test_data = data_sorted.tail(n_test)  # Last 20% of games

        # Validation and training from remaining earlier games
        remaining_data = data_sorted.head(n_total - n_test)  # First 80% of games

        # Validation set: Most recent of the remaining games
        val_data = remaining_data.tail(n_val)  # Last 20% of remaining (16% of total)

        # Training set: Earliest games (majority of data)
        train_data = remaining_data.head(len(remaining_data) - n_val)  # First 64% of total

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

        Data cleaning is crucial for reliable ML model performance:

        Missing Value Handling:
        - Replace NaN, positive infinity, negative infinity with 0.0
        - This is conservative but prevents model crashes
        - Alternative: could use median imputation or forward-fill

        Outlier Removal:
        - Fantasy football has extreme outliers (50+ point games)
        - These rare events can skew model training
        - Remove targets beyond 3 standard deviations from mean
        - Only applies to targets (y), not features (X)

        Feature Scaling:
        - Uses RobustScaler instead of StandardScaler
        - RobustScaler is less sensitive to outliers
        - Scales features to have median=0, IQR=1
        - Critical for algorithms like neural networks, SVM, KMeans

        Fit vs Transform:
        - fit_scaler=True: Fit scaler parameters on this data (training only)
        - fit_scaler=False: Apply existing scaler parameters (validation/test)
        - This prevents data leakage from validation/test back to training

        Args:
            X: Feature matrix (samples × features)
            y: Target vector (fantasy points)
            fit_scaler: Whether to fit scalers (True for training data only)

        Returns:
            Tuple of cleaned and scaled (X, y)
        """
        # Step 1: Handle missing values in features
        # Replace NaN, +inf, -inf with 0.0 (conservative approach)
        # Alternative approaches: median imputation, forward-fill, etc.
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 2: Remove extreme outliers from targets
        # Fantasy football has rare extreme games (50+ points) that can skew training
        if len(y) > 0:
            y_mean = np.mean(y)
            y_std = np.std(y)

            # Keep only samples within 3 standard deviations
            # This removes ~0.3% of data (assuming normal distribution)
            outlier_mask = np.abs(y - y_mean) <= 3 * y_std

            # Apply outlier filter to both features and targets
            X_clean = X_clean[outlier_mask]
            y_clean = y[outlier_mask]

            # Log outlier removal for monitoring data quality
            n_outliers = np.sum(~outlier_mask)
            outlier_pct = 100 * np.mean(~outlier_mask)
            logger.info(f"Removed {n_outliers} outliers ({outlier_pct:.1f}%)")
        else:
            y_clean = y

        # Step 3: Scale features for optimal ML performance
        if fit_scaler:
            # Fit scaler on training data only (prevents data leakage)
            # RobustScaler: median=0, IQR=1 (less sensitive to outliers than StandardScaler)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            # Apply previously fitted scaler to validation/test data
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

        This method handles the inference/prediction pipeline:
        1. Extract features for each player based on their recent history
        2. Apply the same cleaning and scaling used during training
        3. Return ML-ready features for model prediction

        Key Differences from Training Data Preparation:
        - No target values (we're predicting the targets)
        - Uses pre-fitted scaler from training (no fitting here)
        - Handles individual prediction failures gracefully
        - Returns mapping of successful player IDs for result interpretation

        Error Handling:
        - If feature extraction fails for a player, logs warning and continues
        - Returns only players with successful feature extraction
        - Ensures system robustness for production deployment

        Args:
            player_ids: List of player IDs to predict for
            game_date: Date of games to predict (usually upcoming week)
            _position: Player position (underscore indicates unused parameter)

        Returns:
            Tuple of:
            - features: Scaled feature matrix ready for model prediction
            - valid_player_ids: List of player IDs that succeeded (same order as features)
        """
        features_list = []
        valid_player_ids = []

        # Process each player individually for fault tolerance
        for player_id in player_ids:
            try:
                # Extract features based on recent performance (5 game lookback)
                # This gives context about player's current form and role
                player_features = self.feature_extractor.extract_player_features(
                    player_id=player_id,
                    target_game_date=game_date,
                    lookback_games=5,  # Look at last 5 games for recent form
                )

                # Convert feature dictionary to list (consistent with training data)
                feature_values = list(player_features.values())
                features_list.append(feature_values)
                valid_player_ids.append(player_id)

            except Exception as e:
                # Log failure but continue with other players
                # This ensures partial success rather than total failure
                logger.warning(f"Failed to extract features for player {player_id}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid features extracted for prediction") from None

        X = np.array(features_list)

        # Apply same cleaning process as training data
        # Handle missing values (NaN, inf) consistently
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply pre-fitted scaler from training (no fitting here to prevent data leakage)
        # This ensures prediction data is scaled consistently with training data
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
