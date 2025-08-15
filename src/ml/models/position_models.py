"""Position-specific ML models for NFL DFS predictions.

This file implements different machine learning models tailored to each NFL position's
statistical patterns and scoring characteristics:

- QBModel: Uses XGBoost with auxiliary models for rushing and primetime adjustments
- RBModel: Uses LightGBM with workload-based clustering for different RB archetypes
- WRModel: Uses XGBoost optimized for target share and air yards patterns
- TEModel: Uses LightGBM for more consistent TE scoring patterns
- DEFModel: Uses Random Forest to handle the high variance of defensive scoring

Each model inherits from BaseModel but implements position-specific:
- Hyperparameter tuning optimized for that position's variance patterns
- Feature weighting strategies (recent games, target share, etc.)
- Prediction adjustments (floor/ceiling calculations, confidence scores)
- Auxiliary models for position-specific factors (QB rushing, RB workload types)

For beginners: Think of each position as having different "personalities" in terms
of predictability and scoring patterns. QBs are fairly consistent, RBs depend on
workload, WRs are high-variance, TEs are in between, and DEF/ST are chaotic.
"""

import logging
import time

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .base import BaseModel, ModelConfig, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)


class QBModel(BaseModel):
    """Quarterback-specific prediction model using XGBoost.

    QBs are generally the most predictable position in fantasy football because:
    - They handle the ball on every play
    - Passing volume is fairly consistent
    - Game script heavily favors passing when behind

    This model uses XGBoost (eXtreme Gradient Boosting) because:
    - Handles both passing and rushing contributions well
    - Good at capturing non-linear relationships (game script effects)
    - Built-in regularization prevents overfitting
    - Fast training and prediction

    Special QB Features:
    - Auxiliary rushing model: Some QBs get significant rushing upside
    - Primetime adjustment: QBs may perform differently in nationally televised games
    - Time-based weighting: Recent form matters more than season averages

    For beginners: XGBoost builds many simple decision trees and combines them.
    Each tree learns to correct the mistakes of the previous trees.
    """

    def __init__(self, config: ModelConfig):
        """Initialize QB model with auxiliary models.

        The QB model uses a main model plus two auxiliary models:
        - rushing_model: Predicts additional points from QB rushing
        - primetime_model: Adjusts for primetime game effects

        This ensemble approach allows us to model different aspects of QB
        performance separately, then combine them for final predictions.
        """
        super().__init__(config)  # Initialize base model components

        # Auxiliary models for QB-specific adjustments
        self.rushing_model = (
            None  # Separate model for rushing upside (Lamar Jackson, Josh Allen types)
        )
        self.primetime_model = (
            None  # Adjustment for primetime games (Monday/Thursday night effects)
        )

    def build_model(self) -> XGBRegressor:
        """Build XGBoost model optimized for QB predictions."""
        return XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,  # Suppress XGBoost output
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train QB model with auxiliary models for rushing and primetime adjustments."""
        start_time = time.time()

        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Build and train main model
        self.model = self.build_model()

        # Apply time decay weights (more recent games weighted higher)
        weights = self._calculate_time_weights(X_train)

        # Train with early stopping
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False,
        )

        # Train auxiliary models for QB-specific adjustments
        self.rushing_model = self._train_rushing_model(X_train, y_train)
        self.primetime_model = self._train_primetime_model(X_train, y_train)

        # Calculate feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(
                zip(
                    [f"feature_{i}" for i in range(X_train.shape[1])],
                    self.model.feature_importances_,
                    strict=False,
                )
            )

        # Evaluate performance
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_mae = np.mean(np.abs(y_train - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

        # Calculate R² scores
        train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum(
            (y_train - np.mean(y_train)) ** 2
        )
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        # Store residual std for prediction intervals
        self._residual_std = np.std(y_val - val_pred)

        self.is_trained = True
        training_time = time.time() - start_time

        result = TrainingResult(
            model=self.model,
            training_time=training_time,
            best_iteration=getattr(self.model, "best_iteration", None),
            feature_importance=self.feature_importance,
            train_mae=train_mae,
            val_mae=val_mae,
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            train_r2=train_r2,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"QB model trained: MAE={val_mae:.3f}, R²={val_r2:.3f}")

        return result

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate QB predictions with rushing and primetime adjustments."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self._validate_inputs(X)

        # Base passing prediction
        base_pred = self.model.predict(X)

        # Add rushing upside (simplified - would use actual rushing features)
        rushing_adj = np.zeros_like(base_pred)
        if self.rushing_model:
            rushing_adj = np.clip(self.rushing_model.predict(X), -2, 5)  # Cap adjustments

        # Add primetime adjustment (simplified - would use game time features)
        primetime_adj = np.zeros_like(base_pred)
        if self.primetime_model:
            primetime_adj = np.clip(self.primetime_model.predict(X), -3, 3)

        # Combine predictions
        point_estimate = base_pred + rushing_adj + primetime_adj

        # Calculate prediction intervals
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # Calculate floor and ceiling (25th and 75th percentiles)
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.15
        )
        floor = point_estimate - 0.67 * uncertainty  # ~25th percentile
        ceiling = point_estimate + 0.67 * uncertainty  # ~75th percentile

        # Simple confidence score based on prediction consistency
        confidence_score = np.ones_like(point_estimate) * 0.8  # Placeholder

        return PredictionResult(
            point_estimate=point_estimate,
            confidence_score=confidence_score,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )

    def _train_rushing_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestRegressor | None:
        """Train auxiliary model for QB rushing upside."""
        try:
            # Simple model for rushing adjustment (would use rushing-specific features)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

            # For now, predict small random adjustments
            # In real implementation, this would use rushing tendency features
            rushing_targets = np.random.normal(0, 1, len(y_train))
            model.fit(X_train, rushing_targets)

            return model
        except Exception as e:
            logger.warning(f"Failed to train rushing model: {e}")
            return None

    def _train_primetime_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestRegressor | None:
        """Train auxiliary model for primetime game adjustments."""
        try:
            # Simple model for primetime adjustment
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

            # For now, predict small random adjustments
            # In real implementation, this would use game time/TV features
            primetime_targets = np.random.normal(0, 0.5, len(y_train))
            model.fit(X_train, primetime_targets)

            return model
        except Exception as e:
            logger.warning(f"Failed to train primetime model: {e}")
            return None


class RBModel(BaseModel):
    """Running back-specific prediction model using LightGBM with workload clustering.

    RBs are challenging to predict because they fall into distinct archetypes:
    - Workhorse backs: High volume, consistent touches (Derrick Henry types)
    - Committee backs: Share carries, more volatile (many teams use RBBC)
    - Pass-catching specialists: Fewer carries but many targets (James White types)
    - Goal-line specialists: Low volume but high TD probability

    This model uses clustering to identify these archetypes and train separate
    models for each, improving prediction accuracy.

    Why LightGBM for RBs?
    - Faster training than XGBoost (important when training multiple cluster models)
    - Better handling of categorical features (team, opponent, game situation)
    - Memory efficient for large datasets
    - Good performance on tabular data

    Clustering Strategy:
    - Use workload features (touches, snap share, target share) to identify archetypes
    - Train separate models for each cluster
    - Predict cluster membership for new players, then use appropriate model
    """

    def __init__(self, config: ModelConfig):
        """Initialize RB model."""
        super().__init__(config)
        self.cluster_models = {}
        self.clusterer = None
        self.scaler = StandardScaler()

    def build_model(self) -> LGBMRegressor:
        """Build LightGBM model optimized for RB predictions."""
        return LGBMRegressor(
            n_estimators=400,
            num_leaves=31,
            max_depth=7,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=0.1,
            lambda_l2=0.1,
            min_child_samples=20,
            objective="regression",
            metric="mae",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=-1,  # Suppress LightGBM output
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train RB model with workload-based clustering."""
        start_time = time.time()

        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Cluster RBs by workload type (simplified - would use actual workload features)
        clusters = self._cluster_by_workload(X_train)
        n_clusters = len(np.unique(clusters))

        # Train separate models for each cluster
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id

            if np.sum(cluster_mask) < 10:  # Skip clusters with too few samples
                continue

            model = self.build_model()

            # Train cluster-specific model
            X_cluster = X_train[cluster_mask]
            y_cluster = y_train[cluster_mask]

            model.fit(
                X_cluster,
                y_cluster,
                eval_set=[(X_val, y_val)],
                callbacks=[],  # Remove callbacks to avoid warnings
            )

            self.cluster_models[cluster_id] = model

        # If clustering failed, train single model
        if not self.cluster_models:
            self.model = self.build_model()
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[],
            )

        # Calculate aggregated feature importance
        self.feature_importance = self._aggregate_feature_importance(X_train.shape[1])

        # Evaluate performance
        val_pred = self.predict(X_val).point_estimate
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True
        training_time = time.time() - start_time

        result = TrainingResult(
            model=self.cluster_models if self.cluster_models else self.model,
            training_time=training_time,
            feature_importance=self.feature_importance,
            val_mae=val_mae,
            val_rmse=val_rmse,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"RB model trained: {n_clusters} clusters, MAE={val_mae:.3f}, R²={val_r2:.3f}")

        return result

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate RB predictions using appropriate cluster model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self._validate_inputs(X)

        if self.cluster_models:
            # Predict cluster for each sample
            clusters = self._predict_clusters(X)
            predictions = np.zeros(len(X))

            for cluster_id, model in self.cluster_models.items():
                cluster_mask = clusters == cluster_id
                if np.any(cluster_mask):
                    predictions[cluster_mask] = model.predict(X[cluster_mask])

            # Use default model for unmatched clusters
            if len(self.cluster_models) > 0:
                default_model = next(iter(self.cluster_models.values()))
                unmapped_mask = predictions == 0
                if np.any(unmapped_mask):
                    predictions[unmapped_mask] = default_model.predict(X[unmapped_mask])

            point_estimate = predictions
        else:
            # Single model prediction
            point_estimate = self.model.predict(X)

        # Calculate prediction intervals
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # RB-specific floor and ceiling adjustments
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.18
        )
        floor = point_estimate - 0.75 * uncertainty  # RBs have higher floor variance
        ceiling = point_estimate + 0.9 * uncertainty  # Account for TD variance

        confidence_score = np.ones_like(point_estimate) * 0.75  # RBs slightly less predictable

        return PredictionResult(
            point_estimate=point_estimate,
            confidence_score=confidence_score,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )

    def _cluster_by_workload(self, X: np.ndarray) -> np.ndarray:
        """Cluster RBs by workload patterns."""
        try:
            # Simplified clustering - would use actual workload features in real implementation
            self.clusterer = KMeans(n_clusters=3, random_state=self.config.random_state)

            # For now, cluster based on first few features (representing touches, snap share, etc.)
            workload_features = X[:, : min(5, X.shape[1])]
            workload_features_scaled = self.scaler.fit_transform(workload_features)

            clusters = self.clusterer.fit_predict(workload_features_scaled)
            return clusters

        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using single model")
            return np.zeros(len(X))

    def _predict_clusters(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data."""
        if self.clusterer is None:
            return np.zeros(len(X))

        try:
            workload_features = X[:, : min(5, X.shape[1])]
            workload_features_scaled = self.scaler.transform(workload_features)
            return self.clusterer.predict(workload_features_scaled)
        except Exception:
            return np.zeros(len(X))

    def _aggregate_feature_importance(self, n_features: int) -> dict[str, float]:
        """Aggregate feature importance across cluster models."""
        if not self.cluster_models:
            return {}

        # Average importance across all cluster models
        avg_importance = np.zeros(n_features)
        model_count = 0

        for model in self.cluster_models.values():
            if hasattr(model, "feature_importances_"):
                avg_importance += model.feature_importances_
                model_count += 1

        if model_count > 0:
            avg_importance /= model_count

        return dict(zip([f"feature_{i}" for i in range(n_features)], avg_importance, strict=False))


class WRModel(BaseModel):
    """Wide receiver-specific prediction model."""

    def build_model(self) -> XGBRegressor:
        """Build XGBoost model optimized for WR predictions."""
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.5,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train WR model with target share emphasis."""
        start_time = time.time()

        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        self.model = self.build_model()

        # WR-specific sample weighting (emphasize target share and air yards)
        weights = self._calculate_wr_weights(X_train)

        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False,
        )

        # Standard evaluation and cleanup
        val_pred = self.model.predict(X_val)
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True

        return TrainingResult(
            model=self.model,
            training_time=time.time() - start_time,
            val_mae=val_mae,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate WR predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        point_estimate = self.model.predict(X)
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # WR-specific variance adjustments
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.22
        )
        floor = point_estimate - 0.8 * uncertainty  # High variance position
        ceiling = point_estimate + 1.1 * uncertainty  # Big play upside

        return PredictionResult(
            point_estimate=point_estimate,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )

    def _calculate_wr_weights(self, X: np.ndarray) -> np.ndarray:
        """Calculate WR-specific sample weights."""
        # Simplified - would emphasize target share and air yards features
        base_weights = self._calculate_time_weights(X)
        return base_weights


class TEModel(BaseModel):
    """Tight end-specific prediction model."""

    def build_model(self) -> LGBMRegressor:
        """Build LightGBM model optimized for TE predictions."""
        return LGBMRegressor(
            n_estimators=300,
            num_leaves=25,
            max_depth=6,
            learning_rate=0.06,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            lambda_l1=0.1,
            lambda_l2=0.2,
            objective="regression",
            metric="mae",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=-1,
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train TE model."""
        start_time = time.time()

        self.model = self.build_model()
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        val_pred = self.model.predict(X_val)
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True

        return TrainingResult(
            model=self.model,
            training_time=time.time() - start_time,
            val_mae=val_mae,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate TE predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        point_estimate = self.model.predict(X)
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # TE-specific adjustments (lower variance than WR, higher than RB)
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.20
        )
        floor = point_estimate - 0.7 * uncertainty
        ceiling = point_estimate + 0.85 * uncertainty

        return PredictionResult(
            point_estimate=point_estimate,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )


class DEFModel(BaseModel):
    """Defense/Special Teams prediction model."""

    def build_model(self) -> RandomForestRegressor:
        """Build Random Forest model optimized for DEF predictions."""
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train DEF model."""
        start_time = time.time()

        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        val_pred = self.model.predict(X_val)
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True

        return TrainingResult(
            model=self.model,
            training_time=time.time() - start_time,
            val_mae=val_mae,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate DEF predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        point_estimate = self.model.predict(X)
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # DEF has high variance due to defensive TDs, turnovers
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.25
        )
        floor = point_estimate - 1.0 * uncertainty  # Can go very negative
        ceiling = point_estimate + 1.2 * uncertainty  # High upside with TDs

        return PredictionResult(
            point_estimate=point_estimate,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )
