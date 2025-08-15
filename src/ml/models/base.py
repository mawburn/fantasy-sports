"""Base model classes and data structures for ML models."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Set PyTorch for CPU optimization
torch.set_num_threads(8)  # Adjust based on CPU cores
torch.manual_seed(42)


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # Model identification
    model_name: str
    position: str  # QB, RB, WR, TE, DEF
    version: str = "1.0"

    # Training parameters
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

    # CPU optimization settings
    n_jobs: int = -1  # Use all available cores
    use_early_stopping: bool = True
    early_stopping_rounds: int = 50

    # Feature engineering
    feature_selection: bool = True
    max_features: int | None = None
    min_feature_importance: float = 0.001

    # Model persistence
    save_model: bool = True
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # Performance thresholds
    min_r2_score: float = 0.3
    max_mae_threshold: float = 5.0

    # Validation
    cross_validation_folds: int = 5

    def __post_init__(self):
        """Ensure model directory exists."""
        if self.save_model:
            self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Result from model training."""

    model: Any
    training_time: float
    best_iteration: int | None = None
    feature_importance: dict[str, float] | None = None

    # Performance metrics
    train_mae: float = 0.0
    val_mae: float = 0.0
    train_rmse: float = 0.0
    val_rmse: float = 0.0
    train_r2: float = 0.0
    val_r2: float = 0.0

    # Training metadata
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0

    # Model artifacts
    model_path: Path | None = None
    preprocessor_path: Path | None = None


@dataclass
class PredictionResult:
    """Result from model prediction."""

    # Core predictions
    point_estimate: np.ndarray
    confidence_score: np.ndarray | None = None

    # Uncertainty quantification
    prediction_intervals: tuple[np.ndarray, np.ndarray] | None = None
    floor: np.ndarray | None = None  # 25th percentile
    ceiling: np.ndarray | None = None  # 75th percentile

    # Feature importance for predictions
    feature_contributions: dict[str, np.ndarray] | None = None

    # Metadata
    prediction_date: datetime = field(default_factory=datetime.utcnow)
    model_version: str | None = None


@dataclass
class EvaluationMetrics:
    """Model evaluation metrics."""

    # Regression metrics
    mae: float
    rmse: float
    r2: float
    mape: float  # Mean Absolute Percentage Error

    # Distribution metrics
    prediction_std: float
    residual_std: float

    # Consistency metrics
    consistency_score: float  # How consistent are predictions

    # Outlier metrics
    outlier_percentage: float  # Percentage of large errors

    # Sample information
    n_samples: int

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"MAE: {self.mae:.3f}, RMSE: {self.rmse:.3f}, R²: {self.r2:.3f}, MAPE: {self.mape:.3f}"
        )


class BaseModel(ABC):
    """Abstract base class for all position-specific models."""

    def __init__(self, config: ModelConfig):
        """Initialize base model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_names: list[str] | None = None
        self.feature_importance: dict[str, float] | None = None
        self.training_history: list[dict[str, Any]] = []
        self.is_trained = False

        # Set random seed for reproducibility
        np.random.seed(config.random_state)

    @abstractmethod
    def build_model(self) -> Any:
        """Build the model architecture.

        Returns:
            Configured model instance
        """
        pass

    @abstractmethod
    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training result with metrics and artifacts
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions.

        Args:
            X: Features for prediction

        Returns:
            Prediction results with confidence intervals
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Comprehensive evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X_test)
        y_pred = predictions.point_estimate

        # Basic regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # MAPE with protection against division by zero
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1))) * 100

        # Distribution metrics
        prediction_std = np.std(y_pred)
        residuals = y_test - y_pred
        residual_std = np.std(residuals)

        # Consistency score (1 / (1 + coefficient of variation))
        cv = prediction_std / np.mean(y_pred) if np.mean(y_pred) != 0 else 0
        consistency_score = 1 / (1 + cv)

        # Outlier percentage (errors > 2 standard deviations)
        outlier_threshold = 2 * residual_std
        outlier_percentage = np.mean(np.abs(residuals) > outlier_threshold) * 100

        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            prediction_std=prediction_std,
            residual_std=residual_std,
            consistency_score=consistency_score,
            outlier_percentage=outlier_percentage,
            n_samples=len(y_test),
        )

    def save_model(self, path: Path | None = None) -> Path:
        """Save trained model to disk.

        Args:
            path: Optional custom save path

        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.position}_{self.config.model_name}_{timestamp}.pkl"
            path = self.config.model_dir / filename

        # Save model and metadata
        model_data = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "config": self.config,
            "training_history": self.training_history,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

        return path

    def load_model(self, path: Path) -> None:
        """Load trained model from disk.

        Args:
            path: Path to saved model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.preprocessor = model_data.get("preprocessor")
        self.feature_names = model_data.get("feature_names")
        self.feature_importance = model_data.get("feature_importance")
        self.training_history = model_data.get("training_history", [])
        self.is_trained = True

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_importance is None:
            return {}

        return self.feature_importance

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Validate input arrays.

        Args:
            X: Feature array
            y: Optional target array
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")

            if y.ndim != 1:
                raise ValueError("y must be 1-dimensional")

            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")

    def _calculate_time_weights(self, X: np.ndarray) -> np.ndarray:
        """Calculate time-based weights for training samples.

        More recent samples get higher weights.

        Args:
            X: Training features

        Returns:
            Sample weights array
        """
        n_samples = len(X)
        # Linear decay from 1.0 to 0.5
        weights = np.linspace(0.5, 1.0, n_samples)
        return weights

    def _calculate_prediction_intervals(
        self, X: np.ndarray, point_estimate: np.ndarray, confidence_level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals using quantile regression approach.

        Args:
            X: Features
            point_estimate: Point predictions
            confidence_level: Confidence level for intervals

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        # Simple approach: use historical residual distribution
        # In a more sophisticated implementation, this would use quantile regression

        if not hasattr(self, "_residual_std"):
            # Fallback: assume 20% of point estimate as uncertainty
            uncertainty = point_estimate * 0.2
        else:
            uncertainty = self._residual_std

        1 - confidence_level
        z_score = 1.96  # 95% confidence interval

        margin = z_score * uncertainty
        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        return lower_bound, upper_bound
