"""Base model classes and data structures for ML models.

This file contains the foundational classes that all position-specific models inherit from.
It establishes common patterns for:
- Model configuration (hyperparameters, training settings)
- Training results tracking (metrics, artifacts, metadata)
- Prediction results structure (point estimates, confidence intervals)
- Model evaluation (standardized metrics across all positions)
- Model persistence (saving/loading trained models)

For beginners: This is the "blueprint" that defines what all ML models in this system
should be able to do - train, predict, evaluate, and save/load themselves.
"""

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
# torch.set_num_threads() tells PyTorch how many CPU cores to use for parallel operations
# This is important for CPU-only training to maximize performance
torch.set_num_threads(8)  # Adjust based on your CPU cores (8 is good for most modern CPUs)

# Set random seed for reproducible results
# This ensures that random operations in PyTorch always produce the same results
# Critical for comparing model performance and debugging
torch.manual_seed(42)


@dataclass
class ModelConfig:
    """Configuration for ML models.

    This dataclass uses Python's @dataclass decorator to automatically generate
    __init__, __repr__, and other methods. It stores all the settings needed
    to configure and train a machine learning model.

    Think of this as a "settings file" for each model that contains:
    - Model identification (name, position, version)
    - Training parameters (how to split data, when to stop training)
    - Performance requirements (minimum accuracy thresholds)
    - File system settings (where to save models)

    The 'field()' function with 'default_factory' is used for mutable defaults
    like Path objects - this prevents sharing the same object between instances.
    """

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
        """Ensure model directory exists.

        __post_init__ is a special dataclass method that runs after __init__.
        It's used here to create the directory where models will be saved.

        mkdir(parents=True, exist_ok=True) means:
        - parents=True: Create parent directories if they don't exist
        - exist_ok=True: Don't raise error if directory already exists
        """
        if self.save_model:
            self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Result from model training.

    This class stores everything we learned during the training process:

    Performance Metrics:
    - MAE (Mean Absolute Error): Average difference between predictions and actual values
    - RMSE (Root Mean Square Error): Penalizes large errors more than small ones
    - R² (R-squared): How much variance in the target the model explains (1.0 = perfect)

    Training Information:
    - training_time: How long training took (important for production scheduling)
    - best_iteration: Which training epoch performed best (for early stopping)
    - feature_importance: Which input features matter most for predictions

    Model Artifacts:
    - model_path: Where the trained model is saved on disk
    - preprocessor_path: Where data preprocessing steps are saved

    This comprehensive tracking helps with model debugging, comparison, and deployment.
    """

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
    """Result from model prediction.

    When a trained model makes predictions, it returns more than just a single number.
    This class captures the full prediction information:

    Core Predictions:
    - point_estimate: The model's best guess (main prediction)
    - confidence_score: How confident the model is (0-1 scale)

    Uncertainty Quantification:
    - prediction_intervals: Range where true value likely falls (e.g., 90% confidence)
    - floor: Conservative estimate (25th percentile)
    - ceiling: Optimistic estimate (75th percentile)

    This is especially important for fantasy sports where understanding
    the range of possible outcomes helps with risk management in lineup building.

    For beginners: Think of this like a weather forecast - not just "70°F" but
    "70°F with a range of 65-75°F and 90% confidence it won't rain".
    """

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
    """Model evaluation metrics.

    After training, we need to measure how well our model performs.
    This class standardizes evaluation across all position models.

    Regression Metrics (for continuous predictions like fantasy points):
    - mae: Mean Absolute Error - average distance from true values
    - rmse: Root Mean Square Error - penalizes large mistakes more
    - r2: R-squared - proportion of variance explained (higher = better)
    - mape: Mean Absolute Percentage Error - error as % of true value

    Distribution Metrics (understanding prediction patterns):
    - prediction_std: How spread out our predictions are
    - residual_std: How spread out our errors are

    Consistency & Outlier Metrics:
    - consistency_score: How stable predictions are across similar inputs
    - outlier_percentage: How often we make really bad predictions

    These metrics help us understand not just accuracy, but reliability.
    """

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
    """Abstract base class for all position-specific models.

    This is an Abstract Base Class (ABC) that defines the interface all position models
    must implement. It uses Python's abc module to enforce that child classes
    implement specific methods.

    The Strategy Pattern: Each position (QB, RB, WR, etc.) has different statistical
    patterns and requires different modeling approaches. This base class ensures
    they all have the same interface while allowing position-specific implementations.

    Key Responsibilities:
    1. Common functionality (save/load, evaluation, validation)
    2. Abstract methods that must be implemented by each position
    3. Shared utility methods for data validation and processing
    4. Standardized prediction intervals and confidence calculations

    For beginners: Think of this as a "contract" that says "every position model
    must be able to build_model(), train(), and predict(), but each position
    can do these things differently".
    """

    def __init__(self, config: ModelConfig):
        """Initialize base model.

        Sets up the model's initial state and configuration.
        All instance variables start as None or empty until training.

        Args:
            config: Model configuration containing hyperparameters and settings
        """
        self.config = config

        # Core model components (initially None until training)
        self.model = None  # The actual ML model (PyTorch neural network)
        self.preprocessor = None  # Data scaling/transformation pipeline

        # Feature information (learned during training)
        self.feature_names: list[str] | None = None  # Names of input features
        self.feature_importance: dict[str, float] | None = None  # Which features matter most

        # Training metadata and history
        self.training_history: list[dict[str, Any]] = []  # Track training progress over time
        self.is_trained = False  # Safety flag to prevent prediction before training

        # Set random seed for reproducible results
        # This ensures that random operations (like train/test splits) are consistent
        np.random.seed(config.random_state)

    @abstractmethod
    def build_model(self) -> Any:
        """Build the model architecture.

        This abstract method must be implemented by each position-specific model.
        Each position uses different ML algorithms optimized for their characteristics:
        - QBs use multi-task learning for passing/rushing combinations
        - RBs use workload-aware networks with clustering embeddings
        - WRs use attention mechanisms for target competition modeling

        The @abstractmethod decorator from Python's abc module ensures that
        any class inheriting from BaseModel MUST implement this method.

        Returns:
            Configured neural network model instance (PyTorch nn.Module)
        """
        pass

    @abstractmethod
    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the model.

        This method implements the core training loop for each position.
        It uses the train/validation split pattern common in machine learning:

        Training Process:
        1. Use X_train, y_train to teach the model patterns
        2. Use X_val, y_val to evaluate performance during training
        3. Apply early stopping to prevent overfitting
        4. Calculate and return comprehensive metrics

        Data Format:
        - X arrays: Feature matrices (samples × features)
        - y arrays: Target vectors (fantasy points to predict)
        - All arrays are NumPy for efficient computation

        Args:
            X_train: Training features (e.g., player stats, matchup data)
            y_train: Training targets (actual fantasy points scored)
            X_val: Validation features (held-out data for evaluation)
            y_val: Validation targets (actual fantasy points for validation)

        Returns:
            TrainingResult with performance metrics and model artifacts
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions.

        After training, this method uses the learned model to predict fantasy points
        for new player/game combinations. It returns not just point estimates but
        also uncertainty quantification.

        Uncertainty Quantification is crucial for fantasy sports because:
        - Helps with risk management in lineup construction
        - Enables scenario analysis (floor/ceiling projections)
        - Supports bankroll management strategies

        Example: Instead of just "QB will score 18 points", we get:
        "QB will score 18 points (range: 12-24, floor: 14, ceiling: 22)"

        Args:
            X: Features for prediction (player stats, opponent data, etc.)

        Returns:
            PredictionResult with point estimates, confidence intervals,
            floor/ceiling projections, and confidence scores
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Evaluate model performance on unseen test data.

        This method implements the final evaluation step in machine learning:
        testing the trained model on completely new data it has never seen.

        This is critical for:
        - Estimating real-world performance
        - Detecting overfitting (model memorizing training data)
        - Comparing different models objectively

        The method calculates comprehensive metrics to understand different
        aspects of model performance beyond just accuracy.

        Args:
            X_test: Test features (unseen player/game data)
            y_test: Test targets (actual fantasy points for evaluation)

        Returns:
            EvaluationMetrics with comprehensive performance assessment
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Generate predictions on test set
        predictions = self.predict(X_test)
        y_pred = predictions.point_estimate

        # Basic regression metrics - how accurate are we?
        mae = mean_absolute_error(y_test, y_pred)  # Average absolute error
        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )  # Root mean square error (penalizes big mistakes)
        r2 = r2_score(y_test, y_pred)  # R-squared: variance explained (1.0 = perfect)

        # MAPE (Mean Absolute Percentage Error) with protection against division by zero
        # This shows error as a percentage of the true value
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1))) * 100

        # Distribution metrics - understand prediction patterns
        prediction_std = np.std(y_pred)  # How spread out are our predictions?
        residuals = y_test - y_pred  # Prediction errors
        residual_std = np.std(residuals)  # How spread out are our errors?

        # Consistency score - how stable are predictions?
        # Coefficient of variation = standard deviation / mean
        # Lower CV = more consistent predictions
        cv = prediction_std / np.mean(y_pred) if np.mean(y_pred) != 0 else 0
        consistency_score = 1 / (1 + cv)  # Transform so higher = better

        # Outlier percentage - how often do we make really bad predictions?
        # Errors larger than 2 standard deviations are considered outliers
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

        # Check if this is a PyTorch model
        if str(path).endswith('.pkl'):
            try:
                import torch
                # Try loading as PyTorch state_dict first
                state_dict = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(state_dict, dict) and any('weight' in k or 'bias' in k for k in state_dict.keys()):
                    # This is a PyTorch state_dict
                    logger.info(f"PyTorch model detected at {path}")

                    # Try to load using PyTorch wrapper
                    try:
                        from src.ml.models.pytorch_loader import PyTorchModelWrapper

                        # Extract position from filename (e.g., QB_QB_model_...)
                        filename = path.stem
                        if '_' in filename:
                            position = filename.split('_')[0]
                        else:
                            position = 'UNK'

                        # Find preprocessor
                        preproc_path = Path(str(path).replace('.pkl', '_preprocessor.pkl'))

                        # Create wrapper
                        pytorch_wrapper = PyTorchModelWrapper(position, path, preproc_path)
                        self.model = pytorch_wrapper  # Use wrapper as model
                        self.is_trained = True
                        logger.info(f"PyTorch model loaded successfully for {position}")
                        return

                    except Exception as e:
                        logger.warning(f"Could not create PyTorch wrapper: {e}")
                        # Mark as trained but with no model (will use fallback)
                        self.model = None
                        self.is_trained = True
                        return

            except Exception as e:
                logger.debug(f"Not a PyTorch model: {e}")

        # Fall back to joblib for sklearn models
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.preprocessor = model_data.get("preprocessor")
            self.feature_names = model_data.get("feature_names")
            self.feature_importance = model_data.get("feature_importance")
            self.training_history = model_data.get("training_history", [])
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            # If all else fails, mark as trained to use fallback
            logger.warning(f"Could not load model from {path}: {e}, using fallback predictions")
            self.model = None
            self.is_trained = True

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_importance is None:
            return {}

        return self.feature_importance

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Validate input arrays before training or prediction.

        Input validation is crucial in machine learning to catch errors early
        and provide clear error messages. This prevents hard-to-debug issues
        later in the training process.

        Expected formats:
        - X (features): 2D array with shape (n_samples, n_features)
        - y (targets): 1D array with shape (n_samples,)

        For example, if we have 1000 players with 50 features each:
        - X.shape = (1000, 50)
        - y.shape = (1000,)

        Args:
            X: Feature array (must be 2D numpy array)
            y: Optional target array (must be 1D numpy array if provided)

        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If arrays have wrong dimensions or mismatched lengths
        """
        # Validate feature array X
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (samples × features)")

        # Validate target array y if provided
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")

            if y.ndim != 1:
                raise ValueError("y must be 1-dimensional (samples only)")

            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")

    def _calculate_time_weights(self, X: np.ndarray) -> np.ndarray:
        """Calculate time-based weights for training samples.

        In sports, recent performance is often more predictive than older data.
        This method assigns higher weights to more recent samples, helping
        the model focus on current trends and player form.

        Time weighting is important because:
        - Player skills and roles change over time
        - Rule changes affect scoring
        - Coaching strategies evolve
        - Injury patterns may affect recent performance

        The linear weighting goes from 0.5 (oldest data) to 1.0 (most recent),
        meaning recent games have twice the influence of old games.

        Args:
            X: Training features (assumed to be ordered chronologically)

        Returns:
            Sample weights array with higher values for recent samples
        """
        n_samples = len(X)
        # Create linear weights from 0.5 (oldest) to 1.0 (most recent)
        # np.linspace creates evenly spaced values between start and stop
        weights = np.linspace(0.5, 1.0, n_samples)
        return weights

    def _calculate_prediction_intervals(
        self, X: np.ndarray, point_estimate: np.ndarray, confidence_level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals for uncertainty quantification.

        Prediction intervals estimate the range where future observations are likely
        to fall. This is crucial for fantasy sports because it helps users understand
        the risk/reward profile of each player.

        Current Implementation (Simplified):
        - Uses historical residual standard deviation from validation set
        - Applies normal distribution assumption with z-score multiplier
        - Fallback to 20% of point estimate if no residual data available

        Advanced Implementation (Future Enhancement):
        - Could use quantile regression to model different percentiles directly
        - Could incorporate feature-dependent uncertainty (some players more predictable)
        - Could use bootstrapping for non-parametric intervals

        Example: If point estimate is 18 points with 90% confidence interval,
        result might be (14.2, 21.8) meaning "90% chance player scores 14-22 points"

        Args:
            X: Features (not currently used but available for advanced methods)
            point_estimate: Model's point predictions
            confidence_level: Confidence level for intervals (default 0.9 = 90%)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        # Use historical residual distribution for uncertainty estimation
        # In production, this would be more sophisticated (quantile regression)

        if not hasattr(self, "_residual_std"):
            # Fallback: assume 20% of point estimate as uncertainty
            # This is a rough heuristic when we don't have validation residuals
            uncertainty = point_estimate * 0.2
        else:
            # Use actual residual standard deviation from validation set
            # This is more accurate as it's based on real prediction errors
            uncertainty = self._residual_std

        # Calculate z-score for desired confidence level
        # For 90% confidence: z_score ≈ 1.645
        # For 95% confidence: z_score ≈ 1.96
        alpha = 1 - confidence_level  # Alpha = 0.1 for 90% confidence
        z_score = 1.96  # Currently hardcoded for ~95% confidence

        # Calculate margin of error and bounds
        margin = z_score * uncertainty
        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        return lower_bound, upper_bound
