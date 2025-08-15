"""Model training coordinator for position-specific models.

This file orchestrates the entire ML model training pipeline:

1. Model Selection: Choose appropriate model class for each position
2. Data Preparation: Coordinate with DataPreparator for clean datasets
3. Training Execution: Run training with proper validation and early stopping
4. Model Evaluation: Comprehensive testing on held-out data
5. Model Persistence: Save trained models and metadata to database
6. Ensemble Training: Coordinate multiple models for improved accuracy

Key Concepts for Beginners:

Training Pipeline: The systematic process of turning raw data into a
deployed ML model. Includes data prep, model training, evaluation, and deployment.

Model Registry: Centralized tracking of model versions, performance metrics,
and metadata. Critical for production ML systems.

Early Stopping: Technique to prevent overfitting by stopping training when
validation performance stops improving.

Ensemble Methods: Combining multiple models for better performance than
any individual model. Common in competitive ML.

Hyperparameter Tuning: Finding optimal model settings (learning rate,
max depth, etc.) through systematic search or expert knowledge.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import ModelMetadata
from src.ml.models.base import BaseModel, ModelConfig
from src.ml.models.ensemble import EnsembleModel
from src.ml.models.position_models import DEFModel, QBModel, RBModel, TEModel, WRModel

from .data_preparation import DataPreparator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Coordinate training of position-specific models.

    This class serves as the central coordinator for all model training activities.
    It handles the complexity of training different types of models for different
    positions while maintaining consistent interfaces and quality standards.

    Architecture Pattern: Strategy + Factory
    - Factory: MODEL_CLASSES maps position strings to model classes
    - Strategy: Each position uses different algorithms/hyperparameters

    Responsibilities:
    - Instantiate appropriate model classes for each position
    - Coordinate data preparation and model training
    - Manage model evaluation and performance tracking
    - Handle model persistence and metadata recording
    - Support both individual and ensemble model training

    The class maintains database connections for loading training data
    and storing model artifacts/metadata.
    """

    # Factory pattern: Map position strings to model classes
    # ClassVar indicates this is a class-level constant shared by all instances
    MODEL_CLASSES: ClassVar[dict[str, type]] = {
        "QB": QBModel,  # Quarterback model using XGBoost
        "RB": RBModel,  # Running back model using LightGBM with clustering
        "WR": WRModel,  # Wide receiver model using XGBoost
        "TE": TEModel,  # Tight end model using LightGBM
        "DEF": DEFModel,  # Defense/Special teams model using Random Forest
    }

    def __init__(self, db_session: Session | None = None, model_dir: Path | None = None):
        """Initialize model trainer with database connection and file system setup.

        The trainer needs:
        - Database session for loading data and storing model metadata
        - File system directory for saving trained model artifacts
        - Data preparator for consistent data preprocessing pipeline

        Args:
            db_session: Optional database session (creates new if None)
            model_dir: Directory to save trained models (defaults to "models/")
        """
        # Database connection for data loading and metadata storage
        self.db = db_session or next(get_db())

        # File system setup for model artifacts
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        # Data preparation pipeline (handles feature extraction, cleaning, scaling)
        self.data_preparator = DataPreparator(self.db)

    def train_position_model(
        self,
        position: str,
        start_date: datetime,
        end_date: datetime,
        config: ModelConfig | None = None,
        save_model: bool = True,
    ) -> dict:
        """Train a model for a specific position.

        This is the main entry point for training individual position models.
        It coordinates the entire training pipeline from data extraction to
        model deployment.

        Training Pipeline:
        1. Validate position and create default config if needed
        2. Prepare training data (extract, clean, split)
        3. Instantiate appropriate model class for position
        4. Train model with validation and early stopping
        5. Evaluate on held-out test set
        6. Save model artifacts and metadata to database
        7. Return comprehensive training results

        Error Handling:
        - Validates position against supported types
        - Handles data preparation failures gracefully
        - Logs comprehensive training progress and results

        Args:
            position: Player position (QB, RB, WR, TE, DEF)
            start_date: Training data start date (earliest games to include)
            end_date: Training data end date (latest games to include)
            config: Optional model configuration (uses defaults if None)
            save_model: Whether to save the trained model (True for production)

        Returns:
            Dictionary containing:
            - model: Trained model instance
            - training_result: Metrics from training process
            - test_metrics: Performance on held-out test set
            - data_metadata: Information about training dataset
            - model_metadata: Database record for model tracking
        """
        # Validate that we support this position
        if position not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unsupported position: {position}. Supported: {list(self.MODEL_CLASSES.keys())}"
            ) from None

        logger.info(f"Starting training for {position} model from {start_date} to {end_date}")

        # Create default configuration if none provided
        # This ensures consistent defaults while allowing customization
        if config is None:
            config = ModelConfig(
                model_name=f"{position}_model",  # Descriptive model name
                position=position,  # Position identifier
                model_dir=self.model_dir,  # Where to save model files
            )

        # Step 1: Prepare training data using consistent pipeline
        # This handles all data preprocessing: extraction, cleaning, splitting, scaling
        data = self.data_preparator.prepare_training_data(
            position=position, start_date=start_date, end_date=end_date
        )

        # Step 2: Initialize position-appropriate model
        # Use factory pattern to get the right model class for this position
        model_class = self.MODEL_CLASSES[position]
        model = model_class(config)

        # Step 3: Train model with validation monitoring
        # Each model implements position-specific training strategies
        training_result = model.train(
            X_train=data["X_train"],  # Training features
            y_train=data["y_train"],  # Training targets (fantasy points)
            X_val=data["X_val"],  # Validation features (for monitoring)
            y_val=data["y_val"],  # Validation targets (for early stopping)
        )

        # Step 4: Evaluate on held-out test set
        # This gives unbiased estimate of real-world performance
        test_metrics = model.evaluate(data["X_test"], data["y_test"])

        # Step 5: Save model artifacts and metadata (if requested)
        model_metadata = None
        if save_model:
            # Save both the model files and database metadata for tracking
            model_metadata = self._save_model_artifacts(
                model=model,  # Trained model instance
                training_result=training_result,  # Training process metrics
                test_metrics=test_metrics,  # Final performance metrics
                data_metadata=data["metadata"],  # Dataset information
                preprocessor=data.get("scaler"),  # Data preprocessing pipeline
            )

        # Step 6: Package comprehensive results for return
        results = {
            "model": model,  # Trained model ready for predictions
            "training_result": training_result,  # Training process details
            "test_metrics": test_metrics,  # Performance on unseen data
            "data_metadata": data["metadata"],  # Dataset characteristics
            "model_metadata": model_metadata,  # Database record (if saved)
        }

        # Log training completion with key performance metrics
        logger.info(f"{position} model training completed successfully:")
        logger.info(f"  Test MAE: {test_metrics.mae:.3f} points")  # Average error
        logger.info(f"  Test R²: {test_metrics.r2:.3f}")  # Variance explained
        logger.info(f"  Test MAPE: {test_metrics.mape:.1f}%")  # Percentage error

        return results

    def train_ensemble_model(
        self,
        position: str,
        model_configs: list[ModelConfig],
        start_date: datetime,
        end_date: datetime,
        save_model: bool = True,
    ) -> dict:
        """Train an ensemble model for a position.

        Ensemble Learning: Combine multiple models to achieve better performance
        than any individual model. Common ensemble strategies:
        - Bagging: Train models on different data subsets (Random Forest)
        - Boosting: Train models sequentially to correct errors (XGBoost)
        - Voting: Combine predictions from diverse models (this implementation)

        Why Ensembles Work:
        - Individual models have different biases and blind spots
        - Combining diverse models reduces overall error
        - Robust to outliers and data quality issues
        - Often used in competitions and production systems

        Ensemble Training Process:
        1. Train multiple base models with different configurations
        2. Combine their predictions using weighted averaging
        3. Train meta-model to learn optimal combination weights
        4. Evaluate ensemble vs best individual model

        Trade-offs:
        - Better accuracy vs increased complexity
        - Slower inference vs more robust predictions
        - Harder to interpret vs better performance

        Args:
            position: Player position (QB, RB, WR, TE, DEF)
            model_configs: List of configurations for base models (diversity is key)
            start_date: Training data start date
            end_date: Training data end date
            save_model: Whether to save the ensemble (recommended for production)

        Returns:
            Dictionary with ensemble training results and performance comparison
        """
        logger.info(f"Training ensemble model for {position} with {len(model_configs)} base models")

        # Prepare data once and share across all base models (efficiency)
        # This ensures all models train on identical data for fair comparison
        data = self.data_preparator.prepare_training_data(
            position=position, start_date=start_date, end_date=end_date
        )

        # Initialize ensemble and train base models
        ensemble = EnsembleModel(position)
        base_models = []  # Track individual model performance for analysis

        # Train each base model with different configuration
        for i, config in enumerate(model_configs):
            logger.info(f"Training base model {i + 1}/{len(model_configs)}")

            # Customize config for this specific base model
            config.model_name = f"{position}_base_{i}"
            config.save_model = False  # Don't save individual models (only ensemble)

            # Instantiate and train base model
            model_class = self.MODEL_CLASSES[position]
            model = model_class(config)

            # Train on shared dataset
            training_result = model.train(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
            )

            # Add trained model to ensemble
            ensemble.add_model(model, name=f"{config.model_name}")
            base_models.append((model, training_result))  # Track for analysis

        # Train the meta-learner (ensemble combination weights)
        # This learns how to optimally combine the base model predictions
        ensemble_metrics = ensemble.train_ensemble(
            X_train=data["X_train"],  # Same training data
            y_train=data["y_train"],  # Same training targets
            X_val=data["X_val"],  # Same validation data
            y_val=data["y_val"],  # Same validation targets
        )

        # Evaluate ensemble on test set
        test_predictions = ensemble.predict(data["X_test"])
        test_mae = np.mean(np.abs(data["y_test"] - test_predictions.point_estimate))
        test_r2 = 1 - np.sum((data["y_test"] - test_predictions.point_estimate) ** 2) / np.sum(
            (data["y_test"] - np.mean(data["y_test"])) ** 2
        )

        # Save ensemble
        ensemble_metadata = None
        if save_model:
            ensemble_metadata = self._save_ensemble_artifacts(
                ensemble=ensemble,
                ensemble_metrics=ensemble_metrics,
                test_mae=test_mae,
                test_r2=test_r2,
                data_metadata=data["metadata"],
            )

        results = {
            "ensemble": ensemble,
            "base_models": base_models,
            "ensemble_metrics": ensemble_metrics,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "data_metadata": data["metadata"],
            "ensemble_metadata": ensemble_metadata,
        }

        # Log ensemble performance vs individual models
        logger.info("Ensemble training completed:")
        logger.info(f"  Ensemble Test MAE: {test_mae:.3f} points")
        logger.info(f"  Ensemble Test R²: {test_r2:.3f}")
        logger.info(f"  Best Individual MAE: {ensemble_metrics['best_individual_mae']:.3f} points")
        logger.info(
            f"  Ensemble Improvement: {ensemble_metrics['ensemble_improvement']:.3f} points"
        )

        # Performance analysis
        if ensemble_metrics["ensemble_improvement"] > 0:
            logger.info("✅ Ensemble outperformed best individual model")
        else:
            logger.warning("⚠️ Ensemble did not improve over best individual model")

        return results

    def train_all_positions(
        self, start_date: datetime, end_date: datetime, use_ensemble: bool = False
    ) -> dict[str, dict]:
        """Train models for all positions.

        Args:
            start_date: Training data start date
            end_date: Training data end date
            use_ensemble: Whether to train ensemble models

        Returns:
            Dictionary mapping positions to training results
        """
        results = {}

        for position in self.MODEL_CLASSES:
            try:
                if use_ensemble:
                    # Create multiple configs for ensemble
                    configs = self._create_ensemble_configs(position)
                    results[position] = self.train_ensemble_model(
                        position=position,
                        model_configs=configs,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    # Train single model
                    results[position] = self.train_position_model(
                        position=position, start_date=start_date, end_date=end_date
                    )

            except Exception as e:
                logger.exception("Failed to train %s model", position)
                results[position] = {"error": str(e)}

        return results

    def _save_model_artifacts(
        self,
        model: BaseModel,
        training_result,
        test_metrics,
        data_metadata: dict,
        preprocessor=None,
    ) -> ModelMetadata:
        """Save model artifacts and metadata to database.

        Model persistence is critical for production ML systems:

        Artifacts Saved:
        - Model file: Serialized model object (pickle format)
        - Preprocessor: Data scaling/transformation pipeline
        - Metadata: Training metrics, data info, hyperparameters

        Why Save Metadata?
        - Track model performance over time
        - Enable model comparison and selection
        - Debug production issues
        - Comply with ML governance requirements
        - Enable model rollback if needed

        File Naming Convention:
        {position}_{model_name}_{timestamp}.pkl
        Example: QB_model_20240315_143022.pkl

        Database Record:
        - Links to file paths for artifacts
        - Stores performance metrics for quick lookup
        - Tracks training configuration for reproducibility

        Args:
            model: Trained model instance
            training_result: Training process results and metrics
            test_metrics: Performance on held-out test set
            data_metadata: Information about training dataset
            preprocessor: Optional data preprocessing pipeline

        Returns:
            ModelMetadata: Database record for tracking and retrieval
        """
        # Generate unique model identifier with timestamp
        # Format: POSITION_MODELNAME_YYYYMMDD_HHMMSS
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model.config.position}_{model.config.model_name}_{timestamp}"

        # Save model artifact to disk
        model_path = self.model_dir / f"{model_id}.pkl"
        model.save_model(model_path)
        logger.info(f"Saved model artifact: {model_path}")

        # Save data preprocessor if provided (critical for consistent inference)
        preprocessor_path = None
        if preprocessor is not None:
            preprocessor_path = self.model_dir / f"{model_id}_preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Saved preprocessor: {preprocessor_path}")

        # Create database record
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model.config.model_name,
            position=model.config.position,
            model_type=type(model).__name__,
            version=model.config.version,
            training_start_date=data_metadata["start_date"],
            training_end_date=data_metadata["end_date"],
            training_data_size=data_metadata["train_samples"],
            validation_data_size=data_metadata["val_samples"],
            hyperparameters=json.dumps(model.config.__dict__, default=str),
            feature_names=",".join(data_metadata["feature_names"]),
            feature_count=data_metadata["feature_count"],
            mae_validation=training_result.val_mae,
            rmse_validation=training_result.val_rmse,
            r2_validation=training_result.val_r2,
            mape_validation=test_metrics.mape,
            status="trained",
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path) if preprocessor_path else None,
        )

        self.db.add(metadata)
        self.db.commit()

        logger.info(f"Saved model metadata with ID: {model_id}")
        return metadata

    def _save_ensemble_artifacts(
        self,
        ensemble: EnsembleModel,
        ensemble_metrics: dict,
        _test_mae: float,
        test_r2: float,
        data_metadata: dict,
    ) -> ModelMetadata:
        """Save ensemble artifacts and metadata.

        Args:
            ensemble: Trained ensemble model
            ensemble_metrics: Ensemble training metrics
            test_mae: Test MAE score
            test_r2: Test R² score
            data_metadata: Training data metadata

        Returns:
            Model metadata record
        """
        # Generate unique ensemble ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{ensemble.position}_ensemble_{timestamp}"

        # Save ensemble
        model_path = self.model_dir / f"{model_id}.pkl"
        joblib.dump(ensemble, model_path)

        # Create metadata record
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=f"{ensemble.position}_ensemble",
            position=ensemble.position,
            model_type="EnsembleModel",
            version="1.0",
            training_start_date=data_metadata["start_date"],
            training_end_date=data_metadata["end_date"],
            training_data_size=data_metadata["train_samples"],
            validation_data_size=data_metadata["val_samples"],
            hyperparameters=json.dumps(ensemble_metrics, default=str),
            feature_names=",".join(data_metadata["feature_names"]),
            feature_count=data_metadata["feature_count"],
            mae_validation=ensemble_metrics["ensemble_mae"],
            rmse_validation=0.0,  # Not calculated for ensemble
            r2_validation=test_r2,
            status="trained",
            model_path=str(model_path),
        )

        self.db.add(metadata)
        self.db.commit()

        logger.info(f"Saved ensemble metadata with ID: {model_id}")
        return metadata

    def _create_ensemble_configs(self, position: str) -> list[ModelConfig]:
        """Create multiple model configurations for ensemble training.

        Args:
            position: Player position

        Returns:
            List of model configurations
        """
        # base_config = ModelConfig(
        #     model_name=f"{position}_base", position=position, model_dir=self.model_dir
        # )

        # Create variations for ensemble diversity
        configs = []

        # Config 1: Standard configuration
        config1 = ModelConfig(
            model_name=f"{position}_standard",
            position=position,
            model_dir=self.model_dir,
            random_state=42,
        )
        configs.append(config1)

        # Config 2: Different random seed
        config2 = ModelConfig(
            model_name=f"{position}_variant1",
            position=position,
            model_dir=self.model_dir,
            random_state=123,
        )
        configs.append(config2)

        # Config 3: Different validation strategy
        config3 = ModelConfig(
            model_name=f"{position}_variant2",
            position=position,
            model_dir=self.model_dir,
            random_state=456,
            validation_size=0.15,  # Smaller validation set
        )
        configs.append(config3)

        return configs

    def load_model(self, model_id: str) -> BaseModel:
        """Load a trained model from database.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model instance
        """
        # Get model metadata
        metadata = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not metadata:
            raise ValueError(f"Model not found: {model_id}") from None

        # Load model file
        model_path = Path(metadata.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}") from None

        if metadata.model_type == "EnsembleModel":
            model = joblib.load(model_path)
        else:
            # Load base model
            model_class = self.MODEL_CLASSES.get(metadata.position)
            if not model_class:
                raise ValueError(f"Unknown position: {metadata.position}") from None

            # Create model config from metadata
            config = ModelConfig(
                model_name=metadata.model_name,
                position=metadata.position,
                version=metadata.version,
                model_dir=self.model_dir,
            )

            model = model_class(config)
            model.load_model(model_path)

        logger.info(f"Loaded model: {model_id}")
        return model
