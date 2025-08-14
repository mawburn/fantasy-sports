"""Model training coordinator for position-specific models."""

import json
import logging
from datetime import datetime
from pathlib import Path

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
    """Coordinate training of position-specific models."""

    MODEL_CLASSES = {"QB": QBModel, "RB": RBModel, "WR": WRModel, "TE": TEModel, "DEF": DEFModel}

    def __init__(self, db_session: Session | None = None, model_dir: Path | None = None):
        """Initialize model trainer.

        Args:
            db_session: Optional database session
            model_dir: Directory to save trained models
        """
        self.db = db_session or next(get_db())
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

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

        Args:
            position: Player position (QB, RB, WR, TE, DEF)
            start_date: Training data start date
            end_date: Training data end date
            config: Optional model configuration
            save_model: Whether to save the trained model

        Returns:
            Dictionary with training results and model metadata
        """
        if position not in self.MODEL_CLASSES:
            raise ValueError(f"Unsupported position: {position}")

        logger.info(f"Starting training for {position} model")

        # Create default config if none provided
        if config is None:
            config = ModelConfig(
                model_name=f"{position}_model", position=position, model_dir=self.model_dir
            )

        # Prepare training data
        data = self.data_preparator.prepare_training_data(
            position=position, start_date=start_date, end_date=end_date
        )

        # Initialize model
        model_class = self.MODEL_CLASSES[position]
        model = model_class(config)

        # Train model
        training_result = model.train(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
        )

        # Evaluate on test set
        test_metrics = model.evaluate(data["X_test"], data["y_test"])

        # Save model and metadata
        model_metadata = None
        if save_model:
            model_metadata = self._save_model_artifacts(
                model=model,
                training_result=training_result,
                test_metrics=test_metrics,
                data_metadata=data["metadata"],
                preprocessor=data.get("scaler"),
            )

        results = {
            "model": model,
            "training_result": training_result,
            "test_metrics": test_metrics,
            "data_metadata": data["metadata"],
            "model_metadata": model_metadata,
        }

        logger.info(f"{position} model training completed:")
        logger.info(f"  Test MAE: {test_metrics.mae:.3f}")
        logger.info(f"  Test R²: {test_metrics.r2:.3f}")
        logger.info(f"  Test MAPE: {test_metrics.mape:.1f}%")

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

        Args:
            position: Player position
            model_configs: List of configurations for base models
            start_date: Training data start date
            end_date: Training data end date
            save_model: Whether to save the ensemble

        Returns:
            Dictionary with ensemble results
        """
        logger.info(f"Training ensemble model for {position} with {len(model_configs)} base models")

        # Prepare data once for all models
        data = self.data_preparator.prepare_training_data(
            position=position, start_date=start_date, end_date=end_date
        )

        # Train base models
        ensemble = EnsembleModel(position)
        base_models = []

        for i, config in enumerate(model_configs):
            logger.info(f"Training base model {i + 1}/{len(model_configs)}")

            # Update config for this specific model
            config.model_name = f"{position}_base_{i}"
            config.save_model = False  # Don't save individual models

            # Train model
            model_class = self.MODEL_CLASSES[position]
            model = model_class(config)

            training_result = model.train(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
            )

            # Add to ensemble
            ensemble.add_model(model, name=f"{config.model_name}")
            base_models.append((model, training_result))

        # Train ensemble
        ensemble_metrics = ensemble.train_ensemble(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
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

        logger.info("Ensemble training completed:")
        logger.info(f"  Test MAE: {test_mae:.3f}")
        logger.info(f"  Test R²: {test_r2:.3f}")
        logger.info(f"  Best individual: {ensemble_metrics['best_individual_mae']:.3f}")
        logger.info(f"  Improvement: {ensemble_metrics['ensemble_improvement']:.3f}")

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

        for position in self.MODEL_CLASSES.keys():
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
                logger.error(f"Failed to train {position} model: {e}")
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

        Args:
            model: Trained model
            training_result: Training results
            test_metrics: Test evaluation metrics
            data_metadata: Training data metadata
            preprocessor: Optional data preprocessor

        Returns:
            Model metadata record
        """
        # Generate unique model ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model.config.position}_{model.config.model_name}_{timestamp}"

        # Save model file
        model_path = self.model_dir / f"{model_id}.pkl"
        model.save_model(model_path)

        # Save preprocessor if provided
        preprocessor_path = None
        if preprocessor is not None:
            preprocessor_path = self.model_dir / f"{model_id}_preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)

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
        test_mae: float,
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
        base_config = ModelConfig(
            model_name=f"{position}_base", position=position, model_dir=self.model_dir
        )

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
            raise ValueError(f"Model not found: {model_id}")

        # Load model file
        model_path = Path(metadata.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if metadata.model_type == "EnsembleModel":
            model = joblib.load(model_path)
        else:
            # Load base model
            model_class = self.MODEL_CLASSES.get(metadata.position)
            if not model_class:
                raise ValueError(f"Unknown position: {metadata.position}")

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
