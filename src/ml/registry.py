"""Model registry and deployment management."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import ModelMetadata
from src.ml.models.base import BaseModel
from src.ml.training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing trained models and deployments."""

    def __init__(self, db_session: Session | None = None):
        """Initialize model registry."""
        self.db = db_session or next(get_db())
        self.trainer = ModelTrainer(self.db)

    def register_model(
        self,
        model: BaseModel,
        model_name: str,
        position: str,
        training_metadata: dict,
        performance_metrics: dict,
        model_path: Path,
        _description: str | None = None,
    ) -> str:
        """Register a trained model in the registry.

        Args:
            model: Trained model instance
            model_name: Name for the model
            position: Player position
            training_metadata: Metadata about training process
            performance_metrics: Model performance metrics
            model_path: Path where model is saved
            description: Optional model description

        Returns:
            Model ID for the registered model
        """
        # Generate unique model ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{position}_{model_name}_{timestamp}"

        # Create metadata record
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            position=position,
            model_type=type(model).__name__,
            version="1.0",
            training_start_date=training_metadata.get("start_date"),
            training_end_date=training_metadata.get("end_date"),
            training_data_size=training_metadata.get("train_samples", 0),
            validation_data_size=training_metadata.get("val_samples", 0),
            hyperparameters="{}",  # Would serialize model config
            feature_names=",".join(training_metadata.get("feature_names", [])),
            feature_count=training_metadata.get("feature_count", 0),
            mae_validation=performance_metrics.get("mae", 0),
            rmse_validation=performance_metrics.get("rmse", 0),
            r2_validation=performance_metrics.get("r2", 0),
            mape_validation=performance_metrics.get("mape", 0),
            status="trained",
            is_active=False,
            model_path=str(model_path),
        )

        self.db.add(metadata)
        self.db.commit()

        logger.info(f"Registered model: {model_id}")
        return model_id

    def deploy_model(
        self, model_id: str, make_active: bool = True, retire_previous: bool = True
    ) -> bool:
        """Deploy a model to production.

        Args:
            model_id: Model to deploy
            make_active: Whether to make this the active model
            retire_previous: Whether to retire previous active models

        Returns:
            True if deployment successful
        """
        # Get model metadata
        model_meta = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not model_meta:
            raise ValueError(f"Model not found: {model_id}") from None

        if model_meta.status != "trained":
            raise ValueError(f"Model {model_id} is not in trained status") from None

        # Retire previous active models for this position if requested
        if retire_previous and make_active:
            previous_active = (
                self.db.query(ModelMetadata)
                .filter(
                    ModelMetadata.position == model_meta.position,
                    ModelMetadata.is_active,
                )
                .all()
            )

            for prev_model in previous_active:
                prev_model.is_active = False
                prev_model.retirement_date = datetime.utcnow()

        # Deploy the model
        model_meta.status = "deployed"
        model_meta.deployment_date = datetime.utcnow()
        if make_active:
            model_meta.is_active = True

        self.db.commit()

        logger.info(f"Deployed model: {model_id} (active: {make_active})")
        return True

    def retire_model(self, model_id: str) -> bool:
        """Retire a deployed model.

        Args:
            model_id: Model to retire

        Returns:
            True if retirement successful
        """
        model_meta = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not model_meta:
            raise ValueError(f"Model not found: {model_id}") from None

        model_meta.status = "retired"
        model_meta.is_active = False
        model_meta.retirement_date = datetime.utcnow()

        self.db.commit()

        logger.info(f"Retired model: {model_id}")
        return True

    def list_models(
        self, position: str | None = None, status: str | None = None, active_only: bool = False
    ) -> list[ModelMetadata]:
        """List models in the registry.

        Args:
            position: Filter by position
            status: Filter by status
            active_only: Only return active models

        Returns:
            List of model metadata
        """
        query = self.db.query(ModelMetadata)

        if position:
            query = query.filter(ModelMetadata.position == position)

        if status:
            query = query.filter(ModelMetadata.status == status)

        if active_only:
            query = query.filter(ModelMetadata.is_active)

        return query.order_by(ModelMetadata.created_at.desc()).all()

    def get_active_model(self, position: str) -> ModelMetadata | None:
        """Get the active model for a position.

        Args:
            position: Player position

        Returns:
            Active model metadata or None
        """
        return (
            self.db.query(ModelMetadata)
            .filter(
                ModelMetadata.position == position,
                ModelMetadata.is_active,
                ModelMetadata.status == "deployed",
            )
            .first()
        )

    def load_model(self, model_id: str) -> BaseModel:
        """Load a model from the registry.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model instance
        """
        return self.trainer.load_model(model_id)

    def compare_models(
        self, model_ids: list[str], metric: str = "mae_validation"
    ) -> list[dict[str, Any]]:
        """Compare multiple models by a metric.

        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare by

        Returns:
            List of model comparisons sorted by metric
        """
        models = self.db.query(ModelMetadata).filter(ModelMetadata.model_id.in_(model_ids)).all()

        comparisons = []
        for model in models:
            comparison = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "position": model.position,
                "mae_validation": model.mae_validation,
                "rmse_validation": model.rmse_validation,
                "r2_validation": model.r2_validation,
                "mape_validation": model.mape_validation,
                "status": model.status,
                "is_active": model.is_active,
                "created_at": model.created_at,
            }
            comparisons.append(comparison)

        # Sort by the specified metric (lower is better for MAE, RMSE, MAPE; higher for R²)
        reverse_sort = metric == "r2_validation"
        return sorted(comparisons, key=lambda x: x.get(metric, 0), reverse=reverse_sort)

    def get_model_performance_history(self, position: str) -> list[dict[str, Any]]:
        """Get performance history for all models of a position.

        Args:
            position: Player position

        Returns:
            List of model performance over time
        """
        models = (
            self.db.query(ModelMetadata)
            .filter(ModelMetadata.position == position)
            .order_by(ModelMetadata.created_at.asc())
            .all()
        )

        history = []
        for model in models:
            history.append(
                {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "created_at": model.created_at,
                    "mae_validation": model.mae_validation,
                    "r2_validation": model.r2_validation,
                    "status": model.status,
                    "is_active": model.is_active,
                }
            )

        return history

    def cleanup_old_models(self, keep_recent: int = 5, keep_active: bool = True) -> int:
        """Clean up old models to save storage space.

        Args:
            keep_recent: Number of recent models to keep per position
            keep_active: Whether to always keep active models

        Returns:
            Number of models cleaned up
        """
        positions = ["QB", "RB", "WR", "TE", "DEF"]
        cleaned_count = 0

        for position in positions:
            # Get all models for position
            models = (
                self.db.query(ModelMetadata)
                .filter(ModelMetadata.position == position)
                .order_by(ModelMetadata.created_at.desc())
                .all()
            )

            # Identify models to delete
            models_to_delete = []

            for i, model in enumerate(models):
                # Skip if it's one of the recent models
                if i < keep_recent:
                    continue

                # Skip if it's active and we want to keep active models
                if keep_active and model.is_active:
                    continue

                # Skip if it's deployed (keep for safety)
                if model.status == "deployed":
                    continue

                models_to_delete.append(model)

            # Delete old models
            for model in models_to_delete:
                # Delete model file if it exists
                if model.model_path:
                    model_file = Path(model.model_path)
                    if model_file.exists():
                        model_file.unlink()
                        logger.info(f"Deleted model file: {model_file}")

                # Delete preprocessor file if it exists
                if model.preprocessor_path:
                    preprocessor_file = Path(model.preprocessor_path)
                    if preprocessor_file.exists():
                        preprocessor_file.unlink()

                # Delete database record
                self.db.delete(model)
                cleaned_count += 1

        self.db.commit()
        logger.info(f"Cleaned up {cleaned_count} old models")
        return cleaned_count

    def validate_model_compatibility(self, model_id: str) -> dict[str, Any]:
        """Validate that a model is compatible with current system.

        Args:
            model_id: Model to validate

        Returns:
            Validation results
        """
        model_meta = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not model_meta:
            return {"valid": False, "error": "Model not found"}

        validation_results = {
            "valid": True,
            "model_id": model_id,
            "checks": {},
        }

        # Check if model file exists
        if model_meta.model_path:
            model_file = Path(model_meta.model_path)
            validation_results["checks"]["file_exists"] = model_file.exists()
            if not model_file.exists():
                validation_results["valid"] = False
        else:
            validation_results["checks"]["file_exists"] = False
            validation_results["valid"] = False

        # Check model status
        validation_results["checks"]["status_valid"] = model_meta.status in ["trained", "deployed"]
        if not validation_results["checks"]["status_valid"]:
            validation_results["valid"] = False

        # Try to load model
        try:
            self.load_model(model_id)
            validation_results["checks"]["loadable"] = True
        except Exception as e:
            validation_results["checks"]["loadable"] = False
            validation_results["checks"]["load_error"] = str(e)
            validation_results["valid"] = False

        return validation_results


class DeploymentPipeline:
    """Automated deployment pipeline for models."""

    def __init__(self, registry: ModelRegistry):
        """Initialize deployment pipeline."""
        self.registry = registry

    def auto_deploy_best_model(
        self, position: str, min_improvement_threshold: float = 0.05
    ) -> dict[str, Any]:
        """Automatically deploy the best performing model for a position.

        Args:
            position: Player position
            min_improvement_threshold: Minimum improvement required to deploy

        Returns:
            Deployment results
        """
        # Get current active model
        current_active = self.registry.get_active_model(position)
        current_mae = current_active.mae_validation if current_active else float("inf")

        # Get all trained models for position
        trained_models = self.registry.list_models(position=position, status="trained")

        if not trained_models:
            return {"deployed": False, "reason": "No trained models available"}

        # Find best model by MAE
        best_model = min(trained_models, key=lambda m: m.mae_validation or float("inf"))
        best_mae = best_model.mae_validation or float("inf")

        # Check improvement threshold
        improvement = (current_mae - best_mae) / current_mae if current_mae > 0 else 0

        if improvement < min_improvement_threshold:
            return {
                "deployed": False,
                "reason": f"Insufficient improvement: {improvement:.3f} < {min_improvement_threshold}",
                "current_mae": current_mae,
                "best_mae": best_mae,
            }

        # Validate model before deployment
        validation = self.registry.validate_model_compatibility(best_model.model_id)
        if not validation["valid"]:
            return {
                "deployed": False,
                "reason": "Model validation failed",
                "validation_errors": validation,
            }

        # Deploy the model
        try:
            self.registry.deploy_model(best_model.model_id, make_active=True, retire_previous=True)

            return {
                "deployed": True,
                "model_id": best_model.model_id,
                "improvement": improvement,
                "current_mae": current_mae,
                "new_mae": best_mae,
                "previous_active": current_active.model_id if current_active else None,
            }

        except Exception as e:
            logger.exception("Deployment failed")
            return {"deployed": False, "reason": f"Deployment error: {e!s}"}

    def rollback_deployment(self, position: str) -> dict[str, Any]:
        """Rollback to the previous active model.

        Args:
            position: Player position

        Returns:
            Rollback results
        """
        # Get current active model
        current_active = self.registry.get_active_model(position)
        if not current_active:
            return {"success": False, "reason": "No active model to rollback from"}

        # Get previous deployed model
        previous_models = (
            self.registry.db.query(ModelMetadata)
            .filter(
                ModelMetadata.position == position,
                ModelMetadata.status == "deployed",
                ModelMetadata.model_id != current_active.model_id,
                ModelMetadata.retirement_date.isnot(None),
            )
            .order_by(ModelMetadata.retirement_date.desc())
            .first()
        )

        if not previous_models:
            return {"success": False, "reason": "No previous model available for rollback"}

        try:
            # Retire current active model
            self.registry.retire_model(current_active.model_id)

            # Reactivate previous model
            previous_models.is_active = True
            previous_models.retirement_date = None
            self.registry.db.commit()

            return {
                "success": True,
                "rolled_back_from": current_active.model_id,
                "rolled_back_to": previous_models.model_id,
            }

        except Exception as e:
            logger.exception("Rollback failed")
            return {"success": False, "reason": f"Rollback error: {e!s}"}
