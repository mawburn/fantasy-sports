"""Model registry and deployment management for production ML systems.

This file implements a comprehensive model lifecycle management system that handles:

1. Model Registration: Catalog trained models with metadata and performance metrics
2. Model Deployment: Promote models from training to production with validation
3. Model Versioning: Track multiple versions of models for rollback and comparison
4. Model Retirement: Safely decommission outdated or poorly performing models
5. Automated Deployment: Intelligent promotion of better-performing models
6. Model Cleanup: Storage management and artifact cleanup

Key Concepts for Beginners:

Model Registry: Centralized catalog of all trained models, similar to a code repository
but for ML models. Tracks model versions, performance, deployment status, and metadata.

Model Lifecycle States:
- 'trained': Model completed training but not yet deployed
- 'deployed': Model is available for predictions in production
- 'retired': Model has been decommissioned and is no longer used

Blue-Green Deployment: Safe deployment strategy where new models are validated
before replacing active models, with ability to quickly rollback if issues arise.

Model Governance: Systematic tracking and management of ML models for compliance,
auditability, and operational safety in production environments.

This system ensures that only validated, high-performing models serve predictions
while maintaining the ability to quickly rollback if performance degrades.
"""

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
    """Registry for managing trained models and deployments.

    The ModelRegistry serves as the central control system for all ML models in production.
    It provides a complete model lifecycle management solution that ensures safe,
    reliable, and auditable model deployments.

    Core Responsibilities:
    1. Model Cataloging: Maintain comprehensive inventory of all trained models
    2. Performance Tracking: Store and compare model performance metrics
    3. Deployment Safety: Validate models before promotion to production
    4. Version Control: Enable rollbacks and A/B testing between model versions
    5. Operational Monitoring: Track model status and deployment history
    6. Storage Management: Clean up old models to optimize disk usage

    Database Schema:
    The registry maintains model metadata in the ModelMetadata table, including:
    - Performance metrics (MAE, R², etc.)
    - Training configuration and hyperparameters
    - File paths to model artifacts
    - Deployment status and dates
    - Feature specifications and version information

    Integration Points:
    - ModelTrainer: Loads and validates model artifacts
    - Database: Persists all model metadata and deployment history
    - File System: Manages model artifact storage and cleanup

    For beginners: Think of this as a "model warehouse" that safely stores,
    catalogs, and manages all your trained models, similar to how a code
    repository manages different versions of your code.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize model registry with database connection and trainer.

        The registry requires:
        - Database session for metadata storage and queries
        - ModelTrainer instance for loading and validating model artifacts

        Args:
            db_session: Optional database session (creates new if None)
        """
        # Database connection for model metadata operations
        self.db = db_session or next(get_db())

        # Model trainer for loading and validating model artifacts
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

        Model registration is the first step in the ML deployment pipeline.
        It creates a permanent record of the trained model with all necessary
        metadata for future deployment, comparison, and governance.

        Registration Process:
        1. Generate unique model identifier with timestamp
        2. Extract and validate training metadata
        3. Store performance metrics for comparison
        4. Record model artifacts and file paths
        5. Set initial status as 'trained' (not yet deployed)
        6. Commit metadata to database for permanent storage

        Why Register Models?
        - Traceability: Track which model generated which predictions
        - Comparison: Compare performance across different model versions
        - Governance: Maintain audit trail for regulatory compliance
        - Deployment Safety: Validate models before production deployment
        - Rollback Capability: Enable quick reversion to previous versions

        Model ID Format: {position}_{model_name}_{timestamp}
        Example: QB_xgboost_20240315_143022

        Args:
            model: Trained model instance (must implement BaseModel interface)
            model_name: Descriptive name for the model (e.g., 'xgboost', 'ensemble')
            position: Player position (QB, RB, WR, TE, DEF)
            training_metadata: Dictionary with training process information
            performance_metrics: Dictionary with validation performance metrics
            model_path: File system path where model artifacts are stored
            _description: Optional human-readable description (unused parameter)

        Returns:
            Unique model ID for future reference and deployment operations
        """
        # Step 1: Generate unique model identifier
        # Format: POSITION_MODELNAME_YYYYMMDD_HHMMSS
        # This ensures no collisions and provides chronological ordering
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{position}_{model_name}_{timestamp}"

        # Step 2: Create comprehensive metadata record for the database
        # This stores all information needed for deployment, comparison, and governance
        metadata = ModelMetadata(
            # Model identification
            model_id=model_id,  # Unique identifier
            model_name=model_name,  # Human-readable name
            position=position,  # Player position (QB, RB, etc.)
            model_type=type(model).__name__,  # Algorithm type (XGBRegressor, etc.)
            version="1.0",  # Version string for compatibility
            # Training process metadata
            training_start_date=training_metadata.get("start_date"),  # Training data period
            training_end_date=training_metadata.get("end_date"),  # Training data period
            training_data_size=training_metadata.get("train_samples", 0),  # Sample counts
            validation_data_size=training_metadata.get("val_samples", 0),  # Sample counts
            # Model configuration (placeholder for hyperparameter serialization)
            hyperparameters="{}",  # TODO: Serialize model config to JSON
            # Feature information for compatibility checking
            feature_names=",".join(training_metadata.get("feature_names", [])),  # Feature list
            feature_count=training_metadata.get("feature_count", 0),  # Feature count
            # Performance metrics for model comparison and selection
            mae_validation=performance_metrics.get("mae", 0),  # Mean Absolute Error
            rmse_validation=performance_metrics.get("rmse", 0),  # Root Mean Square Error
            r2_validation=performance_metrics.get("r2", 0),  # R-squared
            mape_validation=performance_metrics.get("mape", 0),  # Mean Absolute Percentage Error
            # Deployment status tracking
            status="trained",  # Initial status (not yet deployed)
            is_active=False,  # Not active until explicitly deployed
            # File system references
            model_path=str(model_path),  # Path to serialized model file
        )

        # Step 3: Persist metadata to database
        # This creates the permanent record for tracking and deployment
        self.db.add(metadata)
        self.db.commit()  # Ensure data is saved before returning

        # Step 4: Log successful registration for operational monitoring
        logger.info(f"Registered model: {model_id}")
        return model_id

    def deploy_model(
        self, model_id: str, make_active: bool = True, retire_previous: bool = True
    ) -> bool:
        """Deploy a model to production.

        Model deployment is a critical operation that promotes a trained model
        to production status where it can serve live predictions. This implements
        a safe deployment strategy with validation and rollback capabilities.

        Deployment Safety Measures:
        1. Validate model exists and is in 'trained' status
        2. Optionally retire previous active models (prevents conflicts)
        3. Update model status to 'deployed' with timestamp
        4. Activate model for live predictions (if requested)
        5. Maintain deployment audit trail

        Blue-Green Deployment Pattern:
        - Previous models remain available for quick rollback
        - New models are validated before activation
        - Zero-downtime deployment with immediate fallback capability

        Why This Process Matters:
        - Production Safety: Prevents deployment of invalid models
        - Operational Continuity: Ensures always-available prediction service
        - Change Management: Maintains audit trail of all deployments
        - Risk Mitigation: Enables quick rollback if performance degrades

        Args:
            model_id: Unique identifier of model to deploy
            make_active: Whether to activate for live predictions (default: True)
            retire_previous: Whether to deactivate previous active models (default: True)

        Returns:
            True if deployment completed successfully

        Raises:
            ValueError: If model not found or not in 'trained' status
        """
        # Step 1: Validate model exists and is deployable
        model_meta = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        # Ensure model exists in registry
        if not model_meta:
            raise ValueError(f"Model not found: {model_id}") from None

        # Ensure model is in correct state for deployment
        # Only 'trained' models can be deployed (not retired or already deployed)
        if model_meta.status != "trained":
            raise ValueError(
                f"Model {model_id} is not in trained status (current: {model_meta.status})"
            ) from None

        # Step 2: Handle previous active models (Blue-Green deployment)
        if retire_previous and make_active:
            # Find all currently active models for this position
            # This prevents multiple active models causing conflicts
            previous_active = (
                self.db.query(ModelMetadata)
                .filter(
                    ModelMetadata.position == model_meta.position,  # Same position only
                    ModelMetadata.is_active,  # Currently active
                )
                .all()
            )

            # Deactivate previous models with retirement timestamp
            # This maintains audit trail while preventing conflicts
            for prev_model in previous_active:
                prev_model.is_active = False
                prev_model.retirement_date = datetime.utcnow()
                logger.info(f"Retiring previously active model: {prev_model.model_id}")

        # Step 3: Deploy the new model
        # Update status and timestamps to reflect deployment
        model_meta.status = "deployed"  # Mark as production-ready
        model_meta.deployment_date = datetime.utcnow()  # Record deployment time

        if make_active:
            model_meta.is_active = True  # Activate for live predictions

        # Step 4: Commit all changes atomically
        # Either all changes succeed or none do (prevents partial state)
        self.db.commit()

        # Step 5: Log successful deployment for operational monitoring
        logger.info(f"Successfully deployed model: {model_id} (active: {make_active})")
        return True

    def retire_model(self, model_id: str) -> bool:
        """Retire a deployed model.

        Model retirement is the final step in a model's lifecycle. It safely
        removes models from active duty while preserving their historical
        records for audit trails and potential emergency rollbacks.

        When to Retire Models:
        - Performance degradation detected in production
        - Better models have been deployed and proven stable
        - Model becomes incompatible with system changes
        - Scheduled retirement as part of model lifecycle policy
        - Emergency situations requiring immediate model deactivation

        Retirement Process:
        1. Validate model exists in the registry
        2. Update status to 'retired' (prevents future activation)
        3. Deactivate model (stops serving predictions)
        4. Record retirement timestamp (audit trail)
        5. Commit changes atomically

        Safety Considerations:
        - Preserves model files and metadata for historical analysis
        - Maintains database records for compliance and auditing
        - Allows potential reactivation in emergency scenarios
        - Does not delete any artifacts (use cleanup for that)

        Args:
            model_id: Unique identifier of model to retire

        Returns:
            True if retirement completed successfully

        Raises:
            ValueError: If model not found in registry
        """
        # Step 1: Validate model exists
        model_meta = self.db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not model_meta:
            raise ValueError(f"Model not found: {model_id}") from None

        # Step 2: Update model status and deactivate
        model_meta.status = "retired"  # Mark as retired (prevents reactivation)
        model_meta.is_active = False  # Deactivate immediately
        model_meta.retirement_date = datetime.utcnow()  # Record when retired

        # Step 3: Commit changes to database
        self.db.commit()

        # Step 4: Log retirement for operational monitoring
        logger.info(f"Successfully retired model: {model_id}")
        return True

    def list_models(
        self, position: str | None = None, status: str | None = None, active_only: bool = False
    ) -> list[ModelMetadata]:
        """List models in the registry.

        This method provides flexible querying capabilities for exploring the
        model registry. It supports various filtering options to find specific
        models or analyze model populations.

        Common Use Cases:
        - list_models(): Get all models across all positions
        - list_models(position='QB'): Get all QB models
        - list_models(status='deployed'): Get all deployed models
        - list_models(active_only=True): Get currently active models
        - list_models(position='RB', status='trained'): Get trained RB models

        Query Strategy:
        - Builds query incrementally with optional filters
        - Orders by creation date (most recent first)
        - Returns full ModelMetadata objects with all attributes

        Performance Considerations:
        - Uses database indexes on position, status, is_active columns
        - Reasonable for registry sizes up to thousands of models
        - Consider pagination for very large registries

        Args:
            position: Filter by player position (QB, RB, WR, TE, DEF)
            status: Filter by deployment status ('trained', 'deployed', 'retired')
            active_only: If True, only return models with is_active=True

        Returns:
            List of ModelMetadata objects matching the specified filters,
            ordered by creation date (newest first)
        """
        # Start with base query for all models
        query = self.db.query(ModelMetadata)

        # Apply optional filters to narrow results
        if position:
            query = query.filter(ModelMetadata.position == position)

        if status:
            query = query.filter(ModelMetadata.status == status)

        if active_only:
            query = query.filter(ModelMetadata.is_active)

        # Order by creation date (newest first) and execute query
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
