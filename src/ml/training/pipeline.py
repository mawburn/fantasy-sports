"""Training pipeline orchestration for ML models."""

import logging
from datetime import datetime
from typing import Any

from src.ml.training.data_preparation import DataPreparator
from src.ml.training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete ML training pipeline.

    This class coordinates data preparation, model training, evaluation,
    and deployment in a systematic, repeatable process.
    """

    def __init__(self):
        """Initialize the training pipeline."""
        self.data_preparator = DataPreparator()
        self.model_trainer = ModelTrainer()

    def run_full_pipeline(
        self,
        positions: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        ensemble: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the complete training pipeline for specified positions.

        Args:
            positions: List of positions to train models for
            start_date: Start date for training data
            end_date: End date for training data
            ensemble: Whether to create ensemble models
            **kwargs: Additional arguments for training

        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Starting training pipeline for positions: {positions}")

        results = {}
        pipeline_start = datetime.now()

        try:
            # Step 1: Data preparation
            logger.info("Step 1: Preparing training data")
            data_prep_results = self._prepare_data(
                positions=positions, start_date=start_date, end_date=end_date
            )
            results["data_preparation"] = data_prep_results

            # Step 2: Model training
            logger.info("Step 2: Training models")
            training_results = self._train_models(positions=positions, ensemble=ensemble, **kwargs)
            results["training"] = training_results

            # Step 3: Pipeline summary
            pipeline_end = datetime.now()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

            results["pipeline_summary"] = {
                "start_time": pipeline_start.isoformat(),
                "end_time": pipeline_end.isoformat(),
                "duration_seconds": pipeline_duration,
                "positions_trained": positions,
                "model_type": "neural",
                "ensemble": ensemble,
                "success": True,
            }

            logger.info(
                f"Training pipeline completed successfully in {pipeline_duration:.2f} seconds"
            )

        except Exception as e:
            logger.exception("Training pipeline failed")
            results["error"] = str(e)
            results["success"] = False

        return results

    def _prepare_data(
        self, positions: list[str], start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Prepare training data for all positions."""
        logger.info("Preparing training data...")

        # For now, data preparation is handled within model training
        # This is a placeholder for future dedicated data preparation steps
        return {
            "positions": positions,
            "start_date": start_date,
            "end_date": end_date,
            "status": "completed",
        }

    def _train_models(
        self, positions: list[str], ensemble: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Train models for all specified positions."""
        training_results = {}

        for position in positions:
            logger.info(f"Training {position} model...")

            try:
                if ensemble:
                    # Train ensemble model
                    result = self.model_trainer.train_ensemble_model(position=position, **kwargs)
                else:
                    # Train individual model
                    result = self.model_trainer.train_position_model(position=position, **kwargs)

                training_results[position] = {
                    "status": "success",
                    "model_type": "ensemble" if ensemble else "neural",
                    "metrics": result.get("metrics", {}),
                    "model_id": result.get("model_id"),
                }

            except Exception as e:
                logger.exception(f"Failed to train {position} model")
                training_results[position] = {"status": "failed", "error": str(e)}

        return training_results

    def validate_pipeline_config(self, positions: list[str], **kwargs: Any) -> bool:
        """Validate pipeline configuration before running."""
        valid_positions = ["QB", "RB", "WR", "TE", "DEF"]

        for position in positions:
            if position not in valid_positions:
                logger.error(f"Invalid position: {position}. Valid positions: {valid_positions}")
                return False

        return True
