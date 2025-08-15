"""Ensemble model for combining multiple base models to improve predictions.

Ensemble Methods in Machine Learning:

Ensemble methods combine multiple individual models (called "base learners" or "weak learners")
to create a stronger, more robust prediction system. The key insight is that different models
make different types of errors, and by combining them intelligently, we can:
1. Reduce overfitting (variance reduction)
2. Improve generalization to new data
3. Increase prediction accuracy and reliability
4. Provide better uncertainty estimates

Types of Ensemble Methods:
1. Bagging: Train models on different data samples (Random Forest)
2. Boosting: Train models sequentially, focusing on previous errors (XGBoost)
3. Stacking: Use a meta-model to learn how to combine base model predictions
4. Voting/Averaging: Simple combination by voting or weighted averaging

This Implementation:
Combines stacking (meta-model) with weighted averaging for robust predictions.
The ensemble learns optimal weights based on each model's validation performance
and uses a Ridge regression meta-model to capture non-linear combinations.

For Fantasy Sports:
Different models excel at different aspects:
- XGBoost: Captures feature interactions well
- LightGBM: Handles categorical features effectively
- Random Forest: Robust to outliers and noise
- Neural Networks: Can learn complex patterns

Combining them provides more reliable player projections.
"""

import logging  # For tracking ensemble training and prediction processes

import numpy as np  # Numerical operations for combining predictions
from sklearn.linear_model import Ridge  # Meta-model for stacking ensemble
from sklearn.metrics import mean_absolute_error  # Evaluation metric

from .base import BaseModel, PredictionResult  # Base model interface and result structure

# Set up logging for ensemble operations
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of multiple model types for improved predictions."""

    def __init__(self, position: str):
        """Initialize ensemble model.

        Args:
            position: Position this ensemble is for (QB, RB, WR, TE, DEF)
        """
        self.position = position
        self.base_models: list[dict] = []
        self.meta_model: Ridge | None = None
        self.weights: np.ndarray | None = None
        self.is_trained = False

    def add_model(self, model: BaseModel, weight: float | None = None, name: str | None = None):
        """Add a model to the ensemble.

        Args:
            model: Trained base model to add
            weight: Optional fixed weight for this model
            name: Optional name for the model
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before adding to ensemble")

        model_info = {
            "model": model,
            "weight": weight,
            "name": name or f"model_{len(self.base_models)}",
        }

        self.base_models.append(model_info)
        logger.info(f"Added {model_info['name']} to ensemble (weight: {weight})")

    def train_ensemble(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict:
        """Train ensemble using stacking or weighted average.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training metrics dictionary
        """
        if len(self.base_models) < 2:
            raise ValueError("Need at least 2 models for ensemble")

        # Get predictions from all base models
        train_preds = []
        val_preds = []

        for model_info in self.base_models:
            model = model_info["model"]

            # Get predictions
            train_pred = model.predict(X_train).point_estimate
            val_pred = model.predict(X_val).point_estimate

            train_preds.append(train_pred)
            val_preds.append(val_pred)

        # Stack predictions as features
        train_meta = np.column_stack(train_preds)
        val_meta = np.column_stack(val_preds)

        # Train meta-model (stacking)
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(train_meta, y_train)

        # Calculate optimal weights using validation performance
        self.weights = self._optimize_weights(val_meta, y_val)

        # Evaluate ensemble performance
        meta_pred = self.meta_model.predict(val_meta)
        weighted_pred = np.average(val_meta, weights=self.weights, axis=1)

        # Combine both approaches (60% meta-model, 40% weighted average)
        ensemble_pred = 0.6 * meta_pred + 0.4 * weighted_pred

        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        meta_mae = mean_absolute_error(y_val, meta_pred)
        weighted_mae = mean_absolute_error(y_val, weighted_pred)

        # Compare with individual model performance
        individual_maes = []
        for i, pred in enumerate(val_preds):
            mae = mean_absolute_error(y_val, pred)
            individual_maes.append(mae)
            logger.info(f"{self.base_models[i]['name']} MAE: {mae:.4f}")

        self.is_trained = True

        metrics = {
            "ensemble_mae": ensemble_mae,
            "meta_mae": meta_mae,
            "weighted_mae": weighted_mae,
            "individual_maes": individual_maes,
            "best_individual_mae": min(individual_maes),
            "ensemble_improvement": min(individual_maes) - ensemble_mae,
            "weights": self.weights.tolist(),
            "meta_coefficients": self.meta_model.coef_.tolist(),
        }

        logger.info(f"Ensemble trained for {self.position}:")
        logger.info(f"  Best individual MAE: {metrics['best_individual_mae']:.4f}")
        logger.info(f"  Ensemble MAE: {ensemble_mae:.4f}")
        logger.info(f"  Improvement: {metrics['ensemble_improvement']:.4f}")
        logger.info(
            f"  Model weights: {dict(zip([m['name'] for m in self.base_models], self.weights, strict=False))}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate ensemble predictions.

        Args:
            X: Features for prediction

        Returns:
            Ensemble prediction result
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Get predictions from all base models
        predictions = []
        confidence_scores = []
        floors = []
        ceilings = []

        for model_info in self.base_models:
            model = model_info["model"]
            result = model.predict(X)

            predictions.append(result.point_estimate)

            if result.confidence_score is not None:
                confidence_scores.append(result.confidence_score)
            if result.floor is not None:
                floors.append(result.floor)
            if result.ceiling is not None:
                ceilings.append(result.ceiling)

        # Stack predictions
        stacked = np.column_stack(predictions)

        # Meta-model prediction
        meta_pred = self.meta_model.predict(stacked)

        # Weighted average prediction
        weighted_pred = np.average(stacked, weights=self.weights, axis=1)

        # Combine both approaches
        final_pred = 0.6 * meta_pred + 0.4 * weighted_pred

        # Ensemble confidence (weighted average of individual confidences)
        if confidence_scores:
            ensemble_confidence = np.average(
                np.column_stack(confidence_scores), weights=self.weights, axis=1
            )
        else:
            ensemble_confidence = self._calculate_prediction_confidence(stacked)

        # Ensemble floor and ceiling
        if floors and ceilings:
            ensemble_floor = np.average(np.column_stack(floors), weights=self.weights, axis=1)
            ensemble_ceiling = np.average(np.column_stack(ceilings), weights=self.weights, axis=1)
        else:
            # Calculate based on prediction variance
            pred_std = np.std(stacked, axis=1)
            ensemble_floor = final_pred - 0.75 * pred_std
            ensemble_ceiling = final_pred + 0.75 * pred_std

        # Calculate prediction intervals based on model disagreement
        lower_bound, upper_bound = self._calculate_ensemble_intervals(stacked, final_pred)

        return PredictionResult(
            point_estimate=final_pred,
            confidence_score=ensemble_confidence,
            prediction_intervals=(lower_bound, upper_bound),
            floor=ensemble_floor,
            ceiling=ensemble_ceiling,
            feature_contributions={
                "individual_predictions": dict(
                    zip([m["name"] for m in self.base_models], predictions, strict=False)
                )
            },
        )

    def _optimize_weights(self, val_predictions: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights based on validation performance.

        Args:
            val_predictions: Validation predictions from each model [n_samples, n_models]
            y_val: Validation targets

        Returns:
            Optimized weights array
        """
        n_models = val_predictions.shape[1]

        # Calculate individual model MAEs
        maes = []
        for i in range(n_models):
            mae = mean_absolute_error(y_val, val_predictions[:, i])
            maes.append(mae)

        # Convert MAEs to weights (inverse relationship)
        # Better models (lower MAE) get higher weights
        mae_array = np.array(maes)

        # Avoid division by zero
        mae_array = np.maximum(mae_array, 1e-6)

        # Inverse weighting with softmax for stability
        inv_maes = 1.0 / mae_array
        weights = inv_maes / np.sum(inv_maes)

        # Apply temperature to control weight distribution
        temperature = 2.0  # Higher = more uniform weights
        weights = np.exp(np.log(weights) / temperature)
        weights = weights / np.sum(weights)

        return weights

    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence based on model agreement.

        Args:
            predictions: Predictions from all models [n_samples, n_models]

        Returns:
            Confidence scores [n_samples]
        """
        # Calculate coefficient of variation across models
        pred_mean = np.mean(predictions, axis=1)
        pred_std = np.std(predictions, axis=1)

        # Avoid division by zero
        cv = pred_std / np.maximum(pred_mean, 0.1)

        # Convert to confidence (higher agreement = higher confidence)
        confidence = 1.0 / (1.0 + cv)

        # Ensure confidence is in [0, 1] range
        confidence = np.clip(confidence, 0.0, 1.0)

        return confidence

    def _calculate_ensemble_intervals(
        self, predictions: np.ndarray, point_estimate: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals based on model disagreement.

        Args:
            predictions: Individual model predictions [n_samples, n_models]
            point_estimate: Ensemble point estimates

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Use the range of individual predictions as uncertainty measure
        pred_min = np.min(predictions, axis=1)
        pred_max = np.max(predictions, axis=1)
        pred_range = pred_max - pred_min

        # Prediction intervals based on model disagreement
        # More disagreement = wider intervals
        margin = 0.5 * pred_range + 0.1 * np.abs(point_estimate)

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        return lower_bound, upper_bound

    def get_model_weights(self) -> dict[str, float]:
        """Get the weights assigned to each model.

        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_trained or self.weights is None:
            return {}

        return dict(zip([model["name"] for model in self.base_models], self.weights, strict=False))

    def get_model_info(self) -> list[dict]:
        """Get information about all models in the ensemble.

        Returns:
            List of model information dictionaries
        """
        info = []
        for i, model_info in enumerate(self.base_models):
            model = model_info["model"]
            info.append(
                {
                    "name": model_info["name"],
                    "type": type(model).__name__,
                    "weight": self.weights[i] if self.weights is not None else None,
                    "is_trained": model.is_trained,
                    "feature_count": len(model.feature_names) if model.feature_names else None,
                }
            )
        return info
