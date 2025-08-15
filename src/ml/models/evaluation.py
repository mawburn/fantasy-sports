"""Model evaluation framework for fantasy sports predictions."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import (
    BacktestResult,
)
from src.ml.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    mae: float
    rmse: float
    r2: float
    mape: float
    accuracy_within_5: float  # Percentage of predictions within 5 points
    accuracy_within_10: float  # Percentage of predictions within 10 points
    consistency_score: float  # How consistent are the predictions
    calibration_score: float  # How well calibrated are confidence intervals
    prediction_bias: float  # Average prediction error (signed)
    total_predictions: int


@dataclass
class BacktestConfiguration:
    """Configuration for backtesting runs."""

    start_date: datetime
    end_date: datetime
    min_predictions_per_week: int = 10
    confidence_interval: float = 0.95
    save_detailed_results: bool = True


class ModelEvaluator:
    """Evaluate ML model performance with comprehensive metrics."""

    def __init__(self, db_session: Session | None = None):
        """Initialize model evaluator."""
        self.db = db_session or next(get_db())

    def evaluate_model(
        self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate a trained model on test data.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Comprehensive evaluation metrics
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Generate predictions
        predictions = model.predict(X_test)
        y_pred = predictions.point_estimate

        # Calculate core metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE (handling zero targets)
        mape = self._calculate_mape(y_test, y_pred)

        # Accuracy metrics
        accuracy_5 = np.mean(np.abs(y_test - y_pred) <= 5.0) * 100
        accuracy_10 = np.mean(np.abs(y_test - y_pred) <= 10.0) * 100

        # Prediction consistency
        consistency_score = self._calculate_consistency_score(y_test, y_pred)

        # Calibration score (if confidence intervals available)
        calibration_score = self._calculate_calibration_score(
            y_test, predictions.prediction_intervals
        )

        # Prediction bias
        prediction_bias = np.mean(y_pred - y_test)

        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            accuracy_within_5=accuracy_5,
            accuracy_within_10=accuracy_10,
            consistency_score=consistency_score,
            calibration_score=calibration_score,
            prediction_bias=prediction_bias,
            total_predictions=len(y_test),
        )

    def backtest_model(
        self,
        model: BaseModel,
        position: str,
        config: BacktestConfiguration,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive backtesting of a model.

        Args:
            model: Trained model to backtest
            position: Player position
            config: Backtesting configuration
            save_results: Whether to save results to database

        Returns:
            Dictionary with backtest results and detailed analysis
        """
        logger.info(
            f"Starting backtest for {position} model from {config.start_date} to {config.end_date}"
        )

        # Collect historical predictions and actuals
        predictions_data = self._collect_backtest_data(position, config)

        if predictions_data.empty:
            raise ValueError("No prediction data found for backtesting period")

        # Evaluate performance
        y_actual = predictions_data["actual_points"].values
        y_pred = predictions_data["predicted_points"].values

        metrics = self._calculate_backtest_metrics(y_actual, y_pred)

        # Weekly performance analysis
        weekly_performance = self._analyze_weekly_performance(predictions_data)

        # Player-level analysis
        player_analysis = self._analyze_player_performance(predictions_data)

        # Financial simulation (simplified ROI calculation)
        financial_metrics = self._simulate_financial_performance(predictions_data)

        results = {
            "config": config,
            "metrics": metrics,
            "weekly_performance": weekly_performance,
            "player_analysis": player_analysis,
            "financial_metrics": financial_metrics,
            "detailed_predictions": predictions_data if config.save_detailed_results else None,
        }

        # Save to database if requested
        if save_results:
            self._save_backtest_results(model, position, results)

        logger.info(f"Backtest completed: MAE={metrics.mae:.2f}, R²={metrics.r2:.3f}")
        return results

    def compare_models(
        self, models: dict[str, BaseModel], X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """Compare multiple models on the same test set.

        Args:
            models: Dictionary mapping model names to trained models
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with comparison results
        """
        results = []

        for name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test)
                results.append(
                    {
                        "model_name": name,
                        "mae": metrics.mae,
                        "rmse": metrics.rmse,
                        "r2": metrics.r2,
                        "mape": metrics.mape,
                        "accuracy_5pt": metrics.accuracy_within_5,
                        "accuracy_10pt": metrics.accuracy_within_10,
                        "consistency": metrics.consistency_score,
                        "bias": metrics.prediction_bias,
                        "total_predictions": metrics.total_predictions,
                    }
                )
            except Exception as e:
                logger.exception(f"Failed to evaluate model {name}")
                results.append({"model_name": name, "error": str(e)})

        df = pd.DataFrame(results)

        # Rank models by MAE (lower is better)
        if "mae" in df.columns:
            df["mae_rank"] = df["mae"].rank()

        return df.sort_values("mae") if "mae" in df.columns else df

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if not np.any(non_zero_mask):
            return 0.0

        mape = (
            np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
            * 100
        )
        return float(mape)

    def _calculate_consistency_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prediction consistency score (1 - coefficient of variation of errors)."""
        errors = np.abs(y_true - y_pred)
        if np.mean(errors) == 0:
            return 1.0

        cv = np.std(errors) / np.mean(errors)
        return max(0.0, 1.0 - cv)

    def _calculate_calibration_score(
        self, y_true: np.ndarray, prediction_intervals: tuple[np.ndarray, np.ndarray] | None
    ) -> float:
        """Calculate how well prediction intervals are calibrated."""
        if prediction_intervals is None:
            return 0.0

        lower_bound, upper_bound = prediction_intervals

        # Check what percentage of actuals fall within prediction intervals
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(within_interval)

        # For 95% confidence intervals, we expect ~95% coverage
        expected_coverage = 0.95
        calibration_error = abs(coverage - expected_coverage)

        # Convert to score (1 is perfect calibration)
        return max(0.0, 1.0 - calibration_error * 2)

    def _collect_backtest_data(self, position: str, config: BacktestConfiguration) -> pd.DataFrame:
        """Collect historical prediction data for backtesting."""
        # Query prediction results from database
        query = """
        SELECT
            pr.predicted_points,
            pr.actual_points,
            pr.prediction_error,
            pr.absolute_error,
            pr.confidence_score,
            pr.predicted_floor,
            pr.predicted_ceiling,
            pr.game_date,
            pr.player_id,
            p.display_name as player_name,
            g.week
        FROM prediction_results pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        WHERE pr.position = :position
        AND pr.game_date >= :start_date
        AND pr.game_date <= :end_date
        AND pr.actual_points IS NOT NULL
        ORDER BY pr.game_date, pr.player_id
        """

        result = self.db.execute(
            query,
            {
                "position": position,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
        )

        df = pd.DataFrame(result.fetchall())

        if df.empty:
            logger.warning(f"No backtest data found for {position} in specified date range")

        return df

    def _calculate_backtest_metrics(
        self, y_actual: np.ndarray, y_pred: np.ndarray
    ) -> EvaluationMetrics:
        """Calculate comprehensive metrics for backtest results."""
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = self._calculate_mape(y_actual, y_pred)

        accuracy_5 = np.mean(np.abs(y_actual - y_pred) <= 5.0) * 100
        accuracy_10 = np.mean(np.abs(y_actual - y_pred) <= 10.0) * 100

        consistency_score = self._calculate_consistency_score(y_actual, y_pred)
        prediction_bias = np.mean(y_pred - y_actual)

        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            accuracy_within_5=accuracy_5,
            accuracy_within_10=accuracy_10,
            consistency_score=consistency_score,
            calibration_score=0.0,  # Would need confidence intervals from stored predictions
            prediction_bias=prediction_bias,
            total_predictions=len(y_actual),
        )

    def _analyze_weekly_performance(self, predictions_data: pd.DataFrame) -> dict:
        """Analyze model performance by week."""
        if "week" not in predictions_data.columns:
            return {}

        weekly_stats = predictions_data.groupby("week").agg(
            {
                "prediction_error": ["mean", "std", "count"],
                "absolute_error": ["mean", "median"],
            }
        )

        weekly_stats.columns = ["_".join(col).strip() for col in weekly_stats.columns]

        return {
            "weekly_mae": weekly_stats["absolute_error_mean"].to_dict(),
            "weekly_bias": weekly_stats["prediction_error_mean"].to_dict(),
            "weekly_std": weekly_stats["prediction_error_std"].to_dict(),
            "weekly_count": weekly_stats["prediction_error_count"].to_dict(),
            "most_accurate_week": weekly_stats["absolute_error_mean"].idxmin(),
            "least_accurate_week": weekly_stats["absolute_error_mean"].idxmax(),
        }

    def _analyze_player_performance(self, predictions_data: pd.DataFrame) -> dict:
        """Analyze model performance by player."""
        if "player_name" not in predictions_data.columns:
            return {}

        player_stats = predictions_data.groupby("player_name").agg(
            {
                "absolute_error": ["mean", "count"],
                "predicted_points": "mean",
                "actual_points": "mean",
            }
        )

        player_stats.columns = ["_".join(col).strip() for col in player_stats.columns]

        # Filter players with sufficient predictions
        min_predictions = 3
        qualified_players = player_stats[player_stats["absolute_error_count"] >= min_predictions]

        if qualified_players.empty:
            return {"message": "No players with sufficient predictions for analysis"}

        return {
            "best_predicted_player": qualified_players["absolute_error_mean"].idxmin(),
            "worst_predicted_player": qualified_players["absolute_error_mean"].idxmax(),
            "most_predicted_player": player_stats["absolute_error_count"].idxmax(),
            "player_count": len(qualified_players),
            "avg_mae_per_player": qualified_players["absolute_error_mean"].mean(),
        }

    def _simulate_financial_performance(self, predictions_data: pd.DataFrame) -> dict:
        """Simulate financial performance if using predictions for DFS."""
        if predictions_data.empty:
            return {}

        # Simple simulation: assume we can generate value by predicting outperformers
        predictions_data["prediction_error"] = (
            predictions_data["actual_points"] - predictions_data["predicted_points"]
        )

        # Simulate strategy: pick players predicted to score above average
        avg_predicted = predictions_data["predicted_points"].mean()
        high_confidence_picks = predictions_data[
            predictions_data["predicted_points"] > avg_predicted
        ]

        if high_confidence_picks.empty:
            return {"message": "No high-confidence picks available for simulation"}

        # Calculate hit rate for high-confidence picks
        avg_actual = predictions_data["actual_points"].mean()
        hits = high_confidence_picks[high_confidence_picks["actual_points"] > avg_actual]
        hit_rate = len(hits) / len(high_confidence_picks) if len(high_confidence_picks) > 0 else 0

        # Simplified ROI calculation
        # Assume we make money when our picks outperform average
        outperformance = high_confidence_picks["actual_points"].mean() - avg_actual
        simulated_roi = outperformance * 0.1  # Simplified multiplier

        return {
            "hit_rate": hit_rate * 100,
            "high_confidence_picks": len(high_confidence_picks),
            "avg_outperformance": outperformance,
            "simulated_roi": simulated_roi,
            "total_weeks_analyzed": (
                predictions_data["week"].nunique() if "week" in predictions_data.columns else 0
            ),
        }

    def _save_backtest_results(self, model: BaseModel, position: str, results: dict) -> None:
        """Save backtest results to database."""
        try:
            # Generate unique backtest ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backtest_id = f"{position}_backtest_{timestamp}"

            metrics = results["metrics"]
            config = results["config"]
            financial = results["financial_metrics"]

            backtest_result = BacktestResult(
                backtest_id=backtest_id,
                model_id=getattr(model, "model_id", f"{position}_model"),
                backtest_name=f"{position} Model Backtest",
                position=position,
                start_date=config.start_date,
                end_date=config.end_date,
                total_predictions=metrics.total_predictions,
                mae=metrics.mae,
                rmse=metrics.rmse,
                r2_score=metrics.r2,
                mape=metrics.mape,
                consistency_score=metrics.consistency_score,
                calibration_score=metrics.calibration_score,
                roi_simulated=financial.get("simulated_roi", 0.0),
                profit_simulated=0.0,  # Would calculate from actual contest simulation
                sharpe_ratio=0.0,  # Would calculate from return series
                execution_time_seconds=0.0,  # Would track actual execution time
                memory_usage_mb=0.0,  # Would track memory usage
            )

            self.db.add(backtest_result)
            self.db.commit()

            logger.info(f"Saved backtest results with ID: {backtest_id}")

        except Exception:
            logger.exception("Failed to save backtest results")
            self.db.rollback()

    def generate_evaluation_report(
        self, model: BaseModel, position: str, test_metrics: EvaluationMetrics
    ) -> str:
        """Generate a human-readable evaluation report."""
        report = f"""
=== {position} Model Evaluation Report ===

Performance Metrics:
- Mean Absolute Error (MAE): {test_metrics.mae:.2f} points
- Root Mean Square Error (RMSE): {test_metrics.rmse:.2f} points
- R-squared (R²): {test_metrics.r2:.3f}
- Mean Absolute Percentage Error (MAPE): {test_metrics.mape:.1f}%

Accuracy Metrics:
- Predictions within 5 points: {test_metrics.accuracy_within_5:.1f}%
- Predictions within 10 points: {test_metrics.accuracy_within_10:.1f}%
- Prediction consistency score: {test_metrics.consistency_score:.3f}
- Prediction bias: {test_metrics.prediction_bias:+.2f} points

Total Predictions Evaluated: {test_metrics.total_predictions}

Model Quality Assessment:
"""

        # Add quality assessment
        if test_metrics.mae < 5.0:
            report += "✓ Excellent prediction accuracy (MAE < 5.0)\n"
        elif test_metrics.mae < 7.0:
            report += "✓ Good prediction accuracy (MAE < 7.0)\n"
        else:
            report += "⚠ Consider model improvements (MAE ≥ 7.0)\n"

        if test_metrics.r2 > 0.3:
            report += "✓ Strong explanatory power (R² > 0.3)\n"
        elif test_metrics.r2 > 0.1:
            report += "✓ Moderate explanatory power (R² > 0.1)\n"
        else:
            report += "⚠ Low explanatory power (R² ≤ 0.1)\n"

        if abs(test_metrics.prediction_bias) < 1.0:
            report += "✓ Low prediction bias\n"
        else:
            report += f"⚠ High prediction bias ({test_metrics.prediction_bias:+.2f})\n"

        if test_metrics.consistency_score > 0.7:
            report += "✓ Consistent predictions\n"
        else:
            report += "⚠ Inconsistent predictions\n"

        return report
