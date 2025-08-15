"""Model evaluation framework for fantasy sports predictions.

This file provides comprehensive model evaluation tools for assessing the performance
of trained ML models in the fantasy sports domain. It goes beyond basic accuracy
metrics to provide insights into real-world model performance.

Key Evaluation Concepts for Beginners:

Backtesting: Testing a model's performance on historical data to simulate
how it would have performed in the past. This is the gold standard for
evaluating time series and prediction models.

Metrics Explained:
- MAE (Mean Absolute Error): Average distance between predictions and actual values
- RMSE (Root Mean Square Error): Like MAE but penalizes large errors more heavily
- R² (R-squared): Proportion of variance explained by the model (1.0 = perfect)
- MAPE (Mean Absolute Percentage Error): Error as a percentage of actual value
- Calibration: How well prediction confidence intervals match reality

Why Comprehensive Evaluation Matters:
- Accuracy alone isn't enough - we need consistency, calibration, and bias analysis
- Fantasy sports have unique challenges (high variance, outliers, seasonal patterns)
- Financial implications require understanding prediction reliability
- Model deployment decisions require confidence in performance estimates

This framework supports both single model evaluation and model comparison,
with detailed reporting for stakeholders and automated database storage
for tracking model performance over time.
"""

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
    """Container for comprehensive model evaluation metrics.

    This dataclass stores all the metrics we use to evaluate model performance.
    Each metric provides different insights into how well the model performs:

    Accuracy Metrics (How close are predictions?):
    - mae: Mean Absolute Error - average distance from true values (lower = better)
    - rmse: Root Mean Square Error - penalizes large errors (lower = better)
    - r2: R-squared - proportion of variance explained (higher = better, 1.0 = perfect)
    - mape: Mean Absolute Percentage Error - error as % of true value (lower = better)

    Fantasy-Specific Accuracy:
    - accuracy_within_5: % of predictions within 5 points (fantasy-relevant threshold)
    - accuracy_within_10: % of predictions within 10 points (broader accuracy measure)

    Reliability Metrics (How trustworthy are predictions?):
    - consistency_score: How stable predictions are across similar inputs (higher = better)
    - calibration_score: How well confidence intervals match reality (higher = better)
    - prediction_bias: Average prediction error with sign (close to 0 = unbiased)

    Volume:
    - total_predictions: Number of predictions evaluated (larger = more reliable metrics)

    For beginners: Think of this as a "report card" for the model that covers
    not just accuracy but also reliability and consistency - critical for
    real-world deployment where we need to trust the predictions.
    """

    # Core regression metrics
    mae: float  # Mean Absolute Error (points)
    rmse: float  # Root Mean Square Error (points)
    r2: float  # R-squared coefficient (0-1, higher is better)
    mape: float  # Mean Absolute Percentage Error (%)

    # Fantasy-specific accuracy thresholds
    accuracy_within_5: float  # Percentage of predictions within 5 points
    accuracy_within_10: float  # Percentage of predictions within 10 points

    # Reliability and trust metrics
    consistency_score: float  # How consistent are the predictions (0-1)
    calibration_score: float  # How well calibrated are confidence intervals (0-1)
    prediction_bias: float  # Average prediction error with sign (should be ~0)

    # Sample size for statistical confidence
    total_predictions: int


@dataclass
class BacktestConfiguration:
    """Configuration for backtesting runs.

    Backtesting is the process of testing a model's performance on historical data
    to understand how it would have performed in the past. This is crucial for
    fantasy sports models because:

    1. Time Series Nature: Fantasy sports are sequential - we predict future games
       based on past performance, so we need to test in chronological order

    2. Realistic Performance Estimates: Random train/test splits give overly
       optimistic results because they allow "future" information to leak into training

    3. Seasonal Patterns: NFL seasons have different patterns (early season adjustments,
       playoff implications) that only show up in proper time-ordered testing

    Configuration Parameters:
    - Date range defines the backtesting period (usually 1-2 seasons)
    - Minimum predictions ensure statistical reliability
    - Confidence intervals help assess prediction uncertainty
    - Detailed results storage enables deep-dive analysis

    For beginners: Think of this as "time travel testing" - we pretend we're back
    in time and see how well our model would have predicted games we now know the
    results of, testing week by week in the order they actually happened.
    """

    # Time period for backtesting
    start_date: datetime  # First week to include in backtest
    end_date: datetime  # Last week to include in backtest

    # Quality control parameters
    min_predictions_per_week: int = 10  # Minimum predictions needed per week for reliability
    confidence_interval: float = 0.95  # Confidence level for interval evaluation (95%)

    # Output control
    save_detailed_results: bool = True  # Whether to store individual prediction results


class ModelEvaluator:
    """Evaluate ML model performance with comprehensive metrics.

    This class provides a complete toolkit for evaluating fantasy sports ML models.
    It handles everything from basic accuracy metrics to complex backtesting and
    financial performance simulation.

    Key Responsibilities:
    1. Single Model Evaluation: Comprehensive metrics on test sets
    2. Model Comparison: Head-to-head performance analysis
    3. Backtesting: Historical performance simulation
    4. Financial Analysis: ROI and profitability estimation
    5. Reporting: Human-readable performance summaries
    6. Persistence: Database storage of evaluation results

    The evaluator connects to the database to access historical prediction
    data for backtesting and to store evaluation results for tracking
    model performance over time.

    For beginners: This is like a "testing laboratory" for ML models.
    Just as you might test a new car's performance (acceleration, braking,
    fuel efficiency), this class tests ML models across multiple dimensions
    to understand their strengths and weaknesses before deployment.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize model evaluator with database connection.

        The evaluator needs database access to:
        - Retrieve historical prediction data for backtesting
        - Store evaluation results for tracking and comparison
        - Access player and game metadata for detailed analysis

        Args:
            db_session: Optional database session (creates new if None)
        """
        # Database connection for accessing historical data and storing results
        self.db = db_session or next(get_db())

    def evaluate_model(
        self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate a trained model on test data.

        This method performs a comprehensive evaluation of a trained model using
        held-out test data. It calculates multiple types of metrics to provide
        a complete picture of model performance.

        Evaluation Process:
        1. Generate predictions on test set
        2. Calculate core regression metrics (MAE, RMSE, R²)
        3. Compute fantasy-specific accuracy measures
        4. Assess prediction reliability and calibration
        5. Analyze prediction bias and consistency

        Why Multiple Metrics?
        - MAE/RMSE: Different sensitivity to outliers
        - R²: Understanding of explained variance
        - Accuracy thresholds: Fantasy-relevant performance
        - Consistency: Reliability across different scenarios
        - Calibration: Trust in confidence estimates
        - Bias: Systematic over/under-prediction

        Args:
            model: Trained model to evaluate (must have is_trained = True)
            X_test: Test features - player/game data not seen during training
            y_test: Test targets - actual fantasy points for comparison

        Returns:
            EvaluationMetrics object with comprehensive performance assessment

        Raises:
            ValueError: If model is not trained
        """
        # Safety check: ensure model is ready for evaluation
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Step 1: Generate predictions on test set
        # This gives us point estimates and (optionally) confidence intervals
        predictions = model.predict(X_test)
        y_pred = predictions.point_estimate  # Extract the main predictions

        # Step 2: Calculate core regression metrics
        # These are standard ML metrics used across all regression problems
        mae = mean_absolute_error(y_test, y_pred)  # Average absolute error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root mean square error
        r2 = r2_score(y_test, y_pred)  # R-squared: variance explained

        # Step 3: Calculate MAPE (handling division by zero gracefully)
        # MAPE shows error as percentage of actual value - useful for comparing
        # across different score ranges (comparing 5-point vs 25-point performances)
        mape = self._calculate_mape(y_test, y_pred)

        # Step 4: Fantasy-specific accuracy metrics
        # These thresholds are meaningful in fantasy sports context
        # Being within 5 points is very good, within 10 is acceptable
        accuracy_5 = np.mean(np.abs(y_test - y_pred) <= 5.0) * 100  # % within 5 points
        accuracy_10 = np.mean(np.abs(y_test - y_pred) <= 10.0) * 100  # % within 10 points

        # Step 5: Prediction reliability metrics
        # Consistency: How stable are predictions across similar inputs?
        consistency_score = self._calculate_consistency_score(y_test, y_pred)

        # Step 6: Calibration assessment
        # If model provides confidence intervals, check if they're well-calibrated
        # (i.e., 90% confidence intervals contain actual values 90% of the time)
        calibration_score = self._calculate_calibration_score(
            y_test, predictions.prediction_intervals
        )

        # Step 7: Prediction bias analysis
        # Positive bias = model consistently over-predicts
        # Negative bias = model consistently under-predicts
        # Zero bias = unbiased predictions (ideal)
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

        Backtesting is the gold standard for evaluating time series prediction models.
        Instead of random train/test splits, we test the model's ability to predict
        future games based on past data, exactly as it would be used in production.

        Backtesting Process:
        1. Collect historical predictions from database (what model predicted)
        2. Compare with actual results (what actually happened)
        3. Calculate comprehensive performance metrics
        4. Analyze patterns by week and player
        5. Simulate financial performance (ROI estimation)
        6. Store results for tracking model evolution over time

        Why Backtesting Matters:
        - Realistic Performance: Tests model in real-world sequential order
        - Seasonal Patterns: Captures NFL season dynamics (weather, playoffs, etc.)
        - Data Leakage Prevention: No future information can "leak" into predictions
        - Production Simulation: Exactly how model would be used live

        Output Analysis:
        - Overall metrics: How accurate was the model across all predictions?
        - Weekly analysis: Which weeks were most/least predictable?
        - Player analysis: Which players were easiest/hardest to predict?
        - Financial simulation: Could this model generate profit in DFS?

        Args:
            model: Trained model to backtest (needs historical prediction data)
            position: Player position being evaluated (QB, RB, WR, TE, DEF)
            config: Backtesting configuration (time period, thresholds, etc.)
            save_results: Whether to persist results to database for tracking

        Returns:
            Comprehensive dictionary containing:
            - metrics: Overall performance statistics
            - weekly_performance: Week-by-week analysis
            - player_analysis: Player-specific insights
            - financial_metrics: Simulated profitability analysis
            - detailed_predictions: Raw prediction data (if requested)

        Raises:
            ValueError: If no prediction data found for the specified time period
        """
        logger.info(
            f"Starting backtest for {position} model from {config.start_date} to {config.end_date}"
        )

        # Step 1: Collect historical prediction data from database
        # This retrieves stored predictions and actual results for comparison
        predictions_data = self._collect_backtest_data(position, config)

        # Validate we have data to work with
        if predictions_data.empty:
            raise ValueError("No prediction data found for backtesting period")

        # Step 2: Extract predictions and actual results for metric calculation
        y_actual = predictions_data["actual_points"].values  # What actually happened
        y_pred = predictions_data["predicted_points"].values  # What model predicted

        # Step 3: Calculate comprehensive performance metrics
        # This gives us overall model performance across the entire backtest period
        metrics = self._calculate_backtest_metrics(y_actual, y_pred)

        # Step 4: Analyze performance patterns by time period
        # Which weeks were most/least predictable? Seasonal patterns?
        weekly_performance = self._analyze_weekly_performance(predictions_data)

        # Step 5: Analyze performance patterns by player
        # Which players are consistently easy/hard to predict?
        player_analysis = self._analyze_player_performance(predictions_data)

        # Step 6: Simulate financial performance for business case
        # If we used these predictions in DFS, would we have made money?
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

        Model comparison is critical for selecting the best performing model
        for production deployment. By evaluating multiple models on identical
        test data, we get an apples-to-apples comparison.

        Comparison Process:
        1. Evaluate each model on the same test set
        2. Calculate comprehensive metrics for each model
        3. Rank models by primary metric (MAE)
        4. Handle evaluation failures gracefully
        5. Return sortable comparison table

        Use Cases:
        - A/B testing different algorithms (XGBoost vs LightGBM)
        - Hyperparameter optimization (comparing different configurations)
        - Feature engineering validation (old vs new feature sets)
        - Ensemble selection (choosing best base models)

        The returned DataFrame enables easy sorting, filtering, and
        visualization of model performance differences.

        Args:
            models: Dictionary mapping descriptive names to trained model instances
            X_test: Test features (same for all models for fair comparison)
            y_test: Test targets (same for all models for fair comparison)

        Returns:
            DataFrame with rows for each model and columns for each metric,
            sorted by MAE (best performing model first)
        """
        # Store results for each model
        results = []

        # Evaluate each model individually with error handling
        for name, model in models.items():
            try:
                # Calculate comprehensive metrics for this model
                metrics = self.evaluate_model(model, X_test, y_test)

                # Package results in consistent format
                results.append(
                    {
                        "model_name": name,
                        "mae": metrics.mae,  # Primary ranking metric
                        "rmse": metrics.rmse,  # Error magnitude sensitivity
                        "r2": metrics.r2,  # Variance explained
                        "mape": metrics.mape,  # Percentage error
                        "accuracy_5pt": metrics.accuracy_within_5,  # Fantasy-relevant accuracy
                        "accuracy_10pt": metrics.accuracy_within_10,  # Broader accuracy
                        "consistency": metrics.consistency_score,  # Prediction stability
                        "bias": metrics.prediction_bias,  # Systematic over/under-prediction
                        "total_predictions": metrics.total_predictions,  # Sample size
                    }
                )
            except Exception as e:
                # Log error but continue with other models
                # This prevents one broken model from breaking entire comparison
                logger.exception(f"Failed to evaluate model {name}")
                results.append({"model_name": name, "error": str(e)})

        # Convert to DataFrame for easy analysis and visualization
        df = pd.DataFrame(results)

        # Add ranking column based on primary metric (MAE - lower is better)
        if "mae" in df.columns:
            df["mae_rank"] = df["mae"].rank()  # 1 = best, 2 = second best, etc.

        # Return sorted by performance (best model first)
        return df.sort_values("mae") if "mae" in df.columns else df

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.

        MAPE (Mean Absolute Percentage Error) expresses prediction error as a
        percentage of the actual value. This is useful for comparing model
        performance across different scales.

        For example:
        - Predicting 15 instead of 10 points: 50% error
        - Predicting 30 instead of 25 points: 20% error

        The first error is relatively larger even though both are 5 points off.

        Division by Zero Handling:
        Fantasy football scores can be zero (player injured, benched, etc.).
        We exclude zero scores to avoid division by zero, focusing on games
        where players actually participated.

        Formula: MAPE = mean(|actual - predicted| / |actual|) * 100%

        Args:
            y_true: Actual fantasy points scored
            y_pred: Predicted fantasy points

        Returns:
            MAPE as percentage (lower is better)
        """
        # Filter out zero scores to avoid division by zero
        # Zero scores typically mean player didn't play (injury, benching, etc.)
        non_zero_mask = y_true != 0

        # If all scores are zero, return 0 (no meaningful errors to calculate)
        if not np.any(non_zero_mask):
            return 0.0

        # Calculate percentage errors only for non-zero actual values
        # |actual - predicted| / |actual| gives fractional error
        # Multiply by 100 to convert to percentage
        percentage_errors = np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
        )
        mape = np.mean(percentage_errors) * 100

        return float(mape)

    def _calculate_consistency_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prediction consistency score (1 - coefficient of variation of errors).

        Consistency measures how stable the model's errors are. A consistent model
        makes similarly-sized errors across different predictions, while an
        inconsistent model might be very accurate sometimes but wildly off other times.

        Mathematical Approach:
        1. Calculate absolute errors for all predictions
        2. Compute coefficient of variation (CV) = std_dev / mean
        3. Transform to score: consistency = 1 - CV

        Interpretation:
        - CV = 0: All errors are identical (perfect consistency) → score = 1.0
        - CV = 1: Error std dev equals mean error (moderate consistency) → score = 0.0
        - CV > 1: Very inconsistent errors → score < 0.0 (clamped to 0.0)

        Why Consistency Matters:
        - High consistency: Users can trust the model's confidence levels
        - Low consistency: Model might be accurate on average but unreliable
        - Fantasy sports: Helps with bankroll management and risk assessment

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Consistency score from 0.0 (inconsistent) to 1.0 (perfectly consistent)
        """
        # Calculate absolute errors (distance from true values)
        errors = np.abs(y_true - y_pred)

        # Handle edge case where all predictions are perfect (all errors = 0)
        mean_error = np.mean(errors)
        if mean_error == 0:
            return 1.0  # Perfect consistency if no errors

        # Calculate coefficient of variation: std deviation / mean
        # This normalizes variability by the average error size
        error_std = np.std(errors)
        cv = error_std / mean_error

        # Transform to score: 1.0 - CV (higher CV = lower consistency)
        # Clamp to [0, 1] range - negative consistency doesn't make sense
        consistency_score = max(0.0, 1.0 - cv)

        return consistency_score

    def _calculate_calibration_score(
        self, y_true: np.ndarray, prediction_intervals: tuple[np.ndarray, np.ndarray] | None
    ) -> float:
        """Calculate how well prediction intervals are calibrated.

        Calibration measures whether prediction confidence intervals are trustworthy.
        A well-calibrated model's 90% confidence intervals should contain the actual
        value 90% of the time.

        Calibration Example:
        - Model predicts "18 points with 90% confidence interval [12, 24]"
        - If actual values fall in [12, 24] exactly 90% of the time → well calibrated
        - If actual values fall in [12, 24] only 60% of the time → under-confident
        - If actual values fall in [12, 24] 99% of the time → over-confident

        Why Calibration Matters:
        - Fantasy sports: Helps users understand prediction uncertainty
        - Risk management: Enables appropriate bankroll sizing
        - Trust: Well-calibrated intervals build user confidence
        - Lineup optimization: Better uncertainty estimates improve decisions

        Scoring Method:
        1. Calculate actual coverage (% of true values within intervals)
        2. Compare to expected coverage (usually 95%)
        3. Convert calibration error to score (1.0 = perfect calibration)

        Args:
            y_true: Actual values that occurred
            prediction_intervals: Tuple of (lower_bound, upper_bound) arrays

        Returns:
            Calibration score from 0.0 (poorly calibrated) to 1.0 (perfectly calibrated)
            Returns 0.0 if no prediction intervals available
        """
        # If model doesn't provide prediction intervals, can't assess calibration
        if prediction_intervals is None:
            return 0.0

        # Unpack interval bounds
        lower_bound, upper_bound = prediction_intervals

        # Calculate actual coverage: what % of true values fall within intervals?
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        actual_coverage = np.mean(within_interval)

        # Expected coverage for confidence intervals (typically 95%)
        # This should match the confidence level used to generate the intervals
        expected_coverage = 0.95

        # Calculate calibration error: how far off is actual vs expected coverage?
        calibration_error = abs(actual_coverage - expected_coverage)

        # Transform error to score (1.0 = perfect calibration, 0.0 = terrible)
        # Multiply by 2 to make the penalty more sensitive
        calibration_score = max(0.0, 1.0 - calibration_error * 2)

        return calibration_score

    def _collect_backtest_data(self, position: str, config: BacktestConfiguration) -> pd.DataFrame:
        """Collect historical prediction data for backtesting.

        This method retrieves stored prediction results from the database to enable
        backtesting analysis. The data represents real predictions that were made
        in the past, along with the actual results that occurred.

        Data Requirements:
        - Predictions must have corresponding actual results (completed games)
        - Data must span the requested backtesting time period
        - Sufficient volume for statistical reliability

        Query Strategy:
        - Joins prediction_results with players and games tables
        - Filters by position and date range
        - Orders chronologically to maintain time series structure
        - Excludes predictions without actual outcomes

        Args:
            position: Player position to analyze
            config: Backtesting configuration with date range

        Returns:
            DataFrame with historical predictions and actual results,
            or empty DataFrame if no data found
        """
        # SQL query to retrieve historical prediction vs actual data
        # This complex join gets all the information needed for comprehensive analysis
        query = """
        SELECT
            pr.predicted_points,        -- What the model predicted
            pr.actual_points,           -- What actually happened
            pr.prediction_error,        -- Signed error (predicted - actual)
            pr.absolute_error,          -- Unsigned error |predicted - actual|
            pr.confidence_score,        -- Model's confidence in prediction
            pr.predicted_floor,         -- Conservative estimate (25th percentile)
            pr.predicted_ceiling,       -- Optimistic estimate (75th percentile)
            pr.game_date,               -- When the game occurred
            pr.player_id,               -- Player identifier
            p.display_name as player_name, -- Human-readable player name
            g.week                      -- NFL week number
        FROM prediction_results pr
        JOIN players p ON pr.player_id = p.id      -- Get player details
        JOIN games g ON pr.game_id = g.id          -- Get game details
        WHERE pr.position = :position              -- Filter to specific position
        AND pr.game_date >= :start_date            -- Within backtest date range
        AND pr.game_date <= :end_date              -- Within backtest date range
        AND pr.actual_points IS NOT NULL           -- Only completed games
        ORDER BY pr.game_date, pr.player_id        -- Chronological order
        """

        # Execute query with parameterized values (prevents SQL injection)
        result = self.db.execute(
            query,
            {
                "position": position,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
        )

        # Convert query results to pandas DataFrame for analysis
        df = pd.DataFrame(result.fetchall())

        # Log warning if no data found (helps with debugging)
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
        """Analyze model performance by week.

        Weekly analysis helps identify temporal patterns in model performance:
        - Are certain weeks consistently harder to predict?
        - Do we see seasonal effects (weather, playoff implications)?
        - Are there systematic biases at different points in the season?

        NFL Season Patterns to Look For:
        - Week 1: Hard to predict (offseason changes, rust)
        - Weeks 2-10: Most predictable (established patterns)
        - Weeks 11-17: Weather effects, playoff implications
        - Fantasy playoffs: Resting star players, changed motivations

        This analysis helps with:
        - Model improvement: Focus on problematic weeks
        - Confidence adjustment: Lower confidence for historically difficult weeks
        - Feature engineering: Add week-specific features
        - Risk management: Adjust bankroll for unpredictable periods

        Args:
            predictions_data: DataFrame with prediction results and week information

        Returns:
            Dictionary with weekly performance statistics and insights
        """
        # Validate required column exists
        if "week" not in predictions_data.columns:
            return {}  # Can't analyze without week information

        # Calculate comprehensive statistics for each week
        # GroupBy week and aggregate multiple metrics simultaneously
        weekly_stats = predictions_data.groupby("week").agg(
            {
                "prediction_error": ["mean", "std", "count"],  # Bias, variance, sample size
                "absolute_error": ["mean", "median"],  # Average and median accuracy
            }
        )

        # Flatten multi-level column names (pandas groupby creates hierarchical columns)
        # Convert from ('prediction_error', 'mean') to 'prediction_error_mean'
        weekly_stats.columns = ["_".join(col).strip() for col in weekly_stats.columns]

        # Package results in interpretable format
        return {
            "weekly_mae": weekly_stats["absolute_error_mean"].to_dict(),  # Accuracy by week
            "weekly_bias": weekly_stats["prediction_error_mean"].to_dict(),  # Bias by week
            "weekly_std": weekly_stats["prediction_error_std"].to_dict(),  # Consistency by week
            "weekly_count": weekly_stats["prediction_error_count"].to_dict(),  # Sample sizes
            "most_accurate_week": weekly_stats["absolute_error_mean"].idxmin(),  # Best week
            "least_accurate_week": weekly_stats["absolute_error_mean"].idxmax(),  # Worst week
        }

    def _analyze_player_performance(self, predictions_data: pd.DataFrame) -> dict:
        """Analyze model performance by player.

        Player-level analysis reveals which types of players the model handles
        well versus poorly. This is crucial for understanding model strengths
        and limitations.

        Insights from Player Analysis:
        - Consistent players (e.g., elite QBs) should be easier to predict
        - High-variance players (e.g., boom/bust WRs) should have higher errors
        - Injured or aging players may show systematic prediction bias
        - Role changes (new team, scheme) may cause prediction difficulties

        Business Applications:
        - Confidence scoring: Higher confidence for historically well-predicted players
        - Feature engineering: Special handling for problematic player types
        - User experience: Warn users about less predictable players
        - Model improvement: Focus training data collection on problem players

        Statistical Reliability:
        - Require minimum prediction count to avoid small sample bias
        - Report sample sizes for statistical confidence assessment

        Args:
            predictions_data: DataFrame with prediction results and player information

        Returns:
            Dictionary with player-level performance insights and statistics
        """
        # Validate required column exists
        if "player_name" not in predictions_data.columns:
            return {}  # Can't analyze without player information

        # Calculate performance statistics for each player
        # Group by player and compute accuracy, prediction volume, and average performance
        player_stats = predictions_data.groupby("player_name").agg(
            {
                "absolute_error": ["mean", "count"],  # Accuracy and sample size
                "predicted_points": "mean",  # Average prediction level
                "actual_points": "mean",  # Average actual performance
            }
        )

        # Flatten multi-level column names for easier access
        player_stats.columns = ["_".join(col).strip() for col in player_stats.columns]

        # Filter for statistical reliability: require minimum predictions
        # Players with too few predictions give unreliable statistics
        min_predictions = 3  # Minimum for meaningful analysis
        qualified_players = player_stats[player_stats["absolute_error_count"] >= min_predictions]

        # Handle case where no players have sufficient data
        if qualified_players.empty:
            return {"message": "No players with sufficient predictions for analysis"}

        # Extract key insights from the analysis
        return {
            "best_predicted_player": qualified_players[
                "absolute_error_mean"
            ].idxmin(),  # Most accurate
            "worst_predicted_player": qualified_players[
                "absolute_error_mean"
            ].idxmax(),  # Least accurate
            "most_predicted_player": player_stats[
                "absolute_error_count"
            ].idxmax(),  # Highest volume
            "player_count": len(qualified_players),  # Total analyzed
            "avg_mae_per_player": qualified_players[
                "absolute_error_mean"
            ].mean(),  # Overall accuracy
        }

    def _simulate_financial_performance(self, predictions_data: pd.DataFrame) -> dict:
        """Simulate financial performance if using predictions for DFS.

        This method provides a simplified simulation of how the model's predictions
        could translate into daily fantasy sports profitability. While not a complete
        DFS simulation, it gives stakeholders a business-relevant performance metric.

        Simulation Strategy:
        1. Identify "high-confidence" picks (predicted above average)
        2. Calculate hit rate (how often these picks actually outperformed)
        3. Measure average outperformance of successful picks
        4. Estimate simplified ROI based on outperformance

        Limitations of This Simulation:
        - Doesn't account for salary constraints (DFS salary cap)
        - Ignores lineup construction optimization
        - Simplified ROI calculation (real DFS has complex payout structures)
        - No transaction costs or platform fees
        - Doesn't consider player ownership percentages

        Business Value:
        - Provides stakeholder-friendly performance metric
        - Helps justify model development investment
        - Identifies potential profitability of predictions
        - Enables comparison across different model versions

        Real-World Usage:
        For actual DFS deployment, this would be replaced with comprehensive
        lineup optimization and contest-specific simulation.

        Args:
            predictions_data: Historical predictions with actual results

        Returns:
            Dictionary with financial simulation metrics and business insights
        """
        # Validate we have data to work with
        if predictions_data.empty:
            return {}

        # Calculate prediction errors for analysis
        # Positive error = under-predicted (good for us), Negative = over-predicted
        predictions_data["prediction_error"] = (
            predictions_data["actual_points"] - predictions_data["predicted_points"]
        )

        # Strategy Simulation: Pick players predicted to score above average
        # This is a simplified DFS strategy - in reality we'd use more sophisticated selection
        avg_predicted = predictions_data["predicted_points"].mean()
        high_confidence_picks = predictions_data[
            predictions_data["predicted_points"] > avg_predicted
        ]

        # Handle edge case where no picks meet criteria
        if high_confidence_picks.empty:
            return {"message": "No high-confidence picks available for simulation"}

        # Calculate success rate of our strategy
        # "Hit" = player we picked actually outperformed the average
        avg_actual = predictions_data["actual_points"].mean()
        hits = high_confidence_picks[high_confidence_picks["actual_points"] > avg_actual]
        hit_rate = len(hits) / len(high_confidence_picks) if len(high_confidence_picks) > 0 else 0

        # Calculate average outperformance of our picks
        # This represents the "edge" our predictions provide
        outperformance = high_confidence_picks["actual_points"].mean() - avg_actual

        # Simplified ROI calculation
        # In reality, DFS ROI depends on contest type, field size, payout structure
        # This uses a simple multiplier to estimate profitability
        simulated_roi = outperformance * 0.1  # Conservative 10% conversion rate

        # Package results for business stakeholders
        return {
            "hit_rate": hit_rate * 100,  # Success rate as percentage
            "high_confidence_picks": len(high_confidence_picks),  # Volume of strategic picks
            "avg_outperformance": outperformance,  # Average edge per pick
            "simulated_roi": simulated_roi,  # Estimated return on investment
            "total_weeks_analyzed": (  # Time period covered
                predictions_data["week"].nunique() if "week" in predictions_data.columns else 0
            ),
        }

    def _save_backtest_results(self, model: BaseModel, position: str, results: dict) -> None:
        """Save backtest results to database.

        Persisting backtest results is crucial for:
        - Tracking model performance over time
        - Comparing different model versions
        - Regulatory compliance and audit trails
        - Historical analysis and reporting
        - A/B testing and experimentation

        Database Storage Benefits:
        - Queryable history of all model evaluations
        - Automated reporting and dashboards
        - Performance trend analysis
        - Model regression detection
        - Stakeholder transparency

        Error Handling:
        - Comprehensive try/catch to prevent backtest failure
        - Database rollback on errors to maintain consistency
        - Detailed logging for debugging

        Args:
            model: The evaluated model (for identification)
            position: Player position being evaluated
            results: Complete backtest results dictionary
        """
        try:
            # Generate unique identifier for this backtest run
            # Format: POSITION_backtest_YYYYMMDD_HHMMSS
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backtest_id = f"{position}_backtest_{timestamp}"

            # Extract key components from results for database storage
            metrics = results["metrics"]
            config = results["config"]
            financial = results["financial_metrics"]

            # Create database record with comprehensive backtest information
            backtest_result = BacktestResult(
                backtest_id=backtest_id,  # Unique identifier
                model_id=getattr(model, "model_id", f"{position}_model"),  # Model being tested
                backtest_name=f"{position} Model Backtest",  # Human-readable name
                position=position,  # Player position
                start_date=config.start_date,  # Test period start
                end_date=config.end_date,  # Test period end
                total_predictions=metrics.total_predictions,  # Sample size
                mae=metrics.mae,  # Core accuracy metric
                rmse=metrics.rmse,  # Error magnitude metric
                r2_score=metrics.r2,  # Variance explained
                mape=metrics.mape,  # Percentage error
                consistency_score=metrics.consistency_score,  # Prediction stability
                calibration_score=metrics.calibration_score,  # Confidence reliability
                roi_simulated=financial.get("simulated_roi", 0.0),  # Business metric
                # Placeholders for future enhancements
                profit_simulated=0.0,  # Would calculate from actual contest simulation
                sharpe_ratio=0.0,  # Would calculate from return series
                execution_time_seconds=0.0,  # Would track actual execution time
                memory_usage_mb=0.0,  # Would track memory usage
            )

            # Persist to database with transaction safety
            self.db.add(backtest_result)
            self.db.commit()  # Commit the transaction

            logger.info(f"Saved backtest results with ID: {backtest_id}")

        except Exception:
            # Comprehensive error handling to prevent backtest failure
            logger.exception("Failed to save backtest results")
            self.db.rollback()  # Undo any partial changes to maintain database consistency

    def generate_evaluation_report(
        self, model: BaseModel, position: str, test_metrics: EvaluationMetrics
    ) -> str:
        """Generate a human-readable evaluation report.

        This method creates a comprehensive, stakeholder-friendly report that
        translates technical metrics into actionable insights. The report is
        designed for both technical and non-technical audiences.

        Report Structure:
        1. Performance Metrics: Core numerical results
        2. Accuracy Assessments: Fantasy-relevant thresholds
        3. Quality Assessment: Automated interpretation with recommendations
        4. Statistical Context: Sample size and confidence indicators

        Quality Thresholds (based on fantasy sports domain knowledge):
        - MAE < 5.0: Excellent (most predictions within 1 standard deviation)
        - MAE < 7.0: Good (acceptable for most use cases)
        - MAE ≥ 7.0: Needs improvement (high prediction variance)

        - R² > 0.3: Strong explanatory power (model captures key patterns)
        - R² > 0.1: Moderate (some predictive value)
        - R² ≤ 0.1: Weak (little better than random guessing)

        Args:
            model: The evaluated model (for context)
            position: Player position being evaluated
            test_metrics: Comprehensive evaluation results

        Returns:
            Formatted string report suitable for logging, emails, or dashboards
        """
        # Create structured report with clear sections
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

        # Automated quality assessment with clear thresholds and recommendations
        # MAE Assessment: Core prediction accuracy
        if test_metrics.mae < 5.0:
            report += "✓ Excellent prediction accuracy (MAE < 5.0) - Deploy with confidence\n"
        elif test_metrics.mae < 7.0:
            report += "✓ Good prediction accuracy (MAE < 7.0) - Suitable for production\n"
        else:
            report += "⚠ Consider model improvements (MAE ≥ 7.0) - May need retraining\n"

        # R² Assessment: Explanatory power and model usefulness
        if test_metrics.r2 > 0.3:
            report += "✓ Strong explanatory power (R² > 0.3) - Model captures key patterns\n"
        elif test_metrics.r2 > 0.1:
            report += "✓ Moderate explanatory power (R² > 0.1) - Some predictive value\n"
        else:
            report += "⚠ Low explanatory power (R² ≤ 0.1) - Consider feature engineering\n"

        # Bias Assessment: Systematic prediction errors
        if abs(test_metrics.prediction_bias) < 1.0:
            report += "✓ Low prediction bias - Well-calibrated predictions\n"
        else:
            bias_direction = (
                "over-predicting" if test_metrics.prediction_bias > 0 else "under-predicting"
            )
            report += f"⚠ High prediction bias ({test_metrics.prediction_bias:+.2f}) - Systematically {bias_direction}\n"

        # Consistency Assessment: Prediction reliability
        if test_metrics.consistency_score > 0.7:
            report += "✓ Consistent predictions - Reliable error patterns\n"
        else:
            report += "⚠ Inconsistent predictions - Consider ensemble methods\n"

        return report
