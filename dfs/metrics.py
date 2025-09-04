"""Metrics and evaluation utilities for DFS models.

Consolidates metric calculations used across different models to reduce duplication.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain at k).

    This metric is ideal for ranking problems where we care about the relative
    ordering of predictions, especially for the top-k items.

    Args:
        y_true: True target values (actual fantasy points)
        y_pred: Predicted values (predicted fantasy points)
        k: Number of top items to consider (default: 20)

    Returns:
        NDCG@k score between 0 and 1, where 1 is perfect ranking
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    # Ensure we have valid arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Handle NaN values by setting them to minimum
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        y_true = np.nan_to_num(
            y_true, nan=np.nanmin(y_true) if not np.all(np.isnan(y_true)) else 0.0
        )
        y_pred = np.nan_to_num(
            y_pred, nan=np.nanmin(y_pred) if not np.all(np.isnan(y_pred)) else 0.0
        )

    # Limit k to the number of available items
    n_items = len(y_true)
    k = min(k, n_items)

    if k <= 0:
        return 0.0

    # Handle negative fantasy points using exponential transformation
    # This preserves relative differences better than shifting
    y_true_relevance = np.power(2, y_true / 10) - 1
    # Ensure non-negative (handles very negative values)
    y_true_relevance = np.maximum(y_true_relevance, 0)

    # Sort indices by predicted values (descending order)
    predicted_order = np.argsort(y_pred)[::-1]

    # Get the top-k predictions and their true relevance scores
    top_k_indices = predicted_order[:k]
    top_k_true_scores = y_true_relevance[top_k_indices]

    # Calculate DCG@k (Discounted Cumulative Gain)
    dcg = 0.0
    for i, relevance in enumerate(top_k_true_scores):
        dcg += relevance / np.log2(i + 2)

    # Calculate ideal DCG@k (sort by true relevance scores)
    ideal_order = np.argsort(y_true_relevance)[::-1]
    ideal_top_k = y_true_relevance[ideal_order][:k]

    idcg = 0.0
    for i, relevance in enumerate(ideal_top_k):
        idcg += relevance / np.log2(i + 2)

    # Calculate NDCG@k
    if idcg == 0.0:
        # If all relevance scores are 0, return 0
        return 0.0

    ndcg = dcg / idcg

    # Ensure result is in valid range [0, 1]
    return max(0.0, min(1.0, ndcg))


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman rank correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2:
        return 0.0

    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) < 2:
        return 0.0

    return stats.spearmanr(y_true[mask], y_pred[mask])[0]


def calculate_metrics_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position: Optional[str] = None
) -> Dict[str, float]:
    """Calculate a comprehensive suite of metrics for model evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values
        position: Optional position name for position-specific metrics

    Returns:
        Dictionary of metric names and values
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Ranking metrics
    spearman = calculate_spearman_correlation(y_true, y_pred)
    ndcg_20 = calculate_ndcg_at_k(y_true, y_pred, k=20)
    ndcg_10 = calculate_ndcg_at_k(y_true, y_pred, k=10)

    # Error percentiles
    errors = np.abs(y_true - y_pred)
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "ndcg@20": ndcg_20,
        "ndcg@10": ndcg_10,
        "median_error": median_error,
        "p90_error": p90_error,
    }

    # Add position-specific metrics if applicable
    if position:
        metrics["position"] = position

        # Calculate what percentage of predictions are within reasonable bounds
        from models import POSITION_RANGES
        if position in POSITION_RANGES:
            min_val, max_val = POSITION_RANGES[position]
            in_bounds = np.mean((y_pred >= min_val) & (y_pred <= max_val))
            metrics["predictions_in_bounds"] = in_bounds

    return metrics


def calculate_validation_stats(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate statistics from multiple validation runs.

    Args:
        results: List of result dictionaries from multiple runs

    Returns:
        Dictionary with mean and std statistics
    """
    if not results:
        return {}

    # Extract metrics
    metric_names = results[0].keys()
    stats_dict = {}

    for metric in metric_names:
        values = [r[metric] for r in results if metric in r]
        if values:
            stats_dict[f"mean_{metric}"] = np.mean(values)
            stats_dict[f"std_{metric}"] = np.std(values)

    return stats_dict


def print_metrics_report(
    metrics: Dict[str, float],
    title: Optional[str] = None,
    comparison_metrics: Optional[Dict[str, float]] = None
):
    """Print a formatted metrics report.

    Args:
        metrics: Dictionary of metrics to display
        title: Optional title for the report
        comparison_metrics: Optional baseline metrics for comparison
    """
    from helpers import format_percentage

    if title:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    # Group metrics by type
    regression_metrics = ["mae", "rmse", "r2", "median_error", "p90_error"]
    ranking_metrics = ["spearman", "ndcg@20", "ndcg@10"]
    other_metrics = [k for k in metrics.keys()
                     if k not in regression_metrics + ranking_metrics]

    def print_metric(name: str, value: float, baseline: Optional[float] = None):
        """Helper to print a single metric with optional comparison."""
        display_name = name.replace("_", " ").title()

        # Format based on metric type
        if "ndcg" in name or "r2" in name or "spearman" in name:
            formatted = f"{value:.4f}"
        elif "error" in name or "mae" in name or "rmse" in name:
            formatted = f"{value:.3f}"
        else:
            formatted = f"{value:.3f}"

        output = f"  {display_name:20s}: {formatted}"

        # Add comparison if baseline provided
        if baseline is not None:
            if "error" in name or "mae" in name or "rmse" in name:
                # Lower is better for error metrics
                improvement = (baseline - value) / max(abs(baseline), 0.001) * 100
            else:
                # Higher is better for other metrics
                improvement = (value - baseline) / max(abs(baseline), 0.001) * 100

            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            output += f" ({arrow} {abs(improvement):.1f}%)"

        print(output)

    # Print regression metrics
    if any(m in metrics for m in regression_metrics):
        print("\nRegression Metrics:")
        for metric in regression_metrics:
            if metric in metrics:
                baseline = comparison_metrics.get(metric) if comparison_metrics else None
                print_metric(metric, metrics[metric], baseline)

    # Print ranking metrics
    if any(m in metrics for m in ranking_metrics):
        print("\nRanking Metrics:")
        for metric in ranking_metrics:
            if metric in metrics:
                baseline = comparison_metrics.get(metric) if comparison_metrics else None
                print_metric(metric, metrics[metric], baseline)

    # Print other metrics
    if other_metrics:
        print("\nOther Metrics:")
        for metric in other_metrics:
            if metric not in ["position"]:  # Skip non-numeric fields
                baseline = comparison_metrics.get(metric) if comparison_metrics else None
                print_metric(metric, metrics[metric], baseline)

    print()


def evaluate_predictions_by_percentile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentiles: List[int] = [25, 50, 75, 90, 95]
) -> Dict[str, Dict[str, float]]:
    """Evaluate prediction quality across different percentiles.

    Useful for understanding model performance on low/medium/high scorers.

    Args:
        y_true: True values
        y_pred: Predicted values
        percentiles: List of percentiles to evaluate

    Returns:
        Dictionary mapping percentile ranges to metrics
    """
    results = {}

    # Sort by true values
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Add overall metrics
    results["overall"] = calculate_metrics_suite(y_true, y_pred)

    # Evaluate each percentile range
    prev_cutoff = 0
    for p in percentiles:
        cutoff = int(len(y_true) * p / 100)
        if cutoff > prev_cutoff:
            slice_true = y_true_sorted[prev_cutoff:cutoff]
            slice_pred = y_pred_sorted[prev_cutoff:cutoff]

            if len(slice_true) > 1:
                range_name = f"p{prev_cutoff*100//len(y_true)}-p{p}"
                results[range_name] = calculate_metrics_suite(slice_true, slice_pred)

        prev_cutoff = cutoff

    return results


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calculate calibration metrics for probabilistic predictions.

    Args:
        y_true: True values
        y_pred_mean: Predicted mean values
        y_pred_std: Optional predicted standard deviations
        n_bins: Number of bins for calibration plot

    Returns:
        Dictionary of calibration metrics
    """
    metrics = {}

    # Basic calibration: predicted vs actual by bins
    bin_edges = np.percentile(y_pred_mean, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_pred_mean, bin_edges[:-1]) - 1

    calibration_error = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            predicted_mean = np.mean(y_pred_mean[mask])
            actual_mean = np.mean(y_true[mask])
            calibration_error += np.abs(predicted_mean - actual_mean) * np.sum(mask)

    metrics["mean_calibration_error"] = calibration_error / len(y_true)

    # If uncertainty provided, calculate coverage
    if y_pred_std is not None:
        # Check what percentage fall within 1 and 2 standard deviations
        within_1std = np.mean(
            np.abs(y_true - y_pred_mean) <= y_pred_std
        )
        within_2std = np.mean(
            np.abs(y_true - y_pred_mean) <= 2 * y_pred_std
        )

        metrics["coverage_1std"] = within_1std
        metrics["coverage_2std"] = within_2std
        metrics["expected_1std"] = 0.683  # Expected for normal distribution
        metrics["expected_2std"] = 0.954

        # Sharpness: average predicted uncertainty
        metrics["mean_uncertainty"] = np.mean(y_pred_std)
        metrics["uncertainty_cv"] = np.std(y_pred_std) / np.mean(y_pred_std)

    return metrics
