"""Safe helper functions for DFS system.

These helpers consolidate repeated patterns without affecting any DFS-specific
calculations or model behavior. They only handle:
1. Data type conversions
2. Logging formatting
3. File path validation

NO business logic or DFS calculations are included here.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default fallback.

    Args:
        val: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely convert value to integer with default fallback.

    Args:
        val: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    try:
        return int(val) if pd.notna(val) and val != "" else default
    except (ValueError, TypeError):
        return default


def log_model_metrics(
    model_name: str,
    phase: str,
    mae: float,
    r2: float,
    rmse: Optional[float] = None
) -> None:
    """Standardized logging for model metrics.

    Args:
        model_name: Name of the model (e.g., "QB", "RB")
        phase: Training phase (e.g., "Training", "Validation", "Test")
        mae: Mean Absolute Error
        r2: R-squared score
        rmse: Root Mean Square Error (optional)
    """
    metrics_str = f"{model_name} {phase}: MAE={mae:.3f}, R²={r2:.3f}"
    if rmse is not None:
        metrics_str += f", RMSE={rmse:.3f}"
    logger.info(metrics_str)


def validate_file_path(file_path: Union[str, Path], file_type: str = "File") -> Path:
    """Validate that a file path exists.

    Args:
        file_path: Path to validate
        file_type: Description for error message (e.g., "Model", "CSV")

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} not found: {file_path}")
    return path


def format_currency(amount: float) -> str:
    """Format number as currency for display.

    Args:
        amount: Dollar amount

    Returns:
        Formatted string (e.g., "$50,000")
    """
    return f"${amount:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format decimal as percentage.

    Args:
        value: Decimal value (e.g., 0.234)
        decimals: Number of decimal places

    Returns:
        Formatted percentage (e.g., "23.4%")
    """
    return f"{value * 100:.{decimals}f}%"
