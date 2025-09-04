"""Safe helper functions for DFS system.

These helpers consolidate repeated patterns without affecting any DFS-specific
calculations or model behavior. They only handle:
1. Data type conversions
2. Logging formatting
3. File path validation
4. Currency and percentage formatting
5. Progress display for long-running operations
6. Date parsing utilities

NO business logic or DFS calculations are included here.
"""

import logging
from datetime import datetime
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


class ProgressDisplay:
    """Simple progress display that updates in place.

    Usage:
        progress = ProgressDisplay("Loading data")
        for i, item in enumerate(items):
            progress.update(i + 1, len(items))
            # process item
        progress.finish("Data loaded successfully!")
    """

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.last_percentage = -1
        self._finished = False

    def update(self, current: int, total: int):
        """Update progress display with current/total."""
        if self._finished or total == 0:
            return

        percentage = int((current / total) * 100)

        # Only update if percentage changed to reduce output
        if percentage != self.last_percentage:
            print(f"\r{self.description}: {percentage}%", end="", flush=True)
            self.last_percentage = percentage

    def finish(self, message: str = None):
        """Clear the progress line and optionally print a completion message."""
        if self._finished:
            return

        self._finished = True
        if message:
            print(f"\r{message}")
        else:
            print()  # Just add newline to clear the line


def parse_date_flexible(date_value: Any) -> datetime:
    """Parse various date formats flexibly.

    Args:
        date_value: Date in various formats (datetime object, string, etc.)

    Returns:
        datetime object

    Raises:
        ValueError: If the date cannot be parsed
    """
    if isinstance(date_value, datetime):
        return date_value

    if not date_value:
        raise ValueError("Date value is empty")

    # Convert to string if needed
    date_str = str(date_value)

    # Remove any time component if present
    date_str = date_str.split(" ")[0]

    # Try common date formats
    formats = [
        "%Y-%m-%d",      # ISO format: 2024-01-15
        "%m/%d/%Y",      # American: 1/15/2024 or 01/15/2024
        "%Y/%m/%d",      # Alternative: 2024/01/15
        "%d-%m-%Y",      # European: 15-01-2024
        "%d/%m/%Y",      # European: 15/01/2024
        "%Y%m%d",        # Compact: 20240115
        "%m-%d-%Y",      # Alternative American: 01-15-2024
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unable to parse date: '{date_str}'. "
        f"Expected formats: YYYY-MM-DD, M/D/YYYY, or other common formats"
    )
