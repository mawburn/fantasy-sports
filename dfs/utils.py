"""Minimal utilities for the simplified DFS system.

This module contains only essential helper functions and configurations:
1. Logging configuration
2. Configuration management
3. Essential helper functions

No abstractions or classes - just simple functions that work.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up basic logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file with sensible defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "database": {"path": "data/nfl_dfs.db"},
        "models": {
            "directory": "models",
            "training_seasons": 3,
            "validation_split": 0.2,
        },
        "optimization": {
            "salary_cap": 50000,
            "lineup_count": 5,
            "default_strategy": "balanced",
        },
        "data_collection": {"seasons": [2022, 2023, 2024], "lookback_weeks": 4},
        "logging": {"level": "INFO", "file": None},
    }

    # Try to load user config
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)

            # Merge with defaults (user config overwrites defaults)
            config = merge_dicts(default_config, user_config)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            config = default_config
    else:
        config = default_config
        logging.info("Using default configuration")

    return config


def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config to {config_path}: {e}")


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge two dictionaries.

    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (overwrites dict1 values)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def format_currency(amount: float) -> str:
    """Format amount as currency.

    Args:
        amount: Amount to format

    Returns:
        Formatted currency string
    """
    return f"${amount:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage.

    Args:
        value: Value to format (0.0 to 1.0)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_current_season() -> int:
    """Get current NFL season based on date.

    Returns:
        Current season year
    """
    now = datetime.now()

    # NFL season starts in September and ends in February of the following year
    if now.month >= 9:  # September onwards is current year season
        return now.year
    elif now.month <= 2:  # January-February is previous year season
        return now.year - 1
    else:  # March-August is upcoming season
        return now.year


def validate_position(position: str) -> bool:
    """Validate NFL position.

    Args:
        position: Position string

    Returns:
        True if valid position
    """
    valid_positions = {"QB", "RB", "WR", "TE", "DEF", "DST", "K"}
    return position.upper() in valid_positions


def validate_team_abbr(team_abbr: str) -> bool:
    """Validate NFL team abbreviation.

    Args:
        team_abbr: Team abbreviation

    Returns:
        True if valid team
    """
    valid_teams = {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LV",
        "LAC",
        "LAR",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SF",
        "SEA",
        "TB",
        "TEN",
        "WAS",
    }
    return team_abbr.upper() in valid_teams


def calculate_value(projected_points: float, salary: int) -> float:
    """Calculate value (points per $1000 salary).

    Args:
        projected_points: Projected fantasy points
        salary: Player salary

    Returns:
        Value score
    """
    if salary <= 0:
        return 0.0
    return projected_points / (salary / 1000)


def get_week_from_date(date_str: str) -> int:
    """Get NFL week from date string (simplified).

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Estimated NFL week (1-18)
    """
    try:
        date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")

        # NFL season typically starts first week of September
        season_start = datetime(date_obj.year, 9, 1)
        if date_obj < season_start:
            # Previous season (January-August)
            season_start = datetime(date_obj.year - 1, 9, 1)

        days_since_start = (date_obj - season_start).days
        week = max(1, min(18, (days_since_start // 7) + 1))

        return week
    except ValueError:
        return 1


def format_runtime(seconds: float) -> str:
    """Format runtime in seconds to human readable format.

    Args:
        seconds: Runtime in seconds

    Returns:
        Formatted runtime string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_environment() -> str:
    """Get current environment (dev, test, prod).

    Returns:
        Environment string
    """
    return os.getenv("ENVIRONMENT", "dev").lower()


def is_debug_mode() -> bool:
    """Check if debug mode is enabled.

    Returns:
        True if debug mode enabled
    """
    return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")


# Configuration singleton
_config = None


def get_config() -> Dict[str, Any]:
    """Get global configuration (singleton pattern).

    Returns:
        Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config_value(key_path: str, value: Any) -> None:
    """Set configuration value using dot notation.

    Args:
        key_path: Dot-separated key path (e.g., 'database.path')
        value: Value to set
    """
    config = get_config()
    keys = key_path.split(".")

    # Navigate to parent dictionary
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set value
    current[keys[-1]] = value


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation.

    Args:
        key_path: Dot-separated key path (e.g., 'database.path')
        default: Default value if key not found

    Returns:
        Configuration value
    """
    config = get_config()
    keys = key_path.split(".")

    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


# Simple performance timer context manager
class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logging.info(f"{self.name} completed in {format_runtime(duration)}")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# Simple health check function
def health_check() -> Dict[str, Any]:
    """Perform basic system health check.

    Returns:
        Health check results
    """
    import sqlite3

    import torch

    checks = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_available": True,
        "sqlite_available": True,
        "config_loaded": _config is not None,
        "errors": [],
    }

    # Check PyTorch
    try:
        torch.tensor([1.0])
    except Exception as e:
        checks["torch_available"] = False
        checks["errors"].append(f"PyTorch error: {e}")

    # Check SQLite
    try:
        conn = sqlite3.connect(":memory:")
        conn.close()
    except Exception as e:
        checks["sqlite_available"] = False
        checks["errors"].append(f"SQLite error: {e}")

    # Check database
    db_path = get_config_value("database.path", "data/nfl_dfs.db")
    checks["database_exists"] = os.path.exists(db_path)

    # Check models directory
    models_dir = get_config_value("models.directory", "models")
    checks["models_directory_exists"] = os.path.exists(models_dir)

    return checks


# Version information
__version__ = "1.0.0"
__author__ = "DFS System"


def get_version_info() -> Dict[str, str]:
    """Get version information.

    Returns:
        Version information dictionary
    """
    return {
        "version": __version__,
        "author": __author__,
        "python_version": sys.version,
        "build_date": datetime.now().isoformat(),
    }
