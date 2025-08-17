"""Application settings and configuration management.

This file implements a centralized configuration system using Pydantic Settings.
It handles environment variables, default values, and configuration validation
for the entire fantasy football application.

Key Benefits:
- Type safety: All settings have defined types with validation
- Environment integration: Automatically loads from .env files
- Documentation: Clear descriptions of what each setting controls
- Flexibility: Easy to override for different environments

For beginners:

Pydantic Settings: A Python library that automatically validates configuration
and loads values from environment variables, .env files, and defaults.

Environment Variables: System variables that configure applications without
changing code. Example: DATABASE_URL=sqlite:///mydb.db

Configuration Patterns: Centralized config management prevents hardcoded
values scattered throughout the codebase and makes deployment easier.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    This class defines all configuration options for the fantasy football application.
    It uses Pydantic BaseSettings to automatically handle:
    - Loading from environment variables
    - Loading from .env files
    - Type validation and conversion
    - Default value management

    Configuration Sources (in priority order):
    1. Environment variables (highest priority)
    2. .env file values
    3. Default values defined here (lowest priority)

    Example Usage:
    - In code: `settings.database_url`
    - Environment variable: `DATABASE_URL=sqlite:///prod.db`
    - .env file: `database_url=sqlite:///dev.db`
    """

    # Pydantic configuration for settings behavior
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file in project root
        env_file_encoding="utf-8",  # Handle unicode characters in env file
        case_sensitive=False,  # Allow DATABASE_URL or database_url
    )

    # API Configuration - FastAPI web server settings
    api_host: str = "127.0.0.1"  # Host to bind server (localhost for development)
    api_port: int = 8000  # Port number for API server
    api_workers: int = 1  # Number of worker processes (1 for dev, more for prod)
    api_reload: bool = False  # Auto-reload on code changes (True for development)

    # Database Configuration - SQLAlchemy connection settings
    database_url: str = "sqlite:///data/database/nfl_dfs.db"  # Database connection string
    database_pool_size: int = 5  # Connection pool size (max concurrent connections)
    database_echo: bool = False  # Log all SQL queries (True for debugging, False for prod)

    # ML Model Configuration - Machine learning and PyTorch settings
    model_path: Path = Path("models/production")  # Directory for trained model files
    enable_cpu_optimization: bool = True  # Enable CPU performance optimizations
    num_cpu_threads: int = 8  # CPU threads for ML operations (adjust for your CPU)
    batch_size: int = 32  # Batch size for ML inference
    # Future GPU support - currently disabled (CPU-only for now)
    gpu_available: bool = False  # Whether to use GPU acceleration

    # Cache Configuration - Performance optimization through caching
    cache_backend: Literal["memory", "redis", "file"] = "memory"  # Caching strategy
    cache_ttl: int = 3600  # Cache time-to-live in seconds (1 hour)
    redis_url: str = "redis://localhost:6379"  # Redis connection for distributed caching

    # NFL Data Configuration - Settings for NFL data collection and processing
    nfl_seasons_to_load: int = 3  # How many seasons of data to load (3 years typical)
    nfl_data_cache_dir: Path = Path("data/cache")  # Directory for cached NFL data files
    data_refresh_interval: int = 3600  # How often to refresh data (seconds)

    # DraftKings Configuration - Contest format settings
    dk_classic_salary_cap: int = 50000  # Salary cap for classic contests ($50,000)
    dk_showdown_salary_cap: int = 50000  # Salary cap for showdown contests

    # ML Configuration - Machine learning experiment tracking and deployment
    mlflow_tracking_uri: str = "file:///data/mlflow"  # MLflow experiment tracking location
    default_model_path: Path = Path("models/production")  # Default location for production models

    # Feature Flags - Toggle advanced features on/off
    enable_self_tuning: bool = True  # Enable automatic hyperparameter tuning
    enable_monitoring: bool = True  # Enable performance monitoring and metrics
    enable_caching: bool = True  # Enable caching for better performance

    # Logging Configuration - Application logging settings
    log_level: str = "INFO"  # Log level (DEBUG, INFO, WARNING, ERROR)
    log_file: Path = Path("data/logs/nfl_dfs.log")  # Log file location

    # External APIs (optional) - Third-party service integrations
    weather_api_key: str | None = None  # Weather API key for weather-dependent predictions
    odds_api_key: str | None = None  # The Odds API key for betting lines and market data

    # Data Collection Configuration - Settings for various data collectors
    collect_vegas_odds: bool = True  # Enable/disable Vegas odds collection
    collect_stadium_data: bool = True  # Enable/disable stadium data collection
    odds_collection_interval: int = 14400  # How often to collect odds (4 hours in seconds)
    stadium_data_refresh_interval: int = 2592000  # Stadium data refresh (30 days in seconds)

    @property
    def project_root(self) -> Path:
        """Get the project root directory.

        Calculates the project root by going up 3 levels from this file:
        src/config/settings.py -> src/config -> src -> project_root

        This is useful for constructing relative paths that work regardless
        of where the application is run from.

        Returns:
            Path object pointing to the project root directory
        """
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory.

        Returns the standard data directory relative to project root.
        Used for storing databases, cache files, logs, and downloaded data.

        Returns:
            Path object pointing to the data directory
        """
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """Get the models directory.

        Returns the standard models directory relative to project root.
        Used for storing trained ML models, model metadata, and model artifacts.

        Returns:
            Path object pointing to the models directory
        """
        return self.project_root / "models"


# Global settings instance - singleton pattern for application-wide configuration
# This creates a single Settings instance that can be imported and used throughout the app
# Example: from src.config.settings import settings; print(settings.database_url)
settings = Settings()
