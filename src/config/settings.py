"""Application settings and configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False

    # Database Configuration
    database_url: str = "sqlite:///data/database/nfl_dfs.db"
    database_pool_size: int = 5
    database_echo: bool = False

    # Model Configuration
    model_path: Path = Path("models/production")
    enable_cpu_optimization: bool = True
    num_cpu_threads: int = 8
    batch_size: int = 32
    # Future GPU support - currently disabled
    gpu_available: bool = False

    # Cache Configuration
    cache_backend: Literal["memory", "redis", "file"] = "memory"
    cache_ttl: int = 3600
    redis_url: str = "redis://localhost:6379"

    # NFL Data Configuration
    nfl_seasons_to_load: int = 3
    nfl_data_cache_dir: Path = Path("data/cache")
    data_refresh_interval: int = 3600

    # DraftKings Configuration
    dk_classic_salary_cap: int = 50000
    dk_showdown_salary_cap: int = 50000

    # ML Configuration
    mlflow_tracking_uri: str = "file:///data/mlflow"
    default_model_path: Path = Path("models/production")

    # Feature Flags
    enable_self_tuning: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("data/logs/nfl_dfs.log")

    # External APIs (optional)
    weather_api_key: str | None = None
    odds_api_key: str | None = None

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """Get the models directory."""
        return self.project_root / "models"


# Global settings instance
settings = Settings()
