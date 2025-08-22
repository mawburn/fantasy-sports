"""Application configuration management."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "data/nfl_dfs.db"
    pool_size: int = 5
    timeout: float = 30.0


@dataclass
class ModelConfig:
    """Model configuration."""
    models_dir: str = "models"
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    salary_cap: int = 50000
    lineup_dir: str = "lineups"
    enable_stacking: bool = True
    max_exposure: float = 0.3


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    results_dir: str = "backtest_results"
    include_ownership: bool = True
    min_contests: int = 5


class Config:
    """Application configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or ".env"
        self._load_environment()
        
        self.database = DatabaseConfig()
        self.models = ModelConfig()
        self.optimization = OptimizationConfig()
        self.backtest = BacktestConfig()
        
        # API keys and external services
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
    def _load_environment(self):
        """Load environment variables from .env file."""
        env_path = Path(self.config_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return os.getenv(key, default)
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Global configuration instance
config = Config()