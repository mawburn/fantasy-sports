"""
DFS Optimization System

A modular daily fantasy sports optimization platform with machine learning models,
lineup optimization, and comprehensive backtesting capabilities.
"""

__version__ = "2.0.0"

# Maintain backward compatibility with existing imports
from .models.factory import create_model
from .optimization.engine import optimize_lineup
from .backtesting.engine import run_backtest
from .prediction.pipeline import predict_players

__all__ = [
    "create_model",
    "optimize_lineup", 
    "run_backtest",
    "predict_players"
]