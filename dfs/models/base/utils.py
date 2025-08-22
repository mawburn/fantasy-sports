"""Model utilities and result classes."""

import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple, Optional


@dataclass
class PredictionResult:
    """Results from model prediction."""
    point_estimate: np.ndarray
    confidence_score: np.ndarray
    prediction_intervals: Tuple[np.ndarray, np.ndarray]
    floor: np.ndarray
    ceiling: np.ndarray
    model_version: str


@dataclass
class ModelResult:
    """Results from model training/evaluation."""
    model: Any
    training_time: float
    best_iteration: int
    feature_importance: Optional[np.ndarray]
    train_mae: float
    val_mae: float
    train_rmse: float
    val_rmse: float
    train_r2: float
    val_r2: float
    training_samples: int
    validation_samples: int
    feature_count: int


@dataclass
class ModelConfig:
    """Configuration for model training."""
    position: str
    features: list
    input_size: Optional[int] = None
    hidden_size: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.3
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 500
    weight_decay: float = 0.001
    patience: int = 50