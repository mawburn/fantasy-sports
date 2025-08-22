"""Base neural network model class."""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from dfs.core.logging import get_logger
from dfs.core.exceptions import ModelError
from .utils import ModelConfig, PredictionResult, ModelResult

logger = get_logger("models.base.neural")

# Position-specific fantasy point ranges (DraftKings scoring)
POSITION_RANGES = {
    'QB': 45.0,   # QBs typically score 15-35, max ~45
    'RB': 35.0,   # RBs typically score 8-25, max ~35
    'WR': 30.0,   # WRs typically score 6-22, max ~30
    'TE': 25.0,   # TEs typically score 4-18, max ~25
    'DEF': 20.0,  # DEFs typically score 5-15, max ~20
    'DST': 20.0   # Same as DEF
}

# Device selection optimized for Apple Silicon
def get_optimal_device():
    """Get the best available device for Apple Silicon M-series."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal) acceleration on Apple Silicon")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA acceleration")
        return device
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
        return device

OPTIMAL_DEVICE = get_optimal_device()


class BaseNeuralModel(ABC):
    """Base class for PyTorch neural network models."""

    def __init__(self, config: ModelConfig):
        """Initialize neural network base model."""
        self.config = config
        self.device = OPTIMAL_DEVICE
        self.network: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Training parameters
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.weight_decay = config.weight_decay
        self.patience = config.patience
        
        # Architecture parameters
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.is_trained = False
        self.training_history = []
        
        # Target normalization parameters
        self.y_mean = 0.0
        self.y_std = 1.0
        self.input_size = config.input_size

    def _clip_targets_by_position(self, y: np.ndarray) -> np.ndarray:
        """Clip/winsorize targets by position before training."""
        position = self.config.position.upper()
        
        # Position-specific clipping ranges
        clip_ranges = {
            'QB': (-5, 55),
            'RB': (-5, 45),
            'WR': (-5, 40),
            'TE': (-5, 30),
            'DST': (-5, 30),
            'DEF': (-5, 30)
        }
        
        if position in clip_ranges:
            low, high = clip_ranges[position]
            y_clipped = np.clip(y, low, high)
            logger.info(f"Clipped {position} targets to [{low}, {high}]. "
                       f"Original range: [{y.min():.1f}, {y.max():.1f}], "
                       f"Clipped range: [{y_clipped.min():.1f}, {y_clipped.max():.1f}]")
            return y_clipped
        
        return y

    def _normalize_targets(self, y: np.ndarray) -> np.ndarray:
        """Normalize targets to zero mean, unit variance."""
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        if self.y_std == 0:
            raise ModelError(f"Target standard deviation is zero for {self.config.position}")
        
        y_normalized = (y - self.y_mean) / self.y_std
        logger.info(f"Normalized {self.config.position} targets: mean={self.y_mean:.2f}, std={self.y_std:.2f}")
        return y_normalized

    def _denormalize_predictions(self, pred: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return pred * self.y_std + self.y_mean

    @abstractmethod
    def build_network(self, input_size: int) -> nn.Module:
        """Build position-specific neural network architecture."""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> ModelResult:
        """Train the neural network model."""
        try:
            # Prepare data
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip targets by position
            y_train = self._clip_targets_by_position(y_train)
            y_val = self._clip_targets_by_position(y_val)
            
            # Normalize targets
            y_train_norm = self._normalize_targets(y_train)
            y_val_norm = (y_val - self.y_mean) / self.y_std
            
            # Build network
            input_size = X_train.shape[1]
            self.input_size = input_size
            self.network = self.build_network(input_size)
            self.network.to(self.device)
            
            # Setup optimizer and scheduler
            self.optimizer = optim.AdamW(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            
            # Create data loaders
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0
            
            for epoch in range(self.epochs):
                # Training phase
                self.network.train()
                train_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.network(batch_x)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation phase
                self.network.eval()
                with torch.no_grad():
                    X_val_device = X_val_tensor.to(self.device)
                    y_val_device = y_val_tensor.to(self.device)
                    val_outputs = self.network(X_val_device)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val_device).item()
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 50 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
            self.is_trained = True
            
            # Calculate final metrics
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            
            train_mae = np.mean(np.abs(train_pred.point_estimate - y_train))
            val_mae = np.mean(np.abs(val_pred.point_estimate - y_val))
            train_rmse = np.sqrt(np.mean((train_pred.point_estimate - y_train) ** 2))
            val_rmse = np.sqrt(np.mean((val_pred.point_estimate - y_val) ** 2))
            
            # R² calculation
            train_r2 = 1 - np.sum((y_train - train_pred.point_estimate) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
            val_r2 = 1 - np.sum((y_val - val_pred.point_estimate) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
            
            return ModelResult(
                model=self,
                training_time=0.0,  # Would need to track actual time
                best_iteration=best_epoch,
                feature_importance=None,
                train_mae=train_mae,
                val_mae=val_mae,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                train_r2=train_r2,
                val_r2=val_r2,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=input_size
            )
            
        except Exception as e:
            logger.error(f"Training failed for {self.config.position}: {e}")
            raise ModelError(f"Model training failed: {e}")

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions using the trained model."""
        if not self.is_trained or self.network is None:
            raise ModelError("Model must be trained before making predictions")
        
        # Prepare data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(X_tensor)
            predictions_norm = outputs.squeeze().cpu().numpy()
        
        # Denormalize predictions
        predictions = self._denormalize_predictions(predictions_norm)
        
        # Calculate confidence scores and intervals (simplified)
        confidence = np.ones_like(predictions) * 0.8  # Placeholder
        
        # Floor and ceiling estimates
        floor = predictions * 0.7
        ceiling = predictions * 1.3
        
        prediction_intervals = (predictions * 0.8, predictions * 1.2)
        
        return PredictionResult(
            point_estimate=predictions,
            confidence_score=confidence,
            prediction_intervals=prediction_intervals,
            floor=floor,
            ceiling=ceiling,
            model_version=f"{self.config.position}_v1.0"
        )

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.network is None:
            raise ModelError("No model to save")
        
        state = {
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'input_size': self.input_size
        }
        torch.save(state, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        state = torch.load(filepath, map_location=self.device)
        
        self.config = state['config']
        self.y_mean = state['y_mean']
        self.y_std = state['y_std']
        self.input_size = state['input_size']
        
        # Build and load network
        self.network = self.build_network(self.input_size)
        self.network.load_state_dict(state['network_state_dict'])
        self.network.to(self.device)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")