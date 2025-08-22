"""Simplified PyTorch models for NFL fantasy sports predictions.

This module consolidates all neural network models and correlation features
into a single file for simplicity and maintainability.

Core Components:
1. Position-specific neural networks (QB, RB, WR, TE, DEF)
2. Correlation feature extraction
3. Multi-position correlated model
4. Training and prediction utilities
"""

import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import ProgressDisplay

# Import hyperparameter manager with error handling
try:
    from hyperparameter_manager import get_hyperparameter_manager
    HAS_HYPERPARAMETER_MANAGER = True
except ImportError:
    HAS_HYPERPARAMETER_MANAGER = False
    logger.warning("Hyperparameter manager not available. Using default hyperparameters.")

logger = logging.getLogger(__name__)

# Import gradient boosting libraries for ensemble learning
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available.")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available.")

if not HAS_XGBOOST and not HAS_CATBOOST:
    logger.warning("No gradient boosting libraries available. Ensemble features disabled.")

# Position-specific fantasy point ranges (DraftKings scoring)
POSITION_RANGES = {
    'QB': 45.0,   # QBs typically score 15-35, max ~45
    'RB': 35.0,   # RBs typically score 8-25, max ~35
    'WR': 30.0,   # WRs typically score 6-22, max ~30
    'TE': 25.0,   # TEs typically score 4-18, max ~25
    'DEF': 20.0,  # DEFs typically score 5-15, max ~20
    'DST': 20.0   # Same as DEF
}

# Set PyTorch for reproducible, Apple Silicon M-series optimized training
import os
torch.manual_seed(42)

# Optimize for Apple Silicon M-series chips
cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)  # Use all available cores (P-cores + E-cores)

# Device selection optimized for Apple Silicon
def get_optimal_device():
    """Get the best available device for Apple Silicon M-series."""
    if torch.backends.mps.is_available():
        # MPS (Metal Performance Shaders) for M-series GPU acceleration
        device = torch.device("mps")
        logger.info(f"Using MPS (Metal) acceleration on Apple Silicon with {cpu_count} CPU threads")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA acceleration")
        return device
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU with {cpu_count} threads")
        return device

# Global optimal device
OPTIMAL_DEVICE = get_optimal_device()

# Set deterministic behavior (MPS supports deterministic operations)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Apple Silicon memory optimizations
if OPTIMAL_DEVICE.type == "mps":
    # Enable memory efficient attention for M-series
    torch.backends.mps.enable_fallback = True  # Fallback to CPU for unsupported ops


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    position: str
    version: str = "1.0"
    features: List[str] = None


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
class TrainingResult:
    """Results from model training."""
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


class LRFinder:
    """Learning Rate Range Test implementation for finding optimal learning rates."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device):
        """Initialize LR Finder.

        Args:
            model: PyTorch model to test
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device (cpu/cuda/mps)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store initial model state
        self.initial_state = model.state_dict()

        # Results storage
        self.learning_rates = []
        self.losses = []
        self.smoothed_losses = []

    def range_test(self, train_loader: DataLoader, start_lr: float = 1e-8,
                   end_lr: float = 1.0, num_iter: int = 100, smooth_f: float = 0.98) -> float:
        """Run learning rate range test.

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss

        Returns:
            Optimal learning rate
        """
        # Reset model to initial state
        self.model.load_state_dict(self.initial_state)
        self.model.train()

        # Calculate LR schedule (exponential)
        lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)

        # Initialize tracking
        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0

        # Create iterator from data loader
        data_iter = iter(train_loader)

        for iteration, lr in enumerate(lr_schedule):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Unpack batch
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle dictionary outputs (for models with mean/floor/ceiling)
            if isinstance(outputs, dict):
                outputs = outputs['mean']

            # Ensure outputs and targets have same shape
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)

            loss = self.criterion(outputs, targets)

            # Compute smoothed loss
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * avg_loss + (1 - smooth_f) * loss.item()

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Store results
            self.learning_rates.append(lr)
            self.losses.append(loss.item())
            self.smoothed_losses.append(avg_loss)

            # Check for explosion (loss > 4x best loss)
            if avg_loss > 4 * best_loss:
                logger.info(f"Loss exploded at LR={lr:.2e}, stopping early")
                break

            # Backward pass
            loss.backward()
            self.optimizer.step()

            batch_num += 1

        # Find optimal learning rate
        optimal_lr = self._find_optimal_lr()

        # Reset model to initial state
        self.model.load_state_dict(self.initial_state)

        return optimal_lr

    def _find_optimal_lr(self) -> float:
        """Find optimal learning rate from test results.

        Returns:
            Optimal learning rate (steepest decline point)
        """
        if len(self.smoothed_losses) < 10:
            # Not enough data, return conservative estimate
            return self.learning_rates[len(self.learning_rates) // 2]

        # Calculate gradients of smoothed loss curve
        gradients = np.gradient(self.smoothed_losses)

        # Find steepest negative gradient (biggest improvement)
        min_gradient_idx = np.argmin(gradients)

        # Get learning rate at that point
        optimal_lr = self.learning_rates[min_gradient_idx]

        # Apply safety factor (use slightly lower LR for stability)
        safety_factor = 0.5
        optimal_lr *= safety_factor

        logger.info(f"Found optimal LR: {optimal_lr:.2e} (steepest decline at index {min_gradient_idx})")

        return optimal_lr

    def plot_results(self, save_path: str = None):
        """Plot learning rate vs loss curve.

        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot raw losses
            ax1.semilogx(self.learning_rates, self.losses, label='Raw Loss')
            ax1.semilogx(self.learning_rates, self.smoothed_losses, label='Smoothed Loss')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Loss')
            ax1.set_title('Learning Rate Range Test')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot gradients
            gradients = np.gradient(self.smoothed_losses)
            ax2.semilogx(self.learning_rates, gradients)
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Loss Gradient')
            ax2.set_title('Loss Gradient (for finding optimal LR)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            # Mark optimal LR
            optimal_lr = self._find_optimal_lr()
            for ax in [ax1, ax2]:
                ax.axvline(x=optimal_lr, color='g', linestyle='--',
                          label=f'Optimal LR={optimal_lr:.2e}')
                ax.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"LR finder plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class BatchSizeOptimizer:
    """Optimizer for finding optimal batch size considering memory constraints."""

    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize batch size optimizer.

        Args:
            model: PyTorch model
            device: Device (cpu/cuda/mps)
        """
        self.model = model
        self.device = device

    def find_max_batch_size(self, sample_input: torch.Tensor, max_batch_size: int = 512,
                            min_batch_size: int = 8) -> int:
        """Find maximum batch size that fits in memory.

        Args:
            sample_input: Sample input tensor (single example)
            max_batch_size: Maximum batch size to test
            min_batch_size: Minimum batch size

        Returns:
            Maximum feasible batch size
        """
        # Binary search for max batch size
        left, right = min_batch_size, max_batch_size
        max_feasible = min_batch_size

        while left <= right:
            mid = (left + right) // 2

            if self._test_batch_size(sample_input, mid):
                max_feasible = mid
                left = mid + 1
            else:
                right = mid - 1

        logger.info(f"Maximum feasible batch size: {max_feasible}")
        return max_feasible

    def _test_batch_size(self, sample_input: torch.Tensor, batch_size: int) -> bool:
        """Test if batch size fits in memory.

        Args:
            sample_input: Sample input tensor
            batch_size: Batch size to test

        Returns:
            True if batch size fits, False otherwise
        """
        try:
            # Create batch
            batch = sample_input.unsqueeze(0).repeat(batch_size, 1)
            batch = batch.to(self.device)

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch)

            # Clean up
            del batch
            del output

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                # MPS doesn't have explicit cache clearing
                pass

            return True

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                logger.debug(f"Batch size {batch_size} caused OOM")

                # Clean up
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                return False
            else:
                raise

    def optimize_batch_size(self, train_loader: DataLoader, val_loader: DataLoader,
                           model_trainer, batch_sizes: List[int] = None,
                           epochs_per_test: int = 50) -> int:
        """Find optimal batch size by testing performance.

        Args:
            train_loader: Training data loader (will be recreated with different batch sizes)
            val_loader: Validation data loader
            model_trainer: Model trainer instance with train method
            batch_sizes: List of batch sizes to test (default: [16, 32, 64, 128, 256])
            epochs_per_test: Number of epochs to train for each batch size

        Returns:
            Optimal batch size
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128, 256]

        # Get sample input for memory testing
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 2:
            sample_input = sample_batch[0][0]
        else:
            sample_input = sample_batch[0][0]

        # Find max feasible batch size
        max_feasible = self.find_max_batch_size(sample_input)

        # Filter batch sizes to feasible ones
        feasible_batch_sizes = [bs for bs in batch_sizes if bs <= max_feasible]

        if not feasible_batch_sizes:
            logger.warning(f"No feasible batch sizes found, using minimum: {min(batch_sizes)}")
            return min(batch_sizes)

        results = {}

        for batch_size in feasible_batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Store original epochs setting
            original_epochs = model_trainer.epochs
            model_trainer.epochs = epochs_per_test
            model_trainer.batch_size = batch_size

            # Train with this batch size
            try:
                # Get training data
                train_dataset = train_loader.dataset
                val_dataset = val_loader.dataset

                # Create new loaders with test batch size
                test_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Train for limited epochs
                result = model_trainer._train_epochs(test_train_loader, test_val_loader, epochs_per_test)

                # Store result (use validation loss as metric)
                results[batch_size] = result['val_loss']

                logger.info(f"Batch size {batch_size}: val_loss={result['val_loss']:.4f}")

            except Exception as e:
                logger.warning(f"Failed to test batch size {batch_size}: {e}")
                results[batch_size] = float('inf')

            # Restore original epochs
            model_trainer.epochs = original_epochs

        # Find best batch size (lowest validation loss)
        best_batch_size = min(results.keys(), key=lambda k: results[k])

        logger.info(f"Optimal batch size: {best_batch_size} (val_loss={results[best_batch_size]:.4f})")

        return best_batch_size


# Only import optuna if available
try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True

    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
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
        import numpy as np

        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

        # Ensure we have valid arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Handle NaN values by setting them to minimum
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            y_true = np.nan_to_num(y_true, nan=np.nanmin(y_true) if not np.all(np.isnan(y_true)) else 0.0)
            y_pred = np.nan_to_num(y_pred, nan=np.nanmin(y_pred) if not np.all(np.isnan(y_pred)) else 0.0)

        # Limit k to the number of available items
        n_items = len(y_true)
        k = min(k, n_items)

        if k <= 0:
            return 0.0

        # Handle negative fantasy points by shifting to make minimum = 0
        # This ensures relevance scores are non-negative as required by NDCG
        min_true = np.min(y_true)
        if min_true < 0:
            y_true_shifted = y_true - min_true
        else:
            y_true_shifted = y_true.copy()

        # Sort indices by predicted values (descending order)
        predicted_order = np.argsort(y_pred)[::-1]

        # Get the top-k predictions and their true relevance scores
        top_k_indices = predicted_order[:k]
        top_k_true_scores = y_true_shifted[top_k_indices]

        # Calculate DCG@k (Discounted Cumulative Gain)
        # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
        dcg = 0.0
        for i, relevance in enumerate(top_k_true_scores):
            dcg += relevance / np.log2(i + 2)

        # Calculate ideal DCG@k (sort by true relevance scores)
        ideal_order = np.argsort(y_true_shifted)[::-1]
        ideal_top_k = y_true_shifted[ideal_order][:k]

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

    class HyperparameterTuner:
        """Joint hyperparameter optimization using Optuna."""

        def __init__(self, model_class, model_config: ModelConfig,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     epochs: int = 100):
            """Initialize hyperparameter tuner.

            Args:
                model_class: Model class to optimize
                model_config: Model configuration
                X_train: Training features
                y_train: Training targets
                X_val: Validation features
                y_val: Validation targets
                epochs: Number of epochs to train each trial
            """
            self.model_class = model_class
            self.model_config = model_config
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.epochs = epochs

            # Store best results
            self.best_params = None
            self.best_score = None

        def objective(self, trial: Trial) -> float:
            """Optuna objective function.

            Maximizes average NDCG@20 with MAE guardrail < 6.0.
            Also computes and logs Spearman correlation and R² for monitoring.

            Args:
                trial: Optuna trial

            Returns:
                Average NDCG@20 to maximize (with MAE guardrail)
            """
            # Get search ranges from hyperparameter manager
            from hyperparameter_manager import get_hyperparameter_manager
            hyperparameter_manager = get_hyperparameter_manager()
            search_ranges = hyperparameter_manager.get_search_ranges()

            # Suggest hyperparameters using YAML configuration
            lr_config = search_ranges.get('learning_rate', {})
            lr = trial.suggest_float(
                'learning_rate',
                float(lr_config.get('min', 1e-4)),
                float(lr_config.get('max', 5e-2)),
                log=lr_config.get('log_scale', True)
            )

            # Batch size
            batch_config = search_ranges.get('batch_size', {})
            if 'choices' in batch_config:
                batch_size = trial.suggest_categorical('batch_size', batch_config['choices'])
            elif 'step' in batch_config:
                batch_size = trial.suggest_int(
                    'batch_size',
                    int(batch_config.get('min', 16)),
                    int(batch_config.get('max', 256)),
                    step=int(batch_config['step'])
                )
            else:
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

            # Hidden size
            hidden_config = search_ranges.get('hidden_size', {})
            if 'choices' in hidden_config:
                hidden_size = trial.suggest_categorical('hidden_size', hidden_config['choices'])
            elif 'step' in hidden_config:
                hidden_size = trial.suggest_int(
                    'hidden_size',
                    int(hidden_config.get('min', 64)),
                    int(hidden_config.get('max', 512)),
                    step=int(hidden_config['step'])
                )
            else:
                hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

            # Dropout rate
            dropout_config = search_ranges.get('dropout_rate', {})
            dropout_rate = trial.suggest_float(
                'dropout_rate',
                float(dropout_config.get('min', 0.1)),
                float(dropout_config.get('max', 0.5))
            )

            # Number of layers
            layers_config = search_ranges.get('num_layers', {})
            num_layers = trial.suggest_int(
                'num_layers',
                int(layers_config.get('min', 1)),
                int(layers_config.get('max', 4))
            )

            # MAE guardrail threshold
            mae_guardrail = 6.0

            # Handle DST position (uses CatBoost instead of neural network)
            position = self.model_config.position.upper()
            if position == 'DST':
                # DST uses CatBoost, different hyperparameters
                if HAS_CATBOOST:
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                    depth = trial.suggest_int('depth', 4, 10)
                    iterations = trial.suggest_int('iterations', 100, 1000)
                    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)

                    # Create and train CatBoost model with suggested params
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(
                        learning_rate=learning_rate,
                        depth=depth,
                        iterations=iterations,
                        l2_leaf_reg=l2_leaf_reg,
                        verbose=False,
                        random_seed=42
                    )

                    model.fit(self.X_train, self.y_train,
                             eval_set=(self.X_val, self.y_val),
                             early_stopping_rounds=50,
                             verbose=False)

                    # Evaluate using MAE, NDCG@k, Spearman correlation, and R²
                    from sklearn.metrics import mean_absolute_error, r2_score
                    from scipy.stats import spearmanr
                    import numpy as np

                    val_pred = model.predict(self.X_val)

                    mae = mean_absolute_error(self.y_val, val_pred)
                    ndcg_score = ndcg_at_k(self.y_val, val_pred, k=20)
                    spearman_corr, _ = spearmanr(self.y_val, val_pred)
                    r2 = r2_score(self.y_val, val_pred)

                    # Handle NaN values (treat as 0.0)
                    if spearman_corr is None or np.isnan(spearman_corr):
                        spearman_corr = 0.0
                    if np.isnan(ndcg_score):
                        ndcg_score = 0.0
                    if np.isnan(r2):
                        r2 = 0.0

                    # Store metrics in trial attributes
                    trial.set_user_attr("mae", float(mae))
                    trial.set_user_attr("ndcg_at_k", float(ndcg_score))
                    trial.set_user_attr("spearman", float(spearman_corr))
                    trial.set_user_attr("r2", float(r2))

                    # Apply MAE guardrail
                    if mae >= mae_guardrail:
                        raise optuna.TrialPruned(f"MAE guardrail failed: {mae:.3f} >= {mae_guardrail}")

                    # Return NDCG@k for maximization
                    return float(ndcg_score)
                else:
                    return -1.0  # Penalize when CatBoost unavailable

            # Create model with suggested hyperparameters
            model = self.model_class(self.model_config)
            model.learning_rate = lr
            model.batch_size = batch_size

            # For neural network models, update architecture params
            if hasattr(model, 'hidden_size'):
                model.hidden_size = hidden_size
            if hasattr(model, 'dropout_rate'):
                model.dropout_rate = dropout_rate
            if hasattr(model, 'num_layers'):
                model.num_layers = num_layers

            # Train for specified epochs
            model.epochs = self.epochs

            # Add separation before trial starts
            print(f"\n--- Trial {trial.number + 1} ---")

            try:
                # Train model
                result = model.train(self.X_train, self.y_train, self.X_val, self.y_val)

                # Calculate ALL metrics from the current model state (should be best checkpoint after training)
                from scipy.stats import spearmanr
                from sklearn.metrics import r2_score
                import torch
                import numpy as np

                model.network.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
                    if hasattr(model, 'device'):
                        X_val_tensor = X_val_tensor.to(model.device)

                    val_pred = model.network(X_val_tensor)
                    if isinstance(val_pred, dict):
                        val_pred = val_pred['mean']
                    val_pred = val_pred.detach().cpu().numpy().squeeze()

                # Use MAE from training result (calculated from best checkpoint)
                mae = result.val_mae
                ndcg_score = ndcg_at_k(self.y_val, val_pred, k=20)
                spearman_corr, _ = spearmanr(self.y_val, val_pred)
                r2 = r2_score(self.y_val, val_pred)

                # Debug: Check if model actually has best checkpoint loaded
                logger.info(f"Trial {trial.number + 1}: Using MAE={mae:.3f} from training result")

                # Handle NaN values (treat as 0.0)
                if spearman_corr is None or np.isnan(spearman_corr):
                    spearman_corr = 0.0
                if np.isnan(ndcg_score):
                    ndcg_score = 0.0
                if np.isnan(r2):
                    r2 = 0.0

                # Store metrics in trial attributes for inspection
                trial.set_user_attr("mae", float(mae))
                trial.set_user_attr("ndcg_at_k", float(ndcg_score))
                trial.set_user_attr("r2", float(r2))
                trial.set_user_attr("spearman", float(spearman_corr))

                # Store the best model if this is the best NDCG@k so far AND MAE < guardrail
                if mae < mae_guardrail and (not hasattr(self, 'best_ndcg') or ndcg_score > self.best_ndcg):
                    self.best_ndcg = ndcg_score
                    self.best_model = model
                    # Store best metrics for later use (only for valid trials)
                    self.best_metrics = {
                        'mae': float(mae),
                        'ndcg_at_k': float(ndcg_score),
                        'r2': float(r2),
                        'spearman': float(spearman_corr)
                    }
                    logger.info(f"Trial {trial.number + 1}: New best valid trial stored (MAE={mae:.3f}, NDCG={ndcg_score:.4f})")

                # Apply MAE guardrail - prune if MAE too high
                if mae >= mae_guardrail:
                    logger.info(f"Trial {trial.number + 1}: MAE guardrail failed: {mae:.3f} >= {mae_guardrail}")
                    raise optuna.TrialPruned(f"MAE guardrail failed: {mae:.3f} >= {mae_guardrail}")

                # Add newline to separate from training progress, then log trial progress
                print()  # Ensure newline after training progress
                logger.info(f"Trial {trial.number + 1}: NDCG@20={ndcg_score:.4f}, MAE={mae:.4f}, Spearman={spearman_corr:.4f}, R²={r2:.4f}")

                # Return NDCG@k for maximization
                return float(ndcg_score)

            except optuna.TrialPruned:
                raise  # Re-raise pruned trials
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -1.0  # Return poor correlation for failed trials

        def optimize(self, n_trials: int = 20, timeout: int = 3600) -> Dict[str, Any]:
            """Run hyperparameter optimization.

            Args:
                n_trials: Number of trials to run
                timeout: Maximum time in seconds

            Returns:
                Dictionary with best hyperparameters
            """
            # Create study
            study = optuna.create_study(
                direction='maximize',  # Maximize NDCG@20
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )

            # Initialize best model storage
            self.best_model = None

            # Optimize
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials (optimizing NDCG@20)...")
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False  # Disable progress bar for cleaner logging
            )
            logger.info("Hyperparameter optimization completed.")

            # Store best results
            self.best_params = study.best_params
            self.best_score = study.best_value

            logger.info(f"Best hyperparameters: {self.best_params}")
            logger.info(f"Best NDCG@20: {self.best_score:.4f}")

            return self.best_params

        def get_importance(self, study: optuna.Study) -> Dict[str, float]:
            """Get hyperparameter importance.

            Args:
                study: Completed Optuna study

            Returns:
                Dictionary of parameter importances
            """
            try:
                from optuna.importance import get_param_importances
                importance = get_param_importances(study)
                return importance
            except ImportError:
                logger.warning("Optuna importance module not available")
                return {}

except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not available. Joint hyperparameter optimization disabled.")


class HyperparameterValidator:
    """Validation and A/B testing for hyperparameter optimization."""

    def __init__(self, model_class, model_config: ModelConfig):
        """Initialize hyperparameter validator.

        Args:
            model_class: Model class to validate
            model_config: Model configuration
        """
        self.model_class = model_class
        self.model_config = model_config
        self.results = {}

    def ab_test(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                baseline_params: Dict[str, Any],
                optimized_params: Dict[str, Any],
                num_runs: int = 3) -> Dict[str, Any]:
        """Perform A/B testing between baseline and optimized hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            baseline_params: Baseline hyperparameters
            optimized_params: Optimized hyperparameters
            num_runs: Number of runs for each configuration

        Returns:
            Dictionary with comparison results
        """
        baseline_results = []
        optimized_results = []

        logger.info(f"Starting A/B test for {self.model_config.position} model")
        logger.info(f"Baseline params: {baseline_params}")
        logger.info(f"Optimized params: {optimized_params}")

        # Run baseline configuration
        for run in range(num_runs):
            logger.info(f"Running baseline configuration (run {run + 1}/{num_runs})")

            model = self.model_class(self.model_config)

            # Apply baseline parameters
            for param, value in baseline_params.items():
                if hasattr(model, param):
                    setattr(model, param, value)

            # Train with reduced epochs for speed
            model.epochs = min(model.epochs, 200)

            start_time = time.time()
            result = model.train(X_train, y_train, X_val, y_val)
            training_time = time.time() - start_time

            baseline_results.append({
                'val_mae': result.val_mae,
                'val_r2': result.val_r2,
                'val_rmse': result.val_rmse,
                'training_time': training_time
            })

        # Run optimized configuration
        for run in range(num_runs):
            logger.info(f"Running optimized configuration (run {run + 1}/{num_runs})")

            model = self.model_class(self.model_config)

            # Apply optimized parameters
            for param, value in optimized_params.items():
                if hasattr(model, param):
                    setattr(model, param, value)

            # Train with reduced epochs for speed
            model.epochs = min(model.epochs, 200)

            start_time = time.time()
            result = model.train(X_train, y_train, X_val, y_val)
            training_time = time.time() - start_time

            optimized_results.append({
                'val_mae': result.val_mae,
                'val_r2': result.val_r2,
                'val_rmse': result.val_rmse,
                'training_time': training_time
            })

        # Calculate statistics
        baseline_stats = self._calculate_stats(baseline_results)
        optimized_stats = self._calculate_stats(optimized_results)

        # Calculate improvements
        improvements = {
            'r2_improvement': ((optimized_stats['mean_r2'] - baseline_stats['mean_r2']) /
                              max(abs(baseline_stats['mean_r2']), 0.001)) * 100,
            'mae_improvement': ((baseline_stats['mean_mae'] - optimized_stats['mean_mae']) /
                               max(baseline_stats['mean_mae'], 0.001)) * 100,
            'rmse_improvement': ((baseline_stats['mean_rmse'] - optimized_stats['mean_rmse']) /
                                max(baseline_stats['mean_rmse'], 0.001)) * 100,
            'training_time_change': ((optimized_stats['mean_time'] - baseline_stats['mean_time']) /
                                    max(baseline_stats['mean_time'], 0.001)) * 100
        }

        # Determine if optimization is beneficial
        decision = self._make_decision(improvements)

        results = {
            'baseline': baseline_stats,
            'optimized': optimized_stats,
            'improvements': improvements,
            'decision': decision,
            'num_runs': num_runs
        }

        self._print_results(results)

        return results

    def _calculate_stats(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate statistics from multiple runs.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary with mean and std statistics
        """
        import numpy as np

        mae_values = [r['val_mae'] for r in results]
        r2_values = [r['val_r2'] for r in results]
        rmse_values = [r['val_rmse'] for r in results]
        time_values = [r['training_time'] for r in results]

        return {
            'mean_mae': np.mean(mae_values),
            'std_mae': np.std(mae_values),
            'mean_r2': np.mean(r2_values),
            'std_r2': np.std(r2_values),
            'mean_rmse': np.mean(rmse_values),
            'std_rmse': np.std(rmse_values),
            'mean_time': np.mean(time_values),
            'std_time': np.std(time_values)
        }

    def _make_decision(self, improvements: Dict[str, float]) -> str:
        """Make decision based on improvements.

        Args:
            improvements: Dictionary with improvement percentages

        Returns:
            Decision string (ADOPT, REJECT, or MARGINAL)
        """
        # Decision criteria (as per guide: >10% improvement)
        if improvements['r2_improvement'] > 10:
            return "ADOPT - Significant R² improvement"
        elif improvements['r2_improvement'] > 5 and improvements['mae_improvement'] > 5:
            return "ADOPT - Good overall improvement"
        elif improvements['r2_improvement'] > 0 and improvements['training_time_change'] < 50:
            return "MARGINAL - Small improvement, reasonable training time"
        else:
            return "REJECT - No significant improvement"

    def _print_results(self, results: Dict[str, Any]):
        """Print formatted A/B test results.

        Args:
            results: Results dictionary
        """
        print("\n" + "="*60)
        print(f"A/B Test Results for {self.model_config.position} Model")
        print("="*60)

        print("\nBaseline Performance:")
        baseline = results['baseline']
        print(f"  R²: {baseline['mean_r2']:.4f} ± {baseline['std_r2']:.4f}")
        print(f"  MAE: {baseline['mean_mae']:.3f} ± {baseline['std_mae']:.3f}")
        print(f"  RMSE: {baseline['mean_rmse']:.3f} ± {baseline['std_rmse']:.3f}")
        print(f"  Training Time: {baseline['mean_time']:.1f}s ± {baseline['std_time']:.1f}s")

        print("\nOptimized Performance:")
        optimized = results['optimized']
        print(f"  R²: {optimized['mean_r2']:.4f} ± {optimized['std_r2']:.4f}")
        print(f"  MAE: {optimized['mean_mae']:.3f} ± {optimized['std_mae']:.3f}")
        print(f"  RMSE: {optimized['mean_rmse']:.3f} ± {optimized['std_rmse']:.3f}")
        print(f"  Training Time: {optimized['mean_time']:.1f}s ± {optimized['std_time']:.1f}s")

        print("\nImprovements:")
        improvements = results['improvements']
        print(f"  R² Improvement: {improvements['r2_improvement']:+.1f}%")
        print(f"  MAE Improvement: {improvements['mae_improvement']:+.1f}%")
        print(f"  RMSE Improvement: {improvements['rmse_improvement']:+.1f}%")
        print(f"  Training Time Change: {improvements['training_time_change']:+.1f}%")

        print(f"\nDecision: {results['decision']}")
        print("="*60)

    def cross_validate_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                      hyperparameters: Dict[str, Any],
                                      cv_folds: int = 5) -> Dict[str, Any]:
        """Cross-validate hyperparameters to ensure robustness.

        Args:
            X: Features
            y: Targets
            hyperparameters: Hyperparameters to validate
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []

        logger.info(f"Starting {cv_folds}-fold cross-validation for {self.model_config.position}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            logger.info(f"Training fold {fold}/{cv_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and configure model
            model = self.model_class(self.model_config)

            # Apply hyperparameters
            for param, value in hyperparameters.items():
                if hasattr(model, param):
                    setattr(model, param, value)

            # Train with reduced epochs for speed
            model.epochs = min(model.epochs, 200)

            # Train model
            result = model.train(X_train, y_train, X_val, y_val)

            cv_results.append({
                'fold': fold,
                'val_mae': result.val_mae,
                'val_r2': result.val_r2,
                'val_rmse': result.val_rmse
            })

        # Calculate aggregate statistics
        mae_values = [r['val_mae'] for r in cv_results]
        r2_values = [r['val_r2'] for r in cv_results]
        rmse_values = [r['val_rmse'] for r in cv_results]

        summary = {
            'mean_mae': np.mean(mae_values),
            'std_mae': np.std(mae_values),
            'mean_r2': np.mean(r2_values),
            'std_r2': np.std(r2_values),
            'mean_rmse': np.mean(rmse_values),
            'std_rmse': np.std(rmse_values),
            'fold_results': cv_results
        }

        logger.info(f"Cross-validation complete: R²={summary['mean_r2']:.4f}±{summary['std_r2']:.4f}")

        return summary


class CorrelationFeatureExtractor:
    """Extract correlation features that capture player interactions."""

    def __init__(self, db_path: str = "data/database/nfl_dfs.db"):
        """Initialize with database path."""
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def extract_qb_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract QB features including teammate and opponent correlations."""
        features = {}

        with self._get_connection() as conn:
            # Get game info
            game_query = """
                SELECT game_date, season, week, home_team_id, away_team_id
                FROM games WHERE id = ?
            """
            game_result = conn.execute(game_query, (game_id,)).fetchone()
            if not game_result:
                return features

            game_date, season, week, home_team_id, away_team_id = game_result

            # Get player team
            team_query = "SELECT team_id FROM players WHERE id = ?"
            team_result = conn.execute(team_query, (player_id,)).fetchone()
            if not team_result:
                return features
            team_id = team_result[0]

            # Get opponent team
            opp_team_id = away_team_id if team_id == home_team_id else home_team_id

            # Parse game_date
            if isinstance(game_date, str):
                try:
                    game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Could not parse game_date: {game_date}")
                    return {}

            start_date = game_date - timedelta(weeks=lookback_weeks)

            # Teammate correlations
            teammate_query = """
                SELECT
                    AVG(ps.rushing_yards) as team_rush_ypg,
                    AVG(ps.receiving_yards) as team_rec_ypg,
                    COUNT(DISTINCT CASE WHEN ps.receiving_yards > 75 THEN ps.player_id END) as num_viable_receivers,
                    MAX(ps.targets) as max_target_share,
                    AVG(CASE WHEN p.position = 'RB' THEN ps.targets ELSE 0 END) as rb_target_rate,
                    AVG(CASE WHEN p.position = 'TE' THEN ps.targets ELSE 0 END) as te_target_rate
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.id
                JOIN games g ON ps.game_id = g.id
                WHERE p.team_id = ? AND g.game_date >= ? AND g.game_date < ? AND g.game_finished = 1
            """
            teammate_result = conn.execute(teammate_query, (team_id, start_date, game_date)).fetchone()

            if teammate_result:
                features.update({
                    'team_rush_ypg': teammate_result[0] or 0,
                    'team_rec_ypg': teammate_result[1] or 0,
                    'num_viable_receivers': teammate_result[2] or 0,
                    'max_target_concentration': teammate_result[3] or 0,
                    'rb_involvement_passing': teammate_result[4] or 0,
                    'te_involvement': teammate_result[5] or 0,
                })

            # Stacking features - QB's favorite targets
            stacking_query = """
                SELECT
                    p.position,
                    AVG(ps.targets) as avg_targets,
                    AVG(CAST(ps.targets AS FLOAT) / NULLIF((
                        SELECT SUM(ps_team.targets)
                        FROM player_stats ps_team
                        JOIN players p_team ON ps_team.player_id = p_team.id
                        WHERE p_team.team_id = ? AND ps_team.game_id = ps.game_id
                    ), 0)) as avg_target_share,
                    COUNT(*) as games_together
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.id
                JOIN games g ON ps.game_id = g.id
                WHERE p.team_id = ? AND p.position IN ('WR', 'TE')
                AND g.game_date >= ? AND g.game_date < ?
                AND EXISTS (
                    SELECT 1 FROM player_stats qb_ps
                    WHERE qb_ps.game_id = ps.game_id AND qb_ps.player_id = ?
                )
                GROUP BY p.id, p.position
                ORDER BY avg_targets DESC
                LIMIT 3
            """
            stacking_results = conn.execute(stacking_query, (team_id, team_id, start_date, game_date, player_id)).fetchall()

            for idx, result in enumerate(stacking_results):
                prefix = f"top_target_{idx+1}"
                features[f"{prefix}_share"] = result[2] or 0
                features[f"{prefix}_consistency"] = min(result[3] / lookback_weeks, 1.0)

        return features

    def extract_rb_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract RB-specific correlation features."""
        features = {}

        with self._get_connection() as conn:
            # Get game info
            game_query = """
                SELECT game_date, season, week, home_team_id, away_team_id
                FROM games WHERE id = ?
            """
            game_result = conn.execute(game_query, (game_id,)).fetchone()
            if not game_result:
                return features

            game_date = game_result[0]
            if isinstance(game_date, str):
                try:
                    game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
                except ValueError:
                    return {}

            start_date = game_date - timedelta(weeks=lookback_weeks)

            # RB workload features
            workload_query = """
                SELECT
                    AVG(ps.rushing_attempts) as avg_carries,
                    AVG(ps.targets) as avg_targets,
                    AVG(ps.rushing_attempts + ps.targets) as avg_touches,
                    AVG(CASE WHEN ps.rushing_attempts > 15 THEN 1 ELSE 0 END) as workhorse_rate,
                    SUM(ps.rushing_tds) as recent_rush_tds
                FROM player_stats ps
                JOIN games g ON ps.game_id = g.id
                WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?
            """
            workload_result = conn.execute(workload_query, (player_id, start_date, game_date)).fetchone()

            if workload_result:
                features.update({
                    'rb_avg_touches': workload_result[2] or 0,
                    'rb_workhorse_rate': workload_result[3] or 0,
                    'rb_pass_involvement': workload_result[1] or 0,
                    'rb_td_regression': workload_result[4] or 0,
                })

        return features

    def extract_wr_te_correlation_features(
        self,
        player_id: int,
        game_id: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract WR/TE-specific correlation features including defensive matchups."""
        features = {}

        with self._get_connection() as conn:
            # Get game and team info
            game_query = """
                SELECT g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                       ht.team_abbr as home_abbr, at.team_abbr as away_abbr
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE g.id = ?
            """
            game_result = conn.execute(game_query, (game_id,)).fetchone()
            if not game_result:
                return features

            game_date, season, week, home_team_id, away_team_id, home_abbr, away_abbr = game_result

            # Get player team
            team_query = "SELECT team_id FROM players WHERE id = ?"
            team_result = conn.execute(team_query, (player_id,)).fetchone()
            if not team_result:
                return features
            team_id = team_result[0]

            # Determine opponent
            is_home = team_id == home_team_id
            opponent_abbr = away_abbr if is_home else home_abbr
            team_abbr = home_abbr if is_home else away_abbr

            if isinstance(game_date, str):
                try:
                    game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
                except ValueError:
                    return {}

            start_date = game_date - timedelta(weeks=lookback_weeks)

            # Target competition analysis
            target_comp_query = """
                SELECT
                    COUNT(DISTINCT ps.player_id) as num_receivers,
                    AVG(ps.targets) as avg_team_targets,
                    MAX(ps.targets) as max_individual_targets,
                    AVG(CASE WHEN p.position = 'WR' THEN ps.targets ELSE 0 END) as wr_targets,
                    AVG(CASE WHEN p.position = 'TE' THEN ps.targets ELSE 0 END) as te_targets
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.id
                JOIN games g ON ps.game_id = g.id
                WHERE p.team_id = ? AND p.position IN ('WR', 'TE')
                AND g.game_date >= ? AND g.game_date < ?
                AND ps.targets > 0
            """
            target_comp_result = conn.execute(target_comp_query, (team_id, start_date, game_date)).fetchone()

            if target_comp_result:
                features.update({
                    'receiver_competition': target_comp_result[0] or 0,
                    'target_concentration': (target_comp_result[2] or 0) / max(target_comp_result[1] or 1, 1),
                    'wr_target_dominance': (target_comp_result[3] or 0) / max(target_comp_result[1] or 1, 1),
                    'te_target_share': (target_comp_result[4] or 0) / max(target_comp_result[1] or 1, 1),
                })

            # Defense vs receiver type using PbP data
            def_vs_receivers = conn.execute(
                """SELECT
                    AVG(CASE WHEN complete_pass = 1 THEN yards_gained ELSE 0 END) as avg_completion_yards,
                    AVG(CASE WHEN complete_pass = 1 THEN 1.0 ELSE 0.0 END) as completion_rate,
                    AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as td_rate,
                    COUNT(*) as targets_allowed
                   FROM play_by_play
                   WHERE defteam = ? AND season = ? AND week < ?
                   AND pass_attempt = 1""",
                (opponent_abbr, season, week)
            ).fetchone()

            if def_vs_receivers:
                features.update({
                    'def_completion_yards_allowed': def_vs_receivers[0] or 0,
                    'def_completion_rate_allowed': def_vs_receivers[1] or 0,
                    'def_pass_td_rate_allowed': def_vs_receivers[2] or 0,
                })

        return features

    def extract_def_correlation_features(
        self,
        team_abbr: str,
        opponent_abbr: str,
        season: int,
        week: int,
        lookback_weeks: int = 4
    ) -> Dict[str, float]:
        """Extract defense-specific correlation features from PbP data."""
        features = {}

        with self._get_connection() as conn:
            # Opponent offensive tendencies
            opp_tendencies = conn.execute(
                """SELECT
                    AVG(CASE WHEN pass_attempt = 1 THEN 1.0 ELSE 0.0 END) as pass_rate,
                    AVG(CASE WHEN yardline_100 <= 20 THEN 1.0 ELSE 0.0 END) as rz_play_rate,
                    AVG(CASE WHEN down = 3 THEN 1.0 ELSE 0.0 END) as third_down_rate,
                    AVG(yards_gained) as avg_play_yards
                   FROM play_by_play
                   WHERE posteam = ? AND season = ? AND week < ?
                   AND (pass_attempt = 1 OR rush_attempt = 1)""",
                (opponent_abbr, season, week)
            ).fetchone()

            if opp_tendencies:
                features.update({
                    'opp_pass_tendency': opp_tendencies[0] or 0.5,
                    'opp_rz_frequency': opp_tendencies[1] or 0,
                    'opp_third_down_freq': opp_tendencies[2] or 0,
                    'opp_avg_play_efficiency': opp_tendencies[3] or 0,
                })

            # Defense game script correlation
            game_script = conn.execute(
                """SELECT
                    AVG(CASE WHEN quarter_seconds_remaining < 900 THEN 1.0 ELSE 0.0 END) as late_game_rate,
                    AVG(CASE WHEN down >= 3 THEN 1.0 ELSE 0.0 END) as pressure_down_rate
                   FROM play_by_play
                   WHERE defteam = ? AND season = ? AND week < ?""",
                (team_abbr, season, week)
            ).fetchone()

            if game_script:
                features.update({
                    'def_late_game_exposure': game_script[0] or 0,
                    'def_pressure_situations': game_script[1] or 0,
                })

        return features

    def extract_correlation_features(
        self,
        player_id: int,
        game_id: int,
        position: str
    ) -> Dict[str, float]:
        """Extract correlation features for any position."""
        if position == 'QB':
            return self.extract_qb_correlation_features(player_id, game_id)
        elif position == 'RB':
            return self.extract_rb_correlation_features(player_id, game_id)
        elif position in ['WR', 'TE']:
            return self.extract_wr_te_correlation_features(player_id, game_id)
        else:
            # For DEF, return basic features
            return {}


class BaseNeuralModel(ABC):
    """Base class for PyTorch neural network models."""

    def __init__(self, config: ModelConfig):
        """Initialize neural network base model."""
        self.config = config
        self.device = OPTIMAL_DEVICE  # Use Apple Silicon optimized device
        self.network: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.criterion = nn.HuberLoss(delta=1.0)  # Even more robust to outliers
        # quantile_criterion will be set as a lambda after _quantile_loss is defined
        self.quantile_criterion = None  # Set properly after methods are defined
        self.mse_criterion = nn.MSELoss()  # For fallback if Huber fails
        self.dfs_criterion = DFSLoss()  # DFS-optimized loss function

        # Load hyperparameters from YAML config or use defaults
        if HAS_HYPERPARAMETER_MANAGER:
            hyperparams = get_hyperparameter_manager().get_hyperparameters(config.position)
            # Training parameters from config
            self.batch_size = hyperparams.get('batch_size', 32)
            self.learning_rate = hyperparams.get('learning_rate', 0.001)
            self.epochs = hyperparams.get('epochs', 500)
            self.weight_decay = hyperparams.get('weight_decay', 0.001)
            self.patience = hyperparams.get('patience', 50)

            # Architecture parameters from config
            self.hidden_size = hyperparams.get('hidden_size', 256)
            self.num_layers = hyperparams.get('num_layers', 3)
            self.dropout_rate = hyperparams.get('dropout_rate', 0.3)
        else:
            # Fallback to hardcoded defaults
            self.batch_size = 32
            self.learning_rate = 0.001
            self.epochs = 500
            self.weight_decay = 0.001
            self.patience = 50
            self.hidden_size = 256
            self.num_layers = 3
            self.dropout_rate = 0.3

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.is_trained = False
        self.training_history = []

        # Target normalization parameters
        self.y_mean = 0.0
        self.y_std = 1.0

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
            raise ValueError(f"Target standard deviation is zero for {self.config.position}")

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

    # BaseNeuralModel methods continue below

    def find_optimal_lr(self, X_train: np.ndarray, y_train: np.ndarray,
                       start_lr: float = 1e-8, end_lr: float = 1.0,
                       num_iter: int = 100) -> float:
        """Find optimal learning rate using LR range test.

        Args:
            X_train: Training features
            y_train: Training targets
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations

        Returns:
            Optimal learning rate
        """
        # Prepare data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Build network if not already built
        if self.network is None:
            input_size = X_train.shape[1]
            self.network = self.build_network(input_size)
            self.network.to(self.device)

        # Create temporary optimizer for LR finding
        temp_optimizer = optim.AdamW(self.network.parameters(), lr=start_lr)

        # Create data loader
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Run LR finder
        lr_finder = LRFinder(self.network, temp_optimizer, self.criterion, self.device)
        optimal_lr = lr_finder.range_test(loader, start_lr, end_lr, num_iter)

        # Save plot if matplotlib available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            lr_finder.plot_results(save_path=f"lr_finder_{self.config.position}.png")
        except ImportError:
            pass

        logger.info(f"Found optimal learning rate for {self.config.position}: {optimal_lr:.2e}")
        return optimal_lr

    def optimize_batch_size(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_sizes: List[int] = None) -> int:
        """Find optimal batch size considering memory and performance.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_sizes: List of batch sizes to test

        Returns:
            Optimal batch size
        """
        # Build network if not already built
        if self.network is None:
            input_size = X_train.shape[1]
            self.network = self.build_network(input_size)
            self.network.to(self.device)

        # Create batch size optimizer
        batch_optimizer = BatchSizeOptimizer(self.network, self.device)

        # Create data loaders for testing
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create initial loaders (will be recreated with different batch sizes)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Find optimal batch size
        optimal_batch_size = batch_optimizer.optimize_batch_size(
            train_loader, val_loader, self, batch_sizes, epochs_per_test=50
        )

        logger.info(f"Found optimal batch size for {self.config.position}: {optimal_batch_size}")
        return optimal_batch_size

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 20, timeout: int = 3600,
                            epochs: int = 100) -> Dict[str, Any]:
        """Perform joint hyperparameter optimization using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            epochs: Number of epochs to train each trial

        Returns:
            Dictionary with best hyperparameters
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, using default hyperparameters")
            return {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }

        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            self.__class__, self.config,
            X_train, y_train, X_val, y_val,
            epochs=epochs
        )

        # Run optimization
        best_params = tuner.optimize(n_trials, timeout)

        # Store best MAE for hyperparameter manager
        self._last_mae = tuner.best_score

        # Use the best model from tuning if available
        if hasattr(tuner, 'best_model') and tuner.best_model is not None:
            # Replace current network with the best model's network
            self.network = tuner.best_model.network
            self.device = tuner.best_model.device

            # Calculate individual metrics using the best model
            self.network.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                val_pred_tensor = self.network(X_val_tensor)

                # Handle different output formats
                if isinstance(val_pred_tensor, dict):
                    val_pred_tensor = val_pred_tensor['mean']
                elif val_pred_tensor.dim() > 1 and val_pred_tensor.size(1) == 1:
                    val_pred_tensor = val_pred_tensor.squeeze(1)

                val_pred = val_pred_tensor.cpu().numpy()

            # Calculate all metrics for storage
            from sklearn.metrics import mean_absolute_error
            from scipy.stats import spearmanr
            import numpy as np

            self._last_validation_mae = mean_absolute_error(y_val, val_pred)
            spearman_corr, _ = spearmanr(y_val, val_pred)
            self._last_validation_spearman = spearman_corr if not np.isnan(spearman_corr) else 0

            # Also calculate R² for compatibility
            ss_res = np.sum((y_val - val_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            self._last_validation_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            # Fallback: use the stored best_score as the metrics
            self._last_validation_mae = tuner.best_score
            self._last_validation_spearman = 0.0
            self._last_validation_r2 = 0.0

        # Apply best parameters
        self.learning_rate = best_params.get('learning_rate', self.learning_rate)
        self.batch_size = best_params.get('batch_size', self.batch_size)

        # Apply architecture params if present
        if 'hidden_size' in best_params and hasattr(self, 'hidden_size'):
            self.hidden_size = best_params['hidden_size']
        if 'dropout_rate' in best_params and hasattr(self, 'dropout_rate'):
            self.dropout_rate = best_params['dropout_rate']
        if 'num_layers' in best_params and hasattr(self, 'num_layers'):
            self.num_layers = best_params['num_layers']

        logger.info(f"Applied best hyperparameters for {self.config.position}: {best_params}")

        # Save the optimized hyperparameters to YAML config
        # Use metrics from the best trial stored by the tuner
        best_metrics = getattr(tuner, 'best_metrics', {})
        logger.info(f"Best trial metrics: {best_metrics}")
        validation_mae = best_metrics.get('mae', getattr(self, '_last_validation_mae', None))
        validation_r2 = best_metrics.get('r2', getattr(self, '_last_validation_r2', None))
        validation_spearman = best_metrics.get('spearman', getattr(self, '_last_validation_spearman', None))
        validation_ndcg = best_metrics.get('ndcg_at_k', getattr(self, '_last_validation_ndcg', None))
        logger.info(f"Extracted metrics - MAE: {validation_mae}, R²: {validation_r2}, Spearman: {validation_spearman}, NDCG: {validation_ndcg}")

        # Save to hyperparameter manager if available
        if HAS_HYPERPARAMETER_MANAGER:
            hyperparameter_manager = get_hyperparameter_manager()
            hyperparameter_manager.update_hyperparameters(
                position=self.config.position,
                new_params=best_params,
                validation_r2=validation_r2,
                validation_mae=validation_mae,
                validation_spearman=validation_spearman,
                validation_ndcg=validation_ndcg,
                trials=n_trials
            )

        return best_params

    def _train_epochs(self, train_loader: DataLoader, val_loader: DataLoader,
                      epochs: int) -> Dict[str, float]:
        """Train for a specific number of epochs (used by batch size optimizer).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train

        Returns:
            Dictionary with final metrics
        """
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return {
            'train_loss': train_loss,
            'val_loss': best_val_loss
        }

    def _normalize_features(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Apply feature normalization using stored parameters."""
        if is_training:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            # Use a small epsilon to avoid division by zero for constant features
            self.X_std = np.where(self.X_std == 0, 1e-8, self.X_std)

        X_normalized = (X - self.X_mean) / self.X_std
        return X_normalized

    def _normalize_targets(self, y: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Apply target normalization using stored parameters."""
        if is_training:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            if self.y_std == 0:
                self.y_std = 1.0  # Avoid division by zero

        y_normalized = (y - self.y_mean) / self.y_std
        return y_normalized

    def _denormalize_predictions(self, pred: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return pred * self.y_std + self.y_mean

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor, quantile: float) -> torch.Tensor:
        """Quantile loss (pinball loss) function."""
        errors = targets - predictions
        return torch.mean(torch.max(quantile * errors, (quantile - 1) * errors))

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate input data."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty input data")
        if len(X) != len(y):
            raise ValueError("Feature and target lengths don't match")

    def _create_data_loaders(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders optimized for Apple Silicon."""
        # Create tensors on CPU first, then move to device for better memory management
        train_X = torch.tensor(X_train, dtype=torch.float32).to(self.device, non_blocking=True)
        train_y = torch.tensor(y_train, dtype=torch.float32).to(self.device, non_blocking=True)
        val_X = torch.tensor(X_val, dtype=torch.float32).to(self.device, non_blocking=True)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self.device, non_blocking=True)

        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)

        # Optimize DataLoader settings for Apple Silicon
        num_workers = 0 if self.device.type == "mps" else min(4, os.cpu_count() // 2)
        pin_memory = False  # Disabled since tensors are already moved to device

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False  # Better for MPS
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )

        return train_loader, val_loader

    def _train_epoch(self, train_loader: DataLoader, salaries: torch.Tensor = None) -> float:
        """Train model for one epoch with enhanced loss function for QB models."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:  # With salaries
                batch_X, batch_y, batch_salaries = batch_data
            else:  # Without salaries
                batch_X, batch_y = batch_data
                batch_salaries = None

            self.optimizer.zero_grad()
            predictions = self.network(batch_X)

            # Use DFS loss for QB models, regular loss for others
            if isinstance(predictions, dict) and self.config.position == 'QB':
                loss = self.dfs_criterion(predictions, batch_y, batch_salaries)
            elif isinstance(predictions, dict):
                predictions = predictions['mean']  # Use mean prediction for non-QB models
                loss = self.criterion(predictions, batch_y)
            elif predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, batch_y)
            else:
                loss = self.criterion(predictions, batch_y)

            # Check if loss is valid before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # Moderate gradient clipping to prevent exploding gradients while allowing learning
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Check for NaN gradients after clipping
            has_nan_grad = False
            for param in self.network.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                # Clear gradients and skip update
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate model for one epoch."""
        self.network.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = self.network(batch_X)

                # Handle QB model's dictionary output
                if isinstance(predictions, dict):
                    predictions = predictions['mean']  # Use mean prediction for validation
                elif predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)

                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _compute_validation_r2(self, val_loader: DataLoader, y_val: np.ndarray) -> float:
        """Compute R² score on validation data."""
        self.network.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in val_loader:
                pred = self.network(batch_X)

                # Handle QB model's dictionary output
                if isinstance(pred, dict):
                    pred = pred['mean']
                elif pred.dim() > 1 and pred.size(1) == 1:
                    pred = pred.squeeze(1)

                predictions.extend(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Handle NaN predictions
        if np.isnan(predictions).any():
            return -float("inf")

        # Compute R²
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        ss_res = np.sum((y_val - predictions) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _compute_validation_mae(self, val_loader: DataLoader, y_val: np.ndarray) -> float:
        """Compute MAE score on validation data."""
        self.network.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in val_loader:
                pred = self.network(batch_X)

                # Handle QB model's dictionary output
                if isinstance(pred, dict):
                    pred = pred['mean']
                elif pred.dim() > 1 and pred.size(1) == 1:
                    pred = pred.squeeze(1)

                predictions.extend(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Handle NaN predictions
        if np.isnan(predictions).any():
            return float("inf")  # High MAE for failed predictions

        # Compute MAE
        mae = np.mean(np.abs(y_val - predictions))
        return mae

    def _compute_validation_ndcg(self, val_loader: DataLoader, y_val: np.ndarray, k: int = 20) -> float:
        """Compute NDCG@k score on validation data."""
        self.network.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in val_loader:
                pred = self.network(batch_X)

                # Handle QB model's dictionary output
                if isinstance(pred, dict):
                    pred = pred['mean']
                elif pred.dim() > 1 and pred.size(1) == 1:
                    pred = pred.squeeze(1)

                predictions.extend(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Handle NaN predictions
        if np.isnan(predictions).any():
            return 0.0  # Low NDCG for failed predictions

        # Compute NDCG@k
        ndcg_score = ndcg_at_k(y_val, predictions, k=k)
        return float(ndcg_score)

    def validate_predictions(self, predictions: np.ndarray, position: str = None) -> bool:
        """Ensure predictions are realistic per optimization guide validation."""

        position = position or self.config.position

        if position == 'QB':
            # QB-specific validation as per optimization guide
            assert predictions.min() >= 5, f"QB floor too low: {predictions.min()}"
            assert predictions.max() <= 45, f"QB ceiling too high: {predictions.max()}"
            assert predictions.std() > 3, f"Insufficient variance: {predictions.std()}"

            # Check for reasonable distribution
            median_pred = np.median(predictions)
            assert 15 <= median_pred <= 25, f"Unusual median prediction: {median_pred}"

            # Check for no extreme outliers (more than 3 std devs from mean)
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            outliers = np.abs(predictions - mean_pred) > 3 * std_pred
            outlier_pct = np.mean(outliers)
            assert outlier_pct < 0.05, f"Too many outliers: {outlier_pct:.2%}"

        logger.info(f"{position} predictions validated successfully: "
                   f"range=[{predictions.min():.1f}, {predictions.max():.1f}], "
                   f"mean={predictions.mean():.1f}, std={predictions.std():.1f}")

        return True

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the neural network model."""
        start_time = time.time()

        # Set up quantile_criterion if not already set
        if self.quantile_criterion is None:
            self.quantile_criterion = self._quantile_loss

        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Additional data cleaning before training
        logger.info("Performing final data cleaning before model training...")

        # Initial cleanup
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for extreme values and clip them
        X_train = np.clip(X_train, -1000, 1000)
        y_train = np.clip(y_train, -10, 50)  # DST fantasy points range
        X_val = np.clip(X_val, -1000, 1000)
        y_val = np.clip(y_val, -10, 50)

        # Robust feature scaling with zero-variance handling
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)

        # Handle zero-variance features (set std to 1, mean to 0 for these features)
        zero_var_mask = X_std < 1e-10
        X_std[zero_var_mask] = 1.0
        X_mean[zero_var_mask] = 0.0

        # Apply scaling
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        # Final cleanup after scaling to catch any numerical issues
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

        if self.network is None:
            input_size = X_train.shape[1]
            self.input_size = input_size  # Store input size for later use
            self.network = self.build_network(input_size)
            self.network.to(self.device)

        # Use AdamW with improved weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=self.learning_rate, weight_decay=0.001,
            betas=(0.9, 0.999), eps=1e-8
        )

        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)

        # More stable learning rate schedule
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=15,
            min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_val_ndcg = 0.0  # Track best NDCG@K score (higher is better)
        best_epoch = 0

        logger.info(f"Starting neural network training for {self.config.position}")

        # Training loop with best model checkpointing (no early stopping)
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            self.scheduler.step(val_loss)  # ReduceLROnPlateau takes loss as input

            # Compute validation NDCG@K every epoch for precise checkpointing
            val_ndcg = self._compute_validation_ndcg(val_loader, y_val, k=20)
            # Also compute R² and MAE for logging/tracking
            val_r2 = self._compute_validation_r2(val_loader, y_val)
            val_mae = self._compute_validation_mae(val_loader, y_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Snapshot best model based on NDCG@K score (higher is better) AND MAE < 6.0
            improved = False
            if val_ndcg > best_val_ndcg and val_mae < 6.0:
                best_val_ndcg = val_ndcg
                best_val_loss = val_loss
                best_epoch = epoch
                import copy
                self.best_state_dict = copy.deepcopy(self.network.state_dict())
                improved = True
            elif val_ndcg == best_val_ndcg and val_loss < best_val_loss and val_mae < 6.0:
                # Same NDCG@K, but better loss - still an improvement if MAE is good
                best_val_loss = val_loss
                best_epoch = epoch
                import copy
                self.best_state_dict = copy.deepcopy(self.network.state_dict())
                improved = True

            # Progress reporting - show improvements and periodic updates
            if improved:
                print(f"\r\033[K🎯 Epoch {epoch}: New best NDCG@20 = {val_ndcg:.4f} (MAE={val_mae:.3f})", end="", flush=True)
            elif epoch % 50 == 0:
                print(f"\r\033[KEpoch {epoch}/{self.epochs}: NDCG@20 = {val_ndcg:.4f} (Best: {best_val_ndcg:.4f} @ epoch {best_epoch})", end="", flush=True)

        # Training completed - always run full epochs and use best checkpoint
        print(f"\nTraining completed ({self.epochs} epochs, Best NDCG@20: {best_val_ndcg:.4f} at epoch {best_epoch})")

        if hasattr(self, "best_state_dict") and self.best_state_dict is not None:
            logger.info(f"Loading best checkpoint from epoch {best_epoch}")
            self.network.load_state_dict(self.best_state_dict)
        else:
            logger.warning(f"No best checkpoint available - using final epoch state (this will likely have high MAE)")

        # Calculate final metrics
        self.network.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)

            train_pred_tensor = self.network(X_train_tensor)
            val_pred_tensor = self.network(X_val_tensor)

            # Handle QB model's dictionary output
            if isinstance(train_pred_tensor, dict):
                train_pred_tensor = train_pred_tensor['mean']
            elif train_pred_tensor.dim() > 1 and train_pred_tensor.size(1) == 1:
                train_pred_tensor = train_pred_tensor.squeeze(1)

            if isinstance(val_pred_tensor, dict):
                val_pred_tensor = val_pred_tensor['mean']
            elif val_pred_tensor.dim() > 1 and val_pred_tensor.size(1) == 1:
                val_pred_tensor = val_pred_tensor.squeeze(1)

            train_pred = train_pred_tensor.cpu().numpy()
            val_pred = val_pred_tensor.cpu().numpy()

        # Check for NaN predictions and handle gracefully
        if np.isnan(train_pred).any() or np.isnan(val_pred).any():
            logger.error("NaN predictions detected after training!")
            train_mae = float('nan')
            val_mae = float('nan')  # Set validation MAE to NaN
            train_rmse = val_rmse = float('nan')
            train_r2 = float('nan')
            # Calculate final R² from the best NDCG@K model
            val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / val_ss_tot if val_ss_tot != 0 else 0
        else:
            train_mae = np.mean(np.abs(y_train - train_pred))
            # Calculate final validation MAE from best NDCG@K model
            val_mae = np.mean(np.abs(y_val - val_pred))
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

            # Calculate training R² with safeguards against division by zero
            train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)

            if train_ss_tot == 0:
                train_r2 = 0.0  # If no variance in target, R² is undefined, set to 0
            else:
                train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / train_ss_tot

            # Calculate final validation R² from the best MAE model
            val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / val_ss_tot if val_ss_tot != 0 else 0

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True
        training_time = time.time() - start_time

        result = TrainingResult(
            model=self.network,
            training_time=training_time,
            best_iteration=best_epoch,
            feature_importance=None,
            train_mae=train_mae,
            val_mae=val_mae,  # Final MAE from the best NDCG@K model
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            train_r2=train_r2,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"Training completed: NDCG@20={best_val_ndcg:.3f}, R²={val_r2:.3f}")

        return result

    def predict(self, X: np.ndarray, validate: bool = True) -> PredictionResult:
        """Generate predictions with uncertainty quantification and validation."""
        if not self.is_trained or self.network is None:
            raise ValueError("Model must be trained before making predictions")

        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.network(X_tensor)

            # Handle enhanced QB model's dictionary output
            if isinstance(predictions, dict):
                point_estimate = predictions['mean'].cpu().numpy()
                floor = predictions.get('floor', point_estimate - 5).cpu().numpy()
                ceiling = predictions.get('ceiling', point_estimate + 5).cpu().numpy()
            elif predictions.dim() > 1 and predictions.size(1) == 1:
                point_estimate = predictions.squeeze(1).cpu().numpy()
                floor = point_estimate - 5
                ceiling = point_estimate + 5
            else:
                point_estimate = predictions.cpu().numpy()
                floor = point_estimate - 5
                ceiling = point_estimate + 5

        # Validate predictions if requested
        if validate:
            try:
                self.validate_predictions(point_estimate, self.config.position)
            except AssertionError as e:
                logger.warning(f"Prediction validation failed: {e}")
                # Continue with predictions despite validation failure

        # Calculate prediction intervals
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else np.std(point_estimate) * 0.5
        )

        lower_bound = point_estimate - 1.96 * uncertainty
        upper_bound = point_estimate + 1.96 * uncertainty
        confidence_score = np.ones_like(point_estimate) * 0.7

        return PredictionResult(
            point_estimate=point_estimate,
            confidence_score=confidence_score,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )

    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.network is None:
            raise ValueError("No model to save")
        # Save both state dict and input size
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'input_size': self.input_size
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path} with input_size={self.input_size}")

    def load_model(self, path: str, input_size: int = None):
        """Load trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle both old (state_dict only) and new (checkpoint with metadata) formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            state_dict = checkpoint['model_state_dict']
            saved_input_size = checkpoint.get('input_size', input_size)
            if input_size is None:
                input_size = saved_input_size
            elif input_size != saved_input_size:
                logger.warning(f"Input size mismatch: provided {input_size}, saved {saved_input_size}. Using saved size.")
                input_size = saved_input_size
        else:
            # Old format - just state dict
            state_dict = checkpoint
            if input_size is None:
                # Try to infer from first layer
                first_layer_key = next(iter(state_dict.keys()))
                if 'weight' in first_layer_key:
                    input_size = state_dict[first_layer_key].shape[1]
                else:
                    raise ValueError("Cannot determine input size from old model format")

        if self.network is None:
            self.network = self.build_network(input_size)
            self.input_size = input_size

        self.network.load_state_dict(state_dict)
        self.network.to(self.device)  # Ensure model is on the correct device
        self.network.eval()
        self.is_trained = True
        logger.info(f"Model loaded from {path} with input_size={input_size} to device: {self.device}")

# End of BaseNeuralModel class

class DFSLoss(nn.Module):
    """Custom DFS loss function with ranking and salary weighting as per optimization guide."""

    def __init__(self, alpha: float = 0.2, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            alpha: Weight for range penalty term
            beta: Weight for ranking loss term
            gamma: Weight for salary-weighted accuracy term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor,
                salaries: torch.Tensor = None) -> torch.Tensor:
        """
        DFS-optimized loss function.

        Args:
            predictions: Dict with 'mean', 'floor', 'ceiling' tensors
            targets: Actual fantasy points
            salaries: Player salaries (optional, uses uniform weights if None)
        """
        mean_pred = predictions['mean']
        floor_pred = predictions.get('floor', mean_pred - 5)
        ceiling_pred = predictions.get('ceiling', mean_pred + 5)

        # Main prediction loss (weighted MSE)
        point_loss = F.mse_loss(mean_pred, targets)

        # Range penalty - penalize unrealistic predictions
        range_penalty = torch.mean(torch.relu(5 - (ceiling_pred - floor_pred)))

        # Ranking loss (Spearman correlation approximation)
        rank_loss = self._ranking_loss(mean_pred, targets)

        # Salary-weighted accuracy (more important to get expensive players right)
        if salaries is not None:
            salary_weights = salaries / salaries.mean()
            weighted_loss = F.mse_loss(mean_pred * salary_weights, targets * salary_weights)
        else:
            weighted_loss = torch.tensor(0.0, device=targets.device)

        # Combined loss as per optimization guide formula
        total_loss = (point_loss +
                     self.alpha * range_penalty +
                     self.beta * rank_loss +
                     self.gamma * weighted_loss)

        return total_loss

    def _ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ranking loss to encourage correct player ordering."""
        # Compute pairwise differences
        n = predictions.size(0)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Use a simplified ranking loss based on order consistency
        pred_ranks = torch.argsort(torch.argsort(predictions))
        target_ranks = torch.argsort(torch.argsort(targets))

        # L2 loss between rank positions (normalized)
        rank_diff = (pred_ranks.float() - target_ranks.float()) / n
        return torch.mean(rank_diff ** 2)


class ResidualBlock(nn.Module):
    """Residual block with improved skip connections."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # Skip connection - project if dimensions don't match
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.final_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_connection(x)
        out = self.main_path(x)
        return self.final_activation(out + identity)


class QBNetwork(nn.Module):
    """Enhanced QB Network following optimization guide recommendations with multi-head design."""

    def __init__(self, input_size: int):
        super().__init__()

        # Feature extraction layers with skip connections (increased capacity)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Multi-head for different aspects as per optimization guide
        self.passing_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

        self.rushing_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        self.bonus_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),  # Probability of hitting bonuses
            nn.Dropout(0.1),
            nn.Linear(32, 8)
        )

        # Combine all heads as per optimization guide
        combined_features = 32 + 16 + 8  # 56 total

        self.output_layer = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # [mean, floor_adjustment, ceiling_adjustment]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        features = self.feature_extractor(x)

        # Multi-head processing
        passing = self.passing_head(features)
        rushing = self.rushing_head(features)
        bonus = self.bonus_head(features)

        # Combine heads
        combined = torch.cat([passing, rushing, bonus], dim=1)
        output = self.output_layer(combined)

        # Don't apply fantasy point scaling - targets are normalized during training
        # The model will learn to output in normalized space
        mean_pred = output[:, 0]
        floor_adj = torch.abs(output[:, 1]) * 0.5  # Smaller adjustments in normalized space
        ceiling_adj = torch.abs(output[:, 2]) * 0.5

        floor = mean_pred - floor_adj
        ceiling = mean_pred + ceiling_adj

        # Don't clamp - let the model learn the appropriate range for normalized targets

        # Ensure floor <= mean <= ceiling relationship
        floor = torch.min(floor, mean_pred)
        ceiling = torch.max(ceiling, mean_pred)

        return {
            'mean': mean_pred,
            'floor': floor,
            'ceiling': ceiling,
            'q25': floor,
            'q50': mean_pred,
            'q75': ceiling
        }


class RBNetwork(nn.Module):
    """Neural network architecture for running back predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # RB-specific architecture with proper depth
        self.input_norm = nn.LayerNorm(input_size)

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.15),
        )

        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Softmax(dim=1)
        )

        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )

        self.std_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive std
        )

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        features = self.feature_extractor(x)

        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights

        mean = self.mean_head(features).squeeze(-1)
        std = self.std_head(features).squeeze(-1)

        # Don't clamp here - targets are normalized during training
        # Clamping to fantasy point ranges [3,35] when using normalized targets breaks training!

        # Return mean for compatibility with existing code
        return mean


class WRNetwork(nn.Module):
    """Enhanced neural network architecture for wide receiver predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # WR-specific architecture with target share focus
        self.input_norm = nn.LayerNorm(input_size)

        # Enhanced feature extraction for WR complexity
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 320),  # Larger for WR feature complexity
            nn.LayerNorm(320),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),

            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),

            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
        )

        # Specialized branches for WR prediction
        self.target_branch = nn.Sequential(
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.15),
            nn.Linear(40, 24)
        )

        self.efficiency_branch = nn.Sequential(
            nn.Linear(80, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        self.game_script_branch = nn.Sequential(
            nn.Linear(80, 24),
            nn.LayerNorm(24),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(24, 12)
        )

        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(52, 26),  # 24 + 16 + 12 = 52
            nn.Tanh(),
            nn.Linear(26, 52),
            nn.Softmax(dim=1)
        )

        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(52, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )

        self.std_head = nn.Sequential(
            nn.Linear(52, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        features = self.feature_extractor(x)

        # Specialized branches
        target_features = self.target_branch(features)
        efficiency_features = self.efficiency_branch(features)
        script_features = self.game_script_branch(features)

        # Combine branches
        combined = torch.cat([target_features, efficiency_features, script_features], dim=1)

        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        mean = self.mean_head(attended_features).squeeze(-1)
        std = self.std_head(attended_features).squeeze(-1)

        # Don't clamp here - targets are normalized during training
        # Clamping to fantasy point ranges when using normalized targets breaks training!

        # Return mean for compatibility with existing code
        return mean


class TENetwork(nn.Module):
    """Enhanced multi-head neural network architecture for tight end predictions.

    Specialized architecture focusing on:
    - Red zone target share (most predictive for TEs)
    - Formation usage (2-TE sets vs single TE)
    - Game script dependency (receiving vs blocking role)
    - Opponent-specific matchups
    """

    def __init__(self, input_size: int):
        super().__init__()

        # Multi-head architecture for TE-specific feature processing
        # Red Zone/Target Branch - processes target share, red zone usage, goal line opportunities
        self.redzone_branch = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Formation/Role Branch - processes snap share, blocking vs receiving role, two-TE sets
        self.formation_branch = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Game Script Branch - processes team passing volume, game script factors, vegas correlation
        self.script_branch = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Efficiency Branch - processes catch rate, YAC efficiency, ceiling indicators
        self.efficiency_branch = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Attention mechanism for branch importance
        self.attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)

        # Combined processing after multi-head extraction
        combined_size = 32 + 32 + 32 + 24  # Sum of branch outputs: 120
        self.combination_layers = nn.Sequential(
            nn.Linear(combined_size, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Output layers - NO sigmoid compression (learned from optimization guide)
        self.output = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            # NO Sigmoid - allows full range output (2-40 points)
            nn.ReLU()  # Ensures non-negative output
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    # Small random bias initialization for diversity
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head feature extraction
        redzone_features = self.redzone_branch(x)        # [batch, 32]
        formation_features = self.formation_branch(x)    # [batch, 32]
        script_features = self.script_branch(x)          # [batch, 32]
        efficiency_features = self.efficiency_branch(x)  # [batch, 24]

        # Apply attention to the main predictive branches (redzone, formation, script)
        # Stack the three 32-dim branches for attention
        attention_input = torch.stack([redzone_features, formation_features, script_features], dim=1)  # [batch, 3, 32]
        attended_features, attention_weights = self.attention(
            attention_input, attention_input, attention_input
        )
        # Flatten attention output: [batch, 3, 32] -> [batch, 96]
        attended_flat = attended_features.flatten(start_dim=1)

        # Combine all features: attended (96) + efficiency (24) = 120 total
        combined = torch.cat([attended_flat, efficiency_features], dim=1)

        # Process combined features
        processed = self.combination_layers(combined)

        # Generate output with proper scaling (2-40 point range for TEs)
        raw_output = self.output(processed)

        # Apply scaling similar to other position networks for consistency
        scaled_output = raw_output * POSITION_RANGES['TE']

        return scaled_output


class DEFNetwork(nn.Module):
    """Enhanced neural network architecture for defense predictions with research-based improvements."""

    def __init__(self, input_size: int):
        super().__init__()

        # Increased capacity for better DST pattern recognition (Research Finding: DST needs more complex modeling)
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_size, 128),  # Increased from 64
            nn.LayerNorm(128),  # Better than BatchNorm for smaller datasets
            nn.GELU(),  # Better activation for complex patterns
            nn.Dropout(0.4),
            nn.Linear(128, 96),  # Additional layer for complexity
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
        )

        # Enhanced specialized branches based on research insights
        # Research Finding: Pressure rate is #1 predictor
        self.pressure_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 20),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Research Finding: Turnovers correlate more with scoring than points allowed
        self.turnover_branch = nn.Sequential(
            nn.Linear(64, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Points allowed with non-linear tiered scoring
        self.points_branch = nn.Sequential(
            nn.Linear(64, 20),
            nn.LayerNorm(20),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(20, 12),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Game script and special teams branch (research finding: critical for DST)
        self.gamescript_branch = nn.Sequential(
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(16, 10),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Enhanced output processing with attention mechanism
        combined_size = 20 + 16 + 12 + 10  # 58 total
        self.attention = nn.MultiheadAttention(combined_size, num_heads=2, dropout=0.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(combined_size)

        self.output = nn.Sequential(
            nn.Linear(combined_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.feature_extraction(x)

        # Specialized branches
        pressure_features = self.pressure_branch(features)
        turnover_features = self.turnover_branch(features)
        points_features = self.points_branch(features)
        gamescript_features = self.gamescript_branch(features)

        # Combine all branches
        combined = torch.cat([pressure_features, turnover_features, points_features, gamescript_features], dim=1)

        # Apply attention for feature importance weighting
        if combined.dim() == 2:
            combined_expanded = combined.unsqueeze(1)  # Add sequence dimension for attention
            attended, _ = self.attention(combined_expanded, combined_expanded, combined_expanded)
            attended = attended.squeeze(1)  # Remove sequence dimension
            combined = self.attention_norm(attended + combined)  # Residual connection

        # Final output
        output = self.output(combined)
        scaled_output = output * POSITION_RANGES['DEF']
        return scaled_output


class QBNeuralModel(BaseNeuralModel):
    """Neural network model for quarterback predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # QB-specific parameters are now loaded from hyperparameters.yaml via parent class

    def build_network(self, input_size: int) -> nn.Module:
        return QBNetwork(input_size)

    def validate_qb_predictions(self, predictions: np.ndarray) -> bool:
        """QB-specific prediction validation."""
        return self.validate_predictions(predictions, 'QB')


class RBNeuralModel(BaseNeuralModel):
    """Enhanced neural network model for running back predictions with research-based optimizations."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # RB-specific parameters are now loaded from hyperparameters.yaml via parent class
        # Only override if you need RB-specific adjustments not covered by tuning

    def build_network(self, input_size: int) -> nn.Module:
        return RBNetwork(input_size)

    def train_rb_model(self, X_train, y_train, X_val, y_val, salaries_train=None):
        """Special training for RB models with validation."""

        # Validate input data
        assert y_train.min() >= 0, "Negative fantasy points in training"
        assert y_train.max() <= 60, "Unrealistic max points in training"
        assert y_train.std() > 2, "Insufficient variance in training labels"

        # Add sample weights based on recency
        sample_weights = np.ones(len(y_train))
        # Weight recent games more heavily if week data is available
        if len(X_train.shape) > 1 and X_train.shape[1] > 10:  # Assume week is in features
            # Use simple time-based weighting
            sample_weights = 1 + 0.1 * np.linspace(-0.5, 0.5, len(y_train))
            sample_weights = np.clip(sample_weights, 0.5, 1.5)

        # Custom loss for RB
        def rb_loss_fn(predictions, targets):
            if isinstance(predictions, dict):
                mean_pred = predictions['mean']
            else:
                mean_pred = predictions

            mean_loss = F.smooth_l1_loss(mean_pred, targets)

            # Penalty for unrealistic predictions
            range_penalty = torch.mean(
                F.relu(3 - mean_pred) * 2 +  # Heavy penalty below 3
                F.relu(mean_pred - 35) * 2   # Heavy penalty above 35
            )

            # Ensure proper variance
            batch_std = torch.std(mean_pred)
            variance_loss = F.relu(3 - batch_std) * 0.5

            total_loss = mean_loss + 0.1 * range_penalty + variance_loss
            return total_loss

        # Override the criterion for RB
        original_criterion = self.criterion
        self.criterion = rb_loss_fn

        # Train with lower LR
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4
            )

        # Train normally but with validation checks
        try:
            history = self.train(X_train, y_train, X_val, y_val)

            # Validate final predictions
            self.network.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_val).to(self.device)
                val_preds = self.network(val_tensor)
                val_preds_np = val_preds.cpu().numpy()

                # Check predictions are reasonable
                assert val_preds_np.min() >= 2, f"RB predictions too low: {val_preds_np.min()}"
                assert val_preds_np.max() <= 40, f"RB predictions too high: {val_preds_np.max()}"
                assert val_preds_np.std() > 2, f"RB predictions lack variance: {val_preds_np.std()}"

            return history
        finally:
            # Restore original criterion
            self.criterion = original_criterion


class WRNeuralModel(BaseNeuralModel):
    """Enhanced neural network model for wide receiver predictions with target share optimization."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # WR-specific parameters are now loaded from hyperparameters.yaml via parent class

    def build_network(self, input_size: int) -> nn.Module:
        return WRNetwork(input_size)

    def train_wr_model(self, X_train, y_train, X_val, y_val, salaries_train=None):
        """Special training for WR models with target share validation."""

        # Validate input data
        assert y_train.min() >= 0, "Negative fantasy points in training"
        assert y_train.max() <= 50, "Unrealistic max points for WR"
        assert y_train.std() > 3, "Insufficient variance in WR training labels"

        # Custom loss for WR with target share importance
        def wr_loss_fn(predictions, targets):
            if isinstance(predictions, dict):
                mean_pred = predictions['mean']
            else:
                mean_pred = predictions

            # Base loss
            mean_loss = F.smooth_l1_loss(mean_pred, targets)

            # WR-specific penalties
            range_penalty = torch.mean(
                F.relu(2 - mean_pred) * 3 +  # Heavy penalty below 2
                F.relu(mean_pred - 40) * 2   # Heavy penalty above 40
            )

            # Target consistency - WRs should have reasonable variance
            batch_std = torch.std(mean_pred)
            variance_loss = F.relu(4 - batch_std) * 0.3  # WRs more volatile than RBs

            # Ceiling preservation - some WRs should project high
            ceiling_loss = F.relu(15 - torch.max(mean_pred)) * 0.1

            total_loss = mean_loss + 0.1 * range_penalty + variance_loss + ceiling_loss
            return total_loss

        # Override criterion
        original_criterion = self.criterion
        self.criterion = wr_loss_fn

        # WR-optimized training parameters
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5
            )

        # Train with validation
        try:
            history = self.train(X_train, y_train, X_val, y_val)

            # Validate WR predictions
            self.network.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_val).to(self.device)
                val_preds = self.network(val_tensor)
                val_preds_np = val_preds.cpu().numpy()

                # WR validation checks
                assert val_preds_np.min() >= 1, f"WR predictions too low: {val_preds_np.min()}"
                assert val_preds_np.max() <= 45, f"WR predictions too high: {val_preds_np.max()}"
                assert val_preds_np.std() > 3, f"WR predictions lack variance: {val_preds_np.std()}"
                assert np.percentile(val_preds_np, 90) > 15, "No WR ceiling players projected"

            return history
        finally:
            self.criterion = original_criterion


class TENeuralModel(BaseNeuralModel):
    """Neural network model for tight end predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # TE-specific parameters are now loaded from hyperparameters.yaml via parent class

    def build_network(self, input_size: int) -> nn.Module:
        return TENetwork(input_size)

    def train_te_model(self, X_train, y_train, X_val, y_val, salaries_train=None):
        """Specialized training for TE models focusing on TE-specific patterns."""

        # Validate input data for TE-specific ranges
        assert y_train.min() >= 0, "Negative fantasy points in training"
        assert y_train.max() <= 50, "Unrealistic max points in training (TEs typically 2-35)"
        assert y_train.std() > 1.5, "Insufficient variance in TE training labels"

        # Special handling for zero variance issue (seen in performance log)
        if y_train.std() < 1.0:
            logger.warning(f"TE model has very low target variance: {y_train.std():.3f}")
            # Add small amount of noise to break zero variance
            noise = np.random.normal(0, 0.5, len(y_train))
            y_train = y_train + noise
            y_train = np.clip(y_train, 0, 50)  # Ensure valid range

        # Sample weights emphasizing recent data and high-scoring games
        sample_weights = np.ones(len(y_train))
        # Weight higher-scoring games more (TEs have high ceiling variance)
        high_score_mask = y_train > np.percentile(y_train, 70)
        sample_weights[high_score_mask] *= 1.3

        # Custom loss function for TE-specific characteristics
        def te_loss_fn(predictions, targets):
            if isinstance(predictions, dict):
                mean_pred = predictions['mean']
            else:
                mean_pred = predictions

            # Base loss - smooth L1 for robustness to outliers
            mean_loss = F.smooth_l1_loss(mean_pred, targets)

            # Range penalty for TE-appropriate scoring (2-35 points typically)
            range_penalty = torch.mean(
                F.relu(2 - mean_pred) * 3 +     # Heavy penalty below 2
                F.relu(mean_pred - 40) * 2      # Moderate penalty above 40
            )

            # Ceiling preservation - TEs need ability to predict high scores (15+ points)
            batch_mean = torch.mean(mean_pred)
            ceiling_penalty = F.relu(4 - batch_mean) * 0.5  # Penalize if avg prediction too low

            # Variance preservation - critical for TE position
            batch_std = torch.std(mean_pred)
            variance_penalty = F.relu(2 - batch_std) * 0.8  # TEs need at least 2-point std dev

            # Red zone dependency - bonus for predicting TDs accurately
            # Approximate TD prediction from points (6 points per TD)
            estimated_tds = mean_pred / 15.0  # Rough estimate
            td_regularization = torch.mean(torch.abs(estimated_tds - torch.round(estimated_tds)) * 0.1)

            # Total loss combining all TE-specific factors
            total_loss = (mean_loss +
                         0.15 * range_penalty +
                         0.1 * ceiling_penalty +
                         0.2 * variance_penalty +
                         0.05 * td_regularization)

            return total_loss

        # Override criterion for TE-specific training
        original_criterion = self.criterion
        self.criterion = te_loss_fn

        # Optimizer with TE-tuned parameters
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=2e-4,  # Slightly higher regularization for complex TE architecture
                betas=(0.9, 0.999)
            )

        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15, verbose=True
        )

        try:
            # Train with validation monitoring
            history = self.train(X_train, y_train, X_val, y_val)

            # Post-training validation for TE-specific issues
            self.network.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_val).to(self.device)
                val_preds = self.network(val_tensor)
                val_preds_np = val_preds.cpu().numpy()

                # Check for zero variance issue (major problem in previous TE model)
                pred_std = val_preds_np.std()
                pred_mean = val_preds_np.mean()

                logger.info(f"TE Validation - Mean: {pred_mean:.3f}, Std: {pred_std:.3f}, "
                           f"Range: {val_preds_np.min():.3f}-{val_preds_np.max():.3f}")

                # Strict validation checks
                assert val_preds_np.min() >= 1.5, f"TE predictions too low: {val_preds_np.min()}"
                assert val_preds_np.max() <= 45, f"TE predictions too high: {val_preds_np.max()}"
                assert pred_std > 1.5, f"TE predictions lack variance (zero variance issue): {pred_std}"
                assert not np.allclose(val_preds_np, val_preds_np[0]), "All predictions identical - model broken"

                # Check for reasonable score distribution
                high_score_count = np.sum(val_preds_np > 12)
                assert high_score_count > len(val_preds_np) * 0.1, f"Too few high predictions: {high_score_count}/{len(val_preds_np)}"

            return history

        finally:
            # Restore original criterion
            self.criterion = original_criterion



class DEFCatBoostModel:
    """CatBoost-only model for DST predictions - optimized for small, volatile datasets."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.best_params = None  # Will be set by tune_hyperparameters

        # Load hyperparameters from YAML if available
        if HAS_HYPERPARAMETER_MANAGER:
            from hyperparameter_manager import get_hyperparameter_manager
            hyperparameter_manager = get_hyperparameter_manager()
            stored_params = hyperparameter_manager.get_hyperparameters('DST')

            # Extract CatBoost-specific parameters if they exist
            if stored_params:
                logger.info(f"Loading DST hyperparameters from YAML")
                self.best_params = {
                    'iterations': stored_params.get('catboost_iterations', 4000),
                    'learning_rate': stored_params.get('catboost_learning_rate', 0.04),
                    'depth': stored_params.get('catboost_depth', 7),
                    'l2_leaf_reg': stored_params.get('catboost_l2_leaf_reg', 6),
                    'bagging_temperature': stored_params.get('catboost_bagging_temperature', 0.5),
                    'random_strength': stored_params.get('catboost_random_strength', 1),
                    'border_count': stored_params.get('catboost_border_count', 128),
                    'min_data_in_leaf': stored_params.get('catboost_min_data_in_leaf', 3)
                }
                # Compatibility attributes
                self.learning_rate = self.best_params['learning_rate']
                self.batch_size = None  # Not applicable for CatBoost
            else:
                # Default values if no stored params
                self.learning_rate = 0.04
                self.batch_size = None
        else:
            # Default values if hyperparameter manager not available
            self.learning_rate = 0.04
            self.batch_size = None

        if not HAS_CATBOOST:
            raise ImportError("CatBoost is required for DST model but not available")

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train CatBoost model optimized for DST volatility."""
        start_time = time.time()

        # Use best params from tuning if available, otherwise use defaults
        if self.best_params is not None:
            logger.info("Using tuned CatBoost parameters for DST")
            self.model = CatBoostRegressor(
                iterations=self.best_params.get('iterations', 4000),
                learning_rate=self.best_params.get('learning_rate', 0.04),
                depth=self.best_params.get('depth', 7),
                l2_leaf_reg=self.best_params.get('l2_leaf_reg', 6),
                bagging_temperature=self.best_params.get('bagging_temperature', 0.5),
                random_strength=self.best_params.get('random_strength', 1),
                border_count=self.best_params.get('border_count', 128),
                min_data_in_leaf=self.best_params.get('min_data_in_leaf', 3),
                random_seed=42,
                loss_function='MAE',
                eval_metric='MAE',
                use_best_model=True,
                verbose=100,
                allow_writing_files=False,
                bootstrap_type='Bayesian',
                od_type='IncToDec',
                od_wait=50,
                max_ctr_complexity=4,
                model_size_reg=0.5,
                leaf_estimation_iterations=5
            )
        else:
            # Default CatBoost parameters for DST
            self.model = CatBoostRegressor(
                iterations=4000,             # More iterations needed for complex features
                learning_rate=0.04,          # Lower LR for stable convergence with more features
                depth=7,                     # Deeper trees to capture feature interactions
                l2_leaf_reg=6,              # Stronger regularization to prevent overfitting
                random_seed=42,              # Reproducibility
                loss_function='MAE',         # MAE better for DST outliers vs RMSE
                eval_metric='MAE',           # Primary metric for DST prediction
                use_best_model=True,         # Use best model from validation
                verbose=100,                 # Less frequent updates for longer training
                allow_writing_files=False,   # No temp files
                # Advanced parameters for enhanced model
                bootstrap_type='Bayesian',   # Better uncertainty estimation
                bagging_temperature=0.5,     # Reduce overfitting with Bayesian bootstrap
                od_type='IncToDec',         # Overfitting detection
                od_wait=50,                 # Wait period for overfitting detection
                max_ctr_complexity=4,        # Handle categorical interactions better
                model_size_reg=0.5,         # Regularize model complexity
                # Feature interaction settings
                min_data_in_leaf=3,         # Min samples per leaf (small for DST volatility)
                leaf_estimation_iterations=5 # Better leaf value estimation
            )

        logger.info("Training CatBoost model for DST...")

        # Train with validation set
        self.model.fit(
            X, y,
            eval_set=(X_val, y_val),
            verbose=False
        )

        # Generate predictions
        train_pred = self.model.predict(X)
        val_pred = self.model.predict(X_val)

        # Calculate metrics
        train_mae = np.mean(np.abs(y - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))
        train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

        # Calculate R²
        train_ss_tot = np.sum((y - np.mean(y)) ** 2)
        val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

        train_r2 = 1 - np.sum((y - train_pred) ** 2) / train_ss_tot if train_ss_tot > 0 else 0.0
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / val_ss_tot if val_ss_tot > 0 else 0.0

        self.is_trained = True
        training_time = time.time() - start_time

        # Get feature importance
        feature_importance = self.model.get_feature_importance() if hasattr(self.model, 'get_feature_importance') else None

        result = TrainingResult(
            model=self.model,
            training_time=training_time,
            best_iteration=self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else 0,
            feature_importance=feature_importance,
            train_mae=train_mae,
            val_mae=val_mae,
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            train_r2=train_r2,
            val_r2=val_r2,
            training_samples=len(X),
            validation_samples=len(X_val),
            feature_count=X.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"CatBoost training completed: MAE={val_mae:.3f}, R²={val_r2:.3f}")

        return result

    def train_component_models(self, X: np.ndarray, y_components: dict, X_val: np.ndarray, y_val_components: dict) -> dict:
        """Train component-based DST models (sacks, turnovers, PA, TDs) for enhanced predictions."""
        logger.info("Training DST component models...")

        component_models = {}
        component_results = {}

        # Component 1: Sacks Model (Poisson regression)
        logger.info("Training sacks component model...")
        sacks_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=4,
            loss_function='Poisson',  # Poisson loss for count data
            eval_metric='Poisson',
            use_best_model=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=100
        )

        sacks_model.fit(
            X, y_components['sacks'],
            eval_set=(X_val, y_val_components['sacks'])
        )
        component_models['sacks'] = sacks_model

        # Component 2: Turnovers Model (Poisson regression)
        logger.info("Training turnovers component model...")
        turnovers_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=4,
            loss_function='Poisson',  # Poisson for count data
            eval_metric='Poisson',
            use_best_model=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=100
        )

        turnovers_model.fit(
            X, y_components['turnovers'],
            eval_set=(X_val, y_val_components['turnovers'])
        )
        component_models['turnovers'] = turnovers_model

        # Component 3: Points Allowed Bucket Model (Multiclass)
        logger.info("Training PA bucket component model...")
        pa_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=4,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            use_best_model=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=100
        )

        pa_model.fit(
            X, y_components['pa_bucket'],
            eval_set=(X_val, y_val_components['pa_bucket'])
        )
        component_models['pa_bucket'] = pa_model

        # Component 4: Defensive TD Model (Binary classifier with class weighting)
        logger.info("Training defensive TD component model...")
        td_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=4,
            loss_function='Logloss',
            eval_metric='Logloss',
            use_best_model=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=100,
            class_weights={0: 1, 1: 5}  # Handle imbalanced TD data
        )

        td_model.fit(
            X, y_components['td'],
            eval_set=(X_val, y_val_components['td'])
        )
        component_models['td'] = td_model

        # Store component models
        self.component_models = component_models

        logger.info("DST component models training completed")
        return component_models

    def predict_components(self, X: np.ndarray) -> dict:
        """Generate component-based predictions and combine into final DST score."""
        if not hasattr(self, 'component_models') or not self.component_models:
            raise ValueError("Component models must be trained before making component predictions")

        # Get component predictions
        sack_pred = self.component_models['sacks'].predict(X)
        turnover_pred = self.component_models['turnovers'].predict(X)
        pa_bucket_pred = self.component_models['pa_bucket'].predict(X)
        td_prob = self.component_models['td'].predict_proba(X)[:, 1]  # Probability of TD

        # Convert PA buckets to points using DraftKings scoring
        def pa_bucket_to_points(bucket):
            """Convert PA tier bucket to DraftKings points."""
            pa_points_map = {
                0: 10,    # 0 points allowed
                1: 7,     # 1-6 points
                2: 4,     # 7-13 points
                3: 1,     # 14-20 points
                4: 0,     # 21-27 points
                5: -1,    # 28-34 points
                6: -4     # 35+ points
            }
            return pa_points_map.get(int(bucket), 0)

        # Calculate combined DST fantasy points
        dst_points = []
        for i in range(len(X)):
            # DraftKings DST scoring formula
            points = 0.0

            # Sacks: 1 point each
            points += sack_pred[i] * 1.0

            # Turnovers: 2 points each (INT + FR)
            points += turnover_pred[i] * 2.0

            # Points allowed tier bonus
            points += pa_bucket_to_points(pa_bucket_pred[i])

            # Defensive TDs: 6 points each (weighted by probability)
            points += td_prob[i] * 6.0

            dst_points.append(points)

        return {
            'combined_points': np.array(dst_points),
            'sacks': sack_pred,
            'turnovers': turnover_pred,
            'pa_bucket': pa_bucket_pred,
            'td_probability': td_prob
        }

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 20, timeout: int = 3600,
                            epochs: int = 100) -> Dict[str, Any]:
        """Perform real hyperparameter optimization for CatBoost DST model using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds

        Returns:
            Dictionary of best hyperparameters found
        """
        logger.info(f"Starting CatBoost hyperparameter optimization for DST ({n_trials} trials)")

        # Check if Optuna is available
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available, using default CatBoost parameters")
            # Fall back to training with default params
            result = self.train(X_train, y_train, X_val, y_val)
            return {
                'iterations': 4000,
                'learning_rate': 0.04,
                'depth': 7,
                'l2_leaf_reg': 6,
                'model_type': 'CatBoost'
            }

        def objective(trial):
            """Optuna objective function for CatBoost hyperparameter tuning."""
            # Suggest hyperparameters
            params = {
                'iterations': trial.suggest_int('iterations', 1000, 5000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10),
            }

            # Train model with suggested parameters
            model = CatBoostRegressor(
                iterations=params['iterations'],
                learning_rate=params['learning_rate'],
                depth=params['depth'],
                l2_leaf_reg=params['l2_leaf_reg'],
                bagging_temperature=params['bagging_temperature'],
                random_strength=params['random_strength'],
                border_count=params['border_count'],
                min_data_in_leaf=params['min_data_in_leaf'],
                loss_function='MAE',
                eval_metric='MAE',
                use_best_model=True,
                early_stopping_rounds=50,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                bootstrap_type='Bayesian'
            )

            # Fit with early stopping
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )

            # Calculate validation metrics
            val_pred = model.predict(X_val)
            val_mae = np.mean(np.abs(y_val - val_pred))

            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(y_val, val_pred)

            # Use Spearman if valid, otherwise fall back to R²
            if not np.isnan(spearman_corr):
                metric_value = spearman_corr
                metric_name = "Spearman"
            else:
                from sklearn.metrics import r2_score
                r2 = r2_score(y_val, val_pred)
                metric_value = r2 if not np.isnan(r2) else -1.0
                metric_name = "R²"

            # Log trial progress
            logger.info(f"Trial {trial.number + 1}: MAE={val_mae:.4f}, {metric_name}={metric_value:.4f}")

            # Optimize on Spearman/R² (higher is better, so return negative)
            return -metric_value

        # Create study and optimize
        study = optuna.create_study(direction='minimize', study_name='dst_catboost_tuning')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Get best parameters
        best_params = study.best_params
        best_mae = study.best_value  # Best MAE achieved

        logger.info(f"Best CatBoost parameters found (MAE={best_mae:.3f}): {best_params}")

        # Train final model with best parameters to get R²
        final_model = CatBoostRegressor(
            iterations=best_params['iterations'],
            learning_rate=best_params['learning_rate'],
            depth=best_params['depth'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            bagging_temperature=best_params['bagging_temperature'],
            random_strength=best_params['random_strength'],
            border_count=best_params['border_count'],
            min_data_in_leaf=best_params['min_data_in_leaf'],
            loss_function='MAE',
            eval_metric='MAE',
            use_best_model=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            bootstrap_type='Bayesian'
        )

        final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        val_pred = final_model.predict(X_val)

        # Calculate final metrics
        final_mae = np.mean(np.abs(y_val - val_pred))
        from scipy.stats import spearmanr
        final_spearman, _ = spearmanr(y_val, val_pred)
        final_spearman = final_spearman if not np.isnan(final_spearman) else 0

        # Also calculate R² for compatibility
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        val_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Update hyperparameter manager if available
        if HAS_HYPERPARAMETER_MANAGER:
            from hyperparameter_manager import get_hyperparameter_manager
            hyperparameter_manager = get_hyperparameter_manager()

            # Update with best CatBoost parameters
            yaml_params = {
                'learning_rate': best_params['learning_rate'],
                'batch_size': 64,  # Keep for compatibility, not used
                'epochs': best_params['iterations'],  # Maps to iterations
                'hidden_size': 128,  # Keep for compatibility, not used
                'num_layers': 2,  # Keep for compatibility, not used
                'dropout_rate': 0.2,  # Keep for compatibility, not used
                'weight_decay': best_params['l2_leaf_reg'],  # Maps to l2_leaf_reg
                # CatBoost specific
                'catboost_depth': best_params['depth'],
                'catboost_l2_leaf_reg': best_params['l2_leaf_reg'],
                'catboost_iterations': best_params['iterations'],
                'catboost_learning_rate': best_params['learning_rate'],
                'catboost_bagging_temperature': best_params['bagging_temperature'],
                'catboost_random_strength': best_params['random_strength'],
                'catboost_border_count': best_params['border_count'],
                'catboost_min_data_in_leaf': best_params['min_data_in_leaf']
            }

            hyperparameter_manager.update_hyperparameters(
                position='DST',
                new_params=yaml_params,
                validation_r2=val_r2,
                validation_mae=final_mae,
                validation_spearman=final_spearman,
                trials=n_trials
            )
            logger.info(f"Updated DST hyperparameters in YAML (MAE: {final_mae:.3f}, Spearman: {final_spearman:.3f}, R²: {val_r2:.4f})")

        # Store best params for use in training
        self.best_params = best_params

        # Return parameters in format compatible with training
        return best_params

    def find_optimal_lr(self, X_train: np.ndarray, y_train: np.ndarray,
                       start_lr: float = 1e-8, end_lr: float = 1.0,
                       num_iter: int = 100) -> float:
        """Stub method - CatBoost DST uses fixed learning rate."""
        return 0.04

    def optimize_batch_size(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> int:
        """Stub method - CatBoost doesn't use batch size."""
        return 0  # N/A for CatBoost

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with CatBoost."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        point_estimate = self.model.predict(X)

        # Simple uncertainty estimation for CatBoost
        uncertainty = np.std(point_estimate) * 0.5  # Conservative uncertainty
        lower_bound = point_estimate - 1.96 * uncertainty
        upper_bound = point_estimate + 1.96 * uncertainty
        floor = point_estimate - 0.8 * uncertainty
        ceiling = point_estimate + 1.0 * uncertainty
        confidence_score = np.ones_like(point_estimate) * 0.75  # Higher confidence for CatBoost

        return PredictionResult(
            point_estimate=point_estimate,
            confidence_score=confidence_score,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )

    def save_model(self, path: str):
        """Save CatBoost model."""
        if self.model is None:
            raise ValueError("No model to save")

        # CatBoost has its own save format
        catboost_path = path.replace('.pth', '_catboost.cbm')
        self.model.save_model(catboost_path)
        logger.info(f"CatBoost model saved to {catboost_path}")

    def load_model(self, path: str):
        """Load CatBoost model."""
        # Try different CatBoost file naming conventions
        import os
        catboost_paths = [
            path.replace('.pth', '_nn_catboost.cbm'),  # New convention: dst_model_nn_catboost.cbm
            path.replace('.pth', '_catboost.cbm'),     # Old convention: dst_model_catboost.cbm
        ]

        catboost_path = None
        for candidate_path in catboost_paths:
            if os.path.exists(candidate_path):
                catboost_path = candidate_path
                break

        if catboost_path is None:
            raise FileNotFoundError(f"CatBoost model not found. Tried: {catboost_paths}")

        self.model = CatBoostRegressor()
        self.model.load_model(catboost_path)
        self.is_trained = True
        logger.info(f"CatBoost model loaded from {catboost_path}")


# Use CatBoost-only for DST (winner of LightGBM comparison)
class DEFNeuralModel(DEFCatBoostModel):
    """CatBoost-only DST model (winner over LightGBM comparison)."""
    pass


# Correlated Multi-Position Model
class GameContextEncoder(nn.Module):
    """Encode game-level context that affects all players."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        encoded_seq = encoded.unsqueeze(0)
        attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        output = encoded + attended.squeeze(0)
        return output


class PositionSpecificHead(nn.Module):
    """Position-specific prediction head."""

    def __init__(self, context_dim: int, player_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(context_dim + player_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, game_context: torch.Tensor, player_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([game_context, player_features], dim=1)
        fused = self.fusion(combined)
        prediction = self.predictor(fused)
        uncertainty_params = self.uncertainty(fused)
        return prediction.squeeze(), uncertainty_params


class EnsembleModel:
    """Ensemble model combining neural network with XGBoost."""

    def __init__(self, neural_model: BaseNeuralModel, position: str):
        """Initialize ensemble with a trained neural network."""
        self.neural_model = neural_model
        self.position = position
        self.xgb_model = None
        self.is_trained = False

        if not HAS_XGBOOST:
            logger.warning("XGBoost not available. Falling back to neural network only.")

    @property
    def input_size(self):
        """Expose neural network's input size for feature alignment."""
        return getattr(self.neural_model, 'input_size', None)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train ensemble model."""
        # First ensure neural network is trained
        if not self.neural_model.is_trained:
            neural_result = self.neural_model.train(X_train, y_train, X_val, y_val)
        else:
            # Get existing neural network result metrics
            neural_result = TrainingResult(
                model=self.neural_model.network,
                training_time=0.0,
                best_iteration=0,
                feature_importance=None,
                train_mae=0.0,
                val_mae=0.0,
                train_rmse=0.0,
                val_rmse=0.0,
                train_r2=0.0,
                val_r2=0.0,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=X_train.shape[1]
            )

        if not HAS_XGBOOST:
            return neural_result

        logger.info(f"Training XGBoost for {self.position} ensemble...")

        # Get neural network predictions for ensemble features
        nn_train_pred = self.neural_model.predict(X_train).point_estimate
        nn_val_pred = self.neural_model.predict(X_val).point_estimate

        # Create ensemble features: original features + neural network predictions
        X_train_ensemble = np.column_stack([X_train, nn_train_pred])
        X_val_ensemble = np.column_stack([X_val, nn_val_pred])

        # Configure XGBoost for the position
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }

        # Position-specific XGBoost tuning (separate num_boost_round)
        if self.position == 'QB':
            xgb_params.update({
                'max_depth': 7,
                'learning_rate': 0.05
            })
            num_rounds = 500
        elif self.position in ['RB', 'WR']:
            xgb_params.update({
                'max_depth': 6,
                'learning_rate': 0.08
            })
            num_rounds = 400
        else:  # TE, DST
            xgb_params.update({
                'max_depth': 5,
                'learning_rate': 0.1
            })
            num_rounds = 300

        # Train XGBoost for full rounds
        dtrain = xgb.DMatrix(X_train_ensemble, label=y_train)
        dval = xgb.DMatrix(X_val_ensemble, label=y_val)

        self.xgb_model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=False
        )

        # Get ensemble predictions
        xgb_train_pred = self.xgb_model.predict(dtrain)
        xgb_val_pred = self.xgb_model.predict(dval)

        # Calculate metrics
        train_mae = np.mean(np.abs(y_train - xgb_train_pred))
        val_mae = np.mean(np.abs(y_val - xgb_val_pred))
        train_rmse = np.sqrt(np.mean((y_train - xgb_train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - xgb_val_pred) ** 2))

        # Calculate R²
        train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

        train_r2 = 1 - np.sum((y_train - xgb_train_pred) ** 2) / train_ss_tot if train_ss_tot > 0 else 0.0
        val_r2 = 1 - np.sum((y_val - xgb_val_pred) ** 2) / val_ss_tot if val_ss_tot > 0 else 0.0

        self.is_trained = True

        logger.info(f"XGBoost ensemble trained: MAE={val_mae:.3f}, R²={val_r2:.3f}")

        return TrainingResult(
            model=self.xgb_model,
            training_time=0.0,  # XGBoost training time not tracked here
            best_iteration=self.xgb_model.best_iteration if hasattr(self.xgb_model, 'best_iteration') else 0,
            feature_importance=self.xgb_model.get_score(importance_type='weight') if self.xgb_model else None,
            train_mae=train_mae,
            val_mae=val_mae,
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            train_r2=train_r2,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train_ensemble.shape[1]
        )

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 20, timeout: int = 3600,
                            epochs: int = 100) -> Dict[str, Any]:
        """Perform hyperparameter optimization for ensemble model.

        First tunes the neural network, then uses default XGBoost parameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds

        Returns:
            Dictionary with best hyperparameters
        """
        logger.info(f"Tuning hyperparameters for {self.position} ensemble model...")

        # First tune the neural network component
        if hasattr(self.neural_model, 'tune_hyperparameters'):
            nn_params = self.neural_model.tune_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials, timeout, epochs
            )
            logger.info(f"Neural network hyperparameters tuned: {nn_params}")
        else:
            logger.warning("Neural model doesn't support hyperparameter tuning")
            nn_params = {}

        # XGBoost hyperparameters are position-specific and fixed for now
        # In a future version, we could also tune XGBoost parameters
        # Use different key names to avoid collision with neural network params
        xgb_params = {
            'xgb_max_depth': 6,
            'xgb_learning_rate': 0.1,
            'xgb_subsample': 0.8,
            'xgb_colsample_bytree': 0.8
        }

        if self.position == 'QB':
            xgb_params.update({
                'xgb_max_depth': 7,
                'xgb_learning_rate': 0.05
            })
        elif self.position in ['RB', 'WR']:
            xgb_params.update({
                'xgb_max_depth': 6,
                'xgb_learning_rate': 0.08
            })
        else:  # TE, DST
            xgb_params.update({
                'xgb_max_depth': 5,
                'xgb_learning_rate': 0.1
            })

        # Return combined parameters without collisions
        combined_params = {**nn_params, **xgb_params}
        logger.info(f"Ensemble hyperparameters: {combined_params}")

        # Save the optimized hyperparameters to YAML config if available
        # Only save neural network parameters (not XGBoost ones) to avoid confusion
        if HAS_HYPERPARAMETER_MANAGER:
            nn_only_params = {k: v for k, v in combined_params.items() if not k.startswith('xgb_')}
            hyperparameter_manager = get_hyperparameter_manager()
            hyperparameter_manager.update_hyperparameters(
                position=self.position,
                new_params=nn_only_params,
                validation_r2=None,  # Will be updated after training
                trials=n_trials
            )

        return combined_params

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Get neural network predictions
        nn_result = self.neural_model.predict(X)

        if not HAS_XGBOOST or self.xgb_model is None:
            return nn_result

        # Create ensemble features
        X_ensemble = np.column_stack([X, nn_result.point_estimate])

        # Get XGBoost predictions
        dtest = xgb.DMatrix(X_ensemble)
        xgb_pred = self.xgb_model.predict(dtest)

        # Ensemble prediction: weighted average (XGBoost gets higher weight due to better performance)
        ensemble_pred = 0.7 * xgb_pred + 0.3 * nn_result.point_estimate

        # Apply QB scaling if this is a QB model
        if hasattr(self.config, 'position') and self.config.position == 'QB':
            ensemble_pred = ensemble_pred * POSITION_RANGES['QB']

        # Use neural network uncertainty estimates with slight adjustment
        uncertainty = nn_result.confidence_score * 0.9  # Slightly more confident with ensemble

        lower_bound = ensemble_pred - 1.96 * uncertainty
        upper_bound = ensemble_pred + 1.96 * uncertainty
        floor = ensemble_pred - 0.8 * uncertainty
        ceiling = ensemble_pred + 1.0 * uncertainty

        return PredictionResult(
            point_estimate=ensemble_pred,
            confidence_score=uncertainty,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=f"ensemble_v{self.neural_model.config.version}",
        )

    def save_model(self, path: str):
        """Save ensemble model."""
        # Save neural network
        nn_path = path.replace('.pth', '_nn.pth')
        self.neural_model.save_model(nn_path)

        # Save XGBoost model
        if HAS_XGBOOST and self.xgb_model:
            xgb_path = path.replace('.pth', '_xgb.json')
            self.xgb_model.save_model(xgb_path)
            logger.info(f"Ensemble model saved: {nn_path}, {xgb_path}")
        else:
            logger.info(f"Neural network saved: {nn_path}")

    def load_model(self, path: str, input_size: int = None):
        """Load ensemble model."""
        # Load neural network
        nn_path = path.replace('.pth', '_nn.pth')
        self.neural_model.load_model(nn_path, input_size)

        # Load XGBoost model if available
        if HAS_XGBOOST:
            xgb_path = path.replace('.pth', '_xgb.json')
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                self.is_trained = True
                logger.info(f"Ensemble model loaded: {nn_path}, {xgb_path}")
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}. Using neural network only.")
                self.xgb_model = None
        else:
            logger.info(f"Neural network loaded: {nn_path}")


class CorrelatedFantasyModel(nn.Module):
    """Neural network that models player correlations for fantasy football."""

    def __init__(
        self,
        game_feature_dim: int,
        position_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.positions = list(position_feature_dims.keys())
        self.game_encoder = GameContextEncoder(game_feature_dim, hidden_dim)

        self.position_heads = nn.ModuleDict({
            pos: PositionSpecificHead(
                context_dim=hidden_dim,
                player_dim=feat_dim,
                hidden_dim=hidden_dim // 2
            )
            for pos, feat_dim in position_feature_dims.items()
        })

        self.correlation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * len(self.positions), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.stack_factors = nn.Parameter(
            torch.randn(len(self.positions), len(self.positions)) * 0.1
        )

    def forward(
        self,
        game_features: torch.Tensor,
        player_features: Dict[str, torch.Tensor],
        return_correlations: bool = False
    ) -> Dict[str, torch.Tensor]:
        game_context = self.game_encoder(game_features)

        predictions = {}
        position_embeddings = []

        for pos in self.positions:
            if pos in player_features:
                pred, unc = self.position_heads[pos](game_context, player_features[pos])
                pos_range = POSITION_RANGES.get(pos.upper(), 30.0)
                scaled_pred = pred * pos_range

                predictions[pos] = scaled_pred
                position_embeddings.append(scaled_pred.unsqueeze(1))

        # Model correlations between positions
        if len(position_embeddings) > 1:
            stacked = torch.cat(position_embeddings, dim=1)
            correlated = torch.matmul(stacked, self.stack_factors[:len(position_embeddings), :len(position_embeddings)])

            for i, pos in enumerate(predictions.keys()):
                correlation_adjustment = correlated[:, i]
                predictions[pos] = predictions[pos] + 0.1 * correlation_adjustment

        if return_correlations:
            return predictions, self.stack_factors

        return predictions


def create_model(position: str, config: ModelConfig = None, use_ensemble: bool = False) -> BaseNeuralModel:
    """Factory function to create position-specific models."""
    if config is None:
        config = ModelConfig(position=position)

    models = {
        'QB': QBNeuralModel,
        'RB': RBNeuralModel,
        'WR': WRNeuralModel,
        'TE': TENeuralModel,
        'DEF': DEFNeuralModel,
        'DST': DEFNeuralModel,  # Use same as DEF
    }

    if position not in models:
        raise ValueError(f"Unknown position: {position}")

    base_model = models[position](config)

    if use_ensemble:
        return EnsembleModel(base_model, position)

    return base_model


def load_trained_model(position: str, model_path: str, input_size: int) -> BaseNeuralModel:
    """Load a trained model from disk."""
    model = create_model(position)
    model.load_model(model_path, input_size)
    return model
