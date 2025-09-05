"""Training utilities for DFS neural network models.

Provides common training loops, early stopping, and optimization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional, Any, Callable, List
from pathlib import Path

from helpers import log_model_metrics
from metrics import calculate_metrics_suite, print_metrics_report

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler with patience and delta threshold."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0001,
        mode: str = "min",
        restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped = False

    def __call__(
        self,
        score: float,
        model: nn.Module,
        epoch: int
    ) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score
            model: Model to save weights from
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best:
                self.best_weights = model.state_dict().copy()
            return False

        # Check if improved
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                logger.info(
                    f"Early stopping triggered. Best score: {self.best_score:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True

        return False


class ModelTrainer:
    """Generic trainer for neural network models with common training logic."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        scheduler_patience: int = 10,
        gradient_clip_val: float = 1.0,
        use_amp: bool = False,
        verbose: bool = True
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_patience = scheduler_patience
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp
        self.verbose = verbose

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6
        )

        # Setup AMP if requested
        # Use the new torch.amp API instead of deprecated torch.cuda.amp
        if use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }

    def create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        sample_weights: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays.

        Args:
            X: Feature array
            y: Target array
            shuffle: Whether to shuffle data
            sample_weights: Optional sample weights

        Returns:
            DataLoader instance
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        if sample_weights is not None:
            weights_tensor = torch.FloatTensor(sample_weights)
            dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid CUDA issues
            pin_memory=self.device.type == "cuda"
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Optional[Callable] = None,
        position: Optional[str] = None
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            loss_fn: Loss function (defaults to MSE)
            position: Optional position name for logging

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        for batch in train_loader:
            if len(batch) == 3:
                X_batch, y_batch, weights_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = None

            # Forward pass
            if self.use_amp and self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = self.model(X_batch)
                    if isinstance(outputs, dict):
                        # Multi-head output
                        loss = self._compute_multi_head_loss(
                            outputs, y_batch, loss_fn, weights_batch
                        )
                    else:
                        # Ensure consistent shapes - flatten both to 1D
                        loss = loss_fn(outputs.view(-1), y_batch.view(-1))
                        if weights_batch is not None:
                            loss = (loss * weights_batch).mean()
            else:
                outputs = self.model(X_batch)
                if isinstance(outputs, dict):
                    loss = self._compute_multi_head_loss(
                        outputs, y_batch, loss_fn, weights_batch
                    )
                else:
                    # Ensure consistent shapes - flatten both to 1D
                    loss = loss_fn(outputs.view(-1), y_batch.view(-1))
                    if weights_batch is not None:
                        loss = (loss * weights_batch).mean()

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: Optional[Callable] = None,
        position: Optional[str] = None,
        return_predictions: bool = False
    ) -> Tuple[float, Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Validate the model.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function (defaults to MSE)
            position: Optional position name for metrics
            return_predictions: Whether to return predictions

        Returns:
            Tuple of (validation loss, metrics dict, optional predictions)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch
                else:
                    X_batch, y_batch = batch

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)

                if isinstance(outputs, dict):
                    # Multi-head output - use mean for validation
                    predictions = outputs["mean"]
                    loss = loss_fn(predictions.view(-1), y_batch.view(-1))
                else:
                    predictions = outputs
                    loss = loss_fn(predictions.view(-1), y_batch.view(-1))

                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(predictions.view(-1).cpu().numpy())
                all_targets.extend(y_batch.view(-1).cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        metrics = calculate_metrics_suite(all_targets, all_preds, position)

        avg_loss = total_loss / max(num_batches, 1)

        if return_predictions:
            return avg_loss, metrics, (all_targets, all_preds)
        else:
            return avg_loss, metrics, None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        loss_fn: Optional[Callable] = None,
        sample_weights: Optional[np.ndarray] = None,
        position: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full training loop with validation and early stopping.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            loss_fn: Optional custom loss function
            sample_weights: Optional sample weights for training
            position: Optional position name for logging

        Returns:
            Dictionary with training results and history
        """
        # Create data loaders
        train_loader = self.create_data_loader(
            X_train, y_train, shuffle=True, sample_weights=sample_weights
        )
        val_loader = self.create_data_loader(
            X_val, y_val, shuffle=False
        )

        # Setup best model tracking (without early stopping)
        best_val_loss = float("inf")
        best_metrics = {}
        best_epoch = 0
        best_model_state = None

        # Training loop - run all epochs
        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, loss_fn, position)

            # Validate
            val_loss, val_metrics, _ = self.validate(val_loader, loss_fn, position)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            # Track best model (save state when we find a better one)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()
                if self.verbose:
                    logger.info(f"New best model found at epoch {epoch} with val_loss={val_loss:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            if self.verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val R²: {val_metrics['r2']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

        # Restore best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.verbose:
                logger.info(f"Training completed. Best model from epoch {best_epoch} restored.")

        # Final validation to get best metrics
        _, final_metrics, _ = self.validate(val_loader, loss_fn, position)

        return {
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "history": self.history,
            "epochs_trained": len(self.history["train_loss"]),
            "best_epoch": best_epoch,
            "early_stopped": False,  # No early stopping anymore
            "final_learning_rate": self.optimizer.param_groups[0]["lr"]
        }

    def _compute_multi_head_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        base_loss_fn: Callable,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for multi-head outputs.

        Args:
            outputs: Dictionary of output tensors
            targets: Target tensor
            base_loss_fn: Base loss function to use
            weights: Optional sample weights

        Returns:
            Combined loss
        """
        # Primary loss on mean prediction - ensure consistent shapes
        mean_pred = outputs["mean"].view(-1)
        targets_flat = targets.view(-1)

        mean_loss = base_loss_fn(mean_pred, targets_flat)
        if weights is not None:
            mean_loss = (mean_loss * weights.view(-1)).mean()

        total_loss = mean_loss

        # Add auxiliary losses if present
        if "std" in outputs:
            # Negative log likelihood loss for uncertainty
            std_pred = outputs["std"].view(-1)
            nll_loss = 0.5 * (
                torch.log(std_pred ** 2 + 1e-6) +
                (targets_flat - mean_pred) ** 2 / (std_pred ** 2 + 1e-6)
            ).mean()
            total_loss = total_loss + 0.1 * nll_loss

        if "ceiling" in outputs:
            # Ceiling should be higher than mean
            ceiling_pred = outputs["ceiling"].view(-1)
            ceiling_loss = F.relu(mean_pred - ceiling_pred).mean()
            total_loss = total_loss + 0.05 * ceiling_loss

        if "floor" in outputs:
            # Floor should be lower than mean
            floor_pred = outputs["floor"].view(-1)
            floor_loss = F.relu(floor_pred - mean_pred).mean()
            total_loss = total_loss + 0.05 * floor_loss

        return total_loss


class LearningRateFinder:
    """Find optimal learning rate using the LR range test."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Save initial state
        self.initial_model_state = model.state_dict().copy()
        self.initial_optimizer_state = optimizer.state_dict().copy()

    def find(
        self,
        train_loader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10,
        num_iterations: int = 100
    ) -> float:
        """Find optimal learning rate.

        Args:
            train_loader: Training data loader
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iterations: Number of iterations

        Returns:
            Suggested optimal learning rate
        """
        self.model.train()

        # Learning rate schedule
        lr_schedule = np.logspace(
            np.log10(min_lr),
            np.log10(max_lr),
            num_iterations
        )

        losses = []
        lrs = []

        iterator = iter(train_loader)

        for i, lr in enumerate(lr_schedule):
            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            if len(batch) == 3:
                X_batch, y_batch, _ = batch
            else:
                X_batch, y_batch = batch

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            outputs = self.model(X_batch)
            if isinstance(outputs, dict):
                outputs = outputs["mean"]

            loss = F.mse_loss(outputs.view(-1), y_batch.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record
            losses.append(loss.item())
            lrs.append(lr)

            # Stop if loss explodes
            if loss.item() > min(losses) * 4:
                break

        # Restore initial state
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)

        # Find suggested LR (steepest gradient)
        if len(losses) > 10:
            gradients = np.gradient(losses)
            min_gradient_idx = np.argmin(gradients[:len(gradients)//2])
            suggested_lr = lrs[min_gradient_idx]
        else:
            suggested_lr = min_lr * 10

        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

        return suggested_lr
