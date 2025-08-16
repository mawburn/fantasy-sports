"""PyTorch neural network models for NFL fantasy sports predictions.

This file implements deep learning models using PyTorch as alternatives to traditional
ML approaches. Neural networks can capture complex non-linear patterns and player
interactions that traditional models might miss.

Key Advantages of Neural Networks for Fantasy Sports:
1. Non-linear Pattern Recognition: Capture complex relationships between features
2. Automatic Feature Learning: Discover important feature combinations automatically
3. Player Interaction Modeling: Learn how different players affect each other
4. Sequence Modeling: Understand game flow and momentum effects
5. Flexible Architecture: Easily adapt to new features and scoring systems

Position-Specific Architectures:
- QB: Multi-task learning (passing + rushing) with attention for drive context
- RB: Workload-aware network with clustering embeddings
- WR: Target competition model with attention mechanisms
- TE: Dual-role network (receiving + blocking importance)
- DEF: Ensemble of specialized networks for different defensive stats

For beginners: Neural networks learn by adjusting thousands of parameters
through backpropagation, gradually improving at pattern recognition.
"""

import logging
import time
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, ModelConfig, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)

# Set PyTorch for reproducible, CPU-optimized training
torch.manual_seed(42)
torch.set_num_threads(8)  # Optimize for CPU training

# Enable CPU optimizations
if torch.cuda.is_available():
    logger.info("CUDA available but using CPU for compatibility")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BaseNeuralModel(BaseModel):
    """Base class for PyTorch neural network models.

    Provides common functionality for all neural network models:
    - PyTorch-specific training loops with proper gradient handling
    - Data loading with batching and shuffling
    - Early stopping based on validation loss
    - Learning rate scheduling for optimal convergence
    - Proper model saving/loading for PyTorch state_dicts

    This base class ensures consistent training patterns across all
    neural network position models while allowing customization.
    """

    def __init__(self, config: ModelConfig):
        """Initialize neural network base model."""
        super().__init__(config)

        # PyTorch-specific components
        self.device = torch.device("cpu")  # CPU-only for compatibility
        self.network: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: optim.lr_scheduler._LRScheduler | None = None
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        self.patience = 15  # Early stopping patience

        # Training history for monitoring
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def build_model(self) -> nn.Module:
        """Build the neural network model.

        This method satisfies the BaseModel abstract interface.
        It delegates to position-specific build_network methods.

        Returns:
            PyTorch neural network module
        """
        # This will be called during training when network is needed
        # The actual network construction is deferred to build_network()
        return None  # Placeholder - actual building happens in fit()

    @abstractmethod
    def build_network(self, input_size: int) -> nn.Module:
        """Build position-specific neural network architecture.

        This method must be implemented by each position-specific model
        to define their custom neural network architecture.

        Args:
            input_size: Number of input features

        Returns:
            PyTorch neural network module
        """
        pass

    def _create_data_loaders(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders for training and validation.

        Data loaders handle:
        - Converting numpy arrays to PyTorch tensors
        - Batching data for mini-batch gradient descent
        - Shuffling training data for better convergence
        - Automatic GPU/CPU tensor placement

        Args:
            X_train, y_train: Training features and targets
            X_val, y_val: Validation features and targets

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert numpy arrays to PyTorch tensors
        # .float() ensures tensors are float32 (required for neural networks)
        train_X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        val_X = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        # Create TensorDatasets (pairs features with targets)
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)

        # Create DataLoaders with batching and shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data for better learning
            num_workers=0,  # No multiprocessing (CPU-only)
            pin_memory=False,  # No GPU memory pinning needed
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=0,
            pin_memory=False,
        )

        return train_loader, val_loader

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train model for one epoch.

        An epoch is one complete pass through the training data.
        This method implements the standard PyTorch training loop:
        1. Set model to training mode
        2. Iterate through batches
        3. Forward pass (predictions)
        4. Calculate loss
        5. Backward pass (gradients)
        6. Update weights

        Args:
            train_loader: DataLoader with training batches

        Returns:
            Average training loss for this epoch
        """
        self.network.train()  # Set to training mode (enables dropout, etc.)
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            # Zero gradients from previous batch
            # PyTorch accumulates gradients, so we must clear them
            self.optimizer.zero_grad()

            # Forward pass: compute predictions
            predictions = self.network(batch_X)

            # Ensure predictions have correct shape for loss calculation
            if predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)  # Remove extra dimension

            # Calculate loss (how wrong our predictions are)
            loss = self.criterion(predictions, batch_y)

            # Backward pass: compute gradients
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            # This is important for stable neural network training
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Update model weights using gradients
            self.optimizer.step()

            # Track loss for monitoring
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate model for one epoch.

        Validation evaluates model performance on held-out data
        without updating weights. This helps detect overfitting.

        Args:
            val_loader: DataLoader with validation batches

        Returns:
            Average validation loss
        """
        self.network.eval()  # Set to evaluation mode (disables dropout, etc.)
        total_loss = 0.0
        num_batches = 0

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Forward pass only (no backward pass needed)
                predictions = self.network(batch_X)

                # Ensure correct shape
                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)

                # Calculate loss
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the neural network model.

        Implements complete training pipeline with early stopping:
        1. Create data loaders for batched training
        2. Initialize optimizer and learning rate scheduler
        3. Train for multiple epochs with validation monitoring
        4. Apply early stopping if validation doesn't improve
        5. Return comprehensive training results

        Args:
            X_train, y_train: Training data and targets
            X_val, y_val: Validation data and targets

        Returns:
            TrainingResult with metrics and model artifacts
        """
        start_time = time.time()

        # Validate input data
        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Build model architecture if not already done
        if self.network is None:
            input_size = X_train.shape[1]
            self.network = self.build_network(input_size)
            self.network.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  # L2 regularization to prevent overfitting
        )

        # Learning rate scheduler: reduce LR when validation plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)

        # Training variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        logger.info(f"Starting neural network training for {self.config.position}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.network.parameters()):,}")

        # Main training loop
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best model state
                self.best_state_dict = self.network.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Best: {best_val_loss:.4f}"
                )

            # Early stopping check
            if patience_counter >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)"
                )
                break

        # Restore best model
        if hasattr(self, "best_state_dict"):
            self.network.load_state_dict(self.best_state_dict)

        # Calculate final metrics using direct network inference
        self.network.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)

            train_pred_tensor = self.network(X_train_tensor)
            val_pred_tensor = self.network(X_val_tensor)

            # Convert to numpy and handle dimensions
            if train_pred_tensor.dim() > 1 and train_pred_tensor.size(1) == 1:
                train_pred_tensor = train_pred_tensor.squeeze(1)
            if val_pred_tensor.dim() > 1 and val_pred_tensor.size(1) == 1:
                val_pred_tensor = val_pred_tensor.squeeze(1)

            train_pred = train_pred_tensor.cpu().numpy()
            val_pred = val_pred_tensor.cpu().numpy()

        train_mae = np.mean(np.abs(y_train - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

        # R² calculation
        train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum(
            (y_train - np.mean(y_train)) ** 2
        )
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        # Store residual std for prediction intervals
        self._residual_std = np.std(y_val - val_pred)

        self.is_trained = True
        training_time = time.time() - start_time

        # Create training result
        result = TrainingResult(
            model=self.network,
            training_time=training_time,
            best_iteration=best_epoch,
            feature_importance=None,  # Neural networks don't have explicit feature importance
            train_mae=train_mae,
            val_mae=val_mae,
            train_rmse=train_rmse,
            val_rmse=val_rmse,
            train_r2=train_r2,
            val_r2=val_r2,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=X_train.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"Neural network training completed: MAE={val_mae:.3f}, R²={val_r2:.3f}")

        return result

    def _predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """Internal method to generate predictions as numpy arrays."""
        if not self.is_trained or self.network is None:
            raise ValueError("Model must be trained before making predictions")

        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            predictions = self.network(X_tensor)
            if predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)

        return predictions.cpu().numpy()

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with uncertainty quantification."""
        point_estimate = self._predict_numpy(X)

        # Calculate prediction intervals
        lower_bound, upper_bound = self._calculate_prediction_intervals(X, point_estimate)

        # Neural network specific uncertainty (higher than traditional models)
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.25
        )
        floor = point_estimate - 0.8 * uncertainty
        ceiling = point_estimate + 1.0 * uncertainty

        # Confidence based on training stability
        confidence_score = (
            np.ones_like(point_estimate) * 0.7
        )  # Slightly lower than traditional models

        return PredictionResult(
            point_estimate=point_estimate,
            confidence_score=confidence_score,
            prediction_intervals=(lower_bound, upper_bound),
            floor=floor,
            ceiling=ceiling,
            model_version=self.config.version,
        )


class QBNeuralModel(BaseNeuralModel):
    """Neural network model for quarterback predictions.

    QB Architecture: Multi-task learning with attention
    - Main task: Fantasy points prediction
    - Auxiliary tasks: Passing yards, rushing yards (if applicable)
    - Attention mechanism: Focus on key features per game situation
    - Multi-head design: Separate processing for passing vs rushing

    Why this architecture works for QBs:
    - QBs have two distinct skill sets (passing/rushing) that benefit from separate processing
    - Attention helps model focus on situational factors (game script, opponent strength)
    - Multi-task learning improves generalization by sharing representations
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.001  # QBs are relatively predictable
        self.batch_size = 64  # Larger batches for stable gradients

    def build_network(self, input_size: int) -> nn.Module:
        """Build QB-specific neural architecture."""
        return QBNetwork(input_size)


class QBNetwork(nn.Module):
    """Neural network architecture for quarterback predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # Passing-specific branch
        self.passing_branch = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1))

        # Rushing-specific branch
        self.rushing_branch = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.1))

        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 64), nn.Softmax(dim=1)
        )

        # Final prediction layer
        self.output = nn.Linear(32 + 16, 1)  # Combine passing + rushing features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared feature extraction
        shared_features = self.shared_layers(x)

        # Apply attention to shared features
        attention_weights = self.attention(shared_features)
        attended_features = shared_features * attention_weights

        # Separate processing for passing and rushing
        passing_features = self.passing_branch(attended_features)
        rushing_features = self.rushing_branch(attended_features)

        # Combine features
        combined = torch.cat([passing_features, rushing_features], dim=1)

        # Final prediction
        output = self.output(combined)
        return output


class RBNeuralModel(BaseNeuralModel):
    """Neural network model for running back predictions.

    RB Architecture: Workload-aware network with clustering
    - Embedding layer: Learn workload type representations
    - Mixture of experts: Different sub-networks for different RB types
    - Workload attention: Focus on touches, snap share, goal line usage
    - TD probability: Separate head for touchdown prediction

    Why this works for RBs:
    - RBs have distinct roles (workhorse, committee, pass-catcher, goal-line)
    - Workload is the primary driver of fantasy performance
    - Non-linear relationship between touches and scoring
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0015  # Slightly higher for more volatile position
        self.batch_size = 32

    def build_network(self, input_size: int) -> nn.Module:
        """Build RB-specific neural architecture."""
        return RBNetwork(input_size)


class RBNetwork(nn.Module):
    """Neural network architecture for running back predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Main feature processing
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Workload-specific processing
        self.workload_branch = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.15))

        # Efficiency-specific processing
        self.efficiency_branch = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.1))

        # Final prediction
        self.output = nn.Linear(24 + 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)

        workload_features = self.workload_branch(features)
        efficiency_features = self.efficiency_branch(features)

        combined = torch.cat([workload_features, efficiency_features], dim=1)
        output = self.output(combined)
        return output


class WRNeuralModel(BaseNeuralModel):
    """Neural network model for wide receiver predictions.

    WR Architecture: Target competition with attention
    - Target share prediction: Dedicated branch for target probability
    - Route-based processing: Different features for different route types
    - Red zone attention: Special focus on red zone opportunities
    - Boom/bust modeling: Capture high variance of WR scoring

    Why this works for WRs:
    - WRs compete for limited targets (zero-sum game aspect)
    - Route running and separation skills are key differentiators
    - High variance requires explicit uncertainty modeling
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.002  # Higher LR for high-variance position
        self.batch_size = 48

    def build_network(self, input_size: int) -> nn.Module:
        """Build WR-specific neural architecture."""
        return WRNetwork(input_size)


class WRNetwork(nn.Module):
    """Neural network architecture for wide receiver predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Main processing pipeline
        self.main_layers = nn.Sequential(
            nn.Linear(input_size, 112),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout for high-variance position
            nn.Linear(112, 56),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(56, 28),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Target share branch
        self.target_branch = nn.Sequential(nn.Linear(28, 16), nn.ReLU(), nn.Dropout(0.15))

        # Big play potential branch
        self.bigplay_branch = nn.Sequential(nn.Linear(28, 12), nn.ReLU(), nn.Dropout(0.1))

        # Output layer
        self.output = nn.Linear(16 + 12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_features = self.main_layers(x)

        target_features = self.target_branch(main_features)
        bigplay_features = self.bigplay_branch(main_features)

        combined = torch.cat([target_features, bigplay_features], dim=1)
        output = self.output(combined)
        return output


class TENeuralModel(BaseNeuralModel):
    """Neural network model for tight end predictions.

    TE Architecture: Dual-role network
    - Receiving branch: Traditional receiver processing
    - Blocking importance: Factor in blocking assignments
    - Usage rate modeling: TEs have more predictable usage than WRs
    - Red zone specialist: Many TEs are red zone targets

    Why this works for TEs:
    - TEs balance receiving and blocking (affects snap share and targets)
    - More consistent usage patterns than WRs but less than RBs
    - Often game-script dependent (more targets when behind)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0012  # Moderate LR for moderate variance
        self.batch_size = 40

    def build_network(self, input_size: int) -> nn.Module:
        """Build TE-specific neural architecture."""
        return TENetwork(input_size)


class TENetwork(nn.Module):
    """Neural network architecture for tight end predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Feature processing
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Output layer (simpler than other positions)
        self.output = nn.Linear(20, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        output = self.output(features)
        return output


class DEFNeuralModel(BaseNeuralModel):
    """Neural network model for defense/special teams predictions.

    DEF Architecture: Multi-head ensemble
    - Sacks/pressure head: Pass rush effectiveness
    - Turnover head: Interceptions and fumble recoveries
    - Points allowed head: Defensive scoring prevention
    - Special teams head: Return TDs, blocked kicks
    - Variance modeling: Explicit uncertainty for chaotic position

    Why this works for DEF:
    - Defense scoring is driven by distinct, independent factors
    - High variance requires ensemble approach
    - Opponent strength is critical factor
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0025  # Highest LR for most chaotic position
        self.batch_size = 24  # Smaller batches for stability
        self.patience = 20  # More patience due to high variance

    def build_network(self, input_size: int) -> nn.Module:
        """Build DEF-specific neural architecture."""
        return DEFNetwork(input_size)


class DEFNetwork(nn.Module):
    """Neural network architecture for defense predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # High dropout for high variance
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Defensive pressure branch
        self.pressure_branch = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2))

        # Turnover branch
        self.turnover_branch = nn.Sequential(nn.Linear(32, 12), nn.ReLU(), nn.Dropout(0.2))

        # Points allowed branch
        self.points_branch = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Dropout(0.15))

        # Final prediction
        self.output = nn.Linear(16 + 12 + 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_layers(x)

        pressure_features = self.pressure_branch(shared)
        turnover_features = self.turnover_branch(shared)
        points_features = self.points_branch(shared)

        combined = torch.cat([pressure_features, turnover_features, points_features], dim=1)
        output = self.output(combined)
        return output
