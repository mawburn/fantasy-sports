"""Data utilities for DFS models.

Handles data loading, batch optimization, and dataset management.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DFSDataset(Dataset):
    """Custom dataset for DFS predictions with optional sample weights."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        player_ids: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None
    ):
        """Initialize DFS dataset.

        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,)
            sample_weights: Optional sample weights (n_samples,)
            player_ids: Optional player IDs for tracking
            metadata: Optional dictionary of additional arrays
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        self.sample_weights = (
            torch.FloatTensor(sample_weights)
            if sample_weights is not None
            else None
        )

        self.player_ids = player_ids
        self.metadata = metadata or {}

        # Validate shapes
        assert len(self.features) == len(self.targets), (
            f"Feature/target length mismatch: {len(self.features)} != {len(self.targets)}"
        )

        if self.sample_weights is not None:
            assert len(self.sample_weights) == len(self.features), (
                "Sample weights length mismatch"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.sample_weights is not None:
            return (
                self.features[idx],
                self.targets[idx],
                self.sample_weights[idx]
            )
        else:
            return self.features[idx], self.targets[idx]


class BatchSizeOptimizer:
    """Optimize batch size for GPU memory efficiency."""

    def __init__(self, device: torch.device):
        self.device = device

    def find_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        min_batch: int = 8,
        max_batch: int = 512,
        safety_factor: float = 0.9
    ) -> int:
        """Find the maximum batch size that fits in memory.

        Args:
            model: Model to test
            sample_input: Sample input tensor
            min_batch: Minimum batch size to test
            max_batch: Maximum batch size to test
            safety_factor: Memory safety factor (0-1)

        Returns:
            Optimal batch size
        """
        if self.device.type != "cuda":
            # For CPU, use a reasonable default
            return 32

        model.to(self.device)
        model.eval()

        # Binary search for maximum batch size
        left, right = min_batch, max_batch
        optimal = min_batch

        while left <= right:
            mid = (left + right) // 2

            # Create test batch
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                test_batch = sample_input.unsqueeze(0).repeat(mid, 1).to(self.device)

                # Forward pass
                with torch.no_grad():
                    _ = model(test_batch)

                # If successful, try larger batch
                optimal = mid
                left = mid + 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Too large, try smaller
                    right = mid - 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

        # Apply safety factor
        final_batch_size = int(optimal * safety_factor)
        final_batch_size = max(min_batch, final_batch_size)

        logger.info(f"Optimal batch size: {final_batch_size} (device: {self.device})")

        return final_batch_size

    def optimize_for_dataset(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        target_batches: int = 100
    ) -> int:
        """Optimize batch size for a specific dataset size.

        Args:
            model: Model to optimize for
            dataset: Dataset to process
            target_batches: Target number of batches (for efficiency)

        Returns:
            Optimal batch size
        """
        dataset_size = len(dataset)

        # Get a sample for memory testing
        sample_input, _ = dataset[0] if len(dataset) > 0 else (None, None)

        if sample_input is None:
            return 32

        # Find memory-limited batch size
        memory_batch_size = self.find_optimal_batch_size(
            model, sample_input
        )

        # Calculate ideal batch size for target number of batches
        ideal_batch_size = max(1, dataset_size // target_batches)

        # Use smaller of memory-limited and ideal
        optimal = min(memory_batch_size, ideal_batch_size)

        # Ensure reasonable bounds
        optimal = max(8, min(optimal, 256))

        return optimal


class DataManager:
    """Centralized data management for DFS models."""

    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_workers = num_workers
        self.pin_memory = pin_memory and self.device.type == "cuda"

        self.batch_optimizer = BatchSizeOptimizer(self.device)

    def create_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        optimize_batch_size: bool = False,
        model: Optional[torch.nn.Module] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            sample_weights: Optional training sample weights
            optimize_batch_size: Whether to optimize batch size
            model: Model for batch size optimization

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = DFSDataset(X_train, y_train, sample_weights)
        val_dataset = DFSDataset(X_val, y_val)

        # Optimize batch size if requested
        batch_size = self.batch_size
        if optimize_batch_size and model is not None:
            batch_size = self.batch_optimizer.optimize_for_dataset(
                model, train_dataset
            )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        logger.info(
            f"Created data loaders - Train: {len(train_dataset)} samples, "
            f"Val: {len(val_dataset)} samples, Batch size: {batch_size}"
        )

        return train_loader, val_loader

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.2,
        random_seed: Optional[int] = None,
        stratify: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets.

        Args:
            X: Feature array
            y: Target array
            val_ratio: Validation set ratio
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify split by target quantiles

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        from sklearn.model_selection import train_test_split

        if stratify:
            # Create stratification bins based on target quantiles
            n_bins = 10
            y_bins = np.digitize(
                y,
                bins=np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1])
            )
            stratify_array = y_bins
        else:
            stratify_array = None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=stratify_array
        )

        logger.info(
            f"Split data - Train: {len(X_train)}, Val: {len(X_val)} "
            f"(ratio: {val_ratio:.1%})"
        )

        return X_train, y_train, X_val, y_val

    def create_temporal_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create temporal train/validation splits for time series validation.

        Args:
            X: Feature array
            y: Target array
            timestamps: Timestamp array (e.g., season-week combinations)
            n_splits: Number of temporal splits

        Returns:
            List of (X_train, y_train, X_val, y_val) tuples
        """
        # Sort by timestamp
        sorted_idx = np.argsort(timestamps)
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        timestamps_sorted = timestamps[sorted_idx]

        # Unique timestamps for splitting
        unique_timestamps = np.unique(timestamps_sorted)
        n_timestamps = len(unique_timestamps)

        if n_timestamps < n_splits + 1:
            logger.warning(
                f"Not enough unique timestamps ({n_timestamps}) for {n_splits} splits"
            )
            n_splits = max(1, n_timestamps - 1)

        splits = []
        val_size = n_timestamps // (n_splits + 1)

        for i in range(n_splits):
            # Define validation window
            val_start_idx = (i + 1) * val_size
            val_end_idx = min(val_start_idx + val_size, n_timestamps)

            val_timestamps = unique_timestamps[val_start_idx:val_end_idx]

            # Create masks
            train_mask = timestamps_sorted < val_timestamps[0]
            val_mask = np.isin(timestamps_sorted, val_timestamps)

            if np.sum(train_mask) > 0 and np.sum(val_mask) > 0:
                splits.append((
                    X_sorted[train_mask],
                    y_sorted[train_mask],
                    X_sorted[val_mask],
                    y_sorted[val_mask]
                ))

        logger.info(f"Created {len(splits)} temporal splits")

        return splits

    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.01,
        augment_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data with noise for regularization.

        Args:
            X: Feature array
            y: Target array
            noise_level: Standard deviation of Gaussian noise
            augment_ratio: Ratio of augmented samples to add

        Returns:
            Tuple of augmented (X, y)
        """
        n_samples = len(X)
        n_augment = int(n_samples * augment_ratio)

        if n_augment == 0:
            return X, y

        # Select random samples to augment
        augment_idx = np.random.choice(n_samples, n_augment, replace=True)
        X_augment = X[augment_idx].copy()
        y_augment = y[augment_idx].copy()

        # Add noise to features
        noise = np.random.normal(0, noise_level, X_augment.shape)
        X_augment += noise * np.std(X, axis=0, keepdims=True)

        # Combine original and augmented
        X_combined = np.vstack([X, X_augment])
        y_combined = np.hstack([y, y_augment])

        logger.info(f"Augmented data from {n_samples} to {len(X_combined)} samples")

        return X_combined, y_combined

    def create_sample_weights(
        self,
        y: np.ndarray,
        method: str = "inverse_frequency",
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create sample weights for training.

        Args:
            y: Target array
            method: Weighting method ('inverse_frequency', 'temporal', 'uniform')
            timestamps: Optional timestamps for temporal weighting

        Returns:
            Sample weights array
        """
        n_samples = len(y)

        if method == "uniform":
            weights = np.ones(n_samples)

        elif method == "inverse_frequency":
            # Weight samples inversely to target frequency
            bins = np.percentile(y, np.linspace(0, 100, 11))
            bin_indices = np.digitize(y, bins[1:-1])

            # Count samples in each bin
            bin_counts = np.bincount(bin_indices, minlength=10)
            bin_weights = 1.0 / (bin_counts + 1)
            bin_weights /= bin_weights.mean()

            # Assign weights
            weights = bin_weights[bin_indices]

        elif method == "temporal" and timestamps is not None:
            # Weight recent samples more heavily
            time_rank = np.argsort(np.argsort(timestamps))
            weights = 1 + 0.5 * (time_rank / n_samples)

        else:
            weights = np.ones(n_samples)

        # Normalize weights
        weights = weights / weights.mean()

        return weights
