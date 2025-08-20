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

logger = logging.getLogger(__name__)

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
        self.criterion = nn.MSELoss()

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        self.patience = 15

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def build_network(self, input_size: int) -> nn.Module:
        """Build position-specific neural network architecture."""
        pass

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
        pin_memory = self.device.type == "cuda"  # Only useful for CUDA

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

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train model for one epoch."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            # Check for NaN in input data
            if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                logger.warning("NaN detected in batch input, skipping...")
                continue

            self.optimizer.zero_grad()
            predictions = self.network(batch_X)

            if predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)

            # Check for NaN in predictions
            if torch.isnan(predictions).any():
                logger.warning("NaN detected in predictions, skipping batch...")
                continue

            loss = self.criterion(predictions, batch_y)

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch...")
                continue

            loss.backward()
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

            # Check for NaN gradients
            has_nan_grad = False
            for param in self.network.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                logger.warning("NaN gradients detected, skipping update...")
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

                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)

                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the neural network model."""
        start_time = time.time()

        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Additional data cleaning before training
        logger.info("Performing final data cleaning before model training...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for extreme values and clip them
        X_train = np.clip(X_train, -1000, 1000)
        y_train = np.clip(y_train, -10, 50)  # DST fantasy points range
        X_val = np.clip(X_val, -1000, 1000)
        y_val = np.clip(y_val, -10, 50)

        # Feature scaling for stability
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0) + 1e-8  # Add epsilon to prevent division by zero
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        if self.network is None:
            input_size = X_train.shape[1]
            self.network = self.build_network(input_size)
            self.network.to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        logger.info(f"Starting neural network training for {self.config.position}")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_state_dict = self.network.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if hasattr(self, "best_state_dict"):
            self.network.load_state_dict(self.best_state_dict)

        # Calculate final metrics
        self.network.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)

            train_pred_tensor = self.network(X_train_tensor)
            val_pred_tensor = self.network(X_val_tensor)

            if train_pred_tensor.dim() > 1 and train_pred_tensor.size(1) == 1:
                train_pred_tensor = train_pred_tensor.squeeze(1)
            if val_pred_tensor.dim() > 1 and val_pred_tensor.size(1) == 1:
                val_pred_tensor = val_pred_tensor.squeeze(1)

            train_pred = train_pred_tensor.cpu().numpy()
            val_pred = val_pred_tensor.cpu().numpy()

        # Check for NaN predictions and handle gracefully
        if np.isnan(train_pred).any() or np.isnan(val_pred).any():
            logger.error("NaN predictions detected after training!")
            train_mae = val_mae = float('nan')
            train_rmse = val_rmse = float('nan')
            train_r2 = val_r2 = float('nan')
        else:
            train_mae = np.mean(np.abs(y_train - train_pred))
            val_mae = np.mean(np.abs(y_val - val_pred))
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

            # Calculate R² with safeguards against division by zero
            train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

            if train_ss_tot == 0:
                train_r2 = 0.0  # If no variance in target, R² is undefined, set to 0
            else:
                train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / train_ss_tot

            if val_ss_tot == 0:
                val_r2 = 0.0  # If no variance in target, R² is undefined, set to 0
            else:
                val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / val_ss_tot

        self._residual_std = np.std(y_val - val_pred)
        self.is_trained = True
        training_time = time.time() - start_time

        result = TrainingResult(
            model=self.network,
            training_time=training_time,
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
            feature_count=X_train.shape[1],
        )

        self.training_history.append(result.__dict__)
        logger.info(f"Training completed: MAE={val_mae:.3f}, R�={val_r2:.3f}")

        return result

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with uncertainty quantification."""
        if not self.is_trained or self.network is None:
            raise ValueError("Model must be trained before making predictions")

        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.network(X_tensor)
            if predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)

        point_estimate = predictions.cpu().numpy()

        # Calculate prediction intervals
        uncertainty = (
            self._residual_std if hasattr(self, "_residual_std") else point_estimate * 0.25
        )

        lower_bound = point_estimate - 1.96 * uncertainty
        upper_bound = point_estimate + 1.96 * uncertainty
        floor = point_estimate - 0.8 * uncertainty
        ceiling = point_estimate + 1.0 * uncertainty
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


class QBNetwork(nn.Module):
    """Neural network architecture for quarterback predictions, optimized for Apple Silicon."""

    def __init__(self, input_size: int):
        super().__init__()

        # Use BatchNorm instead of LayerNorm for better stability with sparse features
        # Set eps higher to prevent division by zero
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128, eps=1e-3),  # Higher eps for stability
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.passing_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rushing_branch = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.output = nn.Sequential(
            nn.Linear(32 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add small epsilon to prevent issues with sparse features
        x = x + 1e-8

        shared_features = self.feature_layers(x)

        passing_features = self.passing_branch(shared_features)
        rushing_features = self.rushing_branch(shared_features)
        combined = torch.cat([passing_features, rushing_features], dim=1)

        output = self.output(combined)
        # Use direct linear output - let the network learn the proper scaling
        return output.squeeze(-1)


class RBNetwork(nn.Module):
    """Neural network architecture for running back predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Optimized for Apple Silicon MPS
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.LayerNorm(96),  # LayerNorm works better than BatchNorm on MPS
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.workload_branch = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.15))
        self.efficiency_branch = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.1))

        self.output = nn.Sequential(
            nn.Linear(24 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        workload_features = self.workload_branch(features)
        efficiency_features = self.efficiency_branch(features)
        combined = torch.cat([workload_features, efficiency_features], dim=1)
        output = self.output(combined)
        scaled_output = output * POSITION_RANGES['RB']
        return scaled_output


class WRNetwork(nn.Module):
    """Neural network architecture for wide receiver predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        self.main_layers = nn.Sequential(
            nn.Linear(input_size, 112),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(112, 56),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(56, 28),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.target_branch = nn.Sequential(nn.Linear(28, 16), nn.ReLU(), nn.Dropout(0.15))
        self.bigplay_branch = nn.Sequential(nn.Linear(28, 12), nn.ReLU(), nn.Dropout(0.1))

        self.output = nn.Sequential(
            nn.Linear(16 + 12, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_features = self.main_layers(x)
        target_features = self.target_branch(main_features)
        bigplay_features = self.bigplay_branch(main_features)
        combined = torch.cat([target_features, bigplay_features], dim=1)
        output = self.output(combined)
        scaled_output = output * POSITION_RANGES['WR']
        return scaled_output


class TENetwork(nn.Module):
    """Neural network architecture for tight end predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        # Optimized for Apple Silicon MPS
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.LayerNorm(80),  # LayerNorm works better than BatchNorm on MPS
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(40, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.output = nn.Sequential(
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        output = self.output(features)
        scaled_output = output * POSITION_RANGES['TE']
        return scaled_output


class DEFNetwork(nn.Module):
    """Neural network architecture for defense predictions."""

    def __init__(self, input_size: int):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.pressure_branch = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2))
        self.turnover_branch = nn.Sequential(nn.Linear(32, 12), nn.ReLU(), nn.Dropout(0.2))
        self.points_branch = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Dropout(0.15))

        self.output = nn.Sequential(
            nn.Linear(16 + 12 + 8, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_layers(x)
        pressure_features = self.pressure_branch(shared)
        turnover_features = self.turnover_branch(shared)
        points_features = self.points_branch(shared)
        combined = torch.cat([pressure_features, turnover_features, points_features], dim=1)
        output = self.output(combined)
        scaled_output = output * POSITION_RANGES['DEF']
        return scaled_output


class QBNeuralModel(BaseNeuralModel):
    """Neural network model for quarterback predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.00015
        self.batch_size = 50
        self.epochs = 200

    def build_network(self, input_size: int) -> nn.Module:
        return QBNetwork(input_size)


class RBNeuralModel(BaseNeuralModel):
    """Neural network model for running back predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.00001
        self.batch_size = 50
        self.epochs = 250

    def build_network(self, input_size: int) -> nn.Module:
        return RBNetwork(input_size)


class WRNeuralModel(BaseNeuralModel):
    """Neural network model for wide receiver predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0001
        self.batch_size = 50
        self.epochs = 200

    def build_network(self, input_size: int) -> nn.Module:
        return WRNetwork(input_size)


class TENeuralModel(BaseNeuralModel):
    """Neural network model for tight end predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.000
        self.batch_size = 50
        self.epochs = 200

    def build_network(self, input_size: int) -> nn.Module:
        return TENetwork(input_size)


class DEFNeuralModel(BaseNeuralModel):
    """Neural network model for defense predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Optimized for smaller DST dataset (~1K samples)
        self.learning_rate = 0.001  # Lower learning rate for stability
        self.batch_size = 16        # Smaller batch size for small dataset
        self.epochs = 200           # More epochs since dataset is small
        self.patience = 25          # More patience for small dataset

    def build_network(self, input_size: int) -> nn.Module:
        return DEFNetwork(input_size)


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


def create_model(position: str, config: ModelConfig = None) -> BaseNeuralModel:
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

    return models[position](config)


def load_trained_model(position: str, model_path: str, input_size: int) -> BaseNeuralModel:
    """Load a trained model from disk."""
    model = create_model(position)
    model.load_model(model_path, input_size)
    return model
