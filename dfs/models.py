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

logger = logging.getLogger(__name__)

# Import gradient boosting libraries for ensemble learning
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available.")

try:
    from catboost import CatBoostRegressor
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
        self.quantile_criterion = self._quantile_loss  # Custom quantile loss function
        self.mse_criterion = nn.MSELoss()  # For fallback if Huber fails

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 100
        self.patience = 15

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

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor, quantile: float) -> torch.Tensor:
        """Quantile loss (pinball loss) function."""
        errors = targets - predictions
        return torch.mean(torch.max(quantile * errors, (quantile - 1) * errors))

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
            self.optimizer.zero_grad()
            predictions = self.network(batch_X)

            # Handle QB model's dictionary output
            if isinstance(predictions, dict):
                predictions = predictions['mean']  # Use mean prediction for training
            elif predictions.dim() > 1 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)

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

            # ReduceLROnPlateau doesn't step per batch

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

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResult:
        """Train the neural network model."""
        start_time = time.time()

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
        best_val_r2 = -float("inf")  # Track best R² score
        best_epoch = 0

        logger.info(f"Starting neural network training for {self.config.position}")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            self.scheduler.step(val_loss)  # ReduceLROnPlateau takes loss as input

            # Compute validation R² every epoch for precise checkpointing
            val_r2 = self._compute_validation_r2(val_loader, y_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Always prioritize R² for model selection (computed every epoch now)
            improved = False
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = val_loss  # Update loss too
                best_epoch = epoch
                self.best_state_dict = self.network.state_dict().copy()
                improved = True
            elif val_r2 == best_val_r2 and val_loss < best_val_loss:
                # Same R², but better loss - still an improvement
                best_val_loss = val_loss
                best_epoch = epoch
                self.best_state_dict = self.network.state_dict().copy()
                improved = True

            # Clean progress reporting - only show when R² improves
            if improved:
                # Clear the line completely, then print new best
                print(f"\r\033[K🎯 Epoch {epoch}: New best R² = {val_r2:.4f}", end="", flush=True)
            elif epoch % 10 == 0:
                # Clear the line completely, then print progress
                print(f"\r\033[KEpoch {epoch}/{self.epochs}: R² = {val_r2:.4f} (Best: {best_val_r2:.4f} @ epoch {best_epoch})", end="", flush=True)

            # Skip the ProgressDisplay since we're handling output manually
            # progress.update(epoch, self.epochs)

        # Training completed - always run full epochs since we have checkpointing
        print(f"\nTraining completed ({self.epochs} epochs, Best R²: {best_val_r2:.4f} at epoch {best_epoch})")

        if hasattr(self, "best_state_dict"):
            self.network.load_state_dict(self.best_state_dict)

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
            # Handle QB model's dictionary output
            if isinstance(predictions, dict):
                predictions = predictions['mean']
            elif predictions.dim() > 1 and predictions.size(1) == 1:
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
    """Enhanced neural network architecture for quarterback predictions with residual connections and improved activations."""

    def __init__(self, input_size: int):
        super().__init__()

        # Moderately increased capacity for better representation learning
        self.input_projection = nn.Linear(input_size, 192)
        self.input_norm = nn.BatchNorm1d(192)

        # Enhanced feature layers with residual connections
        self.feature_layers = nn.ModuleList([
            self._residual_block(192, 160),
            self._residual_block(160, 128),
            self._residual_block(128, 96),
            self._residual_block(96, 64)
        ])

        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(64, num_heads=4, dropout=0.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)

        # Enhanced branches with residual connections
        self.passing_branch = nn.Sequential(
            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.rushing_branch = nn.Sequential(
            nn.Linear(64, 24),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Enhanced output heads
        combined_size = 32 + 16
        self.mean_head = nn.Sequential(
            nn.Linear(combined_size, 24),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 1)
        )

        # Quantile heads with shared layers for efficiency
        self.quantile_shared = nn.Sequential(
            nn.Linear(combined_size, 24),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.q25_head = nn.Linear(24, 1)
        self.q50_head = nn.Linear(24, 1)
        self.q75_head = nn.Linear(24, 1)

    def _residual_block(self, input_dim: int, output_dim: int):
        """Create a residual block with batch norm and GELU activation."""
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # Skip connection (project if dimensions don't match)
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        return block

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Add small epsilon to prevent issues with sparse features
        x = x + 1e-8

        # Input projection
        x = F.gelu(self.input_norm(self.input_projection(x)))

        # Apply residual blocks (skip connections handled inside blocks)
        for i, block in enumerate(self.feature_layers):
            x = block(x)
            x = F.gelu(x)

        # Apply attention mechanism
        x_attended, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = self.attention_norm(x + x_attended.squeeze(1))

        # Branch processing
        passing_features = self.passing_branch(x)
        rushing_features = self.rushing_branch(x)
        combined = torch.cat([passing_features, rushing_features], dim=1)

        # Output predictions
        mean_output = self.mean_head(combined).squeeze(-1)

        # Quantile outputs through shared layer
        quantile_features = self.quantile_shared(combined)

        return {
            'mean': mean_output,
            'q25': self.q25_head(quantile_features).squeeze(-1),
            'q50': self.q50_head(quantile_features).squeeze(-1),
            'q75': self.q75_head(quantile_features).squeeze(-1)
        }


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
        # Conservative optimizations for R² improvement
        self.learning_rate = 0.00003  # Lower learning rate for stability
        self.batch_size = 128         # Keep reasonable batch size
        self.epochs = 800            # More epochs for better results
        self.patience = 50           # Keep patience higher to avoid stopping too early

    def build_network(self, input_size: int) -> nn.Module:
        return QBNetwork(input_size)


class RBNeuralModel(BaseNeuralModel):
    """Enhanced neural network model for running back predictions with research-based optimizations."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # PHASE 4B: Research-optimized RB training parameters (REVERT TO WORKING VERSION)
        # Proven: R² = 0.3532, peaked at epoch 302/800
        self.learning_rate = 0.00003  # Back to working learning rate
        self.batch_size = 128         # Back to working batch size
        self.epochs = 800             # Back to working epoch count

    def build_network(self, input_size: int) -> nn.Module:
        return RBNetwork(input_size)


class WRNeuralModel(BaseNeuralModel):
    """Neural network model for wide receiver predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0001   # Doubled from 0.00005 for better learning
        self.batch_size = 64          # Smaller batch for more frequent updates
        self.epochs = 600             # 3x more epochs for better convergence

    def build_network(self, input_size: int) -> nn.Module:
        return WRNetwork(input_size)


class TENeuralModel(BaseNeuralModel):
    """Neural network model for tight end predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.learning_rate = 0.0001   # Fix: was too low at 0.00005
        self.batch_size = 64          # Smaller batch for more frequent updates
        self.epochs = 600             # More epochs for TE complexity

    def build_network(self, input_size: int) -> nn.Module:
        return TENetwork(input_size)



class DEFCatBoostModel:
    """CatBoost-only model for DST predictions - optimized for small, volatile datasets."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []

        if not HAS_CATBOOST:
            raise ImportError("CatBoost is required for DST model but not available")

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train CatBoost model optimized for DST volatility."""
        start_time = time.time()

        # CatBoost parameters optimized for small, volatile DST dataset
        self.model = CatBoostRegressor(
            iterations=500,              # More iterations for better performance
            learning_rate=0.1,           # Higher learning rate for small dataset
            depth=4,                     # Shallow trees to prevent overfitting
            l2_leaf_reg=3,              # L2 regularization for stability
            random_seed=42,              # Reproducibility
            loss_function='RMSE',        # Good for continuous targets
            eval_metric='R2',           # Track R² directly
            early_stopping_rounds=50,    # Stop if no improvement
            use_best_model=True,         # Use best model from validation
            verbose=50,                  # Progress updates
            allow_writing_files=False    # No temp files
        )

        logger.info("Training CatBoost model for DST...")

        # Train with validation set for early stopping
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

        # Train XGBoost with early stopping
        dtrain = xgb.DMatrix(X_train_ensemble, label=y_train)
        dval = xgb.DMatrix(X_val_ensemble, label=y_val)

        self.xgb_model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
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
