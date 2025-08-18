"""Correlated neural network model that captures player interactions.

This advanced model architecture understands how players, defenses, and
coaching styles interact to produce fantasy points. It uses:

1. Shared layers to learn game context
2. Position-specific heads for predictions
3. Attention mechanisms for dynamic feature importance
4. Correlation modeling between positions
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

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
        
        # Attention to identify important game factors
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode game context with attention.
        
        Args:
            x: Game features [batch_size, input_dim]
            
        Returns:
            Encoded game context [batch_size, hidden_dim]
        """
        # Initial encoding
        encoded = self.encoder(x)
        
        # Self-attention to capture feature interactions
        # Reshape for attention: [seq_len, batch, embed_dim]
        encoded_seq = encoded.unsqueeze(0)
        attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        
        # Combine with residual connection
        output = encoded + attended.squeeze(0)
        
        return output


class PositionSpecificHead(nn.Module):
    """Position-specific prediction head."""
    
    def __init__(self, context_dim: int, player_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Combine game context with player features
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
        
        # Final prediction layers with scaling
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Fantasy points prediction
            nn.Sigmoid()  # Output 0-1 range, will be scaled by position
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Mean and log variance
        )
    
    def forward(
        self,
        game_context: torch.Tensor,
        player_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict fantasy points with uncertainty.
        
        Args:
            game_context: Encoded game context [batch_size, context_dim]
            player_features: Player-specific features [batch_size, player_dim]
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        # Combine context and player features
        combined = torch.cat([game_context, player_features], dim=1)
        fused = self.fusion(combined)
        
        # Make prediction
        prediction = self.predictor(fused)
        
        # Estimate uncertainty (for confidence intervals)
        uncertainty_params = self.uncertainty(fused)
        
        return prediction.squeeze(), uncertainty_params


class CorrelatedFantasyModel(nn.Module):
    """Neural network that models player correlations for fantasy football.
    
    This model understands that:
    - QB performance affects WR/TE production
    - Game script affects RB usage
    - Defensive strength limits offensive output
    - Coaching tendencies create predictable patterns
    """
    
    def __init__(
        self,
        game_feature_dim: int,
        position_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout_rate: float = 0.3
    ):
        """Initialize correlated model.
        
        Args:
            game_feature_dim: Dimension of game-level features
            position_feature_dims: Dict of position -> feature dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.positions = list(position_feature_dims.keys())
        
        # Game context encoder (shared across all positions)
        self.game_encoder = GameContextEncoder(game_feature_dim, hidden_dim)
        
        # Position-specific heads
        self.position_heads = nn.ModuleDict({
            pos: PositionSpecificHead(
                context_dim=hidden_dim,
                player_dim=feat_dim,
                hidden_dim=hidden_dim // 2
            )
            for pos, feat_dim in position_feature_dims.items()
        })
        
        # Correlation modeling layers
        self.correlation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * len(self.positions), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Stack correlation factors (QB-WR, QB-TE, etc.)
        self.stack_factors = nn.Parameter(
            torch.randn(len(self.positions), len(self.positions)) * 0.1
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        game_features: torch.Tensor,
        player_features: Dict[str, torch.Tensor],
        return_correlations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass predicting all positions with correlations.
        
        Args:
            game_features: Game-level features [batch_size, game_feature_dim]
            player_features: Dict of position -> features
            return_correlations: Whether to return correlation matrix
            
        Returns:
            Dict of position -> predictions (and correlations if requested)
        """
        # Encode game context (shared understanding)
        game_context = self.game_encoder(game_features)
        
        # Get initial predictions for each position
        predictions = {}
        uncertainties = {}
        position_embeddings = []
        
        for pos in self.positions:
            if pos in player_features:
                pred, unc = self.position_heads[pos](
                    game_context,
                    player_features[pos]
                )
                # Scale prediction to position-specific range (0-1 -> position max)
                pos_range = POSITION_RANGES.get(pos.upper(), 30.0)
                scaled_pred = pred * pos_range
                
                predictions[pos] = scaled_pred
                uncertainties[pos] = unc
                position_embeddings.append(scaled_pred.unsqueeze(1))
        
        # Model correlations between positions
        if len(position_embeddings) > 1:
            # Stack position predictions
            stacked = torch.cat(position_embeddings, dim=1)  # [batch, n_positions]
            
            # Apply correlation factors (learned interactions)
            correlated = torch.matmul(stacked, self.stack_factors[:len(position_embeddings), :len(position_embeddings)])
            
            # Adjust predictions based on correlations
            for i, pos in enumerate(predictions.keys()):
                correlation_adjustment = correlated[:, i]
                predictions[pos] = predictions[pos] + 0.1 * correlation_adjustment
        
        if return_correlations:
            return predictions, self.stack_factors
        
        return predictions
    
    def predict_single_position(
        self,
        position: str,
        game_features: torch.Tensor,
        player_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict for a single position.
        
        Args:
            position: Position to predict
            game_features: Game context features
            player_features: Player-specific features
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        game_context = self.game_encoder(game_features)
        prediction, uncertainty = self.position_heads[position](
            game_context,
            player_features
        )
        # Scale prediction to position-specific range (0-1 -> position max)
        pos_range = POSITION_RANGES.get(position.upper(), 30.0)
        scaled_prediction = prediction * pos_range
        
        return scaled_prediction, uncertainty


class CorrelatedModelTrainer:
    """Trainer for correlated fantasy model."""
    
    def __init__(
        self,
        model: CorrelatedFantasyModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """Initialize trainer.
        
        Args:
            model: Correlated fantasy model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function with correlation penalty
        self.base_loss = nn.MSELoss()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        correlation_weight: float = 0.1
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            correlation_weight: Weight for correlation regularization
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            game_features, player_features, targets = batch
            
            # Move to device
            game_features = game_features.to(self.device)
            player_features = {
                pos: feat.to(self.device)
                for pos, feat in player_features.items()
            }
            targets = {
                pos: tgt.to(self.device)
                for pos, tgt in targets.items()
            }
            
            # Forward pass
            predictions = self.model(game_features, player_features)
            
            # Calculate loss for each position
            position_losses = []
            for pos in predictions:
                if pos in targets:
                    loss = self.base_loss(predictions[pos], targets[pos])
                    position_losses.append(loss)
            
            # Combined loss
            total_position_loss = sum(position_losses) / len(position_losses)
            
            # Add correlation regularization (encourage realistic correlations)
            correlation_reg = self._correlation_regularization(predictions, targets)
            
            loss = total_position_loss + correlation_weight * correlation_reg
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _correlation_regularization(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate correlation regularization loss.
        
        Encourages realistic correlations (e.g., QB-WR positive correlation).
        
        Args:
            predictions: Model predictions by position
            targets: True values by position
            
        Returns:
            Correlation regularization loss
        """
        reg_loss = 0
        
        # QB-WR should be positively correlated
        if 'QB' in predictions and 'WR' in predictions:
            qb_wr_corr = self._pearson_correlation(predictions['QB'], predictions['WR'])
            reg_loss += (1 - qb_wr_corr) ** 2  # Penalize low correlation
        
        # QB-RB should be slightly negatively correlated (game script)
        if 'QB' in predictions and 'RB' in predictions:
            qb_rb_corr = self._pearson_correlation(predictions['QB'], predictions['RB'])
            reg_loss += (qb_rb_corr + 0.2) ** 2  # Target slight negative correlation
        
        return reg_loss
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation between two tensors."""
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        correlation = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
        )
        
        return correlation
    
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict of position -> validation loss
        """
        self.model.eval()
        position_losses = {pos: 0 for pos in self.model.positions}
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                game_features, player_features, targets = batch
                
                # Move to device
                game_features = game_features.to(self.device)
                player_features = {
                    pos: feat.to(self.device)
                    for pos, feat in player_features.items()
                }
                targets = {
                    pos: tgt.to(self.device)
                    for pos, tgt in targets.items()
                }
                
                # Forward pass
                predictions = self.model(game_features, player_features)
                
                # Calculate loss for each position
                for pos in predictions:
                    if pos in targets:
                        loss = self.base_loss(predictions[pos], targets[pos])
                        position_losses[pos] += loss.item()
                
                n_batches += 1
        
        # Average losses
        for pos in position_losses:
            position_losses[pos] /= n_batches
        
        return position_losses