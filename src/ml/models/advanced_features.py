"""Advanced PyTorch features for fantasy sports ML models.

This file implements sophisticated deep learning techniques specifically designed
for fantasy sports prediction challenges:

1. Custom Loss Functions: Tailored for fantasy scoring patterns and objectives
2. Attention Mechanisms: Model player interactions and situational importance
3. Sequence Modeling: Capture game flow and momentum effects
4. Multi-task Learning: Joint prediction of multiple fantasy-relevant metrics
5. Uncertainty Quantification: Better risk assessment for lineup optimization

These advanced features can significantly improve model performance by:
- Better capturing the unique characteristics of fantasy sports scoring
- Modeling complex player interactions and game situations
- Providing more reliable uncertainty estimates for decision making
- Learning representations that generalize better to new scenarios

For beginners: These are "power tools" for deep learning that solve specific
problems in fantasy sports that standard neural networks struggle with.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FantasyLoss(nn.Module):
    """Custom loss function optimized for fantasy sports scoring patterns.

    Fantasy sports have unique characteristics that standard MSE doesn't capture well:
    1. Floor/Ceiling asymmetry: Predicting floor correctly is more important than ceiling
    2. Slate-relative performance: Players are ranked against each other, not absolutes
    3. Ownership consideration: Low-owned, high-scoring players are most valuable
    4. Risk/reward trade-offs: Different loss penalties for different prediction errors

    This loss function combines multiple objectives to create models that are
    better suited for fantasy decision-making rather than just statistical accuracy.
    """

    def __init__(
        self,
        mse_weight: float = 0.6,
        quantile_weight: float = 0.2,
        ranking_weight: float = 0.1,
        variance_weight: float = 0.1,
        floor_penalty: float = 2.0,
        ceiling_bonus: float = 0.5,
    ):
        """Initialize fantasy-specific loss function.

        Args:
            mse_weight: Weight for standard MSE component (baseline accuracy)
            quantile_weight: Weight for quantile loss component (floor/ceiling modeling)
            ranking_weight: Weight for ranking preservation component (relative ordering)
            variance_weight: Weight for variance penalty (prevent overconfident predictions)
            floor_penalty: Extra penalty for underestimating low scores (missing floors)
            ceiling_bonus: Reduced penalty for overestimating high scores (ceiling upside)
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.ranking_weight = ranking_weight
        self.variance_weight = variance_weight
        self.floor_penalty = floor_penalty
        self.ceiling_bonus = ceiling_bonus

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate fantasy-optimized loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Actual fantasy scores [batch_size]

        Returns:
            Combined loss value optimized for fantasy performance
        """
        # Component 1: Base MSE loss for overall accuracy
        mse_loss = F.mse_loss(predictions, targets)

        # Component 2: Quantile loss for floor/ceiling modeling
        # Penalize underestimating floors more than overestimating ceilings
        residuals = targets - predictions
        quantile_loss = self._quantile_loss(residuals, targets)

        # Component 3: Ranking preservation loss
        # Ensure model preserves relative ordering of players
        ranking_loss = self._ranking_loss(predictions, targets)

        # Component 4: Variance penalty to prevent overconfidence
        # Encourage diverse predictions rather than regression to mean
        variance_penalty = self._variance_penalty(predictions)

        # Combine all components with learned weights
        total_loss = (
            self.mse_weight * mse_loss
            + self.quantile_weight * quantile_loss
            + self.ranking_weight * ranking_loss
            + self.variance_weight * variance_penalty
        )

        return total_loss

    def _quantile_loss(self, residuals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate asymmetric quantile loss for floor/ceiling modeling."""
        # Different penalties for different score ranges
        low_score_mask = targets < 10.0  # Low scoring performances
        high_score_mask = targets > 20.0  # High scoring performances

        # Asymmetric loss: penalize underestimating floors, reward predicting ceilings
        floor_errors = torch.where(
            (residuals > 0) & low_score_mask,  # Underestimated low scores
            self.floor_penalty * torch.abs(residuals),
            torch.abs(residuals),
        )

        ceiling_errors = torch.where(
            (residuals < 0) & high_score_mask,  # Overestimated high scores
            self.ceiling_bonus * torch.abs(residuals),
            torch.abs(residuals),
        )

        return torch.mean(floor_errors + ceiling_errors)

    def _ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate ranking preservation loss using Kendall's tau approximation."""
        # This encourages the model to maintain relative ordering of players
        # Important because fantasy is about ranking players against each other

        n = predictions.size(0)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Create pairwise comparison matrices
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [n, n]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [n, n]

        # Calculate sign agreement (approximation of Kendall's tau)
        pred_sign = torch.sign(pred_diff)
        target_sign = torch.sign(target_diff)

        # Penalize disagreements in ranking (where signs differ)
        disagreements = torch.abs(pred_sign - target_sign)

        # Only consider upper triangle to avoid double counting
        mask = torch.triu(torch.ones_like(disagreements), diagonal=1)
        ranking_loss = torch.sum(disagreements * mask) / torch.sum(mask)

        return ranking_loss

    def _variance_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalize low variance predictions (regression to mean)."""
        # Fantasy models often collapse to predicting similar scores for everyone
        # This penalty encourages maintaining prediction diversity
        pred_var = torch.var(predictions)
        target_var = 25.0  # Approximate expected variance for fantasy scores

        # Penalize when prediction variance is too low
        variance_penalty = (
            torch.max(torch.tensor(0.0, device=predictions.device), target_var - pred_var)
            / target_var
        )

        return variance_penalty


class PlayerAttention(nn.Module):
    """Attention mechanism for modeling player interactions.

    In fantasy sports, players don't exist in isolation - they interact with
    each other in complex ways:
    - QBs and their receivers have positive correlations
    - RBs in committees have negative correlations
    - Opposing players have complex relationships based on game script

    This attention mechanism learns to focus on relevant player interactions
    when making predictions for each individual player.
    """

    def __init__(self, player_dim: int, num_heads: int = 4):
        """Initialize player attention mechanism.

        Args:
            player_dim: Dimension of player feature representations
            num_heads: Number of attention heads for multi-head attention
        """
        super().__init__()
        self.player_dim = player_dim
        self.num_heads = num_heads
        self.head_dim = player_dim // num_heads

        assert player_dim % num_heads == 0, "player_dim must be divisible by num_heads"

        # Linear projections for queries, keys, values
        self.query_proj = nn.Linear(player_dim, player_dim)
        self.key_proj = nn.Linear(player_dim, player_dim)
        self.value_proj = nn.Linear(player_dim, player_dim)

        # Output projection
        self.output_proj = nn.Linear(player_dim, player_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, player_features: torch.Tensor) -> torch.Tensor:
        """Apply player attention mechanism.

        Args:
            player_features: Player features [batch_size, num_players, player_dim]

        Returns:
            Attention-weighted player features [batch_size, num_players, player_dim]
        """
        batch_size, num_players, _ = player_features.shape

        # Generate queries, keys, values
        queries = self.query_proj(player_features)  # [B, P, D]
        keys = self.key_proj(player_features)  # [B, P, D]
        values = self.value_proj(player_features)  # [B, P, D]

        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_players, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_players, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_players, self.num_heads, self.head_dim)

        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # [B, H, P, D//H]
        keys = keys.transpose(1, 2)  # [B, H, P, D//H]
        values = values.transpose(1, 2)  # [B, H, P, D//H]

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Concatenate heads and project to output
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, num_players, self.player_dim)

        output = self.output_proj(attended_values)

        return output


class GameFlowLSTM(nn.Module):
    """LSTM for modeling game flow and momentum effects.

    Fantasy scoring is heavily influenced by game flow and momentum:
    - Teams that get ahead early may run more (affecting RB touches)
    - Teams that fall behind throw more (affecting passing volume)
    - Red zone efficiency varies throughout the game
    - Garbage time can inflate stats for losing teams

    This LSTM processes sequences of game events to capture these temporal
    dynamics and their impact on fantasy scoring.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2
    ):
        """Initialize game flow LSTM.

        Args:
            input_dim: Dimension of input features (game state, drive info, etc.)
            hidden_dim: Hidden dimension of LSTM cells
            num_layers: Number of LSTM layers for depth
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Main LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,  # Only forward direction (can't see future)
        )

        # Output projection layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Single output for momentum score
        )

    def forward(self, game_sequence: torch.Tensor) -> torch.Tensor:
        """Process game flow sequence.

        Args:
            game_sequence: Sequence of game states [batch_size, seq_len, input_dim]

        Returns:
            Game flow momentum scores [batch_size]
        """
        # Process sequence through LSTM
        lstm_output, (hidden, _) = self.lstm(game_sequence)

        # Use final hidden state to represent overall game flow
        final_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Project to momentum score
        momentum_score = self.output_proj(final_hidden)
        momentum_score = momentum_score.squeeze(-1)  # [batch_size]

        return momentum_score


class MultiTaskHead(nn.Module):
    """Multi-task learning head for joint prediction of multiple fantasy metrics.

    Instead of just predicting fantasy points, we can improve model performance
    by jointly learning to predict multiple related metrics:
    - Fantasy points (main target)
    - Touches/targets (volume metrics)
    - Touchdowns (boom/bust events)
    - Yards (efficiency metrics)

    This multi-task approach helps the model learn better representations
    and provides more interpretable predictions.
    """

    def __init__(self, input_dim: int, task_config: dict[str, int]):
        """Initialize multi-task head.

        Args:
            input_dim: Dimension of input features
            task_config: Dictionary mapping task names to output dimensions
                        e.g., {'fantasy_points': 1, 'touches': 1, 'touchdowns': 1}
        """
        super().__init__()
        self.task_config = task_config

        # Shared representation layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        shared_dim = input_dim // 4

        for task_name, output_dim in task_config.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim // 2, output_dim),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through multi-task head.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary mapping task names to predictions
        """
        # Shared representation
        shared_repr = self.shared_layers(x)

        # Task-specific predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(shared_repr)

            # Squeeze single-output tasks
            if predictions[task_name].size(-1) == 1:
                predictions[task_name] = predictions[task_name].squeeze(-1)

        return predictions


class UncertaintyQuantifier(nn.Module):
    """Neural network layer for uncertainty quantification.

    Fantasy sports decision-making benefits greatly from understanding prediction
    uncertainty. This module learns to predict both the expected value and
    the uncertainty of that prediction, enabling better risk assessment.

    Uses aleatoric uncertainty (data noise) and epistemic uncertainty (model uncertainty)
    to provide comprehensive uncertainty estimates.
    """

    def __init__(self, input_dim: int, num_samples: int = 10):
        """Initialize uncertainty quantifier.

        Args:
            input_dim: Dimension of input features
            num_samples: Number of Monte Carlo samples for uncertainty estimation
        """
        super().__init__()
        self.num_samples = num_samples

        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1),
        )

        # Variance prediction head (aleatoric uncertainty)
        self.variance_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, 1),
            nn.Softplus(),  # Ensure positive variance
        )

        # Dropout layers for epistemic uncertainty (Monte Carlo Dropout)
        self.epistemic_dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty quantification.

        Args:
            x: Input features [batch_size, input_dim]
            training: Whether in training mode (affects dropout)

        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        if training:
            # During training, predict mean and variance
            mean_pred = self.mean_head(x).squeeze(-1)
            variance_pred = self.variance_head(x).squeeze(-1)
            return mean_pred, variance_pred
        else:
            # During inference, use Monte Carlo dropout for uncertainty
            mean_predictions = []

            for _ in range(self.num_samples):
                # Enable dropout during inference for epistemic uncertainty
                x_dropout = self.epistemic_dropout(x)
                pred = self.mean_head(x_dropout).squeeze(-1)
                mean_predictions.append(pred)

            # Stack predictions and compute statistics
            predictions = torch.stack(mean_predictions, dim=0)  # [num_samples, batch_size]
            mean_pred = torch.mean(predictions, dim=0)
            epistemic_uncertainty = torch.var(predictions, dim=0)

            # Also compute aleatoric uncertainty
            aleatoric_uncertainty = self.variance_head(x).squeeze(-1)

            # Total uncertainty combines both sources
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

            return mean_pred, total_uncertainty


class PositionalEncoding(nn.Module):
    """Positional encoding for modeling game time and situational context.

    Fantasy scoring varies significantly based on game situation:
    - Quarter (early game vs late game dynamics)
    - Down and distance (passing vs rushing situations)
    - Field position (red zone vs midfield)
    - Score differential (blowout vs close game)

    This module adds learnable positional encodings to capture these effects.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to support
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use sinusoidal positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.

        Args:
            x: Input features [batch_size, seq_len, d_model]
            positions: Position indices [batch_size, seq_len]

        Returns:
            Features with positional encoding added
        """
        # Gather positional encodings for specified positions
        pos_encoding = self.pe[positions]  # [batch_size, seq_len, d_model]

        # Add to input features
        return x + pos_encoding
