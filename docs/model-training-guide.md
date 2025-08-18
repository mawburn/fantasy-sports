# NFL DFS Model Training Guide

This guide covers how to train, configure, customize, and optimize PyTorch neural network models for fantasy football predictions.

## Overview

The system uses **PyTorch neural networks exclusively** with position-specific architectures:

- **QB Model**: Multi-task learning for passing and rushing
- **RB Model**: Workload clustering with efficiency modeling
- **WR Model**: Target competition with attention mechanisms
- **TE Model**: Dual-role processing (receiving + blocking)
- **DEF Model**: Multi-head ensemble for different defensive stats

## Quick Start

### 1. Collect Data First

```bash
# Collect all NFL data including defensive stats and Vegas lines
uv run python -m src.cli.collect_data collect-all --seasons 2021 2022 2023

# Or collect specific data types
uv run python -m src.cli.collect_data collect-teams
uv run python -m src.cli.collect_data collect-players --seasons 2023
uv run python -m src.cli.collect_data collect-stats --seasons 2023
```

### 2. Train Models

```bash
# Train all positions at once
uv run python -m src.cli.train_models train-all-positions

# Train individual position
uv run python -m src.cli.train_models train-position QB

# Train with custom date range
uv run python -m src.cli.train_models train-position RB \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --model-name RB_custom_v1 \
  --evaluate
```

## Neural Network Architecture

### Base Neural Model Structure

All position models inherit from `BaseNeuralModel` and use this general architecture:

```python
class PositionNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super().__init__()

        # Input layer with batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # Hidden layers with dropout for regularization
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(0.1)

        # Output layer
        self.output = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.output(x)
```

### Position-Specific Architectures

#### QB Neural Model

- **Multi-task learning**: Separate branches for passing/rushing
- **Attention mechanism**: Dynamic feature importance
- **Hidden sizes**: [512, 256, 128]
- **Dropout rates**: [0.4, 0.3, 0.2]

#### RB Neural Model

- **Workload embeddings**: Cluster RBs by usage patterns
- **Dual branches**: Workload vs efficiency
- **Hidden sizes**: [384, 192, 96]
- **Dropout rates**: [0.35, 0.25, 0.15]

#### WR Neural Model

- **Target competition**: Models WR competition
- **High variance handling**: Extra dropout layers
- **Hidden sizes**: [448, 224, 112]
- **Dropout rates**: [0.5, 0.4, 0.3]

#### TE Neural Model

- **Dual-role modeling**: Receiving + blocking
- **Simpler architecture**: Less complex patterns
- **Hidden sizes**: [256, 128, 64]
- **Dropout rates**: [0.3, 0.2, 0.1]

#### DEF Neural Model

- **Multi-head ensemble**: Different defensive aspects
- **High variance**: Specialized dropout patterns
- **Hidden sizes**: [320, 160, 80]
- **Dropout rates**: [0.4, 0.3, 0.2]

## Customizing PyTorch Models

### 1. Modifying Network Architecture

To customize the neural network architecture, edit the position-specific model in `src/ml/models/neural_models.py`:

```python
# Example: Add an extra hidden layer to QB model
class QBNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Add your custom layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)  # New layer
        self.output = nn.Linear(64, 1)

        # Add corresponding dropout
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.1)  # New dropout
```

### 2. Adding New Features

#### Step 1: Update Feature Extractor

Edit `src/data/processing/feature_extractor.py`:

```python
def extract_player_features(self, player_id, target_game_date, ...):
    # ... existing code ...

    # Add your custom feature
    features['days_rest'] = self._calculate_days_rest(player_id, target_game_date)
    features['primetime_game'] = self._is_primetime(target_game_date)
    features['revenge_game'] = self._is_revenge_game(player_id, opponent_team_id)

    return features

def _calculate_days_rest(self, player_id, game_date):
    """Calculate days since last game."""
    last_game = self.db.query(PlayerStats).filter(
        PlayerStats.player_id == player_id,
        PlayerStats.game_date < game_date
    ).order_by(PlayerStats.game_date.desc()).first()

    if last_game:
        return (game_date - last_game.game_date).days
    return 7  # Default to normal week
```

#### Step 2: Update Data Preparation

Edit `src/ml/training/data_preparation.py` to include new features:

```python
FEATURE_GROUPS = {
    "basic": [...],
    "recent_performance": [...],
    "opponent": [...],
    "custom": [  # Add new feature group
        "days_rest",
        "primetime_game",
        "revenge_game"
    ]
}
```

### 3. Changing Training Parameters

#### Modify Training Configuration

Edit `src/ml/models/neural_models.py`:

```python
class QBNeuralModel(BaseNeuralModel):
    def get_default_config(self):
        return {
            'learning_rate': 0.001,  # Adjust learning rate
            'batch_size': 64,        # Change batch size
            'num_epochs': 200,        # More epochs
            'early_stopping_patience': 20,  # When to stop
            'optimizer': 'AdamW',     # Try different optimizer
            'weight_decay': 0.01,     # L2 regularization
            'scheduler': 'ReduceLROnPlateau',  # Learning rate scheduling
        }
```

#### Use Custom Config in Training

```python
from src.ml.models.base import ModelConfig

config = ModelConfig(
    model_name="QB_custom",
    position="QB",
    # Add custom parameters
    learning_rate=0.0005,
    batch_size=128,
    num_epochs=300
)

trainer.train_position_model(
    position="QB",
    config=config,
    start_date="2021-01-01",
    end_date="2023-12-31"
)
```

### 4. Advanced PyTorch Techniques

#### Add Attention Mechanism

```python
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return x * weights

class WRNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # ... existing layers ...
        self.attention = AttentionLayer(256)  # Add attention

    def forward(self, x):
        # ... existing forward pass ...
        x = self.attention(x)  # Apply attention
        return self.output(x)
```

#### Add Residual Connections

```python
class RBNetwork(nn.Module):
    def forward(self, x):
        identity = x  # Save input for residual

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        # Add residual connection (if dimensions match)
        if x.shape == identity.shape:
            x = x + identity

        return self.output(x)
```

### 5. Detailed Explanations of Advanced Concepts

#### What Are Attention Mechanisms and Why Use Them?

**Attention in Simple Terms:** Think of attention mechanisms like a spotlight that highlights the most important features for each prediction. When predicting a QB's performance, the model might "pay attention" to passing matchup stats more in good weather but focus on rushing stats in bad weather.

**How It Works:**

1. The model learns to assign importance scores to each feature
1. These scores are normalized (sum to 1) using softmax
1. Features are weighted by their importance scores
1. The model can dynamically adjust what it focuses on

**Fantasy Football Benefits:**

- **Context-Aware**: Different features matter in different situations
- **Interpretable**: You can see what the model is focusing on
- **Better Performance**: Captures complex feature interactions

**Detailed Implementation Example:**

```python
class DetailedAttentionLayer(nn.Module):
    """
    Attention mechanism with detailed comments explaining each step.
    This helps the model focus on relevant features dynamically.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # First layer: Transform input to hidden dimension
        # This allows the model to learn complex feature interactions
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Second layer: Produce a single attention score per input
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Step 1: Calculate attention scores
        # Pass through first layer with activation
        attention_hidden = torch.tanh(self.fc1(x))

        # Get raw attention scores (one per sample)
        attention_scores = self.fc2(attention_hidden)

        # Step 2: Normalize scores with softmax
        # This ensures scores sum to 1 (probability distribution)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Step 3: Apply attention weights
        # Multiply input features by their importance weights
        weighted_features = x * attention_weights

        return weighted_features, attention_weights  # Return weights for interpretability

# Example: Using attention in a QB model
class QBModelWithAttention(nn.Module):
    """QB model showing how attention helps with position-specific patterns."""

    def __init__(self, input_size):
        super().__init__()
        # Standard layers
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)

        # Attention layer - learns what matters for each QB/matchup
        self.attention = DetailedAttentionLayer(512, hidden_dim=128)

        # Output layers
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        # Initial processing
        x = F.relu(self.bn1(self.fc1(x)))

        # Apply attention (get both weighted features and weights)
        x, attention_weights = self.attention(x)

        # Continue to output
        x = F.relu(self.fc2(x))
        prediction = self.output(x)

        # Return prediction and attention weights for analysis
        return prediction, attention_weights
```

#### What Are Residual Connections and Why Use Them?

**Residual Connections in Simple Terms:** Residual connections create "shortcuts" that allow information to skip layers. Instead of learning a complete transformation, the network learns the "change" to apply to the input. This is like saying "keep what you have and add these adjustments" rather than "forget everything and use this instead."

**Mathematical Concept:**

- Without residual: `output = F(input)`
- With residual: `output = F(input) + input`
- The network learns `F(input)` which represents the "residual" or change

**Benefits for Fantasy Predictions:**

- **Preserves Important Information**: Base stats don't get lost in transformations
- **Easier Training**: Gradients flow better through the network
- **Better Performance**: Especially helpful for deeper networks

```python
class ResidualBlockExplained(nn.Module):
    """
    A residual block with detailed explanations.
    This structure helps preserve information while learning transformations.
    """

    def __init__(self, dim):
        super().__init__()
        # Two transformation layers
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        # Normalization for stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Save the original input - this is the "residual"
        residual = x

        # Apply transformations
        # These layers learn what to "add" to the input
        out = self.norm1(x)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)

        # Add the residual (original input) back
        # This means the block only needs to learn the difference
        out = out + residual

        # Final activation
        out = F.relu(out)

        return out
```

#### What is Batch Normalization and Why Use It?

**Batch Normalization in Simple Terms:** Batch normalization is like adjusting the brightness and contrast of your data at each layer. It keeps the values centered and scaled consistently, preventing them from becoming too large or too small during training.

**How It Works:**

1. Calculate mean and variance of each batch
1. Normalize values to have mean=0, variance=1
1. Scale and shift with learnable parameters

**Benefits:**

- **Faster Training**: Can use higher learning rates
- **More Stable**: Less sensitive to initialization
- **Regularization**: Helps prevent overfitting

```python
class NetworkWithBatchNorm(nn.Module):
    """
    Example showing proper batch normalization usage.
    BatchNorm goes after linear layer, before activation.
    """

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # BatchNorm for 256 features

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        # Pattern: Linear -> BatchNorm -> Activation -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)  # Normalize before activation
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return self.output(x)
```

#### Understanding Different Activation Functions

**Common Activation Functions Explained:**

```python
# ReLU: Most common, simple and effective
# Output: max(0, x)
# Use when: Default choice, works well in most cases
x = F.relu(x)

# Leaky ReLU: Allows small negative values
# Output: max(0.01*x, x)
# Use when: Dead neurons are a problem (gradients dying)
x = F.leaky_relu(x, negative_slope=0.01)

# ELU: Smooth version of ReLU
# Use when: You want smoother gradients
x = F.elu(x)

# GELU: Used in transformers, smooth approximation of ReLU
# Use when: Working with attention mechanisms
x = F.gelu(x)

# Tanh: Outputs between -1 and 1
# Use when: You need bounded outputs
x = torch.tanh(x)

# Sigmoid: Outputs between 0 and 1
# Use when: Binary classification or probability outputs
x = torch.sigmoid(x)
```

#### Understanding Dropout and Regularization

**Dropout Explained:** Dropout randomly "turns off" neurons during training, forcing the network to learn redundant representations. It's like training multiple networks and averaging them.

```python
class DropoutStrategies(nn.Module):
    """Different dropout strategies for different positions."""

    def __init__(self, input_size, position):
        super().__init__()

        # Position-specific dropout rates
        dropout_rates = {
            'QB': [0.4, 0.3, 0.2],    # High early dropout for high variance
            'RB': [0.35, 0.25, 0.15],  # Medium dropout
            'WR': [0.5, 0.4, 0.3],     # Highest dropout (most variance)
            'TE': [0.3, 0.2, 0.1],     # Lower dropout (more stable)
            'DEF': [0.4, 0.3, 0.2]     # Medium-high dropout
        }

        rates = dropout_rates.get(position, [0.3, 0.2, 0.1])

        # Build network with position-specific dropout
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(rates[0])

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(rates[1])

        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(rates[2])

        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Randomly zero out neurons

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        return self.output(x)
```

## Training Process

### Understanding the Training Loop

The training process in `BaseNeuralModel.train()`:

1. **Data Preparation**: Scales features, converts to tensors
1. **DataLoader Creation**: Batches data for efficient training
1. **Training Loop**: Forward pass → Loss calculation → Backpropagation
1. **Validation**: Evaluates on validation set each epoch
1. **Early Stopping**: Stops if validation loss doesn't improve

### Key Training Parameters

```python
# In neural_models.py
class BaseNeuralModel:
    def train(self, X_train, y_train, X_val, y_val):
        # Key parameters you can adjust
        learning_rate = 0.001
        batch_size = 32
        num_epochs = 100
        early_stopping_patience = 15

        # Optimizer choice
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Loss function
        criterion = nn.MSELoss()  # For regression
        # Or try: nn.L1Loss(), nn.HuberLoss()
```

## Model Evaluation

### Performance Metrics

The system tracks multiple metrics during training:

```python
# Primary metrics
MAE = Mean Absolute Error  # Average prediction error
R² = R-squared score       # Variance explained
RMSE = Root Mean Square Error  # Penalizes large errors

# Position-specific targets
QB: MAE < 4.0, R² > 0.35
RB: MAE < 3.5, R² > 0.30
WR: MAE < 3.0, R² > 0.25
TE: MAE < 2.5, R² > 0.30
DEF: MAE < 2.0, R² > 0.40
```

### Monitoring Training

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
uv run python -m src.cli.train_models train-position QB

# Watch training progress (logs show each epoch)
# Epoch 10/100, Train Loss: 3.45, Val Loss: 3.89, Val MAE: 4.2
# Epoch 20/100, Train Loss: 2.98, Val Loss: 3.41, Val MAE: 3.8
# Early stopping at epoch 35
```

## Using the New Data Sources

The system now includes rich opponent and game context data:

### Defensive Statistics

- Position-specific rankings (e.g., "tough against RBs")
- Fantasy points allowed by position
- Recent defensive form (4-week rolling averages)

### Vegas Context

- Game totals (high-scoring games = more fantasy points)
- Point spreads (game script predictions)
- Implied team totals

### Example: Using Opponent Data in Custom Features

```python
def extract_matchup_difficulty(self, player_id, game_id):
    """Calculate matchup difficulty score."""

    # Get opponent defensive stats
    opp_stats = self._get_opponent_defensive_stats(...)

    # Position-specific difficulty
    if player.position == "RB":
        difficulty = opp_stats['rb_def_rank'] / 32  # Normalize 0-1
        fantasy_allowed = opp_stats['rb_fantasy_allowed']

    # Combine with Vegas data
    vegas = self._get_vegas_context(game_id)
    game_total = vegas['total']

    # High-scoring games reduce difficulty
    adjusted_difficulty = difficulty * (50 / game_total)

    return adjusted_difficulty
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Converging (Loss not decreasing)

```python
# Solutions:
# - Reduce learning rate: 0.001 → 0.0001
# - Check data scaling (features should be normalized)
# - Increase batch size for stability
# - Simplify architecture (fewer layers/neurons)
```

#### 2. Overfitting (Train loss \<< Validation loss)

```python
# Solutions:
# - Increase dropout rates
# - Add L2 regularization (weight_decay)
# - Reduce model complexity
# - Get more training data
# - Use data augmentation
```

#### 3. Poor Performance on Specific Position

```python
# Solutions:
# - Check position-specific features
# - Adjust architecture for position variance
# - QB/WR: Need more complex models (high variance)
# - TE/DEF: Simpler models work better
```

#### 4. Training Too Slow

```python
# Solutions:
# - Reduce batch size (uses less memory)
# - Use fewer epochs with early stopping
# - Simplify architecture
# - Use GPU if available (set device='cuda')
```

## Advanced Customization

### Custom Loss Functions

```python
class WeightedMSELoss(nn.Module):
    """Weight recent games more heavily."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets)

        squared_diff = (predictions - targets) ** 2
        weighted_loss = (squared_diff * weights).mean()
        return weighted_loss

# Use in training
criterion = WeightedMSELoss()
```

### Ensemble Multiple Neural Networks

```python
from src.ml.models.ensemble import EnsembleModel

# Train multiple models with different architectures
models = []
for i in range(3):
    config = ModelConfig(
        model_name=f"QB_model_{i}",
        hidden_sizes=[512-i*64, 256-i*32, 128-i*16]
    )
    model = trainer.train_position_model("QB", config=config)
    models.append(model)

# Create ensemble
ensemble = EnsembleModel(position="QB")
for model in models:
    ensemble.add_model(model, weight=1/3)
```

### Feature Engineering Tips

1. **Normalize Everything**: Neural networks are sensitive to scale
1. **Create Interaction Features**: `target_share * air_yards`
1. **Use Embeddings**: For categorical variables (team, stadium)
1. **Time-Based Features**: Days rest, weeks into season
1. **Rolling Averages**: Recent form matters more than season totals

## Production Best Practices

### Model Management

```bash
# Deploy best model
uv run python -m src.cli.train_models deploy-model QB_neural_20240315

# Monitor performance
uv run python -m src.cli.train_models evaluate-model QB_neural_20240315

# Rollback if needed
uv run python -m src.cli.train_models rollback QB
```

### Retraining Schedule

- **Weekly**: During NFL season for latest data
- **After Injuries**: When key players are out
- **After Trades**: When team compositions change
- **Performance Degradation**: When MAE increases by >20%

### API Integration

```python
# Get predictions via API
import requests

response = requests.post(
    "http://localhost:8000/api/predictions/player",
    json={
        "player_id": 12345,
        "week": 10,
        "season": 2024
    }
)

prediction = response.json()
print(f"Predicted points: {prediction['point_estimate']}")
print(f"Confidence: {prediction['confidence_score']}")
```

## Summary

The neural network models are highly customizable PyTorch implementations that can be modified at multiple levels:

1. **Architecture**: Change layers, neurons, activation functions
1. **Training**: Adjust learning rates, optimizers, loss functions
1. **Features**: Add new data sources and engineered features
1. **Position-Specific**: Customize each position's unique needs

The key to success is iterative improvement - start simple, measure performance, and gradually add complexity where it helps.
