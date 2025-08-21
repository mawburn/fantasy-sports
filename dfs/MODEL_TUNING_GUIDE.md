# DFS Model Tuning Guide: QB Optimization Journey

This guide documents the phased approach to optimizing the QB prediction model from R² = 0.310 to target R² ≥ 0.38 (ideally 0.40).

## 🎯 Current Status (Latest Update)

- **Phase 4 Feature 1 SUCCESS**: R² = 0.347 (ensemble), Neural = 0.313, MAE = 6.05 (December 2024)
- **Improvement**: +0.007 R² improvement from enhanced game script detection
- **Architecture**: XGBoost ensemble (70%) + Neural network (30%) for QB
- **Data**: 3609 training samples, 77 features, skipping 2020 COVID year
- **Training**: Full 800 epochs, best R² at epoch 678 (proves no early stopping needed)

## ✅ Completed Phases

### Phase 1: Training Parameter Optimization (COMPLETED)

**Goal**: Fix early stopping and basic training issues

**Changes Made**:

- **Learning Rate**: Reduced from 0.00015 → 0.00003 for stability
- **Batch Size**: Increased from 50 → 128 for better gradient estimates
- **Epochs**: Increased from 200 → 800 for more training time
- **Patience**: Increased from 15 → 50 to avoid premature stopping
- **Gradient Clipping**: Adjusted from 0.1 → 1.0 for better gradient flow
- **Optimizer**: Switched to AdamW with better weight decay

**Results**: Fixed early stopping at epoch 21, model now trains for hundreds of epochs

### Phase 2: Architecture Enhancements (COMPLETED)

**Goal**: Improve neural network architecture and feature engineering

**Architecture Changes**:

- **Network Capacity**: Increased to 256→192→128→96→64 layers
- **Activations**: Switched to GELU for better gradient flow
- **Normalization**: Added BatchNorm and LayerNorm layers
- **Attention Mechanism**: Multi-head attention for feature importance

**Feature Engineering**:

- **QB-Specific Features**: completion_pct_trend, yds_per_attempt_trend, td_int_ratio_trend
- **Game Context**: passer_rating_est, passing_volume_trend, dual_threat_factor
- **Situational**: game_script_favorability, pressure_situation, ceiling_indicator

### Phase 3: Model Checkpointing & XGBoost Ensemble (COMPLETED)

**Goal**: Proper checkpointing and ensemble methods

**Checkpointing Fixes**:

- **Every-epoch R² computation** instead of every 5 epochs
- **Best model restoration** based on highest R² score
- **Clear progress logging** showing R² improvements

**XGBoost Ensemble**:

- **EnsembleModel class** combining neural network + XGBoost
- **Feature enrichment** using original features + NN predictions
- **Smart weighting**: 70% XGBoost + 30% Neural Network

**Data Quality**:

- **Always skip 2020** COVID year for all positions
- **Position-specific seasons**: 2018-2019, 2021-2025 for QB/RB/WR/TE

## 🔄 Planned Future Phases

### Phase 4: Advanced Feature Engineering (COMPLETED FEATURE 1)

**Goal**: Leverage odds data and advanced NFL analytics

**Status**: Testing individual features incrementally after Phase 3 baseline

**Feature 1: Enhanced Game Script Detection (✅ COMPLETED - SUCCESS)**

- **Results**: R² = 0.347 (+0.007 improvement), MAE = 6.05
- **Problem Found**: Original `pace_mismatch = 0.0` initialization broke neural network
- **Solution Applied**: Fixed initialization, added bounds checking, safe feature normalization
- **Successful Features Added**:
  - `implied_team_total`: Normalized implied team total from betting lines (bounded 0.3-2.0)
  - `shootout_potential`: Binary flag for high-scoring games (O/U > 50)
  - `defensive_game_script`: Binary flag for low-scoring games (O/U < 42)
  - `garbage_time_upside`: QBs as big underdogs in high-scoring games
  - `blowout_risk`: QBs as big favorites in low-scoring games
  - `game_script_favorability`: Enhanced pace mismatch detection (bounded 0.2-3.0)
- **Key Learning**: Betting data provides significant predictive value when properly bounded

**Additional Odds Features (High Impact Potential)**:

- **Line Movement**: Track how much spread/total have moved (sharp vs public money)
- **Closing Line Value**: How current projections compare to closing lines
- **Steam Moves**: Detect coordinated betting activity (indicates inside info)
- **Weather-Line Correlation**: How weather should affect totals but doesn't in line
- **Situational Spots**: Division games, revenge spots, lookahead games

**Remaining Phase 4 Features to Test**:

- **Feature 2: Recent Form Weighting** - Weight last 4 games more heavily (4x, 3x, 2x, 1x)
- **Feature 3: Enhanced Opponent Matchups** - QB completion rates vs specific defenses
- **Feature 4: Red Zone Efficiency** - TD frequency and reliability metrics

### Phase 5: Advanced Model Architectures (PLANNED)

**Goal**: Cutting-edge ML approaches

**Neural Architecture Search (NAS)**:

- **Automated architecture optimization** for QB-specific patterns
- **Dynamic feature selection** based on game context
- **Temporal attention**: Weight recent games more heavily

**Transformer Architecture**:

- **Sequence modeling**: Model QB performance across season timeline
- **Multi-head attention**: Different aspects (passing, rushing, red zone)
- **Positional encoding**: Account for week-in-season effects

### Phase 5: Multi-Position Ensemble Expansion (COMPLETED)

**Goal**: Extend ensemble approach to other positions

**Completed Position Enhancements**:

1. **QB Ensemble**: R² = 0.347 (baseline neural = 0.313) ✅ 
2. **RB Ensemble**: Extended from QB success (baseline R² = 0.338) ✅
3. **WR Ensemble**: Extended ensemble approach (baseline R² = 0.248) ✅
4. **TE Ensemble**: Extended ensemble approach (baseline R² = 0.267) ✅

**Implementation**: All skill positions (QB, RB, WR, TE) now use XGBoost ensemble (70%) + Neural Network (30%) architecture for improved predictions

**Position-Specific Features** (ready for future enhancement):

- **RB**: Game script, snap share, goal line opportunities
- **WR**: Target quality, coverage matchups, air yards
- **TE**: Red zone usage, blocking vs receiving snaps

### Phase 6: Advanced Model Architectures (PLANNED)

**Goal**: Next-generation ML approaches

**Transformer Models**:

- **Sequence modeling**: Model player performance across season timeline
- **Multi-head attention**: Different aspects (passing, rushing, matchups)
- **Positional encoding**: Account for week-in-season effects

**Neural Architecture Search**:

- **Automated optimization**: Find optimal QB-specific network architectures
- **Dynamic feature selection**: Context-aware feature importance

### Phase 7: Production Enhancement (PLANNED)

**Goal**: Real-world deployment optimization

**Real-Time Integration**:

- **Live injury reports**: Parse and integrate injury news
- **Line movement tracking**: Detect sharp money and adjust projections
- **Weather integration**: Real-time weather impact on game scripts
- **News sentiment**: Parse beat reporter tweets and injury updates

**Validation & Monitoring**:

- **Rolling validation**: Continuous backtesting on recent weeks
- **Performance tracking**: Monitor model degradation in live contests
- **A/B testing**: Compare model versions in actual DFS contests

## 🔧 High Impact, Easy Changes (Historical Reference)

### 1. **Learning Rate & Training Parameters**

**Location**: `models.py` - each position model's `__init__` method

```python
# Current QB settings
self.learning_rate = 0.001
self.batch_size = 64
self.epochs = 100
self.patience = 15
```

**Tuning Options**:

```python
# Try different learning rates
self.learning_rate = 0.005    # Higher for faster learning
self.learning_rate = 0.0005   # Lower for more stable training

# Adjust batch size (affects gradient quality)
self.batch_size = 32          # Smaller = more frequent updates
self.batch_size = 128         # Larger = more stable gradients

# Training duration
self.epochs = 200             # More epochs for complex data
self.patience = 25            # More patience for slow improvements
```

**How to test**: Change values in `models.py`, retrain with `uv run python run.py train --positions QB`

### 2. **Network Architecture Size**

**Location**: `models.py` - `QBNetwork` class

```python
# Current architecture (models.py:769-794)
self.feature_layers = nn.Sequential(
    nn.Linear(input_size, 128),  # 🔧 TUNE: Try 64, 256, 512
    nn.BatchNorm1d(128, eps=1e-3),
    nn.ReLU(),
    nn.Dropout(0.2),             # 🔧 TUNE: Try 0.1, 0.3, 0.4
    nn.Linear(128, 64),          # 🔧 TUNE: Try 32, 128, 256
    # ... more layers
)
```

**Tuning Recommendations**:

- **Wider networks**: 256→128→64 for more capacity
- **Deeper networks**: Add more layers for complex patterns
- **Dropout rates**: 0.1-0.4 range (higher if overfitting)

### 3. **Feature Engineering** ⭐ **HIGHEST IMPACT**

**Location**: `data.py` - feature extraction functions

**Current missing features** (check your database):

```sql
-- Add these if missing
- Snap percentage (playing time)
- Red zone targets/carries
- Target depth (air yards)
- Game script (trailing/leading score)
- Weather conditions (already have collect weather command)
- Defensive rankings by position allowed
```

**Implementation**: Modify `compute_features_from_stats` in `data.py` to add:

```python
# Example additions
'snap_percentage': stats.get('snaps', 0) / max(team_snaps, 1),
'red_zone_touches': stats.get('rz_targets', 0) + stats.get('rz_carries', 0),
'target_depth': stats.get('air_yards', 0) / max(stats.get('targets', 1), 1),
```

## 🎯 Medium Impact Changes

### 4. **Loss Function & Optimization**

**Location**: `models.py` - `BaseNeuralModel.__init__`

```python
# Current
self.criterion = nn.MSELoss()
self.optimizer = optim.Adam(...)

# Try alternatives
self.criterion = nn.HuberLoss(delta=2.0)  # Less sensitive to outliers
self.criterion = nn.L1Loss()              # Mean Absolute Error

# Different optimizers
self.optimizer = AdamW(..., weight_decay=1e-3)  # Better regularization
self.optimizer = optim.SGD(..., momentum=0.9)   # Sometimes more stable
```

### 5. **Data Preprocessing**

**Location**: `data.py` - end of `get_training_data`

**Current clipping** (line 603-606):

```python
X_train = np.clip(X_train, -1000, 1000)
y_train = np.clip(y_train, 0, 100)
```

**Better preprocessing**:

```python
# Standardization (add after clipping)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Or normalization
X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
```

### 6. **Training Data Quality**

**Location**: `data.py` - modify `get_training_data` query

**Current filter** (line 1982):

```sql
WHERE p.position = ? AND g.season IN ({})
AND g.game_finished = 1
```

**Add quality filters**:

```sql
-- Only include players with significant playing time
AND ps.snaps > 10  -- Or whatever snap count makes sense
-- Remove blowout games (skewed data)
AND ABS(g.home_score - g.away_score) < 21
-- Only include recent, relevant data
AND g.season >= 2020  -- Focus on modern NFL
```

## 🚀 Advanced Improvements

### 7. **Multi-Task Learning**

**Location**: New architecture in `models.py`

Instead of just predicting points, predict multiple targets:

```python
class MultiTaskQBNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared_layers = nn.Sequential(...)

        # Multiple prediction heads
        self.points_head = nn.Linear(64, 1)
        self.yards_head = nn.Linear(64, 1)
        self.tds_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_layers(x)
        return {
            'points': self.points_head(features),
            'yards': self.yards_head(features),
            'tds': self.tds_head(features)
        }
```

### 8. **Ensemble Methods**

**Location**: New file `ensemble.py`

Train multiple models and average predictions:

```python
# Train 5 models with different random seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    torch.manual_seed(seed)
    model = create_model('QB', config)
    model.train(X_train, y_train, X_val, y_val)
    models.append(model)

# Average predictions
predictions = [model.predict(X) for model in models]
final_pred = np.mean([p.point_estimate for p in predictions], axis=0)
```

### 9. **Hyperparameter Search**

**Location**: New file `hyperparameter_search.py`

```python
import itertools

# Define search space
learning_rates = [0.0001, 0.0005, 0.001, 0.005]
batch_sizes = [32, 64, 128]
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

best_r2 = -float('inf')
best_params = None

for lr, bs, hs, dr in itertools.product(learning_rates, batch_sizes, hidden_sizes, dropout_rates):
    # Modify model config
    config = ModelConfig(position='QB')
    model = create_model('QB', config)
    model.learning_rate = lr
    model.batch_size = bs
    # ... modify network architecture with hs, dr

    result = model.train(X_train, y_train, X_val, y_val)
    if result.val_r2 > best_r2:
        best_r2 = result.val_r2
        best_params = (lr, bs, hs, dr)
```

## 🎯 Quick Wins to Try First

### Priority 1: Network Size

```python
# In QBNetwork.__init__, try:
nn.Linear(input_size, 256),  # Was 128
nn.Linear(256, 128),         # Was 64
nn.Linear(128, 64),          # Add extra layer
```

### Priority 2: Better Learning Rate

```python
# In QBNeuralModel.__init__, try:
self.learning_rate = 0.005   # Higher learning rate
self.batch_size = 32         # Smaller batches
```

### Priority 3: More Training Data

```python
# In run.py train command, use more seasons:
uv run python run.py train --positions QB --seasons 2020 2021 2022 2023 2024
```

### Priority 4: Feature Normalization

Add to `data.py` after line 2185:

```python
# Standardize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std
```

## 📊 How to Measure Improvement

### Good R² Targets:

- **R² > 0.0**: Model is better than predicting the mean
- **R² > 0.3**: Decent predictive power
- **R² > 0.5**: Strong predictive power
- **R² > 0.7**: Excellent (unlikely for fantasy sports)

### MAE Targets for QB:

- **MAE < 8**: Excellent (within 1 TD of actual)
- **MAE < 10**: Good (current: 10.580)
- **MAE < 12**: Acceptable
- **MAE > 15**: Needs improvement

## 🔄 Systematic Testing Process

1. **Baseline**: Record current R² = -0.753, MAE = 10.580
2. **Single changes**: Test one modification at a time
3. **Best combinations**: Combine the best individual changes
4. **Document**: Keep track of what works

### Test Command:

```bash
# Quick test (2 seasons)
uv run python run.py train --positions QB --seasons 2023 2024

# Full test (5 seasons)
uv run python run.py train --positions QB --seasons 2020 2021 2022 2023 2024
```

## 🎯 Expected Outcomes

With these changes, you should see:

- **R² improvement**: -0.753 → 0.1 to 0.4 (realistic target)
- **MAE improvement**: 10.580 → 7-9 (significant improvement)
- **Training stability**: Consistent results across runs

Start with the Priority 1-4 quick wins, then move to advanced techniques once you achieve positive R².
