# DFS Model Tuning Guide: From R² = -0.753 to Predictive Success

This guide covers all the knobs you can tweak to improve your DFS neural network models, organized by impact and difficulty.

## 🎯 Current Status

- QB Model: MAE = 10.580, R² = -0.753
- Training is stable, no NaN issues
- 38 features, ~1300 training samples

## 🔧 High Impact, Easy Changes

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
