# Hyperparameter Tuning Guide for NFL DFS Models

## Overview

This guide outlines automated methods to find optimal learning rates and batch sizes for your position-specific neural networks, replacing the current fixed values.

## Current State

- **Fixed Learning Rates**: QB=0.0001, RB=0.00003, WR/TE=0.0001
- **Fixed Batch Sizes**: QB=64, RB=128, WR/TE=64
- **Fixed Epochs**: QB=1000, RB=800, WR/TE=600

## Method 1: Learning Rate Range Test (LR Finder) - RECOMMENDED

### Algorithm

1. **Start Small**: Initialize LR at 1e-8
2. **Exponential Increase**: Multiply by ~1.3 each batch
3. **Monitor Loss**: Track smoothed loss vs learning rate
4. **Find Sweet Spot**: Identify steepest decline point
5. **Stop Early**: When loss explodes (>4x min loss)

### Implementation Strategy

```python
class LRFinder:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def range_test(self, train_loader, start_lr=1e-8, end_lr=1, num_iter=100):
        # Exponential LR schedule
        # Track loss vs LR
        # Return optimal LR (steepest decline point)

    def find_optimal_lr(self, losses, lrs):
        # Calculate gradients of loss curve
        # Find point with steepest negative gradient
        # Return LR at that point
```

### Integration Points in models.py

- Add `find_optimal_lr()` method to `BaseNeuralModel`
- Call before training: `optimal_lr = self.find_optimal_lr(train_loader)`
- Update `self.learning_rate = optimal_lr`

## Method 2: Batch Size Optimization

### Approach: Binary Search with Memory Constraints

1. **Start Range**: [8, max_memory_allows]
2. **Test Performance**: Train for 50 epochs at each size
3. **Memory Check**: Monitor GPU/MPS memory usage
4. **Binary Search**: Narrow to optimal range
5. **Select Best**: Choose size with best val_loss

### Memory Calculation for MPS

```python
def find_max_batch_size(model, sample_input):
    # Start with small batch, increase until OOM
    # Return largest feasible batch size

def optimize_batch_size(model, train_loader, val_loader, lr):
    # Test batch sizes: [16, 32, 64, 128, 256]
    # Train for 50 epochs each
    # Return best performing batch size
```

## Method 3: Joint Optimization (Advanced)

### Bayesian Optimization with Optuna

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Train model with these params for 100 epochs
    # Return validation R²

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

## Implementation Plan

### Phase 1: LR Finder

1. Add `LRFinder` class to models.py
2. Add `find_optimal_lr()` method to `BaseNeuralModel`
3. Integrate into training pipeline with `--tune-lr` flag

### Phase 2: Batch Size Optimization

1. Add `BatchSizeOptimizer` class
2. Add memory profiling utilities
3. Integrate with `--tune-batch-size` flag

### Phase 3: Joint Optimization

1. Add Optuna dependency
2. Create `HyperparameterTuner` class
3. Add `--tune-all` flag for full optimization

## Expected Benefits

### Learning Rate Optimization

- **Current**: Manual guessing, suboptimal convergence
- **With LR Finder**: Automated optimal LR, faster convergence
- **Expected Improvement**: 10-20% better R² scores

### Batch Size Optimization

- **Current**: Fixed sizes, may be suboptimal
- **Optimized**: Memory-efficient, best gradient estimates
- **Expected Improvement**: 5-15% training speed boost

### Joint Optimization

- **Current**: Independent hyperparams
- **Optimized**: Synergistic hyperparameter combinations
- **Expected Improvement**: 15-30% overall performance gain

## Usage Examples

```bash
# Find optimal learning rate only
uv run python run.py train --tune-lr

# Optimize batch size for memory efficiency
uv run python run.py train --tune-batch-size

# Full hyperparameter optimization (20 trials)
uv run python run.py train --tune-all --trials 20

# Use found hyperparameters for production training
uv run python run.py train --lr 0.0023 --batch-size 96
```

## Validation Strategy

### A/B Testing

1. **Baseline**: Train with current fixed hyperparameters
2. **Optimized**: Train with auto-tuned hyperparameters
3. **Compare**: R², MAE, training time on same data split
4. **Decision**: Adopt if >10% improvement

### Position-Specific Tuning

- **QB**: Focus on LR (complex architecture needs careful tuning)
- **RB**: Joint optimization (proven working baseline to improve)
- **WR/TE**: Batch size focus (simpler models, memory efficiency)
- **DST**: Skip (CatBoost already optimized)

## Risk Mitigation

### Overfitting Prevention

- Use separate validation set for hyperparameter tuning
- Cross-validate hyperparameter choices
- Early stopping during tuning phases

### Time Management

- Set reasonable trial limits (20 max)
- Use smaller epoch counts during search (100 epochs)
- Cache results to avoid recomputation

### Fallback Strategy

- Always keep current working hyperparameters
- Graceful degradation if tuning fails
- Manual override options for production

## Timeline Estimate

- **LR Finder**: 2-3 hours implementation
- **Batch Size Optimizer**: 1-2 hours implementation
- **Joint Optimization**: 3-4 hours implementation
- **Testing & Validation**: 4-6 hours
- **Total**: 1-2 days for complete system
