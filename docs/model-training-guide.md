# NFL DFS Model Training Guide

This guide covers how to train, configure, and optimize machine learning models for fantasy football predictions.

## Overview

The system supports two types of models:

- **Traditional ML Models**: XGBoost, LightGBM, Random Forest (faster training, good baseline performance)
- **Neural Network Models**: PyTorch deep learning models (longer training, potentially better performance)

## Quick Start

### Train All Models at Once

```bash
# Train traditional models for all positions
uv run python -m src.cli.train_models train-all-positions

# Train with date range
uv run python -m src.cli.train_models train-all-positions --start-date 2020-09-01 --end-date 2023-12-31
```

### Train Individual Position Models

```bash
# Basic training
uv run python -m src.cli.train_models train-position QB
uv run python -m src.cli.train_models train-position RB
uv run python -m src.cli.train_models train-position WR
uv run python -m src.cli.train_models train-position TE
uv run python -m src.cli.train_models train-position DEF

# Train neural network model
uv run python -m src.cli.train_models train-position QB --use-neural

# Train with custom parameters
uv run python -m src.cli.train_models train-position QB \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --model-name QB_experiment_v2 \
  --backtest \
  --evaluate
```

## Model Configuration

### ModelConfig Parameters

Each model can be configured through the `ModelConfig` class. Key parameters:

```python
# Model identification
model_name: str              # Custom name for the model
position: str                # QB, RB, WR, TE, DEF
version: str = "1.0"         # Model version

# Training parameters
random_state: int = 42       # For reproducible results
test_size: float = 0.2       # 20% of data held out for testing
validation_size: float = 0.2 # 20% of training data for validation

# CPU optimization
n_jobs: int = -1             # Use all CPU cores
use_early_stopping: bool = True
early_stopping_rounds: int = 50

# Feature engineering
feature_selection: bool = True
max_features: int | None = None
min_feature_importance: float = 0.001

# Performance thresholds
min_r2_score: float = 0.3    # Minimum R² for acceptable model
max_mae_threshold: float = 5.0  # Maximum MAE for acceptable model
```

### Position-Specific Configurations

Each position uses optimized hyperparameters:

#### QB Model (XGBoost)

```python
XGBRegressor(
    n_estimators=500,        # Number of boosting rounds
    max_depth=8,             # Tree depth (controls complexity)
    learning_rate=0.03,      # Step size (lower = more conservative)
    subsample=0.8,           # Row sampling (prevents overfitting)
    colsample_bytree=0.8,    # Feature sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
)
```

#### RB Model (LightGBM)

```python
LGBMRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=64,           # LightGBM-specific: leaf complexity
    feature_fraction=0.8,
    bagging_fraction=0.7,
    reg_alpha=0.1,
    reg_lambda=0.5,
)
```

#### WR Model (XGBoost)

```python
XGBRegressor(
    n_estimators=600,        # More estimators for high-variance position
    max_depth=10,            # Deeper trees for complex patterns
    learning_rate=0.02,      # Lower learning rate for stability
    subsample=0.7,           # More aggressive sampling
    colsample_bytree=0.7,
    reg_alpha=0.2,           # Higher regularization
    reg_lambda=1.5,
)
```

#### TE Model (LightGBM)

```python
LGBMRegressor(
    n_estimators=300,        # Fewer estimators (TEs more predictable)
    max_depth=5,             # Shallower trees
    learning_rate=0.08,      # Higher learning rate
    num_leaves=32,
    feature_fraction=0.9,    # Use more features
    bagging_fraction=0.8,
    reg_alpha=0.05,          # Less regularization
    reg_lambda=0.3,
)
```

#### DEF Model (Random Forest)

```python
RandomForestRegressor(
    n_estimators=200,        # Balance performance vs speed
    max_depth=12,            # Deep trees for defense complexity
    min_samples_split=10,    # Prevent overfitting
    min_samples_leaf=5,
    max_features="sqrt",     # Feature sampling strategy
    bootstrap=True,          # Enable bagging
)
```

## Neural Network Models

### When to Use Neural Networks

Use `--use-neural` flag when:

- Traditional models plateau in performance
- You have large amounts of training data (5+ seasons)
- Complex player interactions need to be modeled
- You're willing to wait for longer training times

### Neural Network Architectures

#### QB Neural Model

- **Multi-task learning**: Separate branches for passing and rushing
- **Attention mechanism**: Focus on important features dynamically
- **Architecture**: 3 hidden layers with dropout and batch normalization

#### RB Neural Model

- **Workload clustering**: Embeddings for different RB types
- **Dual branches**: Separate processing for workload vs efficiency
- **Batch normalization**: For stable training

#### WR Neural Model

- **Target competition**: Models competition for targets
- **High dropout**: Handles WR volatility
- **Big play potential**: Separate branch for explosive plays

#### TE Neural Model

- **Dual-role modeling**: Receiving and blocking importance
- **Simplified architecture**: Reflects moderate TE variance

#### DEF Neural Model

- **Multi-head ensemble**: Separate heads for pressure, turnovers, points
- **High variance modeling**: Specialized dropout patterns

## Hyperparameter Tuning

### Manual Tuning Guidelines

#### Learning Rate

```bash
# Too high (>0.1): Model may not converge
# Too low (<0.01): Training takes forever
# Good range: 0.02-0.08
```

#### Tree Depth

```bash
# Shallow (3-5): May underfit, good for simple patterns
# Medium (6-8): Good balance for most positions
# Deep (9-12): Risk overfitting, use with regularization
```

#### Regularization

```bash
# Low regularization: Model may memorize training data
# High regularization: Model may be too simple
# L1 (alpha): Feature selection, use 0.1-0.5
# L2 (lambda): Smooth weights, use 0.5-2.0
```

### Automated Tuning (Future Enhancement)

The system is designed to support automated hyperparameter tuning:

```python
# Future implementation
from src.ml.tuning import HyperparameterTuner

tuner = HyperparameterTuner(position="QB")
best_params = tuner.optimize(
    param_grid={
        'n_estimators': [300, 400, 500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.02, 0.03, 0.05]
    },
    cv_folds=5,
    scoring='neg_mean_absolute_error'
)
```

## Model Evaluation and Selection

### Performance Metrics

#### Primary Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in fantasy points

  - Excellent: < 3.0
  - Good: 3.0-5.0
  - Needs improvement: > 5.0

- **R² (R-squared)**: Proportion of variance explained

  - Excellent: > 0.4
  - Good: 0.2-0.4
  - Poor: < 0.2

- **MAPE (Mean Absolute Percentage Error)**: Error as percentage

  - Good: < 15%
  - Acceptable: 15-25%
  - Poor: > 25%

#### Secondary Metrics

- **RMSE**: Penalizes large errors more than MAE
- **Consistency Score**: How stable predictions are
- **Outlier Percentage**: How often large errors occur

### Model Comparison

```bash
# Evaluate specific model
uv run python -m src.cli.train_models evaluate-model QB_model_20240315_143022

# Compare models
uv run python -m src.cli.train_models compare-models QB

# List all trained models
uv run python -m src.cli.train_models list-models

# Get detailed model info
uv run python -m src.cli.train_models model-info QB_model_20240315_143022
```

### Backtesting

Backtesting tests model performance on historical data:

```bash
# Backtest during training
uv run python -m src.cli.train_models train-position QB --backtest

# Backtest existing model
uv run python -m src.cli.train_models evaluate-model QB_model_20240315_143022 --backtest
```

Backtesting metrics include:

- **Hit Rate**: Percentage of successful predictions
- **Simulated ROI**: Return on investment in simulated contests
- **Weekly Performance**: Consistency across different weeks

## Model Deployment

### Manual Deployment

```bash
# Deploy specific model
uv run python -m src.cli.train_models deploy-model QB_model_20240315_143022

# Auto-deploy best model for position
uv run python -m src.cli.train_models auto-deploy QB --min-improvement 0.05

# Validate model before deployment
uv run python -m src.cli.train_models validate-model QB_model_20240315_143022
```

### Rollback and Management

```bash
# Rollback to previous model
uv run python -m src.cli.train_models rollback QB

# Retire old model
uv run python -m src.cli.train_models retire-model QB_model_old

# Clean up old models
uv run python -m src.cli.train_models cleanup-models --keep-recent 5
```

## Best Practices

### Data Preparation

1. **Sufficient Data**: Use at least 2 full seasons for training
1. **Recent Data**: Weight recent games more heavily
1. **Feature Quality**: Ensure all features are clean and validated
1. **Time-Based Splits**: Use chronological train/validation/test splits

### Training Process

1. **Start Simple**: Begin with traditional models before trying neural networks
1. **Cross-Validation**: Use time-series aware cross-validation
1. **Early Stopping**: Prevent overfitting with patience parameter
1. **Regular Retraining**: Retrain models weekly during season

### Model Selection

1. **Validation Performance**: Prioritize validation metrics over training metrics
1. **Consistency**: Choose models with stable performance across different time periods
1. **Interpretability**: Traditional models are easier to debug and understand
1. **Production Requirements**: Consider inference speed and memory usage

### Monitoring

1. **Track Drift**: Monitor when model performance degrades
1. **A/B Testing**: Compare new models against existing ones
1. **Business Metrics**: Track actual contest performance, not just technical metrics
1. **Feature Importance**: Monitor which features drive predictions

## Troubleshooting

### Common Issues

#### Poor Model Performance (MAE > 7.0)

```bash
# Check data quality
uv run python -m src.cli.collect_data status

# Try different algorithms
uv run python -m src.cli.train_models train-position QB --use-neural

# Adjust training period
uv run python -m src.cli.train_models train-position QB --start-date 2022-01-01
```

#### Overfitting (Big gap between train/validation performance)

- Increase regularization parameters
- Reduce model complexity (max_depth, n_estimators)
- Use more training data
- Enable early stopping

#### Underfitting (Poor performance on both train/validation)

- Decrease regularization
- Increase model complexity
- Add more features
- Try neural networks

#### Long Training Times

- Reduce n_estimators
- Use fewer features
- Increase learning_rate
- Use traditional models instead of neural networks

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
uv run python -m src.cli.train_models train-position QB

# Save detailed training metrics
uv run python -m src.cli.train_models train-position QB --evaluate --backtest
```

## Advanced Topics

### Custom Features

Add new features by modifying the feature engineering pipeline:

```python
# In src/data/processing/feature_engineering.py
def add_custom_feature(df):
    df['custom_metric'] = df['target_share'] * df['air_yards']
    return df
```

### Ensemble Models

Train ensemble models for potentially better performance:

```bash
# Train ensemble (combines multiple algorithms)
uv run python -m src.cli.train_models train-all-positions --ensemble
```

### Model Interpretability

```python
# Get feature importance
model = trainer.load_model("QB_model_20240315_143022")
importance = model.get_feature_importance()

# SHAP values for individual predictions
import shap
explainer = shap.Explainer(model.model)
shap_values = explainer(X_test)
```

### Production Optimization

For production deployment:

1. **Model Compression**: Use simpler models for faster inference
1. **Batch Prediction**: Process multiple players at once
1. **Caching**: Cache predictions for repeated queries
1. **Monitoring**: Set up alerts for performance degradation

## API Integration

Models can be used via the REST API:

```bash
# Get predictions for a player
curl -X POST http://localhost:8000/api/predictions/player \
  -H "Content-Type: application/json" \
  -d '{"player_id": "mahomes", "week": 10, "season": 2024}'

# Get predictions for full slate
curl -X POST http://localhost:8000/api/predictions/slate \
  -H "Content-Type: application/json" \
  -d '{"week": 10, "season": 2024}'
```

This guide provides comprehensive coverage of model training in the NFL DFS system. For specific issues or advanced use cases, refer to the source code in `src/ml/` or open an issue in the project repository.
