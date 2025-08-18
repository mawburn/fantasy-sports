# 🏈 Fantasy Football Model Training Guide with Real Correlation Data

This guide shows how to train new fantasy football models with the fixed architecture that includes real correlation features and realistic prediction ranges.

## Quick Start Commands

### Train All Positions (Recommended)
```bash
# Train all positions with correlation features
python -m src.cli.train_models train-all --use-correlations

# Train with custom model name
python -m src.cli.train_models train-all --use-correlations --model-name "correlation_fixed_v1"

# Train with specific date range
python -m src.cli.train_models train-all --start-date 2021-01-01 --end-date 2023-12-31 --use-correlations
```

### Train Individual Positions
```bash
# Train single position with correlation features
python -m src.cli.train_models train-position QB --use-correlations

# Train with evaluation and backtesting
python -m src.cli.train_models train-position RB --use-correlations --evaluate --backtest

# Train with custom name and date range
python -m src.cli.train_models train-position WR --use-correlations --model-name "WR_correlation_v1" --start-date 2021-09-01
```

## Key Parameters

### 🔑 Required Parameter
**`--use-correlations`**: Enables real correlation features
- **Without this flag**: Uses old architecture without correlation features  
- **With this flag**: Uses new architecture with 18 real correlation features

### Optional Parameters
- `--start-date`: Training start date (YYYY-MM-DD format)
- `--end-date`: Training end date (YYYY-MM-DD format)  
- `--model-name`: Custom model name for identification
- `--evaluate`: Run comprehensive evaluation after training
- `--backtest`: Run historical backtesting analysis
- `--save-model`: Save trained model (default: true)

## Expected Training Output

When you run training with `--use-correlations`, you'll see:

```
🏈 Training QB correlation-aware neural network model from 2021-01-01 to 2023-12-31
📝 Model name: QB_model
🔗 Using correlation-aware architecture:
   - Captures player interactions (QB-WR stacks, game script effects)
   - Models defensive matchups and coaching tendencies  
   - Includes attention mechanisms for dynamic feature importance

📊 Training Results:
  Test MAE: 4.245 points (avg prediction error)
  Test RMSE: 6.123 points (error magnitude)
  Test R²: 0.342 (variance explained, higher = better)
  Test MAPE: 23.1% (percentage error)

📈 Model Performance:
  ✅ Excellent accuracy (MAE < 5.0)
  ✅ Strong predictive power (R² > 0.3)

✅ Training completed successfully!
💾 Model saved as: QB_model
```

## Expected Performance Improvements

### Before Fix (Old Models)
- **QB**: MAE ~57 points (unrealistic - 3-4x too high)
- **RB**: MAE ~32 points (unrealistic - 3-4x too high)
- **WR**: MAE ~36 points (unrealistic - 3-4x too high)
- **TE**: MAE ~21 points (unrealistic - 3-4x too high)
- **DEF**: MAE ~21 points (unrealistic - 3-4x too high)

### After Fix (New Models with --use-correlations)
- **QB**: MAE ~4-8 points ✅ (realistic)
- **RB**: MAE ~3-6 points ✅ (realistic)
- **WR**: MAE ~3-6 points ✅ (realistic)
- **TE**: MAE ~3-5 points ✅ (realistic)
- **DEF**: MAE ~3-5 points ✅ (realistic)

## Complete Training Workflow

### Step 1: Train New Models
```bash
# Train all positions with correlation features
python -m src.cli.train_models train-all --use-correlations --model-name "correlation_fixed_v1"
```

### Step 2: Verify Model Performance
```bash
# List all trained models
python -m src.cli.train_models list-models

# Check specific model details
python -m src.cli.train_models model-info [MODEL_ID]
```

### Step 3: Deploy New Models
```bash
# Deploy new models to production (replace MODEL_ID with actual IDs)
python -m src.cli.train_models deploy-model QB_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model RB_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model WR_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model TE_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model DEF_correlation_fixed_v1_20250818_HHMMSS
```

### Step 4: Test Predictions
```bash
# Start API server
make run

# Test predictions (in another terminal)
curl "http://localhost:8000/api/predictions/slate?game_date=2024-12-15T13:00:00"
```

Expected realistic prediction ranges:
- **QB**: 15-35 points
- **RB**: 8-25 points  
- **WR**: 6-22 points
- **TE**: 4-18 points
- **DEF**: 5-15 points

## What's New in Fixed Models

### ✅ Real Correlation Features (18 features)
- **QB**: Teammate WR quality, opponent pass defense, weather conditions
- **RB**: Offensive line strength, game script indicators, opponent run defense  
- **WR**: QB correlation, target competition, route running context
- **TE**: QB relationship, blocking assignments, red zone usage
- **DEF**: Opponent offense strength, pace factors, turnover opportunities

### ✅ Output Scaling Architecture
- Sigmoid activation layers with position-specific scaling
- Automatic prediction range enforcement
- API-level validation for additional safety

### ✅ Enhanced Neural Networks
- Position-specific architectures optimized for each role
- Proper gradient flow and training stability
- Correlation-aware multi-position models available

## Model Management Commands

### List and Inspect Models
```bash
# List all trained models
python -m src.cli.train_models list-models

# Get detailed model information
python -m src.cli.train_models model-info QB_QB_model_20250818_213455

# Validate model integrity
python -m src.cli.train_models validate-model QB_QB_model_20250818_213455
```

### Deploy and Manage Models
```bash
# Deploy model to production
python -m src.cli.train_models deploy-model QB_QB_model_20250818_213455

# Auto-deploy best performing model
python -m src.cli.train_models auto-deploy QB --min-improvement 0.05

# Rollback to previous model
python -m src.cli.train_models rollback QB

# Retire old model
python -m src.cli.train_models retire-model QB_QB_model_20250818_213455
```

### Evaluate Models
```bash
# Comprehensive evaluation with backtesting
python -m src.cli.train_models evaluate-model QB_QB_model_20250818_213455 --backtest

# Compare models for a position
python -m src.cli.train_models compare-models QB
```

### Clean Up
```bash
# Clean up old models (keep 5 most recent per position)
python -m src.cli.train_models cleanup-models --keep-recent 5
```

## Troubleshooting

### Common Issues

1. **High MAE values (>20)**: Model using old architecture without fixes
   - Solution: Ensure you use `--use-correlations` flag

2. **Training fails**: Insufficient data or database connection issues
   - Check database connection: `make db-init`
   - Verify data availability: Check date ranges have sufficient games

3. **Predictions still unrealistic**: Old model still active in API
   - Solution: Deploy new model with `deploy-model` command
   - Verify deployment: `list-models` should show new model as active

### Getting Help
```bash
# Show all available commands
python -m src.cli.train_models --help

# Get help for specific command
python -m src.cli.train_models train-position --help
python -m src.cli.train_models train-all --help
```

## Next Steps After Training

1. **Monitor Performance**: Check MAE values are realistic (< 8 points)
2. **Deploy Models**: Use deploy commands to activate new models
3. **Test API**: Verify predictions via API endpoints
4. **Set up Monitoring**: Track model performance over time
5. **Schedule Retraining**: Set up regular model updates with new data

---

## Key Differences from Before

| Aspect | Before | After (with --use-correlations) |
|--------|--------|--------------------------------|
| Correlation Features | Random noise | Real QB-WR stacks, opponent strength, weather |
| Output Range | Unlimited (50-70+ points) | Position-specific (QB: 0-45, RB: 0-35, etc.) |
| Architecture | Basic neural networks | Enhanced with scaling and validation |
| Training Data | 143 features + 18 random | 143 features + 18 real correlations |
| API Safety | No validation | Range validation and clipping |

Your models are now ready to produce **realistic fantasy point predictions** with **real correlation insights**! 🎯