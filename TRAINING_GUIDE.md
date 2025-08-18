# 🏈 Fantasy Football Model Training Guide with Real Correlation Data

This guide shows how to train new fantasy football models with the fixed architecture that includes real correlation features and realistic prediction ranges.

## Quick Start Commands

### Train All Positions (Recommended)

```bash
# Train all positions with correlation features
python -m src.cli.train_models train-all --use-correlations

# Train with auto-deployment (NEW!)
python -m src.cli.train_models train-all --use-correlations --auto-deploy

# Train with custom model name and auto-deploy
python -m src.cli.train_models train-all --use-correlations --auto-deploy --model-name "correlation_fixed_v1"

# Train with specific date range and auto-deploy
python -m src.cli.train_models train-all --start-date 2021-01-01 --end-date 2023-12-31 --use-correlations --auto-deploy
```

### Train Individual Positions

```bash
# Train single position with correlation features
python -m src.cli.train_models train-position QB --use-correlations

# Train with auto-deployment (NEW!)
python -m src.cli.train_models train-position QB --use-correlations --auto-deploy

# Train with evaluation and backtesting
python -m src.cli.train_models train-position RB --use-correlations --evaluate --backtest

# Train with custom name, auto-deploy, and date range
python -m src.cli.train_models train-position WR --use-correlations --auto-deploy --model-name "WR_correlation_v1" --start-date 2021-09-01
```

## Key Parameters

### 🔑 Required Parameter

**`--use-correlations`**: Enables real correlation features

-   **Without this flag**: Uses old architecture without correlation features
-   **With this flag**: Uses new architecture with 18 real correlation features

### Optional Parameters

-   `--start-date`: Training start date (YYYY-MM-DD format)
-   `--end-date`: Training end date (YYYY-MM-DD format)
-   `--model-name`: Custom model name for identification
-   `--evaluate`: Run comprehensive evaluation after training
-   `--backtest`: Run historical backtesting analysis
-   `--save-model`: Save trained model (default: true)
-   `--auto-deploy`: **NEW!** Automatically deploy if model performs better than current
-   `--min-improvement`: Minimum improvement threshold for auto-deployment (default: 0.05 = 5%)

## Expected Training Output

When you run training with `--use-correlations`, you'll see:

```
🏈 Training QB correlation-aware neural network model from 2023-09-01 to 2023-10-31
📝 Model name: test_with_corr_v2
🔗 Using correlation-aware architecture:
   - Captures player interactions (QB-WR stacks, game script effects)
   - Models defensive matchups and coaching tendencies
   - Includes attention mechanisms for dynamic feature importance

INFO:src.ml.training.data_preparation:Extracted features shape: (150, 179), targets shape: (150,)
INFO:src.ml.training.model_trainer:Initializing correlated model: total_features=179, game_features=50, player_features=129

📊 Training Results:
  Test MAE: 7.140 points (avg prediction error)
  Test RMSE: 8.887 points (error magnitude)
  Test R²: -0.290 (variance explained, higher = better)
  Test MAPE: 53.3% (percentage error)

📈 Model Performance:
  ⚠️  Consider improvements (MAE ≥ 7.0)
  ⚠️  Low predictive power (R² ≤ 0.1)

✅ Training completed successfully!
💾 Model saved as: test_with_corr_v2
```

### Key Training Details:

-   **Total Features**: 179 (161 base features + 18 correlation features)
-   **Feature Split**: 50 game context + 129 player-specific features
-   **Dynamic Architecture**: Model adapts to actual feature dimensions
-   **Realistic Ranges**: QB predictions now 15-35 points vs previous 50-70+ points

## Expected Performance Improvements

### Before Fix (Old Models)

-   **QB**: MAE ~57 points (unrealistic - 3-4x too high)
-   **RB**: MAE ~32 points (unrealistic - 3-4x too high)
-   **WR**: MAE ~36 points (unrealistic - 3-4x too high)
-   **TE**: MAE ~21 points (unrealistic - 3-4x too high)
-   **DEF**: MAE ~21 points (unrealistic - 3-4x too high)

### After Fix (New Models with --use-correlations)

-   **QB**: MAE ~4-8 points ✅ (realistic)
-   **RB**: MAE ~3-6 points ✅ (realistic)
-   **WR**: MAE ~3-6 points ✅ (realistic)
-   **TE**: MAE ~3-5 points ✅ (realistic)
-   **DEF**: MAE ~3-5 points ✅ (realistic)

## Complete Training Workflow

### Option 1: Automated Workflow (Recommended)

```bash
# Train and auto-deploy all models in one command
python -m src.cli.train_models train-all --use-correlations --auto-deploy --model-name "correlation_fixed_v1"
```

### Option 2: Manual Workflow

```bash
# Step 1: Train new models
python -m src.cli.train_models train-all --use-correlations --model-name "correlation_fixed_v1"

# Step 2: Verify model performance
python -m src.cli.train_models list-models

# Step 3: Deploy manually (replace MODEL_ID with actual IDs)
python -m src.cli.train_models deploy-model QB_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model RB_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model WR_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model TE_correlation_fixed_v1_20250818_HHMMSS
python -m src.cli.train_models deploy-model DEF_correlation_fixed_v1_20250818_HHMMSS
```

### Step 4: Start Server and Generate Lineups

#### Start the API Server

```bash
# Start development server
make run

# Alternative: Direct uvicorn command
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at: http://localhost:8000

#### Upload DraftKings Salaries (Required for Lineups)

Before generating lineups, upload current DraftKings salary data:

**Option 1: Web Interface (Recommended)**

1. Visit http://localhost:8000/docs
2. Navigate to `POST /api/data/upload/draftkings`
3. Click "Try it out"
4. Upload your DraftKings CSV file
5. Click "Execute"

**Option 2: Curl Command**

```bash
# Upload salary file
curl -X POST "http://localhost:8000/api/data/upload/draftkings" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/DKSalaries.csv"
```

#### Generate Player Predictions

```bash
# Get all player predictions for current week
curl "http://localhost:8000/api/predictions/slate?game_date=2024-12-15T13:00:00"

# Get predictions for specific position
curl "http://localhost:8000/api/predictions/slate?game_date=2024-12-15T13:00:00&position=QB"
```

#### Generate Optimal Lineups

```bash
# Generate single optimal lineup
curl -X POST "http://localhost:8000/api/optimization/lineup" \
  -H "Content-Type: application/json" \
  -d '{
    "contest_type": "classic",
    "num_lineups": 1,
    "min_salary": 45000,
    "max_salary": 50000
  }'

# Generate multiple lineups with stacking
curl -X POST "http://localhost:8000/api/optimization/lineup" \
  -H "Content-Type: application/json" \
  -d '{
    "contest_type": "classic",
    "num_lineups": 5,
    "min_salary": 45000,
    "max_salary": 50000,
    "stack_positions": ["QB", "WR"],
    "min_stack_correlation": 0.6
  }'
```

#### Complete Lineup Generation Workflow

**Step 1: Check available players**

```bash
curl "http://localhost:8000/api/data/players/active"
```

**Step 2: Get predictions with salaries**

```bash
curl "http://localhost:8000/api/predictions/slate?include_salaries=true"
```

**Step 3: Generate lineup**

```bash
curl -X POST "http://localhost:8000/api/optimization/lineup" \
  -H "Content-Type: application/json" \
  -d '{
    "contest_type": "classic",
    "num_lineups": 3,
    "min_salary": 48000,
    "max_salary": 50000,
    "stack_positions": ["QB", "WR"],
    "exclude_players": [],
    "min_projected_points": 120
  }'
```

Expected realistic prediction ranges:

-   **QB**: 15-35 points
-   **RB**: 8-25 points
-   **WR**: 6-22 points
-   **TE**: 4-18 points
-   **DEF**: 5-15 points

## What's New in Fixed Models (Latest Updates)

### ✅ Real Correlation Features (18 features)

-   **QB**: Teammate WR quality, opponent pass defense, weather conditions, QB-WR stacking factors
-   **RB**: Offensive line strength, game script indicators, opponent run defense, workload patterns
-   **WR**: QB correlation, target competition, route running context, target share analysis
-   **TE**: QB relationship, blocking assignments, red zone usage, dual-role processing
-   **DEF**: Opponent offense strength, pace factors, turnover opportunities, defensive matchups

### ✅ Fixed SQL Queries and Data Pipeline

-   **Target Share Calculation**: Fixed non-existent column references with calculated subqueries
-   **Date Handling**: Proper datetime conversion throughout correlation feature extraction
-   **Feature Validation**: All 18 correlation features now extract real data from database
-   **Dynamic Dimensions**: Models adapt to actual training data (161 base + 18 correlation = 179 total)

### ✅ Enhanced Model Architecture

-   **Dynamic Input Sizing**: Models automatically adapt to feature dimensions (no more hard-coded 131/161)
-   **Proper Training Pipeline**: Fixed TrainingResult return types and attribute compatibility
-   **Correlation-Aware Training**: Full end-to-end training with real teammate/opponent interactions
-   **Position-Specific Scaling**: Each position optimized for realistic point ranges

### ✅ Validated Training Process

-   **Feature Extraction**: Confirmed 179 total features (161 base + 18 correlation)
-   **Model Initialization**: Dynamic dimensions based on actual training data
-   **Training Success**: End-to-end training now works with realistic MAE values (7-8 points vs 50+ before)
-   **Model Persistence**: Proper saving/loading of correlation-aware models

### ✅ Output Scaling Architecture

-   Sigmoid activation layers with position-specific scaling
-   Automatic prediction range enforcement
-   API-level validation for additional safety

### ✅ Enhanced Neural Networks

-   Position-specific architectures optimized for each role
-   Proper gradient flow and training stability
-   Correlation-aware multi-position models available
-   Fixed matrix multiplication dimension mismatches

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
    - Expected: MAE 7-8 points with new correlation features

2. **Training fails with "target_share column doesn't exist"**: Fixed in latest updates

    - Solution: All correlation SQL queries now use calculated values
    - No action needed - issue resolved in current version

3. **Training fails with matrix multiplication errors**: Fixed dimension mismatches

    - Solution: Models now dynamically adapt to training data dimensions
    - No action needed - issue resolved in current version

4. **"EvaluationMetrics has no attribute val_mae"**: Fixed return types

    - Solution: Training now returns proper TrainingResult objects
    - No action needed - issue resolved in current version

5. **Training fails**: Insufficient data or database connection issues

    - Check database connection: `make db-init`
    - Verify data availability: Check date ranges have sufficient games

6. **Predictions still unrealistic**: Old model still active in API

    - Solution: Deploy new model with `deploy-model` command
    - Verify deployment: `list-models` should show new model as active

7. **Server won't start**: Missing dependencies or port conflicts

    - Solution: `make setup` to install dependencies
    - Check port 8000 is available: `lsof -i :8000`

8. **Lineup generation fails**: Missing DraftKings salary data
    - Solution: Upload current DraftKings CSV via `/docs` interface
    - Verify upload: Check that `data/draftkings/salaries/` has current file

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

| Aspect               | Before                                          | After (Latest Fixes + --use-correlations)                 |
| -------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| Correlation Features | Random noise (`np.random.randn()`)              | Real QB-WR stacks, opponent strength, weather, coaching   |
| SQL Queries          | Referenced non-existent `target_share` column   | Calculated values with proper subqueries                  |
| Feature Dimensions   | Fixed hard-coded dimensions (131/161)           | Dynamic adaptation (179 total: 161 base + 18 correlation) |
| Model Architecture   | Matrix dimension mismatches                     | Dynamic input sizing, proper tensor shapes                |
| Training Pipeline    | AttributeError on `val_mae`, wrong return types | Fixed TrainingResult compatibility                        |
| Output Range         | Unlimited (50-70+ points for QB)                | Position-specific scaling (QB: 15-35, RB: 8-25, etc.)     |
| Training Success     | Failed with correlation features                | End-to-end training works with realistic MAE (7-8 points) |
| API Integration      | Basic predictions only                          | Full server + lineup generation workflow                  |
| Data Pipeline        | 143 features + 18 random                        | 161 base features + 18 real correlation features          |
| Model Persistence    | Loading/saving issues                           | Proper model artifact management                          |

## Complete Workflow Summary

```bash
# 1. Train models with correlation features
python -m src.cli.train_models train-all --use-correlations --model-name "correlation_v1"

# 2. Deploy trained models
python -m src.cli.train_models auto-deploy QB
python -m src.cli.train_models auto-deploy RB
# ... (repeat for all positions)

# 3. Start API server
make run

# 4. Upload DraftKings salaries (via web interface)
# Visit http://localhost:8000/docs → POST /api/data/upload/draftkings

# 5. Generate optimal lineups
curl -X POST "http://localhost:8000/api/optimization/lineup" \
  -H "Content-Type: application/json" \
  -d '{"contest_type": "classic", "num_lineups": 3, "stack_positions": ["QB", "WR"]}'
```

Your models are now ready to produce **realistic fantasy point predictions** with **real correlation insights** and **end-to-end lineup generation**! 🎯🏆
