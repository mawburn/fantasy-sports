# Fix Model Predictions - Action Items

## Current State

The ML models are deployed but not functioning correctly:

1. **Feature mismatch**: Models expect 179 features but receive 161
2. **Unrealistic MAE values**: Models show MAE of 20-57 points (10x too high for single-game predictions)
3. **Suspicious metrics**: All models have exactly R² = 0.3, suggesting training issues
4. **Fallback system active**: Currently using hardcoded position averages (QB: 22, RB: 16, WR: 14, TE: 10, DST: 8)

## Critical Issues to Fix

### 1. Feature Engineering Mismatch (HIGHEST PRIORITY)

**Problem**: `X has 161 features, but RobustScaler is expecting 179 features as input`

**Root Cause**: Training and prediction pipelines are creating different feature sets

**Fix Steps**:

```bash
# 1. Debug feature creation
python -c "
from src.ml.training.data_preparation import DataPreparator
from src.data.processing.feature_extractor import FeatureExtractor
from datetime import datetime

# Check training features
prep = DataPreparator()
data = prep.prepare_position_data('RB', datetime(2023,1,1), datetime(2024,12,31))
print(f'Training features: {data[\"feature_names\"]}')
print(f'Training feature count: {len(data[\"feature_names\"])}')

# Check prediction features
extractor = FeatureExtractor()
# Compare feature extraction methods
"

# 2. Save feature names with model
# Update src/ml/training/model_trainer.py:_save_model_artifacts()
# to save feature_names in metadata

# 3. Ensure prediction uses exact same features
# Update src/api/routers/predictions.py to use saved feature list
```

### 2. Model Output Scale Issue

**Problem**: Models appear to predict season totals, not single-game points

**Evidence**:

-   QB MAE: 57 points (typical single game: 15-30)
-   RB MAE: 33 points (typical single game: 8-20)
-   WR MAE: 35 points (typical single game: 6-18)

**Fix Steps**:

```bash
# 1. Verify target variable in training data
python -c "
from src.database.connection import get_db
from src.database.models import PlayerStats
db = next(get_db())
stats = db.query(PlayerStats).limit(10).all()
for s in stats:
    print(f'{s.player_name}: {s.fantasy_points_draftkings} points')
"

# 2. Check if using cumulative vs per-game stats
# Look at src/ml/training/data_preparation.py:_calculate_fantasy_points()

# 3. Retrain with correct target variable
python -m src.cli.train_models train --position QB --start-date 2023-01-01 --end-date 2024-12-31
```

### 3. Correlation Features Not Working

**Problem**: CorrelatedFantasyModel loads but features don't match

**Fix Steps**:

```bash
# 1. Check correlation feature extraction
python -c "
from src.ml.training.correlation_features import CorrelationFeatureExtractor
extractor = CorrelationFeatureExtractor()
# Verify feature generation
"

# 2. Ensure consistent feature ordering
# Add feature name tracking to CorrelatedModelWrapper
```

## Quick Fixes Applied

✅ **Model deployment fixed**: Changed status filter from "trained" to ["trained", "deployed"]
✅ **Fallback projections improved**: Updated from 102 to 130 total points
✅ **Value calculation added**: Now calculating points per $1000 salary

## Recommended Immediate Actions

1. **Use fallback system for now** - It's producing reasonable 130-point lineups
2. **Fix feature mismatch first** - This blocks all ML predictions
3. **Verify data scale** - Ensure models train on single-game, not season totals
4. **Add monitoring** - Log all prediction attempts to diagnose issues

## Testing Commands

```bash
# Test current prediction service
curl -X POST "http://localhost:8000/api/predictions/player" \
  -H "Content-Type: application/json" \
  -d '{"player_id": 436, "game_date": "2025-09-07", "position": "RB"}' | jq .

# Test lineup optimization (uses fallbacks currently)
curl -X POST "http://localhost:8000/api/optimize/lineup" \
  -H "Content-Type: application/json" \
  -d '{"contest_id": 1, "salary_cap": 50000, "min_salary": 45000}' | jq .

# Check model metrics
sqlite3 data/database/nfl_dfs.db \
  "SELECT position, model_id, mae_validation FROM model_metadata WHERE is_active = 1;"
```

## Long-term Improvements

1. **Feature pipeline standardization**: Create single source of truth for feature generation
2. **Model versioning**: Include feature specs in model metadata
3. **Backtesting framework**: Validate predictions against historical contests
4. **Auto-retraining**: Detect and fix data drift automatically
5. **A/B testing**: Compare fallback vs ML predictions in production

## Success Criteria

-   [ ] ML predictions return without errors
-   [ ] MAE values are realistic (QB: 4-8, RB: 3-6, WR: 3-6, TE: 2-5, DST: 2-4)
-   [ ] Lineup projections reach 140-180+ points for top plays
-   [ ] Models outperform fallback system by 10%+
-   [ ] Feature counts match between training and prediction
