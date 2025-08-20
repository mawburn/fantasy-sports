# Feature Schema Documentation

## Overview

The `feature_names.json` file defines the fixed feature ordering and schema for the NFL DFS prediction models. This file is critical for maintaining model compatibility and ensuring consistent feature engineering across training and inference.

## Purpose

1. **Model Compatibility**: Ensures that features are fed to trained models in the exact same order they expect
2. **Feature Validation**: Provides the expected schema for validating input data quality
3. **Version Control**: Tracks changes to the feature set over time
4. **Documentation**: Serves as the definitive reference for all model features

## File Location

```
dfs/feature_names.json
```

## Structure

The file contains a JSON array of 40 feature names in fixed order:

```json
[
  "team_spread",
  "team_spread_abs", 
  "total_line",
  "game_tot_z",
  "team_itt",
  "team_itt_z",
  "is_favorite",
  ...
]
```

## Feature Categories

### 1. Odds Features (7 features)
```json
[
  "team_spread",        // Point spread (negative = favorite)
  "team_spread_abs",    // Absolute value of spread
  "total_line",         // Over/under total points
  "game_tot_z",         // Weekly z-score of total vs other games
  "team_itt",           // Implied team total (total/2 - spread/2)
  "team_itt_z",         // Weekly z-score of implied team total
  "is_favorite"         // Binary flag (1 = favorite, 0 = underdog)
]
```

### 2. Weather Features (7 features)
```json
[
  "temperature_f",      // Temperature in Fahrenheit
  "wind_mph",           // Wind speed in MPH
  "humidity_pct",       // Humidity percentage
  "cold_lt40",          // Binary flag: temperature < 40°F
  "hot_gt85",           // Binary flag: temperature > 85°F
  "wind_gt15",          // Binary flag: wind > 15 MPH
  "dome"                // Binary flag: indoor/dome stadium
]
```

### 3. Injury Features (9 features)
```json
[
  "injury_status_Out",          // One-hot: Player listed as Out
  "injury_status_Doubtful",     // One-hot: Player listed as Doubtful
  "injury_status_Questionable", // One-hot: Player listed as Questionable
  "injury_status_Probable",     // One-hot: Player listed as Probable/Healthy
  "games_missed_last4",         // Count of games missed in last 4 weeks
  "practice_trend",             // Practice participation trend (-1, 0, 1)
  "returning_from_injury",      // Binary: returning after missing games
  "team_injured_starters",      // Count of injured starters on team
  "opp_injured_starters"        // Count of injured starters on opponent
]
```

### 4. Usage/Opportunity Features (8 features)
```json
[
  "targets_ema",        // Exponential moving average of targets
  "routes_run_ema",     // EMA of routes run (WR/TE)
  "rush_att_ema",       // EMA of rushing attempts
  "snap_share_ema",     // EMA of snap share percentage
  "redzone_opps_ema",   // EMA of red zone opportunities
  "air_yards_ema",      // EMA of air yards (passing)
  "adot_ema",           // EMA of average depth of target
  "yprr_ema"            // EMA of yards per route run
]
```

### 5. Efficiency Features (4 features)
```json
[
  "yards_after_contact",    // Average yards after contact
  "missed_tackles_forced",  // Missed tackles forced per touch
  "pressure_rate",          // Pressure rate (QB metric)
  "opp_dvp_pos_allowed"     // Opponent defense vs position ranking
]
```

### 6. Contextual Features (5 features)
```json
[
  "salary",         // DraftKings salary
  "home",           // Binary flag: playing at home
  "rest_days",      // Days of rest since last game
  "travel",         // Travel distance (miles)
  "season_week"     // Week of season (normalized)
]
```

## Usage in Code

### Loading the Schema
```python
from utils_feature_validation import load_expected_schema

# Load expected feature ordering
expected_schema = load_expected_schema("feature_names.json")
print(f"Expected {len(expected_schema)} features")
```

### Feature Validation
```python
from utils_feature_validation import validate_and_prepare_features
import pandas as pd

# Validate and reorder features to match schema
df_validated = validate_and_prepare_features(
    df_features, 
    expected_schema, 
    allow_extra=True
)
```

### Model Training Integration
```python
# In data.py get_training_data()
try:
    expected_schema = load_expected_schema("feature_names.json")
    df = validate_and_prepare_features(df, expected_schema, allow_extra=True)
    
    # Update arrays after validation
    X = df.values.astype(np.float32)
    feature_names = df.columns.tolist()
    logger.info(f"Feature validation passed: {len(feature_names)} features validated")
except Exception as ve:
    logger.warning(f"Feature validation failed: {ve}")
```

## Validation Rules

### Data Quality Checks
- **Column Presence**: All expected features must be present
- **Column Order**: Features are reordered to match schema
- **Data Types**: Numeric features converted to float32
- **Range Validation**: Soft range checks for key features
- **Binary Validation**: Binary flags must be 0 or 1
- **Injury Exclusivity**: Only one injury status can be true per player
- **NaN/Inf Detection**: No invalid values allowed

### Soft Ranges (Warnings Only)
```python
NUMERIC_SOFT_RANGES = {
    "team_spread": (-30, 30),
    "team_spread_abs": (0, 30), 
    "total_line": (20, 75),
    "team_itt": (7, 45),
    "temperature_f": (-10, 120),
    "wind_mph": (0, 60),
    "humidity_pct": (0, 100),
    "salary": (2000, 12000),
    ...
}
```

## Maintenance

### Adding New Features
1. **Update feature_names.json**: Add new features at the end (never reorder existing)
2. **Update validation rules**: Add ranges and validation logic in `utils_feature_validation.py`
3. **Update feature engineering**: Implement feature calculation in `data.py`
4. **Update tests**: Add test cases in `tests/test_features.py`
5. **Retrain models**: Models must be retrained with new feature set

### Removing Features
**⚠️ Warning**: Removing features breaks existing trained models
1. **Train new models** with reduced feature set first
2. **Update schema** to remove deprecated features
3. **Update validation** to handle missing features gracefully

### Version Compatibility
- Models store the feature_names.json they were trained with
- Inference validates that current schema matches model schema
- Mismatches prevent loading/prediction to avoid silent failures

## Best Practices

1. **Never reorder existing features** - always append new ones
2. **Validate before training** - catch data issues early
3. **Test schema changes** - run test suite after modifications
4. **Document feature meanings** - maintain clear descriptions
5. **Use semantic names** - feature names should be self-documenting

## Integration Points

The feature schema is used by:

1. **Feature Engineering** (`data.py`): Ensures consistent feature calculation
2. **Training Pipeline** (`run.py`): Validates data before model training
3. **Prediction Pipeline**: Validates input features match model expectations
4. **Test Suite** (`tests/`): Validates feature engineering correctness
5. **Model Persistence**: Schema versioning for model compatibility

This systematic approach ensures robust, maintainable feature engineering that prevents the common ML problem of train/serve skew.