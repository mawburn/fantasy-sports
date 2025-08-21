# DST Model Optimization Guide - Implementation Plan

## Current State Analysis

### Model Performance Issues

- **R² = 0.008** (essentially no predictive power)
- **MAE = 3.905** (acceptable but lacks variation)
- **Predictions**: All DST projections between 1.6-2.5 points (no meaningful differentiation)
- **Model Type**: Basic CatBoost with only 11 features
- **Data**: 1,708 DST records (2021-2024 seasons)

### Database Assets Available

- **dst_stats**: Game-level DST performance (sacks, INTs, FRs, TDs, PA, fantasy_points)
- **betting_odds**: Spreads, O/U totals, favorites
- **dfs_scores**: Historical DFS points by position
- **play_by_play**: Detailed play data for advanced metrics
- **weather**: Game conditions

## Implementation Strategy

### Phase 1: Enhanced Feature Engineering (Priority: HIGH)

#### A. Vegas & Game Script Features

```sql
-- Create these features from betting_odds + games tables
1. opp_implied_total = (over_under_line / 2) + (spread_favorite / 2) [when team is underdog]
2. game_total_ou = over_under_line
3. spread_signed = team spread (negative = favored)
4. is_home = 1/0
5. is_favorite = 1/0
6. spread_magnitude = ABS(spread)
```

#### B. Opponent Offensive Metrics (from player_stats)

```sql
-- Rolling 3-5 game averages
1. opp_pass_attempts_roll
2. opp_pass_yards_roll
3. opp_turnover_rate_roll (INT + Fumbles / plays)
4. opp_sack_rate_roll
5. opp_scoring_rate_roll
6. opp_explosive_play_rate (plays > 20 yards)
```

#### C. Defensive Team Strength (from dst_stats)

```sql
-- Rolling 3-5 game defensive metrics
1. def_sacks_per_game_roll
2. def_turnovers_per_game_roll
3. def_points_allowed_roll
4. def_fantasy_points_roll
5. def_pressure_rate (if available from play_by_play)
```

#### D. Weather Impact

```sql
-- From weather table
1. wind_speed (>15 mph impacts passing)
2. precipitation (rain/snow increases fumbles)
3. temperature (<32°F affects scoring)
4. dome_game (1/0)
```

### Phase 2: Component Model Architecture (Priority: HIGH)

Instead of predicting total DST fantasy points directly, predict components:

#### Component Models

1. **Sacks Model** (Poisson regression)

   - Features: Pass volume, OL strength, pressure rate
   - Target: sack count

2. **Turnovers Model** (Poisson/Negative Binomial)

   - Features: QB INT rate, weather, game script
   - Target: INT + FR count

3. **Points Allowed Model** (Ordinal classifier)

   - Features: Opponent implied total, defensive strength
   - Target: PA bucket (0, 1-6, 7-13, 14-20, 21-27, 28-34, 35+)

4. **Defensive TD Model** (Binary classifier)
   - Features: Turnover expectation, special teams quality
   - Target: TD probability

#### Recombination Formula

```python
def calculate_dst_points(sacks, turnovers, pa_bucket, td_prob):
    # DraftKings scoring
    points = sacks * 1.0
    points += turnovers * 2.0  # INT + FR
    points += get_pa_points(pa_bucket)  # Bucket mapping
    points += td_prob * 6.0
    return points
```

### Phase 3: Model Implementation Options

#### Option A: CatBoost Ensemble (Recommended - Start Here)

```python
# Direct model for baseline
direct_model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=6,
    loss_function='MAE',
    use_best_model=True
)

# Component models
sacks_model = CatBoostRegressor(loss_function='Poisson', ...)
turnovers_model = CatBoostRegressor(loss_function='Poisson', ...)
pa_model = CatBoostClassifier(loss_function='MultiClass', ...)
td_model = CatBoostClassifier(loss_function='Logloss', ...)
```

#### Option B: Neural Network with Multi-Head Architecture

```python
class DSTComponentNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(...)

        # Task-specific heads
        self.sacks_head = PoissonHead()
        self.turnovers_head = PoissonHead()
        self.pa_head = OrdinalHead(n_classes=7)
        self.td_head = BinaryHead()
```

#### Option C: XGBoost with Custom Objectives

```python
# Custom objective for rare events (TDs)
def rare_event_objective(preds, dtrain):
    # Weighted loss for imbalanced TD prediction
    ...
```

### Phase 4: Feature Creation SQL Queries

```sql
-- Create comprehensive DST features view
CREATE VIEW dst_features AS
WITH rolling_stats AS (
    SELECT
        game_id,
        team_abbr,
        season,
        week,
        -- Rolling defensive stats (last 3 games)
        AVG(sacks) OVER (
            PARTITION BY team_abbr
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as def_sacks_l3,
        AVG(interceptions + fumbles_recovered) OVER (
            PARTITION BY team_abbr
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as def_turnovers_l3,
        AVG(points_allowed) OVER (
            PARTITION BY team_abbr
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as def_pa_l3,
        fantasy_points as target
    FROM dst_stats
),
vegas_features AS (
    SELECT
        game_id,
        CASE
            WHEN favorite_team = home_team THEN away_team_spread
            ELSE home_team_spread
        END as opp_spread,
        over_under_line,
        -- Calculate implied totals
        (over_under_line / 2.0) - (spread_favorite / 2.0) as favorite_implied,
        (over_under_line / 2.0) + (spread_favorite / 2.0) as underdog_implied
    FROM betting_odds
)
SELECT
    r.*,
    v.*,
    -- Add weather, opponent stats, etc.
FROM rolling_stats r
LEFT JOIN vegas_features v ON r.game_id = v.game_id;
```

### Phase 5: Training Pipeline

```python
def train_dst_model_pipeline():
    # 1. Load and prepare data
    features_df = load_dst_features()

    # 2. Create train/val split (time-based)
    X_train, X_val, y_train, y_val = time_based_split(
        features_df,
        test_season=2024,
        test_weeks=[15, 16, 17]
    )

    # 3. Train component models
    models = {}
    models['sacks'] = train_sacks_model(X_train, y_train['sacks'])
    models['turnovers'] = train_turnovers_model(X_train, y_train['turnovers'])
    models['pa_bucket'] = train_pa_model(X_train, y_train['pa_bucket'])
    models['td'] = train_td_model(X_train, y_train['td'])

    # 4. Generate predictions and combine
    preds = combine_component_predictions(models, X_val)

    # 5. Evaluate
    metrics = evaluate_dst_predictions(preds, y_val)

    return models, metrics
```

### Phase 6: Evaluation Metrics

Track these metrics for model improvement:

1. **MAE**: Target ≤ 3.7
2. **R²**: Target ≥ 0.15 (realistic for DST)
3. **Spearman Rank Correlation**: Target ≥ 0.25
4. **Top-5 Precision**: Target ≥ 50%
5. **Prediction Variance**: Should span 0-15 points range

### Phase 7: Implementation Checklist

- [ ] **Week 1: Feature Engineering**

  - [ ] Create Vegas features from betting_odds
  - [ ] Calculate rolling defensive stats
  - [ ] Add opponent offensive metrics
  - [ ] Include weather features

- [ ] **Week 2: Component Models**

  - [ ] Implement Poisson models for sacks/turnovers
  - [ ] Build ordinal classifier for PA buckets
  - [ ] Create rare-event classifier for TDs
  - [ ] Test recombination formula

- [ ] **Week 3: Model Training & Optimization**

  - [ ] Train CatBoost baseline
  - [ ] Train component models
  - [ ] Implement ensemble/stacking
  - [ ] Hyperparameter tuning

- [ ] **Week 4: Validation & Production**
  - [ ] Backtest on 2024 season
  - [ ] Compare with current predictions
  - [ ] Production deployment
  - [ ] Monitor weekly performance

## Quick Start Implementation

### Step 1: Create Enhanced Features (Immediate)

```python
def create_dst_features(conn):
    query = """
    WITH vegas AS (
        SELECT
            game_id,
            over_under_line as total,
            spread_favorite as spread,
            (over_under_line / 2.0) + (ABS(spread_favorite) / 2.0) as dog_implied,
            (over_under_line / 2.0) - (ABS(spread_favorite) / 2.0) as fav_implied
        FROM betting_odds
    ),
    rolling AS (
        SELECT
            *,
            AVG(sacks) OVER (PARTITION BY team_abbr ORDER BY season, week
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as sacks_l5,
            AVG(fantasy_points) OVER (PARTITION BY team_abbr ORDER BY season, week
                                     ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as fp_l5
        FROM dst_stats
    )
    SELECT
        r.*,
        v.total,
        v.spread,
        CASE
            WHEN /* team is favorite */ THEN v.dog_implied
            ELSE v.fav_implied
        END as opp_implied_total
    FROM rolling r
    LEFT JOIN vegas v ON r.game_id = v.game_id
    WHERE r.sacks_l5 IS NOT NULL  -- Ensure we have history
    """
    return pd.read_sql(query, conn)
```

### Step 2: Train CatBoost Baseline (Today)

```python
from catboost import CatBoostRegressor

def train_catboost_dst():
    # Load features
    df = create_dst_features(conn)

    # Define features
    feature_cols = [
        'opp_implied_total', 'total', 'spread',
        'sacks_l5', 'fp_l5', 'turnovers_l5', 'pa_l5'
    ]

    X = df[feature_cols]
    y = df['fantasy_points']

    # Time-based split
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] < 14))
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]

    # Train model
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        loss_function='MAE',
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    preds = model.predict(X_val)

    print(f"MAE: {mean_absolute_error(y_val, preds):.3f}")
    print(f"R²: {r2_score(y_val, preds):.3f}")
    print(f"Prediction range: [{preds.min():.1f}, {preds.max():.1f}]")

    return model
```

### Step 3: Component Models (Next Priority)

```python
def train_component_models():
    models = {}

    # Sacks model (Poisson)
    models['sacks'] = CatBoostRegressor(
        loss_function='Poisson',
        iterations=2000
    )

    # Turnovers model
    models['turnovers'] = CatBoostRegressor(
        loss_function='Poisson',
        iterations=2000
    )

    # Points allowed buckets
    models['pa_bucket'] = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=2000
    )

    # TD probability
    models['td_prob'] = CatBoostClassifier(
        loss_function='Logloss',
        iterations=2000,
        class_weights={0: 1, 1: 5}  # Handle imbalance
    )

    return models
```

## Expected Improvements

After implementing this plan:

- **R² improvement**: 0.008 → 0.15-0.25
- **MAE improvement**: 3.9 → 3.5-3.7
- **Prediction variance**: 1.6-2.5 → 0-15 points
- **Top-5 accuracy**: 20% → 50%+

## Key Success Factors

1. **Vegas features are critical** - Opponent implied totals drive 30-40% of DST performance
2. **Component models outperform direct prediction** - Breaks down variance into manageable parts
3. **Rolling averages matter** - Use 3-5 game windows for stability
4. **Game script dominates** - Favored teams with low opponent totals = DST gold
5. **Weather is a multiplier** - Wind/precipitation significantly impact turnovers

## Next Steps

1. Implement Step 1 (Enhanced Features) immediately
2. Train CatBoost baseline with new features
3. Compare with current model performance
4. Iterate on component models if baseline shows improvement
5. Deploy best performing model to production
