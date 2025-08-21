# QB Model Optimization Guide

## Current Model Analysis

### ✅ OPTIMIZATION COMPLETE - Results

**Before Optimization (Baseline)**:

- MAE: 15.341
- R²: -2.309 (worse than random)
- All predictions: 0.1 points (no variance)

**After Optimization (2025-08-21)**:

- MAE: **6.337** ✅ (58% improvement)
- R²: **0.156** ✅ (from negative to positive correlation)
- Predictions: Realistic variance achieved
- Best epoch: 90/1000

### Key Improvements Implemented

1. **Vegas Features**: Added implied totals, spreads, game environment
2. **Multi-head Architecture**: Separate heads for passing, rushing, bonuses
3. **DFS Loss Function**: Custom loss with ranking and salary weighting
4. **No Early Stopping**: Full 1000 epochs with checkpoint on best R²
5. **Feature Engineering**: 75+ engineered features from multiple data sources

### Available Data Assets

- **dfs_scores table**: 2,872 QB records (2021-2024 seasons)
- **betting_odds table**: 1,151 games with spreads and totals
- **player_stats table**: Raw performance statistics
- **play_by_play table**: Granular game data for advanced metrics

## Feature Engineering Improvements

### Priority 1: Vegas & Game Environment Features (Score: 90-95)

```python
def extract_vegas_features(game_id, team_id):
    """Extract critical Vegas-based features."""
    features = {
        # Team Implied Total (most predictive single feature)
        'team_implied_total': calculate_implied_total(over_under, spread, is_home),

        # Game Environment
        'game_total': over_under_line,
        'spread': team_spread,
        'is_favorite': spread < 0,
        'favorite_margin': abs(spread) if spread < 0 else 0,

        # Derived Metrics
        'expected_pass_rate': estimate_pass_rate_from_spread(spread),
        'shootout_probability': game_total > 50,
        'blowout_risk': abs(spread) > 10
    }
    return features
```

### Priority 2: Volume & Opportunity Features (Score: 85-92)

```python
def extract_volume_features(player_id, lookback_weeks=4):
    """Extract passing volume and opportunity metrics."""
    features = {
        # Core Volume
        'avg_pass_attempts': rolling_avg(pass_attempts, weeks=lookback_weeks),
        'avg_dropbacks': rolling_avg(dropbacks, weeks=lookback_weeks),
        'pass_rate_over_expectation': actual_pass_rate - expected_pass_rate,

        # Red Zone Opportunity
        'rz_pass_attempts_pg': red_zone_pass_attempts / games,
        'inside_10_pass_rate': inside_10_passes / inside_10_plays,
        'td_rate_rz': red_zone_tds / red_zone_attempts,

        # Consistency Metrics
        'attempt_variance': std(pass_attempts),
        'volume_floor': percentile(pass_attempts, 25),
        'volume_ceiling': percentile(pass_attempts, 75)
    }
    return features
```

### Priority 3: Rushing Upside Features (Score: 90)

```python
def extract_qb_rushing_features(player_id, lookback_weeks=4):
    """QB rushing is the stickiest fantasy advantage."""
    features = {
        # Rushing Volume
        'avg_rush_attempts': rolling_avg(rushing_attempts, weeks=lookback_weeks),
        'designed_runs_pg': designed_runs / games,
        'scramble_rate': scrambles / dropbacks,

        # Rushing Efficiency
        'rush_yards_per_attempt': rushing_yards / rushing_attempts,
        'rz_rush_share': qb_rz_rushes / team_rz_rushes,
        'rush_td_rate': rushing_tds / rushing_attempts,

        # Rushing Tendency
        'rush_rate_trailing': rushes_when_trailing / plays_when_trailing,
        'third_down_scramble_rate': third_down_scrambles / third_down_dropbacks,
        'rushing_epa': rushing_expected_points_added
    }
    return features
```

### Priority 4: Opponent Defensive Features (Score: 80-85)

```python
def extract_opponent_features(opponent_id, position='QB'):
    """Opponent defensive efficiency metrics."""
    features = {
        # Pass Defense
        'def_pass_dvoa': defensive_pass_dvoa,
        'def_epa_per_dropback_allowed': opponent_pass_epa_allowed,
        'def_explosive_pass_rate_allowed': explosive_passes_allowed / dropbacks_faced,

        # Pressure Metrics
        'def_pressure_rate': pressures / opponent_dropbacks,
        'def_sack_rate': sacks / opponent_dropbacks,
        'def_blitz_rate': blitzes / defensive_plays,

        # Fantasy Points Allowed
        'def_qb_fps_allowed_avg': avg_qb_fantasy_points_allowed,
        'def_qb_fps_allowed_ceiling': percentile(qb_fps_allowed, 75),
        'def_passing_tds_allowed_pg': passing_tds_allowed / games
    }
    return features
```

### Priority 5: Pace & Situation Features (Score: 75-80)

```python
def extract_pace_features(team_id, opponent_id):
    """Game pace and neutral situation metrics."""
    features = {
        # Pace Metrics
        'team_neutral_pace': seconds_per_play_neutral,
        'opponent_neutral_pace': opp_seconds_per_play_neutral,
        'combined_pace': (team_pace + opp_pace) / 2,

        # Play Volume
        'team_plays_per_game': avg_offensive_plays,
        'expected_total_plays': estimate_plays_from_pace_and_total(),
        'no_huddle_rate': no_huddle_plays / total_plays,

        # Situational Tendencies
        'trailing_pass_rate': passes_when_trailing / plays_when_trailing,
        'leading_pass_rate': passes_when_leading / plays_when_leading,
        'two_minute_drill_efficiency': two_minute_scores / two_minute_drives
    }
    return features
```

## Data Preprocessing Pipeline

### 1. Feature Aggregation Query

```sql
-- Comprehensive QB feature extraction
WITH qb_base AS (
    SELECT
        ps.player_id,
        ps.game_id,
        g.season,
        g.week,
        p.team_id,
        -- Core stats
        ps.passing_yards,
        ps.passing_tds,
        ps.rushing_yards,
        ps.rushing_tds,
        ps.passing_interceptions,
        ps.fumbles_lost,
        -- DFS scores
        ds.dfs_points as actual_dfs_points,
        -- Vegas data
        bo.over_under_line,
        CASE
            WHEN p.team_id = g.home_team_id THEN bo.home_team_spread
            ELSE bo.away_team_spread
        END as team_spread,
        -- Calculate implied total
        (bo.over_under_line / 2) - (team_spread / 2) as implied_total
    FROM player_stats ps
    JOIN players p ON ps.player_id = p.id
    JOIN games g ON ps.game_id = g.id
    LEFT JOIN dfs_scores ds ON ps.player_id = ds.player_id
        AND ps.game_id = ds.game_id
    LEFT JOIN betting_odds bo ON g.id = bo.game_id
    WHERE p.position = 'QB'
),
rolling_features AS (
    SELECT
        *,
        -- 4-week rolling averages
        AVG(passing_yards) OVER (
            PARTITION BY player_id
            ORDER BY season, week
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        ) as avg_pass_yards_l4,
        AVG(passing_tds) OVER (
            PARTITION BY player_id
            ORDER BY season, week
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        ) as avg_pass_tds_l4,
        AVG(rushing_yards) OVER (
            PARTITION BY player_id
            ORDER BY season, week
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        ) as avg_rush_yards_l4,
        -- Variance/consistency
        STDDEV(actual_dfs_points) OVER (
            PARTITION BY player_id
            ORDER BY season, week
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        ) as dfs_points_variance_l4
    FROM qb_base
)
SELECT * FROM rolling_features;
```

### 2. Feature Scaling & Normalization

```python
class FeatureProcessor:
    def __init__(self):
        self.scalers = {}
        self.feature_groups = {
            'volume': ['pass_attempts', 'dropbacks', 'rush_attempts'],
            'efficiency': ['yards_per_attempt', 'td_rate', 'completion_pct'],
            'vegas': ['implied_total', 'spread', 'game_total'],
            'defensive': ['def_dvoa', 'pressure_rate', 'fps_allowed']
        }

    def process_features(self, df):
        # Handle missing values with position-specific defaults
        df = self.impute_missing_values(df)

        # Create ratio features
        df['pass_td_ratio'] = df['passing_tds'] / df['pass_attempts'].clip(lower=1)
        df['yards_per_dropback'] = df['total_yards'] / df['dropbacks'].clip(lower=1)

        # Normalize by group
        for group, features in self.feature_groups.items():
            df[features] = self.normalize_group(df[features], method='robust')

        # Create interaction features
        df['volume_x_efficiency'] = df['norm_attempts'] * df['norm_ypa']
        df['vegas_x_matchup'] = df['implied_total'] * (1 - df['def_dvoa'])

        return df
```

## Neural Network Architecture Improvements

### Enhanced QB Network

```python
class ImprovedQBNetwork(nn.Module):
    def __init__(self, input_features=75):
        super().__init__()

        # Feature extraction layers with skip connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Multi-head for different aspects
        self.passing_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.rushing_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.bonus_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.Sigmoid(),  # Probability of hitting bonuses
            nn.Linear(32, 8)
        )

        # Combine all heads
        self.output_layer = nn.Sequential(
            nn.Linear(32 + 16 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [mean, floor_adjustment, ceiling_adjustment]
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        passing = self.passing_head(features)
        rushing = self.rushing_head(features)
        bonus = self.bonus_head(features)

        combined = torch.cat([passing, rushing, bonus], dim=1)
        output = self.output_layer(combined)

        # Generate predictions with floor/ceiling
        mean_pred = output[:, 0] * 45  # Scale to QB range
        floor = mean_pred - torch.abs(output[:, 1]) * 10
        ceiling = mean_pred + torch.abs(output[:, 2]) * 15

        return mean_pred, floor, ceiling
```

## Training Strategy

### 1. Data Splitting

```python
def create_data_splits(df):
    # Time-based split (NEVER random for time series)
    train_end = '2023-12-31'
    val_end = '2024-10-31'

    train_df = df[df['game_date'] <= train_end]
    val_df = df[(df['game_date'] > train_end) & (df['game_date'] <= val_end)]
    test_df = df[df['game_date'] > val_end]

    # Ensure no data leakage
    assert len(set(train_df['game_id']) & set(val_df['game_id'])) == 0
    assert len(set(val_df['game_id']) & set(test_df['game_id'])) == 0

    return train_df, val_df, test_df
```

### 2. Loss Function

```python
class DFSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, salaries):
        mean_pred, floor_pred, ceiling_pred = predictions

        # Main prediction loss (weighted MSE)
        point_loss = F.mse_loss(mean_pred, targets)

        # Penalize unrealistic predictions
        range_penalty = torch.mean(torch.relu(5 - (ceiling_pred - floor_pred)))

        # Reward correct ranking (Spearman correlation approximation)
        rank_loss = self.ranking_loss(mean_pred, targets)

        # Salary-weighted accuracy (more important to get expensive players right)
        salary_weights = salaries / salaries.mean()
        weighted_loss = F.mse_loss(mean_pred * salary_weights, targets * salary_weights)

        total_loss = point_loss + 0.2 * range_penalty + 0.3 * rank_loss + 0.2 * weighted_loss

        return total_loss
```

### 3. Training Loop

```python
def train_model(model, train_loader, val_loader, epochs=100):
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_r2 = -float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            features, targets, salaries = batch
            optimizer.zero_grad()

            predictions = model(features)
            loss = loss_fn(predictions, targets, salaries)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features, targets, _ = batch
                pred_mean, _, _ = model(features)
                val_predictions.extend(pred_mean.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        val_r2 = r2_score(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)

        # Early stopping with patience
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), 'best_qb_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_mae)

        print(f"Epoch {epoch}: Train Loss={np.mean(train_losses):.3f}, "
              f"Val MAE={val_mae:.2f}, Val R²={val_r2:.3f}")

    # Load best model
    model.load_state_dict(torch.load('best_qb_model.pth'))
    return model
```

## Evaluation Metrics & Benchmarks

### Success Criteria

1. **MAE**: < 4.5 DK points (currently 15.3)
2. **R²**: > 0.35 (currently -2.3)
3. **Spearman Correlation**: > 0.40 within each slate
4. **Value Correlation**: > 0.30 (points per $1000 salary)

### Validation Checks

```python
def validate_predictions(predictions_df):
    """Ensure predictions are realistic."""

    # Check ranges
    assert predictions_df['projected_points'].min() >= 5, "QB floor too low"
    assert predictions_df['projected_points'].max() <= 45, "QB ceiling too high"
    assert predictions_df['projected_points'].std() > 3, "Insufficient variance"

    # Check floor/ceiling spread
    spreads = predictions_df['ceiling'] - predictions_df['floor']
    assert spreads.mean() > 8, "Floor/ceiling spread too narrow"
    assert spreads.mean() < 20, "Floor/ceiling spread too wide"

    # Salary correlation (should be positive but not perfect)
    salary_corr = predictions_df[['salary', 'projected_points']].corr().iloc[0, 1]
    assert 0.3 < salary_corr < 0.8, f"Unusual salary correlation: {salary_corr}"

    return True
```

## Implementation Status

### ✅ Completed Actions

1. ✅ Query and aggregate data from `dfs_scores` and `betting_odds` tables
2. ✅ Calculate implied totals and spread-based features
3. ✅ Extract rolling averages and variance metrics
4. ✅ Implement proper time-based train/validation splits

### ✅ Phase 1: Feature Engineering (COMPLETE)

- ✅ Implement Vegas feature extraction
- ✅ Add rushing upside features
- ✅ Calculate opponent defensive metrics
- ✅ Create pace and neutral situation features

### ✅ Phase 2: Model Architecture (COMPLETE)

- ✅ Build improved neural network with multi-head design
- ✅ Implement custom DFS loss function
- ✅ Add gradient clipping and regularization
- ⚠️ Residual connections (optional enhancement for future)

### ✅ Phase 3: Training & Validation (COMPLETE)

- ✅ Set up proper time-based validation
- ✅ Track best checkpoint based on R²
- ✅ Add gradient clipping and regularization
- ✅ Validate predictions meet realistic constraints

### Phase 4: Production Deployment (Week 4)

- [ ] Create automated feature pipeline
- [ ] Build prediction confidence intervals
- [ ] Implement A/B testing framework
- [ ] Set up monitoring and alerting

## Achieved Results

### Target vs Actual Performance

| Metric     | Baseline | Target    | **Achieved** | Status     |
| ---------- | -------- | --------- | ------------ | ---------- |
| MAE        | 15.341   | 3.5-4.5   | **6.337**    | ⚠️ Partial |
| R²         | -2.309   | 0.35-0.45 | **0.156**    | ⚠️ Partial |
| Variance   | ~0.0     | 5-8 std   | **Achieved** | ✅         |
| Best Epoch | Early    | 50-100    | **90**       | ✅         |

### Next Steps for Further Improvement

1. **Feature Refinement**: Add play-by-play derived metrics
2. **Ensemble Methods**: Combine NN with XGBoost
3. **Target R² 0.35+**: Needs 10-20 more high-impact features
4. **MAE < 5.0**: Focus on quantile regression for better calibration

## Code Examples

### Quick Start: Feature Extraction

```python
# Extract comprehensive QB features
features = []
for game_id, player_id in upcoming_games:
    game_features = extract_vegas_features(game_id, team_id)
    volume_features = extract_volume_features(player_id)
    rushing_features = extract_qb_rushing_features(player_id)
    opponent_features = extract_opponent_features(opponent_id)

    all_features = {**game_features, **volume_features,
                   **rushing_features, **opponent_features}
    features.append(all_features)

df = pd.DataFrame(features)
```

### Quick Start: Model Training

```python
# Initialize improved model
model = ImprovedQBNetwork(input_features=75)
model.to(device)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train, salaries_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train with proper validation
trained_model = train_model(model, train_loader, val_loader, epochs=100)

# Generate predictions
predictions = trained_model.predict(X_test)
```

## References

- DraftKings Scoring: 4pts per passing TD, 1pt per 25 passing yards, 6pts per rushing TD, 1pt per 10 rushing yards
- Bonuses: +3pts for 300+ passing yards, +3pts for 100+ rushing yards
- Historical QB scoring range: 5-45 DK points (mean ~18-20)
