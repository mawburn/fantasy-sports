# WR Model Optimization Guide

## 🚨 Current Problem Analysis

### Model Performance Issues

- **Current R²**: 0.250 (acceptable but room for improvement)
- **Current MAE**: 5.399 (reasonable for WR variance)
- **Critical Issue**: Predictions show **minimal variance** (all predictions clustered around 0.1 points)
- **Root Cause**: Using sigmoid activation with improper scaling, causing severe output compression

### Database Analysis

Available data sources:

- **player_stats**: Basic stats (yards, TDs, receptions, targets)
- **dfs_scores**: Pre-computed DFS points (9,366 WR samples, avg 8.72 points)
- **betting_odds**: Vegas lines (spread, O/U)
- **play_by_play**: Detailed play data for advanced metrics
- **weather**: Game conditions

## 🎯 Optimization Strategy

### Phase 1: Fix Output Scaling (IMMEDIATE)

**Problem**: Current architecture uses `nn.Sigmoid()` then multiplies by 30.0, constraining all outputs to 0-30 range but compressing variance.

**Solution**:

```python
class WRNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # ... existing layers ...

        # REMOVE sigmoid from output
        self.output = nn.Sequential(
            nn.Linear(16 + 12, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Raw linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... existing forward pass ...
        output = self.output(combined)
        # No sigmoid, no artificial scaling
        return output
```

### Phase 2: Feature Engineering (CRITICAL)

Based on available data, implement these features in priority order:

#### Tier 1: Opportunity Metrics (Score: 90-96)

```sql
-- Target metrics (last 3-5 games rolling)
WITH rolling_stats AS (
    SELECT
        player_id,
        game_id,
        AVG(targets) OVER (PARTITION BY player_id ORDER BY game_id ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as avg_targets_l5,
        AVG(targets) OVER (PARTITION BY player_id ORDER BY game_id ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) as avg_targets_l3,
        SUM(targets) OVER (PARTITION BY player_id ORDER BY game_id ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as total_targets_l5,
        AVG(receiving_yards/NULLIF(targets,0)) OVER (PARTITION BY player_id ORDER BY game_id ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as yards_per_target_l5
    FROM player_stats
)

-- Team target share
WITH team_targets AS (
    SELECT
        game_id,
        SUM(targets) as team_total_targets
    FROM player_stats ps
    JOIN players p ON ps.player_id = p.id
    WHERE p.position IN ('WR', 'TE', 'RB')
    GROUP BY game_id, p.team_id
)
SELECT
    ps.targets / tt.team_total_targets as target_share
FROM player_stats ps
JOIN team_targets tt ON ps.game_id = tt.game_id
```

#### Tier 2: Air Yards & Depth (Score: 85-93)

```python
# Estimate air yards from play_by_play data
def calculate_air_yards_metrics(conn, player_id, game_id):
    query = """
    SELECT
        AVG(air_yards) as avg_adot,
        SUM(air_yards) as total_air_yards,
        COUNT(CASE WHEN air_yards > 20 THEN 1 END) as deep_targets,
        SUM(air_yards) / SUM(SUM(air_yards)) OVER (PARTITION BY game_id, posteam) as air_yards_share
    FROM play_by_play
    WHERE receiver_player_id = ?
    AND game_id < ?
    AND play_type = 'pass'
    GROUP BY game_id
    ORDER BY game_id DESC
    LIMIT 5
    """
    # Calculate WOPR = (1.5 * target_share) + (0.7 * air_yards_share)
```

#### Tier 3: Red Zone & TD Access (Score: 88-90)

```python
def calculate_rz_metrics(conn, player_id):
    query = """
    SELECT
        COUNT(CASE WHEN yardline_100 <= 20 THEN 1 END) as rz_targets,
        COUNT(CASE WHEN yardline_100 <= 10 THEN 1 END) as inside_10_targets,
        COUNT(CASE WHEN yardline_100 <= 5 THEN 1 END) as goal_line_targets,
        AVG(CASE WHEN yardline_100 <= 20 THEN 1.0 ELSE 0 END) as rz_target_rate
    FROM play_by_play
    WHERE receiver_player_id = ?
    AND play_type = 'pass'
    AND game_id IN (SELECT game_id FROM games WHERE game_date < CURRENT_DATE ORDER BY game_date DESC LIMIT 5)
    """
```

#### Tier 4: Game Environment (Score: 85-88)

```python
def add_vegas_features(df):
    features = []

    # From betting_odds table
    features.extend([
        'team_implied_total',  # (over_under / 2) + (spread / 2)
        'game_total',          # over_under_line
        'spread',              # home/away_team_spread
        'is_favorite',         # binary
        'expected_positive_script',  # spread > 3
        'shootout_potential'   # over_under > 48 AND abs(spread) < 7
    ])

    return features
```

#### Tier 5: QB & Passing Context (Score: 80-85)

```python
def calculate_qb_context(conn, team_id):
    query = """
    -- Team pass rate over expectation (PROE)
    WITH team_pass_rate AS (
        SELECT
            posteam,
            game_id,
            AVG(CASE WHEN play_type = 'pass' THEN 1.0 ELSE 0 END) as pass_rate,
            COUNT(*) as total_plays
        FROM play_by_play
        WHERE down IN (1,2,3)
        AND wp BETWEEN 0.2 AND 0.8  -- Neutral game script
        GROUP BY posteam, game_id
    )
    SELECT
        AVG(pass_rate) as neutral_pass_rate,
        AVG(total_plays) as avg_plays_per_game
    FROM team_pass_rate
    WHERE posteam = ?
    """
```

### Phase 3: Advanced Features

#### Target Competition Index (TCI)

```python
def calculate_target_concentration(conn, team_id, week):
    """Herfindahl index for target concentration"""
    query = """
    WITH player_shares AS (
        SELECT
            player_id,
            targets * 1.0 / SUM(targets) OVER (PARTITION BY game_id) as target_share
        FROM player_stats ps
        JOIN players p ON ps.player_id = p.id
        WHERE p.team_id = ?
        AND p.position IN ('WR', 'TE', 'RB')
    )
    SELECT
        SUM(target_share * target_share) as tci
    FROM player_shares
    """
```

#### Matchup-Specific Features

```python
def get_defensive_matchup_metrics(conn, opponent_id):
    """Get opponent's defensive tendencies vs WRs"""
    query = """
    SELECT
        AVG(receiving_yards) as avg_yards_allowed,
        AVG(receiving_tds) as avg_tds_allowed,
        AVG(CASE WHEN receiving_yards > 100 THEN 1.0 ELSE 0 END) as boom_rate_allowed,
        STDDEV(fantasy_points) as fantasy_variance_allowed
    FROM player_stats ps
    JOIN games g ON ps.game_id = g.id
    JOIN players p ON ps.player_id = p.id
    WHERE p.position = 'WR'
    AND (g.home_team_id = ? OR g.away_team_id = ?)
    AND ps.player_id IN (
        SELECT id FROM players
        WHERE team_id != ?
    )
    """
```

### Phase 4: Model Architecture Improvements

```python
class ImprovedWRNetwork(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        super().__init__()

        # Feature extraction with residual connections
        self.input_norm = nn.LayerNorm(input_size)

        # Multi-head architecture for different aspects
        self.volume_head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )

        self.efficiency_head = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )

        self.matchup_head = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, 32)
        )

        # Fusion layer
        fusion_size = 64 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Output heads for different targets
        self.mean_output = nn.Linear(32, 1)
        self.floor_output = nn.Linear(32, 1)
        self.ceiling_output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_norm(x)

        volume = self.volume_head(x)
        efficiency = self.efficiency_head(x)
        matchup = self.matchup_head(x)

        combined = torch.cat([volume, efficiency, matchup], dim=1)
        features = self.fusion(combined)

        mean = self.mean_output(features)
        floor = self.floor_output(features)
        ceiling = self.ceiling_output(features)

        return {
            'mean': mean,
            'floor': floor,
            'ceiling': ceiling
        }
```

### Phase 5: Training Improvements

```python
class WRTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Multi-task loss
        self.mse_loss = nn.MSELoss()
        self.quantile_loss = QuantileLoss([0.25, 0.75])  # Floor and ceiling

        # Optimizer with different LRs for different parts
        self.optimizer = torch.optim.AdamW([
            {'params': model.volume_head.parameters(), 'lr': 1e-4},
            {'params': model.efficiency_head.parameters(), 'lr': 5e-5},
            {'params': model.matchup_head.parameters(), 'lr': 5e-5},
            {'params': model.fusion.parameters(), 'lr': 1e-4},
            {'params': [model.mean_output.parameters(),
                       model.floor_output.parameters(),
                       model.ceiling_output.parameters()], 'lr': 2e-4}
        ], weight_decay=1e-5)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

    def compute_loss(self, outputs, targets):
        mean_loss = self.mse_loss(outputs['mean'], targets)
        floor_loss = self.quantile_loss(outputs['floor'], targets, 0.25)
        ceiling_loss = self.quantile_loss(outputs['ceiling'], targets, 0.75)

        # Weighted combination
        total_loss = mean_loss + 0.3 * floor_loss + 0.3 * ceiling_loss

        # Add consistency regularization
        consistency_loss = torch.mean(torch.relu(outputs['floor'] - outputs['mean'])) + \
                          torch.mean(torch.relu(outputs['mean'] - outputs['ceiling']))

        return total_loss + 0.1 * consistency_loss
```

## 📊 Implementation Checklist

### Immediate Actions (Do First)

- [ ] Remove sigmoid activation from WRNetwork output
- [ ] Implement proper data normalization (StandardScaler on targets)
- [ ] Add multi-head outputs (mean, floor, ceiling)

### Feature Engineering Pipeline

- [ ] Create rolling window features (3-5 game averages)
- [ ] Calculate target share and air yards share
- [ ] Add Vegas features (implied totals, game script)
- [ ] Implement red zone and scoring opportunity metrics
- [ ] Add target competition index (TCI)

### Model Training

- [ ] Use 100-150 epochs with no early stopping
- [ ] Implement ReduceLROnPlateau scheduler
- [ ] Track multiple metrics (MAE, R², Spearman rank correlation)
- [ ] Save best checkpoint based on validation R²
- [ ] Implement proper train/validation split (80/20 time-based)

### Validation Metrics

- [ ] Target MAE: 4.5-5.5 DK points
- [ ] Target R²: > 0.30
- [ ] Spearman correlation: > 0.35
- [ ] Prediction variance: std dev > 3.0 points
- [ ] Quantile calibration: P25/P75 coverage

## 🔄 Testing Strategy

```python
def validate_wr_model(model, test_loader):
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['features'])
            predictions.extend(outputs['mean'].cpu().numpy())
            actuals.extend(batch['targets'].cpu().numpy())

    # Key metrics
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    spearman = spearmanr(actuals, predictions)[0]
    pred_std = np.std(predictions)

    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Spearman: {spearman:.3f}")
    print(f"Prediction StdDev: {pred_std:.3f}")

    # Check for compression
    if pred_std < 2.0:
        print("⚠️ WARNING: Low prediction variance - possible output compression")

    # Visual check
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel("Actual DFS Points")
    plt.ylabel("Predicted DFS Points")
    plt.plot([0, 30], [0, 30], 'r--')
    plt.show()
```

## 🚀 Expected Improvements

After implementing these optimizations:

1. **Prediction Variance**: From ~0.1 to 4-6 points std dev
2. **R² Score**: From 0.250 to 0.35-0.45
3. **MAE**: Maintain or improve from 5.399 to 4.5-5.0
4. **Ranking Accuracy**: Spearman correlation > 0.35
5. **Lineup Quality**: More diverse, higher-upside GPP lineups

## 📝 Notes

- The current model's main issue is output compression from sigmoid activation
- Focus on opportunity metrics (targets, air yards) over efficiency metrics
- Vegas features are crucial for game environment context
- Multi-task learning (mean + quantiles) improves overall performance
- Proper feature engineering > complex architectures for tabular data
