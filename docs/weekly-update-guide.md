# Weekly Data Update Guide

This guide explains how to keep your NFL DFS system updated with the latest data throughout the season.

## Quick Update (After Each Week's Games)

Run this command every Tuesday morning after Monday Night Football:

```bash
# Automatic update for current week
uv run python scripts/update_weekly_data.py

# Or specify week manually
uv run python scripts/update_weekly_data.py --season 2024 --week 12

# Check if models need retraining
uv run python scripts/update_weekly_data.py --check-models
```

## What Gets Updated

### 1. **Game Results** (Immediate)

- Final scores
- Game completion status
- Updated Vegas closing lines

### 2. **Player Statistics** (Available ~2 hours after games)

- All player performance stats
- Fantasy points (DK scoring)
- Snap counts and usage

### 3. **Defensive Stats** (Calculated from play-by-play)

- Position-specific fantasy points allowed
- Updated defensive rankings
- 4-week rolling averages

### 4. **Vegas Lines** (Closing lines)

- Final spreads and totals
- Useful for model evaluation

## Model Retraining Strategy

### You DON'T Need to Retrain After Every Week

The models are designed to be robust and don't require weekly retraining. Here's when to retrain:

#### 📅 **Recommended Schedule**

- **Monthly during season**: Retrain once per month for optimal performance
- **After major changes**: Significant trades, coaching changes, or injury to key players
- **Performance degradation**: When MAE increases by >20% from baseline

#### 🚀 **Quick Retrain** (Incremental - 5-10 minutes)

```bash
# Fine-tune existing models with recent data
uv run python -m src.cli.train_models train-incremental --weeks 4
```

#### 🔄 **Full Retrain** (Complete - 30-60 minutes)

```bash
# Complete retraining with all historical data
uv run python -m src.cli.train_models train-all-positions
```

## Automated Weekly Workflow

### Option 1: Cron Job (Linux/Mac)

Add to your crontab (`crontab -e`):

```bash
# Run every Tuesday at 6 AM (after MNF)
0 6 * * 2 cd /path/to/fantasy && uv run python scripts/update_weekly_data.py --check-models

# Run every Sunday at 11 AM (before games) to ensure latest odds
0 11 * * 0 cd /path/to/fantasy && uv run python scripts/update_weekly_data.py --season 2024 --week $(date +%U)
```

### Option 2: GitHub Actions (Automated)

Create `.github/workflows/weekly-update.yml`:

```yaml
name: Weekly Data Update

on:
    schedule:
        # Run Tuesday at 6 AM EST
        - cron: "0 11 * * 2"
    workflow_dispatch: # Allow manual trigger

jobs:
    update:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v3

            - name: Set up Python
              uses: actions/setup-python@v4
              with:
                  python-version: "3.11"

            - name: Install UV
              run: curl -LsSf https://astral.sh/uv/install.sh | sh

            - name: Install dependencies
              run: uv pip install -r requirements.txt

            - name: Update weekly data
              run: uv run python scripts/update_weekly_data.py --check-models

            - name: Check if retraining needed
              id: check
              run: |
                  if uv run python scripts/update_weekly_data.py --check-models | grep -q "consider retraining"; then
                    echo "needs_retraining=true" >> $GITHUB_OUTPUT
                  fi

            - name: Notify if retraining needed
              if: steps.check.outputs.needs_retraining == 'true'
              run: echo "::warning::Models need retraining"
```

### Option 3: Manual Process

1. **Tuesday Morning** (After MNF):

   ```bash
   # Update data
   uv run python scripts/update_weekly_data.py

   # Check model performance
   uv run python scripts/update_weekly_data.py --check-models
   ```

1. **First Tuesday of Month** (Monthly retrain):

   ```bash
   # Full model retrain
   uv run python -m src.cli.train_models train-all-positions

   # Deploy best models
   uv run python -m src.cli.train_models deploy-best
   ```

1. **Thursday** (DFS lineup building):

   ```bash
   # Upload DraftKings CSV
   # Generate predictions
   # Build optimal lineups
   ```

## Data Freshness

Different data sources update at different times:

| Data Type | Update Timing | Source |
| -- | -- | -- |
| Game Results | Immediate | nfl_data_py |
| Player Stats | 2-4 hours after game | nfl_data_py |
| Play-by-play | Next morning | nfl_data_py |
| Vegas Lines | Immediate | nfl_data_py schedules |
| Weather | Before game | nfl_data_py schedules |
| Injuries | Wednesday reports | Manual/API |

## Incremental Learning (Advanced)

For power users who want to update models without full retraining:

```python
# Fine-tune with recent games only (fast)
from src.ml.training.incremental import IncrementalTrainer

trainer = IncrementalTrainer()
trainer.update_with_recent_games(weeks=2, learning_rate=0.0001)
```

Benefits:

- ✅ 10x faster than full retrain
- ✅ Adapts to recent trends
- ✅ Preserves historical knowledge

Drawbacks:

- ⚠️ Can drift over time
- ⚠️ May overfit to recent games
- ⚠️ Requires careful monitoring

## Troubleshooting

### Common Issues

1. **"No data found for current week"**

   - Games haven't been played yet
   - Wait until Tuesday for complete data

1. **"Models performing poorly"**

   - Run full retrain: `make train-models`
   - Check for major roster changes
   - Verify data quality

1. **"Update script fails"**

   - Check internet connection
   - Verify nfl_data_py is working
   - Check database isn't locked

### Performance Monitoring

Track your model performance:

```bash
# View recent prediction accuracy
uv run python -m src.cli.evaluate_models check-recent --weeks 4

# Compare to baseline
uv run python -m src.cli.evaluate_models compare-baseline

# Generate performance report
uv run python -m src.cli.evaluate_models report --output reports/week_$(date +%U).html
```

## Best Practices

1. **Update regularly**: Tuesday mornings for complete weekend data
1. **Monitor performance**: Check MAE weekly, retrain monthly
1. **Version models**: Keep last 2 versions for rollback
1. **Track changes**: Log when players change teams or injuries occur
1. **Validate updates**: Ensure data looks reasonable after updates

## Summary

- **Weekly updates**: Quick, automatic, no retraining needed
- **Monthly retraining**: Keeps models fresh and accurate
- **Incremental updates**: Optional for advanced users
- **Monitoring**: Track performance to know when to retrain

The system is designed to be low-maintenance while maintaining high accuracy!
