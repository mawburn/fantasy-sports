# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simplified NFL Daily Fantasy Sports (DFS) optimization system for DraftKings contests. The project was recently streamlined from a complex multi-module architecture into a focused `dfs/` directory containing the core functionality.

## Architecture

The system has four main components consolidated into single-file modules:

- **`dfs/data.py`**: Data collection, SQLite database operations, feature engineering
- **`dfs/models.py`**: PyTorch neural networks for player predictions (QB, RB, WR, TE, DST)
- **`dfs/optimize.py`**: Lineup optimization using linear programming (PuLP)
- **`dfs/run.py`**: CLI interface with commands: collect, train, predict, optimize

## Common Development Commands

### Setup and Installation

```bash
cd dfs/
pip install -r requirements.txt
```

### Core Workflow Commands

```bash
# 1. Collect NFL data and DraftKings salaries
python run.py collect --seasons 2022 2023 --csv data/DKSalaries.csv

# 2. Train position-specific neural network models
python run.py train

# 3. Generate player predictions for current contest
python run.py predict --contest-id <contest_id> --output predictions.csv

# 4. Build optimal lineups
python run.py optimize --strategy balanced --count 5 --output-dir lineups/
```

### Development and Testing

```bash
# Run the CLI help
python run.py --help

# Test optimization functions
python optimize.py

# Check data quality after collection
python -c "from data import validate_data_quality; print(validate_data_quality('data/nfl_dfs.db'))"
```

## Key Technical Details

### Database Structure

- SQLite database at `dfs/data/nfl_dfs.db`
- Core tables: `games`, `teams`, `players`, `player_stats`, `draftkings_contests`
- Schema defined in `data.py` `DB_SCHEMA` dictionary

### Machine Learning Pipeline

- Position-specific PyTorch neural networks (QBNetwork, RBNetwork, WRNetwork, TENetwork, DEFNetwork)
- Features include recent performance, opponent strength, correlation data
- Models saved as `.pth` files in `dfs/models/` directory
- Training uses 80/20 train/validation split with early stopping

### Optimization Strategy

- Linear programming with PuLP for guaranteed optimal solutions
- Supports multiple strategies: cash games (floor-focused), tournaments (ceiling-focused), contrarian (low-ownership)
- Stacking logic for QB-WR correlations and RB-DEF game script correlations
- Constraint handling: salary cap ($50K), positions (1QB, 2RB, 3WR, 1TE, 1FLEX, 1DST)

### Configuration

- Environment variables in `.env` file (copy from `.env.example`)
- Model hyperparameters defined in respective model classes
- DraftKings scoring rules documented in `DK-NFLClassic-Rules.md`

## Important File Paths

- **Models**: `dfs/models/*.pth` - Trained PyTorch models
- **Data**: `dfs/data/nfl_dfs.db` - SQLite database
- **Salaries**: `dfs/data/DKSalaries.csv` - DraftKings player salaries
- **Lineups**: `dfs/lineups/` - Generated optimal lineups
- **Config**: `dfs/.env` - Environment configuration

## Development Workflow

1. **Data Collection**: Use `collect` command to gather NFL stats and DK salaries
2. **Model Training**: Train position-specific models with `train` command
3. **Prediction**: Generate player projections with `predict` command
4. **Optimization**: Build lineups with various strategies using `optimize` command
5. **Validation**: Always validate lineups meet DraftKings constraints before submission

## Dependencies

Core requirements (see `dfs/requirements.txt`):

- `torch>=2.0.0` - Neural network models
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `pulp>=2.7.0` - Linear programming optimization
- `nfl-data-py>=0.3.0` - NFL statistics collection

## Testing and Validation

The system includes built-in validation:

- Lineup constraint checking (salary cap, position requirements)
- Data quality validation after collection
- Model training metrics (MAE, R², RMSE)
- Feature engineering validation

Always run data quality checks after collection and verify lineup validity before contest submission.
