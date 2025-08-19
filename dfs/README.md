# Simplified DFS System

A streamlined NFL Daily Fantasy Sports prediction and optimization system. This is the simplified version that consolidates 5000+ lines of complex abstractions into ~3000 lines of focused, maintainable code.

## What This Does

1. **Collects NFL data** using nfl_data_py and DraftKings CSV files
2. **Trains PyTorch neural networks** for position-specific fantasy point predictions
3. **Captures complex correlations** between players, teams, and game contexts
4. **Optimizes lineups** using linear programming with PuLP
5. **Generates optimal DFS lineups** for cash games and tournaments

## Quick Start

```bash
# Install dependencies (using UV - fast Rust-based package manager)
uv pip install -r requirements.txt

# Or create virtual environment first
uv venv --python 3.11
uv pip install -r requirements.txt

# Collect NFL data
uv run python run.py collect --seasons 2023 2024

# Collect weather data for outdoor stadiums (optional, batch-optimized)
uv run python run.py weather --limit 50 --rate-limit 1.5

# Train models
uv run python run.py train

# Load DraftKings data and optimize lineups
uv run python run.py collect --csv draftkings.csv
uv run python run.py optimize --strategy tournament --count 5
```

## File Structure

```
simple_dfs/
├── data.py      # Direct SQLite operations + NFL data collection
├── models.py    # PyTorch neural networks + correlation features
├── optimize.py  # PuLP linear programming + stacking algorithms
├── run.py       # Simple CLI interface
├── utils.py     # Essential utilities
└── requirements.txt
```

## Core Features Preserved

### Neural Networks (models.py)

- Position-specific PyTorch models (QB, RB, WR, TE, DEF)
- Complex correlation feature extraction
- Multi-position correlated model for advanced stacking
- CPU-optimized training with early stopping

### Optimization (optimize.py)

- Linear programming with PuLP (guaranteed optimal)
- QB-WR/TE stacking logic
- RB-DEF game script correlations
- Multiple strategies: cash, tournament, contrarian
- Constraint handling and validation

### Data Pipeline (data.py)

- Direct SQLite operations (no ORM overhead)
- nfl_data_py integration for historical data
- DraftKings CSV parsing
- Feature engineering with correlation extraction

## Usage Examples

### Basic Workflow

```bash
# 1. Collect data
uv run python run.py collect --seasons 2022 2023 2024

# 2. Collect weather data (optional, batch-optimized for enhanced predictions)
uv run python run.py weather --limit 25

# 3. Train models
uv run python run.py train --positions QB RB WR TE DEF

# 4. Build optimal lineups (includes predictions automatically)
uv run python run.py optimize --strategy balanced --count 3

# Or save predictions too
uv run python run.py optimize --strategy balanced --count 3 --save-predictions predictions.csv
```

### Optimization Strategies

The system offers different optimization strategies for different contest types:

#### **Balanced** (Default)

- Uses raw projected points for optimization
- Good for general purpose lineups
- No specific adjustments for ceiling/floor or ownership
- Best for: Mixed contests, testing, general use

#### **Tournament** (GPP/Large Field)

- Emphasizes high ceiling players over consistent scorers
- Automatically enables QB-WR stacking for correlation
- Targets players with 25+ point upside potential
- Best for: Large tournaments where you need to differentiate

#### **Cash** (Head-to-Head/50-50s)

- Prioritizes floor and consistency over ceiling
- Uses conservative 80% of projection for safety
- Avoids boom/bust players
- Best for: Cash games where you need to beat ~50% of field

#### **Contrarian**

- Penalizes high-ownership players to find unique lineups
- Good for tournaments when you want to be different
- Reduces player values based on projected ownership
- Best for: Large tournaments with ownership data

### Weather Data Collection

Weather data can enhance predictions for outdoor stadium games. The weather command uses the Visual Crossing Weather API with intelligent batch processing:

```bash
# Optimized batch collection (default - MUCH more efficient)
uv run python run.py weather --limit 25

# Conservative daily collection (stays well within API limits)
uv run python run.py weather --limit 50 --rate-limit 2.0

# Disable batch processing (less efficient, one call per game)
uv run python run.py weather --limit 20 --no-batch

# Unlimited collection (use all available API quota)
uv run python run.py weather --rate-limit 1.5
```

**Batch Processing Benefits:**

- **10x more efficient**: Groups games by stadium and date ranges
- **Fewer API calls**: 1 call can fetch weather for 30 games at same stadium
- **Smart batching**: Processes 30-day chunks per stadium location
- **Incremental**: Only collects missing weather data, resumes where left off

**API Limits (Visual Crossing Free Tier):**

- **1000 requests/day** - resets daily
- **No standard rate limit headers** returned
- **Batch vs individual requests**: Same API cost (1 query per call)

**System Features:**

- Automatically detects rate limits and daily quota exceeded
- Only processes outdoor stadiums (skips domes: ATL, DAL, HOU, etc.)
- Resumes where it left off if interrupted
- Logs progress every 20 successful weather records

### Advanced Usage

```bash
# Tournament strategy with stacking
uv run python run.py optimize \
  --strategy tournament \
  --count 10 \
  --output-dir tournament_lineups/

# Cash game strategy (conservative)
uv run python run.py optimize \
  --strategy cash \
  --output-dir cash_lineup/

# Contrarian strategy (low ownership)
uv run python run.py optimize \
  --strategy contrarian \
  --count 5

# Balanced strategy with predictions export
uv run python run.py optimize \
  --strategy balanced \
  --count 3 \
  --save-predictions predictions.csv
```

## Performance

The simplified system:

- **Faster startup**: No ORM initialization overhead
- **Direct operations**: SQLite queries without abstraction layers
- **Smaller memory footprint**: Fewer dependencies and objects
- **Easier debugging**: Simple call stacks and clear data flow
- **Faster iteration**: No complex build/test infrastructure

## Dependencies

Only 5 essential packages:

- `numpy` + `pandas` for data manipulation
- `torch` for neural networks
- `pulp` for optimization
- `nfl-data-py` for data collection

Total simplified from 20+ complex dependencies to 5 focused ones.

## Migration Benefits

1. **Debuggability**: Easy to trace issues
2. **Maintainability**: Single files for each concern
3. **Performance**: No abstraction overhead
4. **Understanding**: Clear data flow and logic
5. **Reliability**: Fewer points of failure
6. **Speed**: Faster development iteration

This achieves the same results as the original complex system but with 40% fewer lines of code and 75% fewer dependencies.
