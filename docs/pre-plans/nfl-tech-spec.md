# NFL DFS Machine Learning Technical Specification

## Project Overview
Build a machine learning application for NFL DraftKings daily fantasy sports that predicts player performance and optimizes lineups. The system learns from comparing predictions against actual NFL game results to continuously improve accuracy. Contest selection helps identify interesting games to play for entertainment.

**Core Focus:**
- NFL-only implementation for DraftKings scoring
- Local deployment for personal use
- Self-improving models based on prediction accuracy
- Fun-focused contest selection (interesting games, not profit-driven)

## Development Environment Setup

### Package Management with Poetry

UV is the sole package manager for this project. UV is the fastest Python package manager available (10-100x faster than pip/poetry), written in Rust for maximum performance. It provides deterministic lockfiles and serves as a drop-in replacement for pip with minimal system resource usage.

```txt
# requirements.txt - Managed by UV for blazing fast dependency resolution
# UV automatically generates uv.lock for deterministic, reproducible installs
# Python 3.10+ required
python>=3.10
torch = "^2.0.0"
xgboost = "^2.0.0"
lightgbm = "^4.0.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
pyarrow = "^12.0.0"
scikit-learn = "^1.3.0"
fastapi = "^0.100.0"
uvicorn = "^0.23.0"
pydantic = "^2.0.0"
sqlalchemy = "^2.0.0"
mlflow = "^2.5.0"
pulp = "^2.7.0"
nfl-data-py = "^0.3.0"
schedule = "^1.2.0"
joblib = "^1.3.0"
loguru = "^0.7.0"
python-dotenv = "^1.0.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
optuna = "^3.3.0"
shap = "^0.42.0"
scipy = "^1.11.0"

# Development dependencies
pytest = "^7.4.0"
black = "^23.7.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"
httpx = "^0.24.0"
jupyter = "^1.0.0"
```

### Environment Configuration

```bash
# Database settings
DATABASE_URL=sqlite:///data/database/nfl_dfs.db

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# MLflow tracking
MLFLOW_TRACKING_URI=file:///data/mlflow

# Model settings
DEFAULT_MODEL_PATH=models/production/current_best_model
ENABLE_SELF_TUNING=true
TIME_WEIGHT_DECAY_FACTOR=0.95

# NFL Data settings
NFL_SEASONS_TO_LOAD=3  # How many seasons of historical data
NFL_DATA_CACHE_DIR=data/cache

# DraftKings settings
DK_CLASSIC_SALARY_CAP=50000
DK_SHOWDOWN_SALARY_CAP=50000
DK_CAPTAIN_MULTIPLIER=1.5

# Backtesting for model improvement
BACKTEST_START_SEASON=2022
BACKTEST_VALIDATION_SPLIT=0.2
```

## Data Architecture

### File Structure
```
data/
├── database/
│   └── nfl_dfs.db               # SQLite - players, teams, games, schedules
├── nfl_raw/                     # Raw NFL data from nfl_data_py
│   ├── play_by_play/            # Play-by-play data by season
│   ├── player_stats/            # Weekly player statistics
│   ├── rosters/                 # Team rosters
│   └── schedules/               # Game schedules
├── engineered/                  # Processed features
│   ├── qb_features/             # QB-specific metrics (EPA, CPOE, pressure rate)
│   ├── rb_features/             # RB metrics (YPC, broken tackles, goal line carries)
│   ├── wr_te_features/          # Receiving metrics (separation, target share, ADOT)
│   ├── dst_features/            # Defense metrics (points allowed, turnovers)
│   ├── stacking_correlations/   # QB-WR, QB-TE, game correlations
│   └── ownership_projections/   # Predicted ownership by position
├── draftkings/                  # DraftKings specific data
│   ├── salaries/                # Weekly salary CSVs (manual upload)
│   └── historical_salaries/     # Archive of past salaries
├── predictions/                 # Model predictions vs actuals
│   ├── weekly_predictions/      # What model predicted
│   ├── actual_results/          # What actually happened
│   └── error_analysis/          # Prediction errors for learning
├── tensors/                     # Pre-processed PyTorch tensors
│   ├── qb_tensors.pth          # Position-specific tensors
│   ├── rb_tensors.pth
│   ├── wr_te_tensors.pth
│   ├── dst_tensors.pth
│   └── metadata.json            # Feature names, scaling parameters
└── backtest/                    # Backtesting for model improvement
    ├── parameter_search/        # Hyperparameter optimization results
    ├── feature_importance/      # Which features matter most
    └── model_comparison/        # Comparing different model architectures
```

## NFL-Specific Application Architecture

### Core Structure
```
src/
├── config/
│   └── nfl_settings.py         # NFL positions, DK scoring, roster rules
├── data/
│   ├── collection/
│   │   ├── nfl_data_collector.py    # Pull from nfl_data_py
│   │   └── dk_salary_processor.py   # Process uploaded DK CSVs
│   ├── features/
│   │   ├── qb_features.py          # EPA, CPOE, pocket presence
│   │   ├── rb_features.py          # Rushing efficiency, pass catching
│   │   ├── wr_te_features.py       # Route running, separation, YAC
│   │   ├── dst_features.py         # Points allowed, turnovers
│   │   ├── game_script_features.py # Vegas lines, weather, pace
│   │   └── stacking_correlations.py # QB-pass catcher correlations
│   ├── ownership/
│   │   └── ownership_projector.py  # Predict ownership percentages
│   └── storage/
│       ├── database_manager.py     # SQLite operations
│       └── tensor_manager.py       # Fast tensor loading
├── models/
│   ├── position_models/
│   │   ├── qb_model.py            # QB-specific architecture
│   │   ├── rb_model.py            # RB predictions
│   │   ├── wr_te_model.py         # Pass catcher models
│   │   └── dst_model.py           # Defense scoring
│   ├── training/
│   │   ├── position_trainer.py    # Position-specific training
│   │   ├── time_weighted_trainer.py # Recency weighting
│   │   └── ensemble_trainer.py    # Combine XGBoost + NN
│   ├── projections/
│   │   ├── dk_scorer.py          # DraftKings scoring projections
│   │   ├── ceiling_floor_calculator.py # GPP vs cash projections
│   │   └── confidence_intervals.py # Uncertainty estimates
│   └── self_improvement/
│       ├── error_analyzer.py      # Learn from prediction errors
│       ├── feature_adjuster.py    # Adjust feature importance
│       └── parameter_tuner.py     # Auto-tune hyperparameters
├── optimization/
│   ├── dk_lineup_optimizer.py     # Core DraftKings optimizer
│   ├── classic_optimizer.py       # Main slate (QB/RB/RB/WR/WR/WR/TE/FLEX/DST)
│   ├── showdown_optimizer.py      # Single game (CPT + 5 FLEX)
│   ├── stacking/
│   │   ├── qb_stacker.py         # QB + pass catchers
│   │   ├── game_stacker.py       # Multi-team correlations
│   │   └── leverage_finder.py    # Low ownership + correlation
│   └── contest_strategy/
│       ├── cash_game_optimizer.py # Maximize floor
│       └── gpp_optimizer.py      # Maximize ceiling + differentiation
├── game_selection/                # Pick fun/interesting games
│   ├── game_scorer.py            # Score games by entertainment value
│   ├── matchup_analyzer.py       # Find interesting matchups
│   ├── rivalry_detector.py       # Division/rivalry games
│   ├── shootout_predictor.py     # High-scoring game potential
│   └── game_recommender.py       # Recommend fun contests
├── backtesting/
│   ├── model_backtester.py       # Test different model parameters
│   ├── feature_selector.py       # Identify best features
│   ├── parameter_optimizer.py    # Optuna hyperparameter search
│   ├── time_series_validator.py  # Walk-forward validation
│   └── performance_analyzer.py   # Analyze prediction accuracy
├── api/
│   ├── main.py                   # FastAPI application
│   ├── routers/
│   │   ├── predictions.py        # NFL player predictions
│   │   ├── lineups.py           # DK lineup optimization
│   │   ├── games.py             # Game recommendations
│   │   ├── backtesting.py       # Model improvement analysis
│   │   └── data.py              # Salary uploads, data refresh
│   └── schemas/
│       ├── nfl_player.py        # Player data models
│       ├── dk_lineup.py         # DraftKings lineup format
│       └── game_analysis.py     # Game recommendation models
├── analysis/
│   ├── prediction_accuracy.py    # How accurate were predictions
│   ├── error_patterns.py        # Common prediction mistakes
│   ├── feature_importance.py    # SHAP values by position
│   └── model_diagnostics.py     # Model health checks
├── cli/
│   └── upload_dk_salaries.py    # Upload weekly DK CSV files
└── utils/
    ├── dk_scoring.py            # DraftKings scoring rules
    ├── nfl_constants.py         # Positions, teams, schedules
    └── statistical_tests.py     # Significance testing
```

## NFL Feature Engineering

### Position-Specific Features

#### Quarterbacks
- **Passing Efficiency**: EPA/play, CPOE, Success Rate
- **Pressure Metrics**: Passer rating under pressure, sack rate
- **Rushing Upside**: Designed runs, scramble yards, goal line rushes
- **Stacking Potential**: Top 3 target correlation, TE usage in red zone
- **Game Script**: Performance when winning/losing
- **Primetime Performance**: MNF/SNF/TNF adjustments

#### Running Backs
- **Rushing Efficiency**: YPC, yards after contact, broken tackles
- **Pass Game Usage**: Target share, routes run percentage
- **Goal Line Role**: Carries inside 5, red zone share
- **Game Script Impact**: Usage when leading/trailing
- **Backup Threat**: Snap share trends, injury history

#### Wide Receivers & Tight Ends
- **Target Quality**: ADOT, air yards share, catchable targets
- **Separation Metrics**: Average separation, yards after catch
- **Red Zone Role**: Red zone targets, end zone targets
- **Slot vs Outside**: Route location and effectiveness
- **Quarterback Dependency**: Performance with backup QBs

#### Defenses
- **Points Allowed**: Recent trends, home/away splits
- **Turnover Potential**: Interception rate, fumble recoveries
- **Sack Rate**: Pass rush efficiency
- **Opponent Adjustments**: Backup QBs, offensive line injuries
- **Special Teams**: Return TD potential

### Game Environment Features
- **Vegas Lines**: Spread, total, team implied totals
- **Weather Impact**: Wind speed, precipitation, temperature
- **Stadium Effects**: Dome vs outdoor, altitude, turf type
- **Pace Metrics**: Plays per game, seconds per play
- **Division Games**: Historical division performance
- **Rest Advantage**: Thursday games, bye weeks

### DraftKings-Specific Features
- **Salary Changes**: Week-over-week price movement
- **Value Score**: Points per $1000 of salary
- **Ownership Projections**: Based on salary, recent performance, narratives
- **Leverage Score**: Low ownership + high correlation potential
- **Stack Value**: Combined salary efficiency of stacks

## Model Architecture

### Position-Specific Models

Each position has its own model architecture optimized for that position's scoring patterns:

#### QB Model
- XGBoost primary (handles game script dependencies well)
- Neural network for primetime/national game adjustments
- Separate sub-models for rushing QBs vs pocket passers

#### RB Model  
- LightGBM primary (excellent with categorical features like team)
- Separate models for workhorse vs committee backs
- Special handling for pass-catching specialists

#### WR/TE Model
- Ensemble of XGBoost + Neural Network
- Correlation features heavily weighted
- Separate ceiling/floor models for different contest types

#### DST Model
- Simpler linear model with heavy feature engineering
- Focus on opponent quality and game script

### Projection Types

#### Cash Game Projections
- Median outcomes (50th percentile)
- Floor projections (25th percentile)
- Consistency weighting

#### GPP Projections  
- Ceiling projections (75th-90th percentile)
- Boom/bust probability
- Correlation boost factors

#### Showdown Specific
- Captain value multiplier (1.5x scoring at 1.5x salary)
- Single-game correlation adjustments
- Leverage play identification

## Self-Improvement System

### Learning from Prediction Errors

After each NFL week, the system:

1. **Compares Predictions to Actuals**
   - Predicted: Lamar Jackson 24.5 FP
   - Actual: Lamar Jackson 19.2 FP
   - Error: -5.3 FP (overestimated)

2. **Analyzes Error Patterns**
   - Consistently overestimating rushing QBs in bad weather?
   - Undervaluing RBs against specific defenses?
   - Missing injury impact on target distribution?

3. **Adjusts Model Weights**
   - Reduce weather penalty for mobile QBs
   - Increase importance of defensive rankings
   - Add injury cascading effects

4. **Updates Feature Importance**
   - Track which features predicted well
   - Deprecate features that add noise
   - Discover new predictive patterns

### Backtesting for Model Optimization

The backtesting system helps improve models by:

#### Parameter Search
- Test different hyperparameters
- Find optimal model architectures
- Determine best ensemble weights

#### Feature Selection
- Identify most predictive features
- Remove redundant features
- Discover feature interactions

#### Time Decay Analysis
- Find optimal recency weighting
- Balance recent trends vs historical data
- Prevent overfitting to recent weeks

#### Validation Strategy
- Walk-forward validation on historical seasons
- Hold-out weeks for final testing
- Statistical significance testing

## Game Selection System

### Finding Fun Games to Play

Instead of maximizing profit, the system identifies entertaining contests based on:

#### Entertainment Factors
- **Shootout Potential**: High Vegas totals, close spreads
- **Star Players**: Games with exciting players to root for
- **Rivalry Games**: Division matchups with history
- **Primetime Games**: SNF, MNF with better production
- **Playoff Implications**: Late-season meaningful games

#### Personal Preferences
- **Favorite Teams**: Track your preferred teams
- **Player Followings**: Players you enjoy watching
- **Avoid Blowouts**: Skip games with huge spreads
- **Weather Preferences**: Some enjoy snow games, others don't

#### Competitive Balance
- **Multiple Viable Lineups**: Games with many good plays
- **Interesting Decisions**: Not obvious chalk plays
- **Correlation Opportunities**: Fun stacking potential

### Game Scoring Algorithm

Each game gets scored on:
- Entertainment Value (0-100)
- Competitive Balance (0-100)
- DFS Complexity (0-100)
- Personal Interest (0-100)

## Optimization Strategies

### DraftKings Classic Roster Construction
- QB + 2RB + 3WR + TE + FLEX + DST
- $50,000 salary cap
- Correlation rules enforcement

### Stacking Strategies

#### Primary Stacks
- **QB + WR1**: Highest correlation, most common
- **QB + WR2**: Good leverage in tournaments
- **QB + TE**: Valuable in high-total games
- **QB + Multiple**: Full onslaught stacks

#### Secondary Correlations
- **Opposing WR1**: Shootout scenarios
- **RB + DST**: Negative correlation leverage
- **Game Stacks**: 4-5 players from same game

### Contest-Specific Optimization

#### Cash Games (50/50s, Double-Ups)
- Maximize floor projections
- High-ownership players acceptable
- Safe stacks (QB + WR1)
- Minimize variance

#### GPP Tournaments
- Maximize ceiling projections
- Find leverage plays
- Contrarian captain in Showdowns
- Embrace variance

## API Endpoints

### Core Prediction Endpoints
```
GET  /api/v1/nfl/predictions/{week}/{slate_type}     # Get all predictions
GET  /api/v1/nfl/projections/{player_id}            # Individual projection
POST /api/v1/nfl/predictions/generate               # Generate new predictions
GET  /api/v1/nfl/predictions/accuracy/{week}        # How accurate were last week's predictions
```

### DraftKings Lineup Endpoints
```
POST /api/v1/dk/lineups/optimize/classic            # Main slate optimization
POST /api/v1/dk/lineups/optimize/showdown          # Single game optimization
POST /api/v1/dk/lineups/multi-entry                # Generate multiple lineups
GET  /api/v1/dk/lineups/analyze/{lineup_id}       # Analyze specific lineup
```

### Game Selection Endpoints
```
POST /api/v1/games/upload-slates                   # Upload available games/contests
GET  /api/v1/games/recommendations/{week}          # Get fun games to play
GET  /api/v1/games/analysis/{game_id}             # Why this game is interesting
POST /api/v1/games/preferences                     # Set your preferences
```

### Model Improvement Endpoints
```
POST /api/v1/models/learn-from-results             # Update model with actual results
GET  /api/v1/models/error-analysis/{week}         # Analyze prediction errors
POST /api/v1/models/backtest                      # Run backtesting for optimization
GET  /api/v1/models/feature-importance            # Current feature rankings
```

### Data Management Endpoints
```
POST /api/v1/dk/salaries/upload                    # Upload DK salary CSV
POST /api/v1/nfl/data/refresh                      # Pull latest NFL data
GET  /api/v1/nfl/data/status                       # Check data freshness
POST /api/v1/nfl/results/update                    # Update with actual game results
```

## Implementation Priorities

### Phase 1: Core NFL Data Pipeline (Weeks 1-2)
- Set up nfl_data_py integration
- Build position-specific feature engineering
- Create DraftKings salary processor
- Implement basic XGBoost models by position

### Phase 2: DraftKings Optimization (Weeks 3-4)
- Build lineup optimizer with stacking
- Implement cash vs GPP strategies
- Add Showdown captain optimization
- Create correlation matrices from historical data

### Phase 3: Model Self-Improvement (Weeks 5-6)
- Build prediction error tracking
- Implement backtesting framework
- Add automatic parameter tuning
- Create feature importance evolution

### Phase 4: Game Selection & Analysis (Weeks 7-8)
- Build game entertainment scorer
- Add personal preference system
- Create comprehensive error analysis
- Build model interpretation tools

## Success Metrics

### Model Accuracy
- QB predictions: MAE < 3.5 FP
- RB predictions: MAE < 3.0 FP  
- WR predictions: MAE < 2.5 FP
- Continuous improvement week-over-week

### Prediction Improvement
- Error reduction after each week's learning
- Better feature importance rankings over time
- Improved accuracy on specific game situations

### User Experience
- Find engaging games to play
- Generate diverse interesting lineups
- Understand why predictions were made
- Learn from prediction mistakes

### System Performance
- Full slate optimization < 5 seconds
- Weekly model retraining < 10 minutes
- Backtest parameter search < 30 minutes

## Why This Approach

This NFL-specific design focuses on continuous improvement and entertainment:

1. **Self-improving models** learn from comparing predictions to actual results, not contest outcomes
2. **Backtesting** identifies which features and parameters improve prediction accuracy
3. **Game selection** finds fun, interesting games rather than trying to maximize profit
4. **Position-specific models** because QBs, RBs, and WRs have fundamentally different patterns
5. **Correlation-based optimization** because stacking makes DFS more entertaining
6. **Error analysis** helps understand and fix systematic prediction problems

The system is designed for personal enjoyment of NFL DFS, with a focus on getting better at predictions over time rather than treating it as an investment vehicle.