# Complete DFS Application Refactoring Guide

## Current State Analysis

Your DFS application has grown into several massive files with major optimization plans:

### Current Monolithic Structure (12,000+ lines)

- **data.py**: 4774 lines - Data collection, DB ops, feature engineering
- **models.py**: 2070 lines - All ML models and training logic
- **run.py**: 1063 lines - CLI interface with all commands
- **optimize.py**: 1049 lines - Lineup optimization and constraints
- **backtest.py**: 939 lines - Backtesting and validation
- **utils.py**: 509 lines - Mixed utilities

### Planned Major Improvements (From Optimization Guides)

- **QB Model**: Multi-head architecture, DFS loss, Vegas features (COMPLETED - R² 0.156)
- **RB Model**: Red zone features, volume metrics, game script (R² from -1.14 to 0.35+ target)
- **WR Model**: Fix sigmoid scaling, target competition, route analysis (R² 0.25 → 0.40+ target)
- **TE Model**: Role differentiation, TD access, target share (R² from -0.89 to 0.30+ target)
- **DST Model**: Component models, Vegas integration, weather (R² from 0.008 to 0.25+ target)
- **Hyperparameter Tuning**: LR finder, batch size optimization, Bayesian tuning
- **Advanced Features**: 75+ engineered features per position, play-by-play integration

## Proposed Modular Architecture

```
dfs/
├── __init__.py
├── core/                       # Core application logic
│   ├── __init__.py
│   ├── config.py              # Application configuration
│   ├── logging.py             # Centralized logging
│   └── exceptions.py          # Custom exceptions
├── cli/                        # Command-line interface
│   ├── __init__.py
│   ├── main.py                # CLI entry point
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── collect.py         # Data collection commands
│   │   ├── train.py           # Model training commands
│   │   ├── predict.py         # Prediction commands
│   │   ├── optimize.py        # Lineup optimization commands
│   │   └── backtest.py        # Backtesting commands
│   └── utils.py               # CLI utilities
├── data/                       # Data management (split data.py)
│   ├── __init__.py
│   ├── collectors/            # Data collection (1200 lines from data.py)
│   │   ├── __init__.py
│   │   ├── nfl_data.py        # NFL stats collection
│   │   ├── draftkings.py      # DK salary/contest data
│   │   ├── weather.py         # Weather data integration
│   │   ├── injuries.py        # Injury reports
│   │   └── betting.py         # Vegas odds collection
│   ├── processors/            # Data processing (1800 lines from data.py)
│   │   ├── __init__.py
│   │   ├── features/          # Feature engineering (new)
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Base feature extractor
│   │   │   ├── qb_features.py # QB Vegas/passing features
│   │   │   ├── rb_features.py # RB volume/red zone features
│   │   │   ├── wr_features.py # WR target/route features
│   │   │   ├── te_features.py # TE role/target features
│   │   │   ├── dst_features.py# DST component features
│   │   │   └── correlation.py # Player correlation features
│   │   ├── validation.py      # Data quality checks
│   │   ├── transforms.py      # Data transformations
│   │   └── aggregation.py     # Rolling averages, grouping
│   ├── database/              # Database management (800 lines from data.py)
│   │   ├── __init__.py
│   │   ├── manager.py         # DB connection management
│   │   ├── schema.py          # Database schema definitions
│   │   ├── migrations.py      # Schema migrations
│   │   ├── queries.py         # Complex SQL queries
│   │   └── performance.py     # DB performance optimization
│   ├── loaders.py             # Data loading utilities (900 lines from data.py)
│   └── validators.py          # Data validation (270 lines from data.py)
├── models/                     # Machine learning models (split models.py)
│   ├── __init__.py
│   ├── base/                  # Base model components (400 lines from models.py)
│   │   ├── __init__.py
│   │   ├── neural.py          # BaseNeuralModel
│   │   ├── ensemble.py        # Base ensemble logic
│   │   ├── hypertuning.py     # LR finder, batch optimization
│   │   └── utils.py           # Model utilities
│   ├── networks/              # Position-specific networks
│   │   ├── __init__.py
│   │   ├── qb.py             # QB multi-head architecture (300 lines)
│   │   ├── rb.py             # RB volume/efficiency model (200 lines)
│   │   ├── wr.py             # WR fixed scaling architecture (200 lines)
│   │   ├── te.py             # TE role-based model (150 lines)
│   │   └── dst.py            # DST component models (250 lines)
│   ├── training/              # Training orchestration (600 lines from models.py)
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training pipeline
│   │   ├── callbacks.py       # Training callbacks
│   │   ├── metrics.py         # Training metrics
│   │   ├── early_stopping.py  # Early stopping logic
│   │   └── hyperopt.py        # Hyperparameter optimization
│   ├── loss_functions.py      # DFS loss, quantile loss (200 lines)
│   ├── ensemble.py            # Ensemble methods (400 lines from models.py)
│   ├── factory.py             # Model creation factory (100 lines)
│   └── persistence.py         # Model saving/loading (170 lines from models.py)
├── optimization/               # Lineup optimization (split optimize.py)
│   ├── __init__.py
│   ├── engine.py              # Main optimization engine (300 lines)
│   ├── constraints/           # Constraint management (400 lines from optimize.py)
│   │   ├── __init__.py
│   │   ├── salary.py          # Salary cap constraints
│   │   ├── position.py        # Position requirements
│   │   ├── ownership.py       # Ownership constraints
│   │   ├── stacking.py        # QB-WR stacking rules
│   │   └── exposure.py        # Player exposure limits
│   ├── strategies/            # Optimization strategies (250 lines from optimize.py)
│   │   ├── __init__.py
│   │   ├── cash.py            # Cash game strategy
│   │   ├── tournament.py      # Tournament strategy
│   │   └── contrarian.py      # Contrarian strategy
│   ├── validators.py          # Lineup validation (100 lines from optimize.py)
│   └── generators.py          # Lineup generation logic
├── backtesting/                # Backtesting framework (split backtest.py)
│   ├── __init__.py
│   ├── engine.py              # Backtesting engine (300 lines)
│   ├── metrics.py             # Performance metrics (200 lines)
│   ├── reports.py             # Report generation (250 lines)
│   ├── visualizations.py      # Charts and plots (150 lines)
│   └── scenarios.py           # Test scenarios (39 lines from backtest.py)
├── prediction/                 # Prediction pipeline (new - extracted from run.py)
│   ├── __init__.py
│   ├── pipeline.py            # Prediction orchestration
│   ├── preprocessors.py       # Data preprocessing for predictions
│   ├── postprocessors.py      # Prediction post-processing
│   ├── validators.py          # Prediction validation
│   └── exporters.py           # CSV/format export
├── utils/                      # Shared utilities (reorganized from utils.py)
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logging.py             # Logging utilities
│   ├── validation.py          # General validation functions
│   ├── math.py                # Mathematical utilities
│   ├── datetime.py            # Date/time helpers
│   └── io.py                  # File I/O utilities
└── tests/                      # Comprehensive test suite
    ├── __init__.py
    ├── unit/                  # Unit tests by module
    │   ├── test_data/
    │   ├── test_models/
    │   ├── test_optimization/
    │   └── test_backtesting/
    ├── integration/           # Integration tests
    ├── e2e/                   # End-to-end tests
    └── fixtures/              # Test data and fixtures
```

## Key Benefits of Modular Structure

### 1. **Domain Separation**

- **Data concerns**: Collection, processing, validation separate
- **ML concerns**: Models, training, evaluation isolated
- **Business logic**: Optimization, backtesting, prediction distinct
- **Interface**: CLI cleanly separated from core logic

### 2. **Planned Optimization Integration**

- **Feature engineering**: Dedicated modules for 75+ engineered features per position
- **Model improvements**: Space for multi-head QB, component DST models
- **Hyperparameter tuning**: Dedicated training/hyperopt module
- **Advanced backtesting**: Comprehensive metrics and visualization

### 3. **Development Workflow**

- **Parallel development**: Team can work on different positions simultaneously
- **Testing**: Isolated unit tests for each component
- **Debugging**: Clear separation of concerns for troubleshooting
- **Deployment**: Independent model deployment capability

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)

1. **Extract base classes** from models.py → models/base/
2. **Split CLI logic** from run.py → cli/
3. **Create core utilities** from utils.py → utils/
4. **Database management** from data.py → data/database/

### Phase 2: Data Pipeline (Week 2)

1. **Feature engineering** from data.py → data/processors/features/
2. **Data collectors** from data.py → data/collectors/
3. **Validation logic** from data.py → data/validators.py
4. **Implement planned feature sets** per position optimization guides

### Phase 3: Model Architecture (Week 3)

1. **Position networks** from models.py → models/networks/
2. **Training pipeline** from models.py → models/training/
3. **Ensemble methods** from models.py → models/ensemble.py
4. **Integrate optimization improvements** (multi-head QB, component DST)

### Phase 4: Business Logic (Week 4)

1. **Optimization engine** from optimize.py → optimization/
2. **Backtesting framework** from backtest.py → backtesting/
3. **Prediction pipeline** from run.py → prediction/
4. **Integration testing** across all modules

## Implementation Integration with Optimization Guides

### QB Model Integration

```python
# models/networks/qb.py - Multi-head architecture from QB guide
class QBNetwork(nn.Module):
    def __init__(self, input_size: int):
        # Implement multi-head architecture from optimization guide
        self.passing_head = PassingHead()
        self.rushing_head = RushingHead()
        self.bonus_head = BonusHead()
        # Vegas features integration from data/processors/features/qb_features.py
```

### RB Model Integration

```python
# data/processors/features/rb_features.py - Red zone features from RB guide
class RBFeatureExtractor:
    def extract_red_zone_features(self):
        # Implement red_zone_rush_share, inside_10_attempts from RB guide
        # TD access features, volume metrics, game script integration
```

### DST Component Models

```python
# models/networks/dst.py - Component architecture from DST guide
class DSTComponentNetwork(nn.Module):
    def __init__(self):
        self.sacks_head = PoissonHead()      # From DST guide
        self.turnovers_head = PoissonHead()  # Component model approach
        self.pa_head = OrdinalHead()         # Points allowed buckets
        self.td_head = BinaryHead()          # Defensive TD probability
```

### Feature Engineering Pipeline

```python
# data/processors/features/base.py - Base for all position features
class BaseFeatureExtractor:
    def extract_vegas_features(self):
        # team_implied_total, spread, game_total from optimization guides

    def extract_weather_features(self):
        # Wind, precipitation, temperature integration
```

## Backward Compatibility Strategy

### API Preservation

```python
# Main __init__.py maintains existing imports
from .models import create_model, QBNeuralModel  # Still works
from .optimization import optimize_lineup        # Still works
from .backtesting import run_backtest           # Still works
```

### Gradual Migration

1. **Phase 1**: New structure alongside existing files
2. **Phase 2**: Update imports internally, maintain external API
3. **Phase 3**: Remove old monolithic files after validation
4. **Phase 4**: Full migration with enhanced features

## Testing Strategy

### Component Testing

```python
# tests/unit/test_models/test_qb.py
def test_qb_multihead_architecture():
    # Test QB multi-head from optimization guide

def test_qb_vegas_features():
    # Test Vegas feature integration
```

### Integration Testing

```python
# tests/integration/test_prediction_pipeline.py
def test_end_to_end_prediction():
    # Test complete pipeline: data → features → models → predictions
```

### Performance Testing

```python
# tests/performance/test_model_improvements.py
def test_rb_model_r2_improvement():
    # Validate R² improvement from -1.14 to 0.35+

def test_dst_component_models():
    # Validate DST R² improvement from 0.008 to 0.25+
```

## Expected Outcomes

### Development Efficiency

- **File navigation**: 200-400 line files vs 2000+ line monoliths
- **Feature development**: Isolated feature modules vs mixed concerns
- **Model experimentation**: Independent position model development
- **Bug isolation**: Clear module boundaries for debugging

### Performance Improvements (From Optimization Guides)

- **QB Model**: R² 0.156 → maintain with better architecture
- **RB Model**: R² -1.14 → 0.35+ (planned improvement)
- **WR Model**: R² 0.25 → 0.40+ (scaling fixes + features)
- **TE Model**: R² -0.89 → 0.30+ (architecture + features)
- **DST Model**: R² 0.008 → 0.25+ (component models)

### Code Quality

- **Single Responsibility**: Each module has clear purpose
- **Open/Closed**: Easy to extend without modifying existing code
- **Dependency Injection**: Clean interfaces between modules
- **Testability**: Isolated components enable thorough testing

## Implementation Timeline

- **Week 1**: Core infrastructure + CLI separation
- **Week 2**: Data pipeline + feature engineering
- **Week 3**: Model architecture + optimization integration
- **Week 4**: Business logic + comprehensive testing
- **Week 5**: Integration of all optimization guide improvements
- **Week 6**: Performance validation + production deployment

## Success Metrics

### Technical Metrics

- **File count**: 6 large files → 50+ focused modules
- **Average file size**: 1200+ lines → 200-400 lines
- **Test coverage**: <20% → >80%
- **Model performance**: Implementation of all optimization targets

### Development Metrics

- **Feature addition time**: Days → Hours
- **Bug isolation time**: Hours → Minutes
- **Onboarding time**: Weeks → Days
- **Parallel development**: 1 developer → 3+ developers simultaneously

This modular architecture provides the foundation for implementing all your planned optimizations while maintaining clean, testable, and maintainable code structure.
