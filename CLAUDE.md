# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management (UV)

The project uses UV, a fast Rust-based Python package manager:

- `uv venv --python 3.11` - Create virtual environment
- `uv pip install -r requirements-dev.txt` - Install all dependencies including dev tools
- `uv pip sync requirements.txt` - Install production dependencies only
- `uv run <command>` - Run commands in the virtual environment

### Essential Commands

- `make setup` - Complete development setup (installs deps, hooks, creates .env)
- `make run` - Start development server (FastAPI on localhost:8000)
- `make format` - Format code with Ruff, Black, isort, mdformat
- `make lint` - Run all linters (Ruff, mypy, bandit, etc.)
- `make test` - Run pytest test suite
- `make test-cov` - Run tests with coverage reports
- `make clean` - Remove build artifacts and cache

### Database Commands

- `make db-init` - Initialize SQLite database
- `make db-migrate` - Run Alembic migrations
- `uv run python scripts/init_database.py` - Manual database initialization

### CLI Tools

The project includes CLI commands for data collection and ML training:

- `uv run python -m src.cli.collect_data init-db` - Initialize database
- `uv run python -m src.cli.collect_data collect-teams` - Collect NFL team data
- `uv run python -m src.cli.train_models train-position QB` - Train QB model

### Code Quality

- **Ruff**: Primary linter and formatter (fastest, Rust-based)
- **Black**: Secondary Python formatter
- **mypy**: Type checking with strict configuration
- **pytest**: Testing with coverage reports (80% minimum)
- **pre-commit**: Automated hooks for code quality

Run `make help` to see all available commands.

## Architecture Overview

This is an NFL Daily Fantasy Sports (DFS) prediction and optimization system built for personal use with DraftKings contests.

### Core Components

**Data Layer (`src/data/`)**:

- `collection/` - NFL data collection (nfl_data_py) and DraftKings CSV parsing
- `processing/` - Feature engineering and data transformation
- `validation/` - Data quality checks and validation

**Machine Learning (`src/ml/`)**:

- `models/` - Position-specific ML models (QB, RB, WR, TE, DEF)
- `training/` - Model training pipelines with cross-validation
- `registry.py` - Model versioning and deployment system

**Optimization (`src/optimization/`)**:

- `lineup_builder.py` - Linear programming-based lineup optimization using PuLP
- Contest-specific strategies (GPP vs Cash games)
- Stacking logic and exposure controls

**API (`src/api/`)**:

- FastAPI-based REST API with automatic OpenAPI docs
- `routers/data.py` - Data access endpoints
- `routers/predictions.py` - ML prediction endpoints

**Database (`src/database/`)**:

- SQLite with SQLAlchemy ORM
- Comprehensive data models for NFL stats, DFS contests, and ML metadata

### Key Design Patterns

**Position-Specific Modeling**: Each NFL position (QB, RB, WR, TE, DEF) has dedicated models due to different statistical patterns and scoring characteristics.

**Self-Learning System**: Models continuously improve by learning from prediction errors and automatically retraining on new data.

**Local-First Architecture**: All processing and storage is local (no cloud dependencies) for privacy and cost-free operation.

**Entertainment Over Profit**: System optimizes for fun/engaging contests rather than pure expected value maximization.

### Data Flow

1. **Collection**: NFL data via nfl_data_py, DraftKings salaries via CSV upload
1. **Processing**: Feature engineering pipeline creates position-specific features
1. **Training**: Models train on historical data with time-based cross-validation
1. **Prediction**: API serves real-time player projections
1. **Optimization**: Linear programming generates optimal lineups with constraints
1. **Evaluation**: Continuous learning from actual vs predicted performance

### Technology Stack

- **Python 3.11+** with UV package management
- **FastAPI** for API layer with Uvicorn server
- **SQLite + SQLAlchemy** for data persistence
- **PyTorch + XGBoost + LightGBM** for machine learning
- **PuLP** for linear programming optimization
- **Pandas + NumPy** for data processing
- **pytest** for comprehensive testing

### File Structure Highlights

- `src/config/settings.py` - Centralized configuration with environment variables
- `src/database/models.py` - SQLAlchemy models for all entities
- `src/ml/models/position_models.py` - Core ML model implementations
- `docs/architecture/` - Comprehensive system documentation
- `Makefile` - Development workflow automation
- `pyproject.toml` - Project configuration with Ruff/Black/mypy settings

### Development Workflow

1. Use `make setup` for initial environment setup
1. Run `make lint` and `make test` before committing
1. API development: Start with `make run` and use localhost:8000/docs
1. ML development: Use CLI tools for training and evaluation
1. Database changes: Update models and run migrations

### Important Notes

- The system uses UV instead of pip/poetry for significantly faster dependency management
- All models are CPU-optimized (GPU support planned but not required)
- Pre-commit hooks automatically format code and run quality checks
- Comprehensive test coverage with pytest and coverage reporting
- API documentation auto-generated at `/docs` endpoint when running

### Testing

- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Model validation with backtesting framework
- Minimum 80% test coverage enforced
- Use `pytest -v` for verbose test output or `pytest tests/specific_test.py` for specific tests
