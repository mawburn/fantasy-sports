# NFL DFS System

A comprehensive NFL Daily Fantasy Sports (DFS) prediction and optimization system focused on DraftKings contests. Built with Python, PyTorch, and modern ML practices.

## Features

- **ML-Powered Predictions**: Position-specific models for accurate player projections
- **Lineup Optimization**: Advanced algorithms for optimal lineup construction
- **Self-Learning System**: Continuous model improvement based on results
- **Game Selection**: Intelligent contest selection based on expected value
- **CPU-Optimized**: Efficient performance on standard hardware (GPU support planned)

## Quick Start

### Prerequisites

- Python 3.11+
- UV package manager (installed automatically)
- 8GB+ RAM recommended
- 20GB+ disk space

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nfl-dfs-system.git
cd nfl-dfs-system
```

2. Install UV (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Set up the development environment:

```bash
make setup
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:

```bash
make db-init
```

6. **Collect NFL Data** (Required for ML training):

```bash
# Option 1: Collect all data at once (recommended)
uv run python -m src.cli.collect_data collect-all

# Option 2: Collect data step by step
uv run python -m src.cli.collect_data collect-teams
uv run python -m src.cli.collect_data collect-players
uv run python -m src.cli.collect_data collect-schedules
uv run python -m src.cli.collect_data collect-stats

# Option 3: Collect specific seasons (2018-2024 for comprehensive training)
uv run python -m src.cli.collect_data collect-schedules -s 2018 -s 2019 -s 2020 -s 2021 -s 2022 -s 2023 -s 2024
uv run python -m src.cli.collect_data collect-stats -s 2018 -s 2019 -s 2020 -s 2021 -s 2022 -s 2023 -s 2024

# Check data collection status
uv run python -m src.cli.collect_data status
```

**Expected Data Volume:**

- Teams: ~36 NFL teams
- Players: ~3,800+ current/historical players
- Games: ~1,900+ games (2018-2024)
- Player Stats: ~30,000+ performance records

7. Start the development server:

```bash
make run
```

## Development

### Available Commands

```bash
make help           # Show all available commands
make install        # Install production dependencies
make install-dev    # Install all dependencies
make format         # Format code with Ruff/Black
make lint           # Run linters
make test           # Run tests
make test-cov       # Run tests with coverage
```

### Data Collection Commands

```bash
# Data collection (run after database initialization)
uv run python -m src.cli.collect_data collect-all          # Collect all NFL data
uv run python -m src.cli.collect_data collect-teams        # Collect team data
uv run python -m src.cli.collect_data collect-players      # Collect player rosters
uv run python -m src.cli.collect_data collect-schedules    # Collect game schedules
uv run python -m src.cli.collect_data collect-stats        # Collect player statistics
uv run python -m src.cli.collect_data collect-weather      # Collect weather data
uv run python -m src.cli.collect_data status               # Check database status

# Collect specific seasons (useful for updates)
uv run python -m src.cli.collect_data collect-stats -s 2024

# Weather data collection
uv run python -m src.cli.collect_data collect-weather --upcoming  # Collect weather for upcoming games only
uv run python -m src.cli.collect_data collect-weather -s 2024 -w 10  # Collect weather for specific season/week

# DraftKings salary data (CSV upload)
uv run python -m src.cli.collect_data collect-dk           # Process all CSVs in data/draftkings/salaries/
uv run python -m src.cli.collect_data collect-dk --file [path]  # Process single CSV file
```

### ML Model Training Commands

```bash
# Train position-specific models
uv run python -m src.cli.train_models train-position QB    # Train quarterback model
uv run python -m src.cli.train_models train-position RB    # Train running back model
uv run python -m src.cli.train_models train-position WR    # Train wide receiver model
uv run python -m src.cli.train_models train-position TE    # Train tight end model
uv run python -m src.cli.train_models train-position DEF   # Train defense model

# Train with neural networks (PyTorch models)
uv run python -m src.cli.train_models train-position QB --use-neural

# Train all position models at once
uv run python -m src.cli.train_models train-all-positions
```

### Code Quality Tools

This project uses comprehensive linting and formatting:

- **Ruff**: Fast Python linter and formatter (Rust-based, 10-100x faster)
- **Black**: Python code formatter
- **isort**: Import sorting
- **mypy**: Static type checking
- **mdformat**: Markdown formatting
- **yamllint**: YAML linting
- **prettier**: JSON/YAML formatting
- **pre-commit**: Removed - no longer using automated hooks

#### Formatting Commands

```bash
# Format everything at once (recommended)
make format

# Individual formatting commands
ruff format src/ tests/           # Format Python with Ruff (fastest)
ruff check --fix src/ tests/      # Fix Python linting issues
black src/ tests/                 # Format Python with Black
mdformat docs/ *.md                # Format Markdown files
prettier --write "**/*.{json,yaml,yml}"  # Format JSON/YAML files

# Run all formatters manually (pre-commit removed)
ruff format src/ tests/ scripts/
ruff check --fix src/ tests/ scripts/
black src/ tests/ scripts/
isort src/ tests/ scripts/
mdformat --compact-tables docs/ *.md
```

#### Linting Commands

```bash
# Run all linters
make lint

# Individual linting commands
ruff check src/ tests/             # Python linting with Ruff
mypy src/                          # Type checking
bandit -r src/                     # Security linting
yamllint .                         # YAML linting
```

#### Git Hooks (Removed)

Pre-commit hooks have been removed. You can run formatters manually:

```bash
# Run all formatters manually
make format

# Or run individual formatters
ruff format src/ tests/ scripts/
black src/ tests/ scripts/
isort src/ tests/ scripts/
mdformat --compact-tables docs/ *.md
```

### Project Structure

```
nfl-dfs-system/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration management
│   ├── data/              # Data collection and processing
│   ├── database/          # Database models and operations
│   ├── features/          # Feature engineering
│   ├── ml/                # Machine learning models
│   ├── monitoring/        # System monitoring
│   ├── optimization/      # Lineup optimization
│   └── utils/             # Utility functions
├── tests/                  # Test suite
├── data/                   # Data storage
├── models/                 # Trained models
├── scripts/                # Utility scripts
├── docs/                   # Documentation
│   └── architecture/      # System architecture docs
└── requirements.txt        # Dependencies
```

## Architecture

See the [architecture documentation](docs/architecture/) for detailed system design:

- [System Overview](docs/architecture/01-system-overview.md)
- [Data Models](docs/architecture/02-data-models.md)
- [API Specifications](docs/architecture/03-api-specifications.md)
- [ML Pipeline](docs/architecture/07-ml-pipeline.md)
- [Optimization Algorithms](docs/architecture/08-optimization-algorithms.md)

## API Documentation

Once the server is running, visit:

- API Documentation: <http://localhost:8000/docs>
- Alternative API Docs: <http://localhost:8000/redoc>

### DraftKings Data Upload

Upload DraftKings salary CSV files via the web API:

```bash
# Upload CSV file via API
curl -X POST "http://localhost:8000/api/data/upload/draftkings" \
  -F "file=@path/to/salary.csv" \
  -F "contest_name=contest-name-optional"
```

Or use the interactive API documentation at `/docs` for easy file uploads.

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

## Contributing

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes
1. Run tests and linting (`make test && make lint`)
1. Commit your changes (run `make format` and `make lint` before committing)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with UV, the blazing-fast Rust-based Python package manager
- Uses nfl-data-py for NFL data collection
- Powered by PyTorch for machine learning
