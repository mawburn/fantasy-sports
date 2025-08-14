# NFL DFS System - Makefile for common tasks
.PHONY: help install install-dev test format lint clean run setup-hooks

# Variables
PYTHON := python3
UV := uv
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip

# Default target
help:
	@echo "NFL DFS System - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install production dependencies with UV"
	@echo "  make install-dev    - Install all dependencies including dev tools"
	@echo "  make setup-hooks    - Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         - Format code with Ruff and Black"
	@echo "  make lint           - Run all linters (Ruff, mypy, etc.)"
	@echo "  make test           - Run tests with pytest"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Start the development server"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make freeze         - Generate UV lockfile"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build documentation"
	@echo "  make docs-serve     - Serve documentation locally"

# Create virtual environment if it doesn't exist
$(VENV):
	$(UV) venv --python 3.11

# Install production dependencies
install: $(VENV)
	$(UV) pip sync requirements.txt
	@echo "Production dependencies installed!"

# Install all dependencies including dev tools
install-dev: $(VENV)
	$(UV) pip install -r requirements-dev.txt
	@echo "All dependencies installed!"

# Setup pre-commit hooks
setup-hooks: install-dev
	$(VENV)/bin/pre-commit install
	@echo "Pre-commit hooks installed!"

# Format code
format: $(VENV)
	@echo "Formatting Python code with Ruff..."
	$(VENV)/bin/ruff format src/ tests/ scripts/
	$(VENV)/bin/ruff check --fix src/ tests/ scripts/
	@echo "Formatting Python code with Black..."
	$(VENV)/bin/black src/ tests/ scripts/
	@echo "Sorting imports with isort..."
	$(VENV)/bin/isort src/ tests/ scripts/
	@echo "Formatting Markdown files..."
	$(VENV)/bin/mdformat docs/ *.md
	@echo "Formatting complete!"

# Run linters
lint: $(VENV)
	@echo "Running Ruff linter..."
	$(VENV)/bin/ruff check src/ tests/ scripts/
	@echo "Running Black check..."
	$(VENV)/bin/black --check src/ tests/ scripts/
	@echo "Running isort check..."
	$(VENV)/bin/isort --check-only src/ tests/ scripts/
	@echo "Running mypy..."
	$(VENV)/bin/mypy src/
	@echo "Running bandit security checks..."
	$(VENV)/bin/bandit -r src/ -ll
	@echo "Linting complete!"

# Run tests
test: $(VENV)
	$(VENV)/bin/pytest tests/ -v

# Run tests with coverage
test-cov: $(VENV)
	$(VENV)/bin/pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Clean build artifacts and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "Cleanup complete!"

# Generate UV lockfile
freeze: $(VENV)
	$(UV) pip freeze > uv.lock
	@echo "Lockfile generated!"

# Start development server
run: $(VENV)
	$(PYTHON_VENV) -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Build documentation
docs: $(VENV)
	$(VENV)/bin/mkdocs build

# Serve documentation locally
docs-serve: $(VENV)
	$(VENV)/bin/mkdocs serve

# Database operations
db-init: $(VENV)
	$(PYTHON_VENV) scripts/init_database.py

db-migrate: $(VENV)
	$(VENV)/bin/alembic upgrade head

db-backup: $(VENV)
	$(PYTHON_VENV) scripts/backup_system.py

# Install everything and setup for first use
setup: install-dev setup-hooks
	@echo "Creating .env file from example..."
	cp -n .env.example .env || true
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Edit .env file with your configuration"
	@echo "2. Run 'make db-init' to initialize the database"
	@echo "3. Run 'make run' to start the development server"
