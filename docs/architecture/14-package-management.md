# Package Management with UV

## Overview

This document outlines the package management strategy for the NFL DFS system using UV, the fastest Python package manager available. UV is written in Rust and provides 10-100x faster dependency resolution compared to traditional tools like pip and Poetry.

## Why UV?

### Performance Advantages

```yaml
performance_comparison:
  dependency_resolution:
    uv: ~0.1-1 seconds
    pip: ~10-60 seconds
    poetry: ~30-120 seconds
  
  installation_speed:
    uv: 10-100x faster than pip
    cold_cache: Still 8-10x faster
  
  memory_usage:
    uv: Minimal memory footprint
    pip: Higher memory consumption
    poetry: Highest memory usage
```

### Key Features

- **Blazing Fast**: Written in Rust for maximum performance
- **Deterministic Lockfiles**: `uv.lock` ensures reproducible builds across environments
- **Drop-in Replacement**: Compatible with existing pip workflows and requirements.txt
- **Minimal Dependencies**: No Python runtime required for UV itself
- **Cross-platform**: Works on macOS, Linux, and Windows
- **Virtual Environment Management**: Built-in venv creation and management
- **Security Auditing**: Integrated vulnerability scanning with `uv pip audit`

## Installation

### System Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Verification

```bash
# Check UV version
uv --version

# Check available commands
uv --help
```

## Project Setup

### Initial Project Configuration

```bash
# Create new project directory
mkdir nfl-dfs-system
cd nfl-dfs-system

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Create requirements files
touch requirements.txt
touch requirements-dev.txt
```

### Requirements File Structure

```txt
# requirements.txt - Production dependencies
# Managed by UV for deterministic installs

# Core ML/Data Science
torch>=2.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Web Framework
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Database
sqlalchemy>=2.0.0
alembic>=1.11.0

# NFL Data
nfl-data-py>=0.3.0

# Optimization
pulp>=2.7.0
scipy>=1.11.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
schedule>=1.2.0
joblib>=1.3.0

# ML Experiment Tracking
mlflow>=2.5.0

# Visualization (optional)
matplotlib>=3.7.0
seaborn>=0.12.0
```

```txt
# requirements-dev.txt - Development dependencies
# Include production deps
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0

# Code Quality
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.4.0
bandit>=1.7.0

# Development Tools
ipython>=8.14.0
jupyter>=1.0.0
httpx>=0.24.0
```

## Common UV Commands

### Dependency Management

```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Install with exact versions from lock file (recommended for production)
uv pip sync requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt

# Add a new package
uv pip install pandas
# Then manually add to requirements.txt

# Upgrade a package
uv pip install --upgrade pandas

# Upgrade all packages
uv pip install --upgrade -r requirements.txt

# Generate lock file
uv pip freeze > uv.lock

# Install from lock file for reproducible builds
uv pip sync uv.lock
```

### Virtual Environment Management

```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# Deactivate
deactivate

# Remove virtual environment
rm -rf .venv
```

### Security Auditing

```bash
# Audit installed packages for vulnerabilities
uv pip audit

# Audit with detailed output
uv pip audit --desc

# Audit and automatically fix vulnerabilities
uv pip audit --fix

# Audit specific requirements file
uv pip audit -r requirements.txt
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install UV
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH
      
      - name: Create virtual environment
        run: uv venv
      
      - name: Install dependencies
        run: |
          source .venv/bin/activate
          uv pip sync requirements.txt
          uv pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Security audit
        run: |
          source .venv/bin/activate
          uv pip audit
```

### Docker Integration

```dockerfile
# Dockerfile with UV
FROM python:3.11-slim

# Install UV
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get remove -y curl && apt-get autoremove -y
    
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt uv.lock ./

# Create venv and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip sync uv.lock

# Copy application code
COPY . .

# Activate venv and run application
CMD [".venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Best Practices

### 1. Lock File Management

```bash
# Always commit uv.lock for reproducible builds
git add uv.lock
git commit -m "Update dependency lock file"

# Team members should install from lock file
uv pip sync uv.lock  # Instead of uv pip install -r requirements.txt
```

### 2. Development Workflow

```bash
# Standard development setup
git clone <repository>
cd nfl-dfs-system
uv venv
source .venv/bin/activate
uv pip sync uv.lock
uv pip install -r requirements-dev.txt

# Before committing changes
uv pip audit  # Check for vulnerabilities
pytest        # Run tests
black .       # Format code
isort .       # Sort imports
```

### 3. Dependency Updates

```bash
# Weekly dependency update workflow
uv pip install --upgrade -r requirements.txt
uv pip audit
pytest
uv pip freeze > uv.lock
git add uv.lock requirements.txt
git commit -m "Weekly dependency updates"
```

### 4. Performance Optimization

```yaml
optimization_tips:
  parallel_downloads:
    # UV automatically uses optimal parallelism
    default: Auto-configured based on system
  
  cache_management:
    location: ~/.cache/uv
    clear_cache: uv cache clean
    
  network_optimization:
    # Use local PyPI mirror if available
    index_url: https://pypi.company.internal/simple/
```

## Migration from Other Tools

### From pip/requirements.txt

```bash
# No changes needed to requirements.txt
# Just replace pip commands with uv pip

# Old
pip install -r requirements.txt

# New
uv pip install -r requirements.txt
# or better
uv pip sync requirements.txt
```

### From Poetry

```bash
# Export from Poetry
poetry export -f requirements.txt > requirements.txt

# Install with UV
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip freeze > uv.lock
```

### From Pipenv

```bash
# Export from Pipenv
pipenv requirements > requirements.txt

# Install with UV
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip freeze > uv.lock
```

## Troubleshooting

### Common Issues

```bash
# Issue: UV not found after installation
# Solution: Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# Issue: Permission denied
# Solution: Use user installation
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path

# Issue: Conflicts with system Python
# Solution: Always use virtual environments
uv venv
source .venv/bin/activate

# Issue: Package not found
# Solution: Check index URL
uv pip install pandas --index-url https://pypi.org/simple

# Issue: SSL certificate errors
# Solution: Update certificates or use trusted host
uv pip install --trusted-host pypi.org pandas
```

## Performance Benchmarks

### Real-world Comparison

```python
# benchmark_package_managers.py
import time
import subprocess
import statistics

def benchmark_install(manager_cmd, runs=5):
    """Benchmark package manager installation speed"""
    times = []
    
    for i in range(runs):
        # Clear cache
        subprocess.run("rm -rf .venv", shell=True)
        
        # Measure installation time
        start = time.time()
        subprocess.run(manager_cmd, shell=True)
        elapsed = time.time() - start
        times.append(elapsed)
        
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0
    }

# Results on typical NFL DFS dependencies
benchmarks = {
    'uv': benchmark_install('uv venv && uv pip install -r requirements.txt'),
    'pip': benchmark_install('python -m venv .venv && .venv/bin/pip install -r requirements.txt'),
    'poetry': benchmark_install('poetry install')
}

# UV typically shows:
# - 10-100x faster than pip
# - 15-150x faster than Poetry
# - Consistent performance across different dependency sets
```

## Conclusion

UV represents a paradigm shift in Python package management, offering unprecedented speed and reliability. For the NFL DFS system, this translates to:

- **Faster Development Cycles**: Reduced waiting time for dependency installation
- **Improved CI/CD Performance**: Faster build times in continuous integration
- **Better Reproducibility**: Deterministic lockfiles ensure consistent environments
- **Enhanced Security**: Built-in vulnerability scanning keeps dependencies secure
- **Lower Resource Usage**: Minimal memory and CPU footprint

The adoption of UV aligns with the project's goal of maintaining a high-performance, efficient system while keeping development velocity high and operational overhead low.