#!/usr/bin/env python3
"""Verify that the NFL DFS system development environment is set up correctly."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 11:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.11+ required")
        return False


def check_imports():
    """Check that all required packages can be imported."""
    packages = [
        "torch",
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "lightgbm",
        "fastapi",
        "sqlalchemy",
        "pydantic",
        "nfl_data_py",
        "pulp",
        "mlflow",
        "loguru",
    ]

    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            all_ok = False

    return all_ok


def check_directories():
    """Check that project directories exist."""
    base_path = Path(__file__).parent.parent
    directories = [
        "src",
        "tests",
        "data",
        "data/database",
        "data/cache",
        "data/models",
        "data/nfl_raw",
        "data/draftkings",
        "models",
        "scripts",
        "docs/architecture",
    ]

    all_ok = True
    for directory in directories:
        dir_path = base_path / directory
        if dir_path.exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            all_ok = False

    return all_ok


def check_config_files():
    """Check that configuration files exist."""
    base_path = Path(__file__).parent.parent
    files = [
        ".env",
        ".env.example",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".pre-commit-config.yaml",
        ".gitignore",
        "README.md",
        "Makefile",
    ]

    all_ok = True
    for file in files:
        file_path = base_path / file
        if file_path.exists():
            print(f"✓ File exists: {file}")
        else:
            print(f"✗ File missing: {file}")
            all_ok = False

    return all_ok


def check_torch_cpu_optimization():
    """Check PyTorch CPU optimization settings."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of CPU threads: {torch.get_num_threads()}")
    print(f"CPU cores available: {torch.get_num_interop_threads()}")

    if torch.get_num_threads() > 1:
        print("✓ PyTorch CPU optimization enabled")
        return True
    else:
        print("✗ PyTorch CPU optimization may not be optimal")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("NFL DFS System - Environment Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Project Directories", check_directories),
        ("Configuration Files", check_config_files),
        ("PyTorch CPU Optimization", check_torch_cpu_optimization),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Review docs/architecture/ for system design")
        print("2. Start implementing data collection modules")
        print("3. Run 'make run' to start the development server")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
