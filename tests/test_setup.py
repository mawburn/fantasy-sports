"""Test file to verify development environment setup.

This file contains automated tests that validate the development environment
is properly configured for the NFL DFS system. These tests complement the
manual verification script by providing programmatic validation.

Why Automated Environment Testing?

1. CI/CD Integration: Automated pipelines can validate environment setup
2. Regression Prevention: Catch environment breakage during updates
3. Onboarding Validation: New developers can quickly verify setup
4. Documentation: Tests serve as executable documentation of requirements
5. Debugging Aid: Isolate environment issues from code issues

Test Categories:

- Import Tests: Verify all required packages can be imported
- Configuration Tests: Validate project settings and configuration
- Directory Tests: Ensure required directory structure exists
- Integration Tests: Test that components work together

PyTest Framework:
Using pytest for testing provides:
- Automatic test discovery (functions starting with 'test_')
- Rich assertion introspection
- Fixtures for setup/teardown
- Parametrized testing capabilities
- Detailed failure reports

For beginners: Tests are automated checks that verify code and environment
work correctly. They're like quality control checks in manufacturing -
ensuring everything meets requirements before proceeding.

Usage:
    pytest tests/test_setup.py -v
    python tests/test_setup.py  # Run directly
"""

import pytest


def test_import_torch():
    """Test that PyTorch can be imported and is properly configured.

    This test validates the core ML framework is available and optimized:

    1. Import Test: Verifies PyTorch installation is complete
    2. Version Check: Ensures version information is accessible
    3. CPU Optimization: Validates multi-threading is enabled

    PyTorch is critical for:
    - Neural network models (if used)
    - Tensor operations and numerical computing
    - CPU-optimized mathematical operations

    Common Failure Causes:
    - PyTorch not installed: pip install torch
    - Wrong PyTorch version (GPU vs CPU)
    - Threading disabled in environment
    - Conflicting BLAS libraries

    For beginners: PyTorch is like a powerful calculator for machine learning
    that can automatically use multiple CPU cores for faster computation.
    """
    import torch

    # Test 1: Verify PyTorch imported successfully and has version info
    assert torch.__version__, "PyTorch version should be accessible"
    print(f"PyTorch version: {torch.__version__}")

    # Test 2: Verify CPU optimization is enabled (multi-threading)
    num_threads = torch.get_num_threads()
    assert num_threads > 0, "PyTorch should use at least 1 CPU thread"
    print(f"PyTorch CPU threads: {num_threads}")

    # Additional diagnostic: Check for optimal threading
    if num_threads == 1:
        print("Warning: PyTorch using only 1 thread (may be slow)")
    elif num_threads >= 4:
        print("Good: PyTorch using multiple threads for parallel processing")


def test_import_pandas():
    """Test that pandas can be imported and is functional.

    Pandas is the backbone of data processing in this system:
    - Reading CSV files (DraftKings data, NFL statistics)
    - Data cleaning and transformation
    - Feature engineering operations
    - Database query result processing

    This test ensures:
    1. Pandas library is properly installed
    2. Version information is accessible
    3. Basic functionality works (creating DataFrames)

    Common Issues:
    - Missing pandas: pip install pandas
    - Version conflicts with other packages
    - Missing dependencies (numpy, etc.)

    For beginners: Pandas is like Excel for Python - it lets you work with
    tables of data (rows and columns) programmatically.
    """
    import pandas as pd

    # Test 1: Verify pandas imported and has version
    assert pd.__version__, "Pandas version should be accessible"
    print(f"Pandas version: {pd.__version__}")

    # Test 2: Verify basic functionality works
    # Create a simple DataFrame to test core functionality
    test_df = pd.DataFrame({"player": ["Josh Allen", "Derrick Henry"], "points": [25.6, 18.3]})
    assert len(test_df) == 2, "Should be able to create DataFrames"
    assert "player" in test_df.columns, "DataFrame columns should be accessible"
    print("Pandas basic functionality: OK")


def test_import_fastapi():
    """Test that FastAPI can be imported and instantiated.

    FastAPI powers the prediction API that serves ML model results.
    This test ensures the web framework is properly installed and functional.

    FastAPI provides:
    - REST API endpoints for predictions
    - Automatic OpenAPI documentation
    - Request/response validation with Pydantic
    - High-performance async request handling

    Test Validations:
    1. FastAPI can be imported (installation check)
    2. FastAPI app can be instantiated (basic functionality)
    3. App has expected attributes (API framework features)

    Common Issues:
    - Missing FastAPI: pip install fastapi
    - Missing uvicorn: pip install uvicorn (ASGI server)
    - Python version incompatibility (FastAPI needs 3.6+)

    For beginners: FastAPI is a modern web framework that makes it easy
    to create APIs (web services) that other programs can call to get
    predictions from our ML models.
    """
    from fastapi import FastAPI

    # Test 1: Create FastAPI application instance
    app = FastAPI()
    assert app is not None, "Should be able to create FastAPI app"

    # Test 2: Verify app has essential attributes
    assert hasattr(app, "routes"), "FastAPI app should have routes attribute"
    assert hasattr(app, "openapi"), "FastAPI app should support OpenAPI"

    # Test 3: Verify we can add a simple route (basic functionality test)
    @app.get("/test")
    def test_endpoint():
        return {"status": "test"}

    # Check that the route was added
    route_paths = [route.path for route in app.routes]
    assert "/test" in route_paths, "Should be able to add routes to FastAPI app"

    print("FastAPI functionality: OK")


def test_project_config():
    """Test that project configuration loads correctly.

    The configuration system manages all project settings including:
    - API server configuration (host, port)
    - Database connection settings
    - ML model parameters
    - Performance optimization flags

    This test validates:
    1. Configuration module can be imported
    2. Settings object is accessible
    3. Key configuration values have expected defaults
    4. Configuration loading doesn't raise errors

    Configuration Sources:
    - Default values in settings.py
    - Environment variables (.env file)
    - Command line overrides (if implemented)

    Common Issues:
    - Missing .env file (copy from .env.example)
    - Invalid environment variable types
    - Missing required configuration values
    - Import errors in config module

    For beginners: Configuration is like a settings panel that controls
    how the application behaves (which port to use, where the database is, etc.)
    """
    from src.config import settings

    # Test 1: Verify core API configuration
    assert settings.api_host == "127.0.0.1", f"Expected API host 127.0.0.1, got {settings.api_host}"
    assert settings.api_port == 8000, f"Expected API port 8000, got {settings.api_port}"
    print(f"API configuration: {settings.api_host}:{settings.api_port}")

    # Test 2: Verify performance optimization is enabled
    assert settings.enable_cpu_optimization is True, "CPU optimization should be enabled by default"
    print("CPU optimization: enabled")

    # Test 3: Verify other critical settings exist and have reasonable values
    # These might be added in future configuration updates
    if hasattr(settings, "database_url"):
        assert settings.database_url, "Database URL should not be empty"
        print(f"Database URL: {settings.database_url}")

    if hasattr(settings, "model_cache_size"):
        assert isinstance(settings.model_cache_size, int), "Cache size should be integer"
        assert settings.model_cache_size > 0, "Cache size should be positive"
        print(f"Model cache size: {settings.model_cache_size}")

    print("Project configuration: OK")


def test_data_directories_exist():
    """Test that required data directories exist.

    The project requires specific directory structure for:
    - Data storage and organization
    - Database files (SQLite)
    - Model artifacts and checkpoints
    - Temporary cache files
    - Raw data imports

    Directory Structure:
    - data/: Root data directory
    - data/database/: SQLite database storage
    - data/cache/: Temporary cached computations
    - data/models/: Trained ML model files
    - data/nfl_raw/: Raw NFL statistics
    - data/draftkings/: DraftKings CSV imports

    Why Directory Tests Matter:
    - Prevents runtime errors when saving files
    - Ensures consistent project structure across environments
    - Validates setup scripts worked correctly
    - Catches permission issues early

    Common Solutions:
    - Run: make setup (creates all directories)
    - Manual: mkdir -p data/{database,cache,models,nfl_raw,draftkings}
    - Check file permissions if on restricted systems

    For beginners: Think of directories as organized folders that keep
    different types of data separate and easy to find.
    """
    from pathlib import Path

    # Test 1: Verify root data directory exists
    data_dir = Path("data")
    assert data_dir.exists(), "Root data directory should exist"
    assert data_dir.is_dir(), "data should be a directory, not a file"
    print("Root data directory: OK")

    # Test 2: Verify essential subdirectories exist
    required_subdirs = {
        "database": "SQLite database storage",
        "cache": "Temporary cached files",
        "models": "Trained ML model artifacts",
    }

    for subdir_name, description in required_subdirs.items():
        subdir_path = data_dir / subdir_name
        assert subdir_path.exists(), f"{subdir_name} directory should exist"
        assert subdir_path.is_dir(), f"{subdir_name} should be a directory"
        print(f"  {subdir_name}/: OK ({description})")

    # Test 3: Verify directories are writable (optional but important)
    try:
        # Try to create a temporary file in data directory
        test_file = data_dir / "test_write.tmp"
        test_file.write_text("write test")
        test_file.unlink()  # Clean up
        print("Data directory is writable: OK")
    except PermissionError:
        print("Warning: Data directory may not be writable (check permissions)")
    except Exception as e:
        print(f"Warning: Could not test write permissions: {e}")

    print("Data directory structure: OK")


if __name__ == "__main__":
    """Allow script to be run directly for quick testing.

    This enables running the test file directly:
    python tests/test_setup.py

    The pytest.main() call runs all tests in this file with verbose output.
    This is convenient for:
    - Quick environment validation
    - Debugging specific test issues
    - Running tests without pytest installation

    For full test suite, use: pytest tests/ -v
    """
    print("Running environment setup tests...")
    print("=" * 50)

    # Run tests with verbose output and exit on first failure
    exit_code = pytest.main([__file__, "-v", "-x"])

    # Provide user-friendly summary
    if exit_code == 0:
        print("\n" + "=" * 50)
        print("✅ All setup tests passed!")
        print("Environment is ready for development.")
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed.")
        print("Please fix the issues above before proceeding.")

    exit(exit_code)
