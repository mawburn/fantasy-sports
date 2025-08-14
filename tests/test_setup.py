"""Test file to verify development environment setup."""

import pytest


def test_import_torch():
    """Test that PyTorch can be imported."""
    import torch

    assert torch.__version__
    # Verify CPU optimization settings
    assert torch.get_num_threads() > 0


def test_import_pandas():
    """Test that pandas can be imported."""
    import pandas as pd

    assert pd.__version__


def test_import_fastapi():
    """Test that FastAPI can be imported."""
    from fastapi import FastAPI

    app = FastAPI()
    assert app is not None


def test_project_config():
    """Test that project configuration loads."""
    from src.config import settings

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert settings.enable_cpu_optimization is True


def test_data_directories_exist():
    """Test that data directories were created."""
    from pathlib import Path

    data_dir = Path("data")
    assert data_dir.exists()
    assert (data_dir / "database").exists()
    assert (data_dir / "cache").exists()
    assert (data_dir / "models").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
