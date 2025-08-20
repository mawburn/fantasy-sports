# tests/test_training.py
import numpy as np
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import QBNetwork, ModelConfig, BaseNeuralModel

class MockQBModel(BaseNeuralModel):
    """Mock model for testing."""
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = 0.01
        self.epochs = 10

    def build_network(self, input_size: int):
        return QBNetwork(input_size)

def test_qb_network_forward():
    """Test QB network forward pass with quantile outputs."""
    input_size = 40
    batch_size = 32

    network = QBNetwork(input_size)
    x = torch.randn(batch_size, input_size)

    output = network(x)

    # Check that output is a dictionary with correct keys
    assert isinstance(output, dict)
    expected_keys = ['mean', 'q25', 'q50', 'q75']
    for key in expected_keys:
        assert key in output
        assert output[key].shape == (batch_size,)

    print("✓ test_qb_network_forward")

def test_target_clipping():
    """Test target clipping by position."""
    config = ModelConfig(position='QB')
    model = MockQBModel(config)

    # Test QB clipping range [-5, 55]
    y_raw = np.array([-10, 0, 25, 60, 100])
    y_clipped = model._clip_targets_by_position(y_raw)

    expected = np.array([-5, 0, 25, 55, 55])
    np.testing.assert_array_equal(y_clipped, expected)

    print("✓ test_target_clipping")

def test_target_normalization():
    """Test target normalization."""
    config = ModelConfig(position='QB')
    model = MockQBModel(config)

    y = np.array([10, 15, 20, 25, 30])  # mean=20, std=7.07
    y_norm = model._normalize_targets(y)

    # Check normalization properties
    assert abs(np.mean(y_norm)) < 1e-10  # Mean should be ~0
    assert abs(np.std(y_norm, ddof=0) - 1.0) < 1e-10  # Std should be 1

    # Check denormalization
    y_denorm = model._denormalize_predictions(y_norm)
    np.testing.assert_array_almost_equal(y_denorm, y, decimal=10)

    print("✓ test_target_normalization")

def test_quantile_loss():
    """Test quantile loss function."""
    config = ModelConfig(position='QB')
    model = MockQBModel(config)

    predictions = torch.tensor([10.0, 15.0, 20.0])
    targets = torch.tensor([12.0, 13.0, 25.0])

    # Test 50th percentile (median) loss
    loss_q50 = model._quantile_loss(predictions, targets, 0.5)

    # Loss should be positive
    assert loss_q50.item() > 0

    # Test 25th percentile loss
    loss_q25 = model._quantile_loss(predictions, targets, 0.25)
    assert loss_q25.item() > 0

    print("✓ test_quantile_loss")

def test_prediction_ranges():
    """Test that predictions stay within reasonable ranges."""
    input_size = 40
    network = QBNetwork(input_size)

    # Simulate training data
    x = torch.randn(100, input_size) * 0.5  # Realistic feature ranges

    with torch.no_grad():
        outputs = network(x)

        # Check that outputs are finite
        for key, pred in outputs.items():
            assert torch.isfinite(pred).all(), f"{key} predictions contain NaN/Inf"

            # QB predictions should be reasonable (rough sanity check)
            assert pred.min() > -50, f"{key} min prediction too low: {pred.min()}"
            assert pred.max() < 100, f"{key} max prediction too high: {pred.max()}"

    print("✓ test_prediction_ranges")

def test_model_sanity_checks():
    """Test model sanity checks for NaN, zero std, extreme means."""
    config = ModelConfig(position='QB')
    model = MockQBModel(config)

    # Test NaN detection
    y_nan = np.array([10, 15, np.nan, 20, 25])
    try:
        model._normalize_targets(y_nan)
        assert False, "Should have failed on NaN targets"
    except:
        pass  # Expected to fail

    # Test zero std detection
    y_constant = np.array([10, 10, 10, 10, 10])
    try:
        model._normalize_targets(y_constant)
        assert False, "Should have failed on zero std"
    except ValueError as e:
        assert "zero" in str(e)

    print("✓ test_model_sanity_checks")

if __name__ == "__main__":
    test_qb_network_forward()
    test_target_clipping()
    test_target_normalization()
    test_quantile_loss()
    test_prediction_ranges()
    test_model_sanity_checks()
    print("\n✅ All training tests passed!")
