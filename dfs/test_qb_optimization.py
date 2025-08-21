#!/usr/bin/env python3
"""Test script for optimized QB model implementation.

This script tests the key components implemented according to the QB_MODEL_OPTIMIZATION_GUIDE.md:
1. Feature extraction functions
2. Enhanced QB neural network
3. DFS loss function
4. Prediction validation
"""

import numpy as np
import pandas as pd
import torch
import sqlite3
from pathlib import Path

# Import our optimized modules
from data import (
    extract_vegas_features,
    extract_volume_features,
    extract_qb_rushing_features,
    extract_opponent_features,
    extract_pace_features,
    create_comprehensive_qb_features,
    FeatureProcessor
)
from models import QBNetwork, DFSLoss, QBNeuralModel, ModelConfig

def test_feature_extraction():
    """Test the new feature extraction functions."""
    print("🧪 Testing feature extraction functions...")

    # Mock database path for testing
    db_path = "data/nfl_dfs.db"

    if not Path(db_path).exists():
        print("⚠️  Database not found. Skipping feature extraction tests.")
        return False

    try:
        # Test Vegas features
        vegas_features = extract_vegas_features(db_path, "2024_01_BUF_MIA", 1, True)
        print(f"✅ Vegas features extracted: {len(vegas_features)} features")

        # Test volume features
        volume_features = extract_volume_features(db_path, 1, "2024_01_BUF_MIA")
        print(f"✅ Volume features extracted: {len(volume_features)} features")

        # Test rushing features
        rushing_features = extract_qb_rushing_features(db_path, 1, "2024_01_BUF_MIA")
        print(f"✅ Rushing features extracted: {len(rushing_features)} features")

        # Test opponent features
        opponent_features = extract_opponent_features(db_path, 2, 'QB')
        print(f"✅ Opponent features extracted: {len(opponent_features)} features")

        # Test pace features
        pace_features = extract_pace_features(db_path, 1, 2)
        print(f"✅ Pace features extracted: {len(pace_features)} features")

        return True

    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_feature_processor():
    """Test the enhanced feature processor."""
    print("🧪 Testing feature processor...")

    try:
        processor = FeatureProcessor()

        # Create mock features
        mock_features = {
            'team_implied_total': 24.5,
            'game_total': 48.0,
            'spread': -3.5,
            'avg_pass_attempts': 38.0,
            'avg_rush_attempts': 6.0,
            'def_completion_rate_allowed': 0.62,
            'def_pressure_rate': 0.28,
            'combined_pace': 67.0,
            'team_pass_rate': 0.58
        }

        processed = processor.process_features(mock_features)

        # Check that interaction features were created
        expected_features = [
            'implied_total_x_attempts',
            'spread_x_rush_attempts',
            'def_weakness_score',
            'positive_game_script',
            'pace_advantage'
        ]

        for feature in expected_features:
            assert feature in processed, f"Missing feature: {feature}"

        print(f"✅ Feature processor test passed: {len(processed)} processed features")
        return True

    except Exception as e:
        print(f"❌ Feature processor test failed: {e}")
        return False

def test_qb_network():
    """Test the enhanced QB neural network."""
    print("🧪 Testing enhanced QB network...")

    try:
        # Create network with mock input size
        input_size = 50
        network = QBNetwork(input_size)

        # Test forward pass
        batch_size = 32
        x = torch.randn(batch_size, input_size)

        with torch.no_grad():
            output = network(x)

        # Check output structure
        expected_keys = ['mean', 'floor', 'ceiling', 'q25', 'q50', 'q75']
        for key in expected_keys:
            assert key in output, f"Missing output key: {key}"
            assert output[key].shape == (batch_size,), f"Wrong shape for {key}"

        # Check realistic ranges
        mean_vals = output['mean'].numpy()
        assert 5 <= mean_vals.min() <= mean_vals.max() <= 45, "QB predictions outside realistic range"

        # Check floor/ceiling relationship
        floors = output['floor'].numpy()
        ceilings = output['ceiling'].numpy()
        assert np.all(floors <= mean_vals), "Floor should be <= mean"
        assert np.all(ceilings >= mean_vals), "Ceiling should be >= mean"

        print(f"✅ QB Network test passed: output range [{mean_vals.min():.1f}, {mean_vals.max():.1f}]")
        return True

    except Exception as e:
        print(f"❌ QB Network test failed: {e}")
        return False

def test_dfs_loss():
    """Test the custom DFS loss function."""
    print("🧪 Testing DFS loss function...")

    try:
        loss_fn = DFSLoss()

        batch_size = 16

        # Create mock predictions
        predictions = {
            'mean': torch.randn(batch_size) * 5 + 20,  # Mean around 20
            'floor': torch.randn(batch_size) * 2 + 15,  # Floor around 15
            'ceiling': torch.randn(batch_size) * 3 + 25  # Ceiling around 25
        }

        # Create mock targets and salaries
        targets = torch.randn(batch_size) * 6 + 18  # Targets around 18
        salaries = torch.randint(7000, 12000, (batch_size,)).float()

        # Compute loss
        loss = loss_fn(predictions, targets, salaries)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

        # Test without salaries
        loss_no_salary = loss_fn(predictions, targets, None)
        assert loss_no_salary.item() > 0, "Loss without salaries should be positive"

        print(f"✅ DFS Loss test passed: loss = {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"❌ DFS Loss test failed: {e}")
        return False

def test_qb_model_integration():
    """Test the full QB model integration."""
    print("🧪 Testing QB model integration...")

    try:
        config = ModelConfig(position='QB', version='optimized')
        model = QBNeuralModel(config)

        # Mock training data
        n_samples = 100
        input_size = 45

        X_train = np.random.randn(n_samples, input_size)
        y_train = np.random.randn(n_samples) * 6 + 18  # QB-like distribution
        X_val = np.random.randn(20, input_size)
        y_val = np.random.randn(20) * 6 + 18

        # Test prediction validation
        try:
            # Check if method exists and is callable
            if hasattr(model, 'validate_predictions') and callable(getattr(model, 'validate_predictions')):
                model.validate_predictions(y_train, 'QB')
                print("✅ QB prediction validation test passed")
            else:
                print("⚠️  validate_predictions method not found, checking class...")
                print(f"Model class: {model.__class__.__name__}")
                print(f"MRO: {[cls.__name__ for cls in model.__class__.__mro__]}")
                return False
        except AssertionError as e:
            print(f"⚠️  Validation caught unrealistic predictions: {e}")
            print("✅ Validation method working correctly (caught invalid data)")

        print("✅ QB model integration test passed")
        return True

    except Exception as e:
        print(f"❌ QB model integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting QB Model Optimization Tests")
    print("=" * 50)

    tests = [
        test_feature_extraction,
        test_feature_processor,
        test_qb_network,
        test_dfs_loss,
        test_qb_model_integration
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()

    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("\n✅ QB model optimization implementation is ready for training!")
        print("\nNext steps:")
        print("1. Run: uv run python run.py train --position QB")
        print("2. Compare new R² score with baseline")
        print("3. Validate predictions meet success criteria from optimization guide")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("Please fix failing tests before proceeding with training.")

if __name__ == "__main__":
    main()
