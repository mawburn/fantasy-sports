#!/usr/bin/env python3
"""Test script to validate RB model improvements."""

import numpy as np
import sqlite3
from models import RBNeuralModel, ModelConfig
from data import get_rb_specific_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rb_features():
    """Test RB feature extraction."""
    print("Testing RB-specific features...")

    # Connect to database
    conn = sqlite3.connect('data/nfl_dfs.db')

    # Test with a known RB (using placeholder values)
    player_id = 1  # Placeholder
    player_name = "Test Player"
    team_abbr = "SF"
    opponent_abbr = "SEA"
    season = 2024
    week = 17

    features = get_rb_specific_features(
        player_id, player_name, team_abbr, opponent_abbr,
        season, week, conn
    )

    print(f"Extracted {len(features)} RB-specific features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    conn.close()
    return len(features) > 0

def test_rb_network():
    """Test RB network architecture."""
    print("\nTesting RB network architecture...")

    config = ModelConfig(position='RB')
    model = RBNeuralModel(config)

    # Build network with test input size
    input_size = 60  # Approximate feature count
    network = model.build_network(input_size)

    print(f"Network built successfully with input size: {input_size}")
    print(f"Network parameters: {sum(p.numel() for p in network.parameters())}")

    # Test forward pass
    test_input = np.random.random((5, input_size))
    import torch
    with torch.no_grad():
        output = network(torch.FloatTensor(test_input))
        print(f"Test output shape: {output.shape}")
        print(f"Test output range: {output.min():.3f} - {output.max():.3f}")
        print(f"Test output mean: {output.mean():.3f}")

    return True

def validate_rb_improvements():
    """Validate that RB improvements are working."""
    print("\n" + "="*50)
    print("RB MODEL OPTIMIZATION VALIDATION")
    print("="*50)

    # Test 1: Feature extraction
    try:
        features_ok = test_rb_features()
        print(f"✅ Feature extraction: {'PASS' if features_ok else 'FAIL'}")
    except Exception as e:
        print(f"❌ Feature extraction: FAIL - {e}")
        features_ok = False

    # Test 2: Network architecture
    try:
        network_ok = test_rb_network()
        print(f"✅ Network architecture: {'PASS' if network_ok else 'FAIL'}")
    except Exception as e:
        print(f"❌ Network architecture: FAIL - {e}")
        network_ok = False

    # Overall result
    overall_ok = features_ok and network_ok
    print("\n" + "="*50)
    print(f"OVERALL RESULT: {'✅ PASS' if overall_ok else '❌ FAIL'}")
    print("="*50)

    if overall_ok:
        print("\nRB model optimizations are ready for training!")
        print("Next steps:")
        print("1. Run: uv run python run.py train --position RB")
        print("2. Monitor R² score (target: > 0.35)")
        print("3. Check prediction range (target: 5-35 points)")
        print("4. Validate prediction variance (target: > 3.0)")
    else:
        print("\nRB model optimizations need fixes before training.")

    return overall_ok

if __name__ == "__main__":
    validate_rb_improvements()
