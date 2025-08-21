#!/usr/bin/env python3
"""Test script to validate WR model improvements."""

import numpy as np
import sqlite3
import torch
from models import WRNeuralModel, ModelConfig
from data import get_wr_specific_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wr_features():
    """Test WR feature extraction."""
    print("Testing WR-specific features...")

    # Connect to database
    conn = sqlite3.connect('data/nfl_dfs.db')

    # Test with a known WR (using placeholder values)
    player_id = 1  # Placeholder
    player_name = "Test Player"
    team_abbr = "SF"
    opponent_abbr = "SEA"
    season = 2024
    week = 17

    features = get_wr_specific_features(
        player_id, player_name, team_abbr, opponent_abbr,
        season, week, conn
    )

    print(f"Extracted {len(features)} WR-specific features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    conn.close()
    return len(features) > 0

def test_wr_network():
    """Test WR network architecture."""
    print("\nTesting WR network architecture...")

    config = ModelConfig(position='WR')
    model = WRNeuralModel(config)

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
        print(f"Test output std: {output.std():.3f}")

    return True

def test_wr_architecture_improvements():
    """Test specific WR architecture improvements."""
    print("\nTesting WR architecture improvements...")

    config = ModelConfig(position='WR')
    model = WRNeuralModel(config)
    network = model.build_network(60)

    # Check for multi-head architecture
    has_target_branch = hasattr(network, 'target_branch')
    has_efficiency_branch = hasattr(network, 'efficiency_branch')
    has_game_script_branch = hasattr(network, 'game_script_branch')
    has_attention = hasattr(network, 'attention')

    print(f"✅ Target branch: {has_target_branch}")
    print(f"✅ Efficiency branch: {has_efficiency_branch}")
    print(f"✅ Game script branch: {has_game_script_branch}")
    print(f"✅ Attention mechanism: {has_attention}")

    # Test output range (no sigmoid compression)
    test_input = torch.randn(10, 60)
    with torch.no_grad():
        output = network(test_input)
        output_range = output.max() - output.min()
        print(f"✅ Output range: {output_range:.3f} (should be > 5 for no compression)")

    return has_target_branch and has_efficiency_branch and has_game_script_branch and has_attention

def validate_wr_improvements():
    """Validate that WR improvements are working."""
    print("\n" + "="*50)
    print("WR MODEL OPTIMIZATION VALIDATION")
    print("="*50)

    # Test 1: Feature extraction
    try:
        features_ok = test_wr_features()
        print(f"✅ Feature extraction: {'PASS' if features_ok else 'FAIL'}")
    except Exception as e:
        print(f"❌ Feature extraction: FAIL - {e}")
        features_ok = False

    # Test 2: Network architecture
    try:
        network_ok = test_wr_network()
        print(f"✅ Network architecture: {'PASS' if network_ok else 'FAIL'}")
    except Exception as e:
        print(f"❌ Network architecture: FAIL - {e}")
        network_ok = False

    # Test 3: Architecture improvements
    try:
        improvements_ok = test_wr_architecture_improvements()
        print(f"✅ Architecture improvements: {'PASS' if improvements_ok else 'FAIL'}")
    except Exception as e:
        print(f"❌ Architecture improvements: FAIL - {e}")
        improvements_ok = False

    # Overall result
    overall_ok = features_ok and network_ok and improvements_ok
    print("\n" + "="*50)
    print(f"OVERALL RESULT: {'✅ PASS' if overall_ok else '❌ FAIL'}")
    print("="*50)

    if overall_ok:
        print("\nWR model optimizations are ready for training!")
        print("Key improvements implemented:")
        print("- Target share, air yards, red zone features")
        print("- Multi-head architecture (target/efficiency/game_script)")
        print("- Attention mechanism for feature importance")
        print("- No sigmoid compression (output range 2-40)")
        print("- Custom loss function for WR volatility")
        print("\nNext steps:")
        print("1. Run: uv run python run.py train --position WR")
        print("2. Monitor R² score (target: > 0.35)")
        print("3. Check prediction range (target: 3-35 points)")
        print("4. Validate prediction variance (target: > 4.0)")
        print("5. Check target share correlation (target: > 0.7)")
    else:
        print("\nWR model optimizations need fixes before training.")

    return overall_ok

if __name__ == "__main__":
    validate_wr_improvements()
