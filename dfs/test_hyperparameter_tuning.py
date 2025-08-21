#!/usr/bin/env python3
"""
Test script for hyperparameter tuning functionality.

This script demonstrates the various hyperparameter optimization methods:
1. Learning Rate Finder
2. Batch Size Optimization
3. Joint Hyperparameter Tuning with Optuna
4. A/B Testing for validation

Usage:
    python test_hyperparameter_tuning.py [--position POSITION] [--method METHOD]
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import get_training_data, get_db_connection
from models import create_model, ModelConfig, HyperparameterValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_lr_finder(position: str = 'QB'):
    """Test Learning Rate Finder functionality."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing LR Finder for {position}")
    logger.info(f"{'='*60}")
    
    # Get training data
    X, y, feature_names = get_training_data(position, [2022, 2023], "data/nfl_dfs.db")
    
    if len(X) == 0:
        logger.error(f"No training data available for {position}")
        return
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    config = ModelConfig(position=position, features=feature_names)
    model = create_model(position, config, use_ensemble=False)
    
    # Find optimal learning rate
    logger.info("Running LR range test...")
    optimal_lr = model.find_optimal_lr(X_train, y_train, num_iter=50)
    
    logger.info(f"✅ Optimal learning rate found: {optimal_lr:.2e}")
    logger.info(f"   Default LR was: {0.0001:.2e}")
    logger.info(f"   Improvement factor: {optimal_lr/0.0001:.2f}x")
    
    return optimal_lr


def test_batch_size_optimizer(position: str = 'RB'):
    """Test Batch Size Optimizer functionality."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Batch Size Optimizer for {position}")
    logger.info(f"{'='*60}")
    
    # Get training data
    X, y, feature_names = get_training_data(position, [2022, 2023], "data/nfl_dfs.db")
    
    if len(X) == 0:
        logger.error(f"No training data available for {position}")
        return
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    config = ModelConfig(position=position, features=feature_names)
    model = create_model(position, config, use_ensemble=False)
    
    # Find optimal batch size
    logger.info("Testing different batch sizes...")
    optimal_batch_size = model.optimize_batch_size(
        X_train, y_train, X_val, y_val,
        batch_sizes=[16, 32, 64, 128]
    )
    
    logger.info(f"✅ Optimal batch size found: {optimal_batch_size}")
    logger.info(f"   Default batch size was: 32")
    logger.info(f"   Change factor: {optimal_batch_size/32:.2f}x")
    
    return optimal_batch_size


def test_joint_optimization(position: str = 'WR'):
    """Test joint hyperparameter optimization with Optuna."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Joint Hyperparameter Optimization for {position}")
    logger.info(f"{'='*60}")
    
    # Get training data
    X, y, feature_names = get_training_data(position, [2022, 2023], "data/nfl_dfs.db")
    
    if len(X) == 0:
        logger.error(f"No training data available for {position}")
        return
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    config = ModelConfig(position=position, features=feature_names)
    model = create_model(position, config, use_ensemble=False)
    
    # Run hyperparameter tuning (reduced trials for testing)
    logger.info("Running Optuna optimization (5 trials for testing)...")
    best_params = model.tune_hyperparameters(
        X_train, y_train, X_val, y_val,
        n_trials=5, timeout=300
    )
    
    logger.info(f"✅ Best hyperparameters found:")
    for param, value in best_params.items():
        logger.info(f"   {param}: {value}")
    
    return best_params


def test_ab_validation(position: str = 'TE'):
    """Test A/B validation between baseline and optimized hyperparameters."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing A/B Validation for {position}")
    logger.info(f"{'='*60}")
    
    # Get training data
    X, y, feature_names = get_training_data(position, [2022, 2023], "data/nfl_dfs.db")
    
    if len(X) == 0:
        logger.error(f"No training data available for {position}")
        return
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model config
    config = ModelConfig(position=position, features=feature_names)
    
    # Define baseline and optimized hyperparameters
    baseline_params = {
        'learning_rate': 0.0001,
        'batch_size': 32
    }
    
    # Simulate optimized params (in real scenario, these would come from tuning)
    optimized_params = {
        'learning_rate': 0.0023,
        'batch_size': 64
    }
    
    # Create validator
    from models import create_model
    model_class = create_model(position, config, use_ensemble=False).__class__
    validator = HyperparameterValidator(model_class, config)
    
    # Run A/B test
    logger.info("Running A/B test (2 runs each for speed)...")
    results = validator.ab_test(
        X_train, y_train, X_val, y_val,
        baseline_params, optimized_params,
        num_runs=2
    )
    
    logger.info(f"✅ A/B test complete: {results['decision']}")
    
    return results


def test_cross_validation(position: str = 'QB'):
    """Test cross-validation for hyperparameter robustness."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Cross-Validation for {position}")
    logger.info(f"{'='*60}")
    
    # Get training data
    X, y, feature_names = get_training_data(position, [2022, 2023], "data/nfl_dfs.db")
    
    if len(X) == 0:
        logger.error(f"No training data available for {position}")
        return
    
    logger.info(f"Total samples: {len(X)}")
    
    # Create model config
    config = ModelConfig(position=position, features=feature_names)
    
    # Define hyperparameters to validate
    hyperparameters = {
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    # Create validator
    from models import create_model
    model_class = create_model(position, config, use_ensemble=False).__class__
    validator = HyperparameterValidator(model_class, config)
    
    # Run cross-validation
    logger.info("Running 3-fold cross-validation...")
    cv_results = validator.cross_validate_hyperparameters(
        X, y, hyperparameters, cv_folds=3
    )
    
    logger.info(f"✅ Cross-validation complete:")
    logger.info(f"   Mean R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
    logger.info(f"   Mean MAE: {cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f}")
    
    return cv_results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test hyperparameter tuning functionality")
    parser.add_argument(
        "--position",
        choices=['QB', 'RB', 'WR', 'TE', 'DST'],
        help="Position to test (if not specified, tests all)"
    )
    parser.add_argument(
        "--method",
        choices=['lr', 'batch', 'joint', 'ab', 'cv', 'all'],
        default='all',
        help="Method to test: lr (LR Finder), batch (Batch Size), joint (Optuna), ab (A/B Test), cv (Cross-Validation), all"
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    db_path = Path("data/nfl_dfs.db")
    if not db_path.exists():
        logger.error("Database not found. Please run 'python run.py collect' first.")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("NFL DFS Hyperparameter Tuning Test Suite")
    logger.info("="*60)
    
    # Determine which positions to test
    if args.position:
        positions = [args.position]
    else:
        positions = ['QB', 'RB', 'WR', 'TE']  # Skip DST as it uses CatBoost
    
    # Run tests based on method
    for position in positions:
        try:
            if args.method in ['lr', 'all']:
                test_lr_finder(position)
            
            if args.method in ['batch', 'all']:
                test_batch_size_optimizer(position)
            
            if args.method in ['joint', 'all']:
                test_joint_optimization(position)
            
            if args.method in ['ab', 'all']:
                test_ab_validation(position)
            
            if args.method in ['cv', 'all']:
                test_cross_validation(position)
                
        except Exception as e:
            logger.error(f"Error testing {position}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    logger.info("\n" + "="*60)
    logger.info("✅ All tests completed successfully!")
    logger.info("="*60)
    
    # Print usage examples
    print("\n📚 Usage Examples:")
    print("-" * 40)
    print("# Find optimal learning rate only")
    print("uv run python run.py train --tune-lr")
    print()
    print("# Optimize batch size for memory efficiency")
    print("uv run python run.py train --tune-batch-size")
    print()
    print("# Full hyperparameter optimization (20 trials)")
    print("uv run python run.py train --tune-all --trials 20")
    print()
    print("# Use found hyperparameters for production training")
    print("uv run python run.py train --lr 0.0023 --batch-size 96")
    print()
    print("# Position-specific tuning")
    print("uv run python run.py train --positions QB --tune-all")


if __name__ == "__main__":
    main()