"""Model training command."""

import argparse
from dfs.core.logging import get_logger

logger = get_logger("cli.train")


def add_parser(subparsers):
    """Add train command parser."""
    parser = subparsers.add_parser("train", help="Train prediction models")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Seasons to use for training"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=['QB', 'RB', 'WR', 'TE', 'DST'],
        help="Positions to train"
    )
    parser.add_argument(
        "--tune-lr",
        action="store_true",
        help="Find optimal learning rate using LR range test"
    )
    parser.add_argument(
        "--tune-batch-size",
        action="store_true",
        help="Find optimal batch size considering memory constraints"
    )
    parser.add_argument(
        "--tune-all",
        action="store_true",
        help="Perform full hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter optimization (default: 20)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate (if not tuning)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size (if not tuning)"
    )


def run(args):
    """Run model training command."""
    logger.info("Starting model training...")
    
    # Import here to avoid circular imports
    from dfs.models.training.trainer import train_models
    
    train_models(
        seasons=args.seasons,
        positions=args.positions,
        tune_lr=args.tune_lr,
        tune_batch_size=args.tune_batch_size,
        tune_all=args.tune_all,
        trials=args.trials,
        override_lr=args.lr,
        override_batch_size=args.batch_size
    )
    
    logger.info("Model training completed successfully")