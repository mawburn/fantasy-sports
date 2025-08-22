"""Backtesting command."""

import argparse
from dfs.core.logging import get_logger

logger = get_logger("cli.backtest")


def add_parser(subparsers):
    """Add backtest command parser."""
    parser = subparsers.add_parser("backtest", help="Run model backtesting")
    parser.add_argument(
        "--start-date",
        required=True,
        help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick backtest with default settings"
    )
    parser.add_argument(
        "--contest-types",
        nargs="+",
        choices=["cash", "gpp", "satellite"],
        default=["gpp"],
        help="Contest types to test"
    )
    parser.add_argument(
        "--test-correlations",
        action="store_true",
        help="Include correlation strategy testing"
    )
    parser.add_argument(
        "--test-portfolio",
        action="store_true",
        help="Include portfolio strategy testing"
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=20,
        help="Number of entries for portfolio testing"
    )


def run(args):
    """Run backtesting command."""
    logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")
    
    # Import here to avoid circular imports
    from dfs.backtesting.engine import run_backtest_command
    
    run_backtest_command(args)
    
    logger.info("Backtesting completed successfully")