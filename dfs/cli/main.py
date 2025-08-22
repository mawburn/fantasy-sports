"""Main CLI entry point."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from dfs.core.logging import setup_logging
from dfs.core.config import config
from dfs.cli.commands import (
    collect,
    train, 
    predict,
    optimize,
    backtest
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(description="DFS Optimization System")
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add command parsers
    collect.add_parser(subparsers)
    train.add_parser(subparsers)
    predict.add_parser(subparsers)
    optimize.add_parser(subparsers)
    backtest.add_parser(subparsers)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug or config.debug else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Route to appropriate command
    if args.command == "collect":
        collect.run(args)
    elif args.command == "train":
        train.run(args)
    elif args.command == "predict":
        predict.run(args)
    elif args.command == "optimize":
        optimize.run(args)
    elif args.command == "backtest":
        backtest.run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()