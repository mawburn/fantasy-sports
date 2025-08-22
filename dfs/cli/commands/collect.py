"""Data collection command."""

import argparse
from dfs.core.logging import get_logger

logger = get_logger("cli.collect")


def add_parser(subparsers):
    """Add collect command parser."""
    parser = subparsers.add_parser("collect", help="Collect NFL data")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Seasons to collect (e.g., 2022 2023)"
    )
    parser.add_argument(
        "--csv",
        help="Path to DraftKings CSV file for salary integration"
    )
    parser.add_argument(
        "--injuries",
        action="store_true",
        help="Collect injury data from NFL library"
    )
    parser.add_argument(
        "--odds",
        action="store_true",
        help="Collect betting odds data"
    )
    parser.add_argument(
        "--weather",
        action="store_true", 
        help="Collect weather data"
    )


def run(args):
    """Run data collection command."""
    logger.info("Starting data collection...")
    
    # Import here to avoid circular imports
    from dfs.data.collectors.nfl_data import collect_nfl_data
    from dfs.data.collectors.draftkings import load_draftkings_csv
    from dfs.data.collectors.injuries import collect_injury_data
    from dfs.data.collectors.betting import collect_odds_data
    
    seasons = args.seasons or [2023, 2024]
    logger.info(f"Collecting data for seasons: {seasons}")
    
    # Collect NFL data
    collect_nfl_data(seasons)
    
    # Collect DraftKings salaries if provided
    if args.csv:
        logger.info(f"Integrating DraftKings salaries from {args.csv}")
        load_draftkings_csv(args.csv)
    
    # Collect injury data if requested
    if args.injuries:
        logger.info("Collecting injury data...")
        collect_injury_data(seasons)
    
    # Collect odds data if requested  
    if args.odds:
        logger.info("Collecting betting odds...")
        collect_odds_data()
    
    logger.info("Data collection completed successfully")