"""Lineup optimization command."""

import argparse
from dfs.core.logging import get_logger

logger = get_logger("cli.optimize")


def add_parser(subparsers):
    """Add optimize command parser."""
    parser = subparsers.add_parser("optimize", help="Build optimal lineups")
    parser.add_argument(
        "--contest-id",
        help="DraftKings contest ID"
    )
    parser.add_argument(
        "--strategy",
        choices=["cash", "tournament", "balanced"],
        default="balanced",
        help="Optimization strategy"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of lineups to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="lineups",
        help="Directory to save lineups"
    )
    parser.add_argument(
        "--save-predictions",
        help="Optional: Save player predictions to CSV"
    )
    parser.add_argument(
        "--injury-file",
        help="CSV file with injury statuses (columns: player_name, injury_status)"
    )


def run(args):
    """Run lineup optimization command."""
    logger.info(f"Optimizing lineups for contest: {args.contest_id}")
    logger.info(f"Strategy: {args.strategy}, Count: {args.count}")
    
    # Import here to avoid circular imports
    from dfs.optimization.engine import optimize_lineups
    
    optimize_lineups(
        contest_id=args.contest_id,
        strategy=args.strategy,
        num_lineups=args.count,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        injury_file=args.injury_file
    )
    
    logger.info("Lineup optimization completed successfully")