"""Player prediction command."""

import argparse
from dfs.core.logging import get_logger

logger = get_logger("cli.predict")


def add_parser(subparsers):
    """Add predict command parser."""
    parser = subparsers.add_parser("predict", help="Generate player predictions")
    parser.add_argument(
        "--contest-id",
        help="DraftKings contest ID"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file for predictions"
    )
    parser.add_argument(
        "--injury-file",
        help="CSV file with injury statuses (columns: player_name, injury_status)"
    )


def run(args):
    """Run player prediction command."""
    logger.info(f"Generating predictions for contest: {args.contest_id}")
    
    # Import here to avoid circular imports
    from dfs.prediction.pipeline import predict_players_optimized
    
    predictions = predict_players_optimized(
        contest_id=args.contest_id,
        output_file=args.output,
        injury_file=args.injury_file
    )
    
    logger.info(f"Generated predictions for {len(predictions)} players")
    
    if args.output:
        logger.info(f"Predictions saved to {args.output}")