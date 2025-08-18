#!/usr/bin/env python
"""Weekly data update script for NFL DFS system.

This script updates the database with the latest NFL data after games are played.
Run this after each week's games (typically Tuesday morning) to get the latest:
- Player stats
- Game results
- Defensive stats
- Vegas closing lines
- Weather data

No need to retrain models after every update - only retrain when:
- Performance degrades significantly (MAE increases >20%)
- Major roster changes occur (trades, injuries)
- Monthly during the season for best results
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import nfl_data_py as nfl

from src.data.collection.defensive_stats_collector import DefensiveStatsCollector
from src.data.collection.vegas_collector import VegasLinesCollector
from src.database.connection import get_db
from src.database.models import Game, PlayerStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_current_nfl_week():
    """Determine the current NFL week based on date."""
    # NFL season typically starts first week of September
    today = datetime.now()

    # Rough calculation - adjust as needed
    if today.month < 9:
        # Previous season playoffs or offseason
        season = today.year - 1
        week = 18  # Last regular season week
    else:
        season = today.year
        # Calculate week number (rough estimate)
        season_start = datetime(today.year, 9, 7)  # First Thursday of September
        weeks_elapsed = (today - season_start).days // 7
        week = min(weeks_elapsed + 1, 18)

    return season, week


def update_current_week_stats(season: int, week: int):
    """Update stats for the current week only."""
    db = next(get_db())

    logger.info(f"Updating stats for Season {season}, Week {week}")

    # 1. Update game results and scores
    logger.info("Updating game results...")
    schedules = nfl.import_schedules([season])
    week_games = schedules[schedules["week"] == week]

    games_updated = 0
    for _, row in week_games.iterrows():
        game = db.query(Game).filter(Game.game_id == row["game_id"]).first()
        if game and game.home_score is None and row.get("home_score") is not None:
            game.home_score = int(row["home_score"])
            game.away_score = int(row["away_score"])
            game.game_finished = True
            games_updated += 1

    db.commit()
    logger.info(f"  Updated {games_updated} game results")

    # 2. Update player stats for the week
    logger.info("Updating player stats...")
    weekly_stats = nfl.import_weekly_data([season])
    week_stats = weekly_stats[weekly_stats["week"] == week]

    stats_added = 0
    for _, row in week_stats.iterrows():
        if row.get("player_id"):
            # Check if stat exists
            existing = (
                db.query(PlayerStats)
                .filter(
                    PlayerStats.player_id == row["player_id"],
                    (
                        PlayerStats.game_id
                        == db.query(Game).filter(Game.game_id == row.get("game_id")).first().id
                        if db.query(Game).filter(Game.game_id == row.get("game_id")).first()
                        else None
                    ),
                )
                .first()
            )

            if not existing and row.get("game_id"):
                game = db.query(Game).filter(Game.game_id == row["game_id"]).first()
                if game:
                    # Create new stat record
                    stat = PlayerStats(
                        player_id=row["player_id"],
                        game_id=game.id,
                        passing_completions=int(row.get("completions", 0) or 0),
                        passing_attempts=int(row.get("attempts", 0) or 0),
                        passing_yards=int(row.get("passing_yards", 0) or 0),
                        passing_tds=int(row.get("passing_tds", 0) or 0),
                        passing_interceptions=int(row.get("interceptions", 0) or 0),
                        rushing_attempts=int(row.get("carries", 0) or 0),
                        rushing_yards=int(row.get("rushing_yards", 0) or 0),
                        rushing_tds=int(row.get("rushing_tds", 0) or 0),
                        receptions=int(row.get("receptions", 0) or 0),
                        targets=int(row.get("targets", 0) or 0),
                        receiving_yards=int(row.get("receiving_yards", 0) or 0),
                        receiving_tds=int(row.get("receiving_tds", 0) or 0),
                        fantasy_points=float(row.get("fantasy_points", 0) or 0),
                        fantasy_points_ppr=float(row.get("fantasy_points_ppr", 0) or 0),
                    )
                    db.add(stat)
                    stats_added += 1

    db.commit()
    logger.info(f"  Added {stats_added} new player stats")

    # 3. Update defensive stats
    logger.info("Updating defensive stats...")
    def_collector = DefensiveStatsCollector(db)

    # Only process the current week
    try:
        pbp_data = nfl.import_pbp_data([season])
        week_pbp = pbp_data[pbp_data["week"] == week]

        if not week_pbp.empty:
            # Save to cache for defensive stats processing
            cache_dir = Path("data/cache/pbp")
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / f"pbp_{season}_week{week}.parquet"
            week_pbp.to_parquet(cache_file)

            # Process defensive stats for this week
            def_records = def_collector._process_week(week_pbp, season, week)
            logger.info(f"  Updated {def_records} defensive stat records")
    except Exception as e:
        logger.exception("  Error updating defensive stats")

    # 4. Update Vegas closing lines
    logger.info("Updating Vegas lines...")
    vegas_collector = VegasLinesCollector(db)

    # Get updated lines (closing lines)
    schedules_with_lines = nfl.import_schedules([season])
    week_lines = schedules_with_lines[schedules_with_lines["week"] == week]

    lines_updated = vegas_collector._process_lines(week_lines, season)
    logger.info(f"  Updated {lines_updated} Vegas lines")

    return {
        "games_updated": games_updated,
        "stats_added": stats_added,
        "defensive_stats_updated": def_records if "def_records" in locals() else 0,
        "vegas_lines_updated": lines_updated,
    }


def check_model_performance():
    """Check if models need retraining based on recent performance."""
    db = next(get_db())

    # Get recent predictions vs actuals
    # This is a placeholder - implement based on your prediction tracking
    logger.info("Checking model performance...")

    # Simple heuristic: retrain if it's been more than 2 weeks
    # In production, track MAE and retrain if it degrades
    last_training_file = Path("models/production/last_training.txt")

    if last_training_file.exists():
        with last_training_file.open() as f:
            last_training = datetime.fromisoformat(f.read().strip())

        days_since_training = (datetime.now() - last_training).days

        if days_since_training > 14:
            logger.warning(
                f"Models last trained {days_since_training} days ago - consider retraining"
            )
            return False
    else:
        logger.warning("No training timestamp found - models may need training")
        return False

    return True


def incremental_model_update():
    """Perform incremental model update without full retraining.

    This uses the new week's data to fine-tune existing models
    rather than retraining from scratch.
    """
    logger.info("Performing incremental model update...")

    # This is a placeholder for incremental learning
    # In practice, you would:
    # 1. Load existing models
    # 2. Create a small training set from recent weeks
    # 3. Fine-tune with a lower learning rate
    # 4. Validate performance hasn't degraded

    logger.info("  Incremental update not yet implemented - use full retraining if needed")


def main():
    parser = argparse.ArgumentParser(description="Update NFL DFS data after weekly games")
    parser.add_argument("--season", type=int, help="NFL season (default: current)")
    parser.add_argument("--week", type=int, help="NFL week (default: current)")
    parser.add_argument(
        "--full-update", action="store_true", help="Update all weeks in current season"
    )
    parser.add_argument(
        "--check-models", action="store_true", help="Check if models need retraining"
    )
    parser.add_argument(
        "--incremental-train", action="store_true", help="Perform incremental model update"
    )

    args = parser.parse_args()

    # Determine season and week
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = get_current_nfl_week()
        logger.info(f"Auto-detected current NFL week: Season {season}, Week {week}")

    # Update data
    if args.full_update:
        logger.info(f"Performing full season update for {season}...")
        total_updates = {"games_updated": 0, "stats_added": 0}

        for w in range(1, week + 1):
            logger.info(f"\nUpdating Week {w}...")
            results = update_current_week_stats(season, w)
            for key in results:
                total_updates[key] = total_updates.get(key, 0) + results[key]

        logger.info(f"\nFull season update complete: {total_updates}")
    else:
        results = update_current_week_stats(season, week)
        logger.info(f"\nWeek {week} update complete: {results}")

    # Check model performance
    if args.check_models:
        if check_model_performance():
            logger.info("Models performing well - no retraining needed")
        else:
            logger.info("Consider retraining models with: make train-models")

    # Incremental training
    if args.incremental_train:
        incremental_model_update()

    logger.info("\n✅ Weekly update complete!")
    logger.info("Next steps:")
    logger.info("  1. Upload current week's DraftKings CSV")
    logger.info("  2. Generate predictions: make predict")
    logger.info("  3. Build optimal lineups: make optimize")


if __name__ == "__main__":
    main()
