"""CLI commands for collecting NFL data."""

import logging
import sys
from pathlib import Path

import typer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import settings
from src.data.collection.nfl_collector import NFLDataCollector
from src.database.init_db import create_database

app = typer.Typer(help="Data collection commands for NFL data")


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


@app.command()
def init_db():
    """Initialize the database."""
    typer.echo("Initializing database...")
    try:
        create_database()
        typer.echo("✅ Database initialized successfully!")
    except Exception as e:
        typer.echo(f"❌ Database initialization failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_teams(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """Collect NFL team data."""
    setup_logging()
    seasons_list = seasons if seasons else None
    typer.echo(f"Collecting NFL team data for seasons: {seasons_list or 'all available'}")
    try:
        collector = NFLDataCollector()
        teams_added = collector.collect_teams(seasons=seasons_list)
        typer.echo(f"✅ Teams collection complete! Added {teams_added} new teams.")
    except Exception as e:
        typer.echo(f"❌ Teams collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_players(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """Collect NFL player data."""
    setup_logging()
    seasons_list = seasons if seasons else None
    typer.echo(f"Collecting NFL player data for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        players_added = collector.collect_players(seasons_list)
        typer.echo(f"✅ Players collection complete! Added {players_added} new players.")
    except Exception as e:
        typer.echo(f"❌ Players collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_schedules(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """Collect NFL schedule data."""
    setup_logging()
    seasons_list = seasons if seasons else None
    typer.echo(f"Collecting NFL schedule data for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        games_added = collector.collect_schedules(seasons_list)
        typer.echo(f"✅ Schedules collection complete! Added {games_added} new games.")
    except Exception as e:
        typer.echo(f"❌ Schedules collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_stats(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """Collect NFL player statistics."""
    setup_logging()
    seasons_list = seasons if seasons else None
    typer.echo(f"Collecting NFL player stats for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        stats_added = collector.collect_player_stats(seasons_list)
        typer.echo(f"✅ Stats collection complete! Added {stats_added} new stat records.")
    except Exception as e:
        typer.echo(f"❌ Stats collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_pbp(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
    weeks: list[int] = typer.Option([], "--week", "-w", help="Weeks to collect (e.g., -w 1 -w 2)"),
):
    """Collect NFL play-by-play data."""
    setup_logging()
    seasons_list = seasons if seasons else None
    weeks_list = weeks if weeks else None
    typer.echo(
        f"Collecting NFL play-by-play data for seasons: {seasons_list or 'current season'}, weeks: {weeks_list or 'all weeks'}..."
    )
    try:
        collector = NFLDataCollector()
        plays_added = collector.collect_play_by_play(seasons_list, weeks_list)
        typer.echo(f"✅ Play-by-play collection complete! Added {plays_added} new plays.")
    except Exception as e:
        typer.echo(f"❌ Play-by-play collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_dk(
    file: str = typer.Option(None, "--file", "-f", help="Path to DraftKings salary CSV file"),
    directory: str = typer.Option(
        None, "--directory", "-d", help="Directory containing DraftKings CSV files"
    ),
    contest_name: str = typer.Option(
        None, "--contest-name", "-c", help="Contest name (derived from filename if not provided)"
    ),
):
    """Process DraftKings salary data from CSV files."""
    setup_logging()

    if not file and not directory:
        typer.echo("❌ Must specify either --file or --directory")
        raise typer.Exit(1) from None

    if file and directory:
        typer.echo("❌ Cannot specify both --file and --directory")
        raise typer.Exit(1) from None

    try:
        from src.data.collection.dk_collector import DraftKingsCollector

        collector = DraftKingsCollector()

        if file:
            # Process single file
            file_path = Path(file)
            typer.echo(f"Processing DraftKings file: {file_path}")
            results = collector.process_salary_file(file_path, contest_name)
            typer.echo("✅ File processed successfully!")
            typer.echo(f"  Contests: {results.get('contests', 0)}")
            typer.echo(f"  Salaries: {results.get('salaries', 0)}")
            typer.echo(f"  Unmatched players: {results.get('unmatched_players', 0)}")

        else:
            # Process directory
            dir_path = Path(directory)
            typer.echo(f"Processing DraftKings files in: {dir_path}")
            results = collector.bulk_process_files(dir_path)
            typer.echo("✅ Directory processed successfully!")
            typer.echo(f"  Files processed: {results.get('files_processed', 0)}")
            typer.echo(f"  Total contests: {results.get('total_contests', 0)}")
            typer.echo(f"  Total salaries: {results.get('total_salaries', 0)}")
            if results.get("errors"):
                typer.echo(f"  Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    typer.echo(f"    - {error}")

    except Exception as e:
        typer.echo(f"❌ DraftKings collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_all(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """Collect all NFL data (teams, players, schedules, stats)."""
    setup_logging()
    seasons_list = seasons if seasons else None
    typer.echo(f"Starting full data collection for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        results = collector.collect_all_data(seasons_list)

        typer.echo("✅ Full data collection complete!")
        typer.echo("Results:")
        for data_type, count in results.items():
            typer.echo(f"  - {data_type}: {count} new records")

    except Exception as e:
        typer.echo(f"❌ Data collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def status():
    """Show database status and record counts."""
    try:
        from src.database.connection import SessionLocal
        from src.database.models import (
            DraftKingsContest,
            DraftKingsSalary,
            Game,
            PlayByPlay,
            Player,
            PlayerStats,
            Team,
        )

        session = SessionLocal()
        try:
            teams_count = session.query(Team).count()
            players_count = session.query(Player).count()
            games_count = session.query(Game).count()
            stats_count = session.query(PlayerStats).count()
            pbp_count = session.query(PlayByPlay).count()
            contests_count = session.query(DraftKingsContest).count()
            salaries_count = session.query(DraftKingsSalary).count()

            typer.echo("📊 Database Status:")
            typer.echo(f"  Teams: {teams_count}")
            typer.echo(f"  Players: {players_count}")
            typer.echo(f"  Games: {games_count}")
            typer.echo(f"  Player Stats: {stats_count}")
            typer.echo(f"  Play-by-Play: {pbp_count}")
            typer.echo(f"  DK Contests: {contests_count}")
            typer.echo(f"  DK Salaries: {salaries_count}")
        finally:
            session.close()

    except Exception as e:
        typer.echo(f"❌ Failed to get database status: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
