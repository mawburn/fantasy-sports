"""CLI script for collecting NFL data."""

import logging
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import settings
from src.data.collection.nfl_collector import NFLDataCollector
from src.database.init_db import create_database


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(settings.log_file, mode="a")],
    )


@click.group()
def cli():
    """NFL DFS Data Collection CLI."""
    setup_logging()

    # Ensure log directory exists
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)


@cli.command()
def init_db():
    """Initialize the database."""
    click.echo("Initializing database...")
    try:
        create_database()
        click.echo("✅ Database initialized successfully!")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
def collect_teams(seasons: list[int] | None):
    """Collect NFL team data."""
    seasons_list = list(seasons) if seasons else None
    click.echo(f"Collecting NFL team data for seasons: {seasons_list or 'all available'}")
    try:
        collector = NFLDataCollector()
        teams_added = collector.collect_teams(seasons=seasons_list)
        click.echo(f"✅ Teams collection complete! Added {teams_added} new teams.")
    except Exception as e:
        click.echo(f"❌ Teams collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
def collect_players(seasons: list[int] | None):
    """Collect NFL player data."""
    seasons_list = list(seasons) if seasons else None
    click.echo(f"Collecting NFL player data for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        players_added = collector.collect_players(seasons_list)
        click.echo(f"✅ Players collection complete! Added {players_added} new players.")
    except Exception as e:
        click.echo(f"❌ Players collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
def collect_schedules(seasons: list[int] | None):
    """Collect NFL schedule data."""
    seasons_list = list(seasons) if seasons else None
    click.echo(f"Collecting NFL schedule data for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        games_added = collector.collect_schedules(seasons_list)
        click.echo(f"✅ Schedules collection complete! Added {games_added} new games.")
    except Exception as e:
        click.echo(f"❌ Schedules collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
def collect_stats(seasons: list[int] | None):
    """Collect NFL player statistics."""
    seasons_list = list(seasons) if seasons else None
    click.echo(f"Collecting NFL player stats for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        stats_added = collector.collect_player_stats(seasons_list)
        click.echo(f"✅ Stats collection complete! Added {stats_added} new stat records.")
    except Exception as e:
        click.echo(f"❌ Stats collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
@click.option("--weeks", "-w", multiple=True, type=int, help="Weeks to collect (e.g., -w 1 -w 2)")
def collect_pbp(seasons: list[int] | None, weeks: list[int] | None):
    """Collect NFL play-by-play data."""
    seasons_list = list(seasons) if seasons else None
    weeks_list = list(weeks) if weeks else None
    click.echo(
        f"Collecting NFL play-by-play data for seasons: {seasons_list or 'current season'}, weeks: {weeks_list or 'all weeks'}..."
    )
    try:
        collector = NFLDataCollector()
        plays_added = collector.collect_play_by_play(seasons_list, weeks_list)
        click.echo(f"✅ Play-by-play collection complete! Added {plays_added} new plays.")
    except Exception as e:
        click.echo(f"❌ Play-by-play collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Path to DraftKings salary CSV file"
)
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True),
    help="Directory containing DraftKings CSV files",
)
@click.option(
    "--contest-name", "-c", type=str, help="Contest name (derived from filename if not provided)"
)
def collect_dk(file: str | None, directory: str | None, contest_name: str | None):
    """Process DraftKings salary data from CSV files."""
    if not file and not directory:
        click.echo("❌ Must specify either --file or --directory")
        sys.exit(1)

    if file and directory:
        click.echo("❌ Cannot specify both --file and --directory")
        sys.exit(1)

    try:
        from pathlib import Path

        from src.data.collection.dk_collector import DraftKingsCollector

        collector = DraftKingsCollector()

        if file:
            # Process single file
            file_path = Path(file)
            click.echo(f"Processing DraftKings file: {file_path}")
            results = collector.process_salary_file(file_path, contest_name)
            click.echo("✅ File processed successfully!")
            click.echo(f"  Contests: {results.get('contests', 0)}")
            click.echo(f"  Salaries: {results.get('salaries', 0)}")
            click.echo(f"  Unmatched players: {results.get('unmatched_players', 0)}")

        else:
            # Process directory
            dir_path = Path(directory)
            click.echo(f"Processing DraftKings files in: {dir_path}")
            results = collector.bulk_process_files(dir_path)
            click.echo("✅ Directory processed successfully!")
            click.echo(f"  Files processed: {results.get('files_processed', 0)}")
            click.echo(f"  Total contests: {results.get('total_contests', 0)}")
            click.echo(f"  Total salaries: {results.get('total_salaries', 0)}")
            if results.get("errors"):
                click.echo(f"  Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    click.echo(f"    - {error}")

    except Exception as e:
        click.echo(f"❌ DraftKings collection failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--seasons", "-s", multiple=True, type=int, help="Seasons to collect (e.g., -s 2023 -s 2024)"
)
def collect_all(seasons: list[int] | None):
    """Collect all NFL data (teams, players, schedules, stats)."""
    seasons_list = list(seasons) if seasons else None
    click.echo(f"Starting full data collection for seasons: {seasons_list or 'current season'}...")
    try:
        collector = NFLDataCollector()
        results = collector.collect_all_data(seasons_list)

        click.echo("✅ Full data collection complete!")
        click.echo("Results:")
        for data_type, count in results.items():
            click.echo(f"  - {data_type}: {count} new records")

    except Exception as e:
        click.echo(f"❌ Data collection failed: {e}")
        sys.exit(1)


@cli.command()
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

            click.echo("📊 Database Status:")
            click.echo(f"  Teams: {teams_count}")
            click.echo(f"  Players: {players_count}")
            click.echo(f"  Games: {games_count}")
            click.echo(f"  Player Stats: {stats_count}")
            click.echo(f"  Play-by-Play: {pbp_count}")
            click.echo(f"  DK Contests: {contests_count}")
            click.echo(f"  DK Salaries: {salaries_count}")
        finally:
            session.close()

    except Exception as e:
        click.echo(f"❌ Failed to get database status: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
