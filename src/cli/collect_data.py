"""
CLI commands for collecting NFL data.

This module provides a command-line interface for collecting and managing
NFL data using the Typer library, which creates rich CLI applications
with automatic help generation and validation.

Key CLI Patterns Demonstrated:
- Command grouping with typer.Typer()
- Option handling with typer.Option()
- Error handling and exit codes
- Progress feedback with colored output
- Logging configuration for debugging

Data Collection Workflow:
1. Initialize database structure
2. Collect teams (foundational data)
3. Collect players (linked to teams)
4. Collect schedules (games and matchups)
5. Collect statistics (player performance)
6. Optionally collect play-by-play data
7. Process DraftKings salary data

The CLI supports both individual commands and bulk operations.
"""

import logging  # For debugging and monitoring operations
import sys  # For system path manipulation
from pathlib import Path  # For file system operations

import typer  # Modern CLI framework for Python

# Add src directory to Python path for imports
# This allows running the CLI from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration and data collection modules
from src.config.settings import settings
from src.data.collection.nfl_collector import NFLDataCollector
from src.database.init_db import create_database

# Create main CLI application instance
# help= parameter provides description shown with --help
app = typer.Typer(help="Data collection commands for NFL data")


def setup_logging():
    """
    Configure logging for data collection operations.

    Sets up dual logging output:
    - File logging for permanent records
    - Console logging for real-time feedback

    Uses configuration from settings to control log level and file location.
    """
    logging.basicConfig(
        # Convert string log level to logging constant (e.g., "DEBUG" -> logging.DEBUG)
        level=getattr(logging, settings.log_level),
        # Standard log format with timestamp, logger name, level, and message
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Write logs to file for permanent record
            logging.FileHandler(settings.log_file),
            # Also output to console for real-time monitoring
            logging.StreamHandler(sys.stdout),
        ],
    )


# ========== DATABASE MANAGEMENT COMMANDS ==========


@app.command()
def init_db():
    """
    Initialize the database with required tables and structure.

    This command must be run first before any data collection.
    It creates all necessary tables based on SQLAlchemy models.

    Example usage:
        python -m src.cli.collect_data init-db
    """
    typer.echo("Initializing database...")
    try:
        # Call database initialization function
        create_database()
        # Use emoji and color for user-friendly output
        typer.echo("✅ Database initialized successfully!")
    except Exception as e:
        # Show error and exit with non-zero code for script detection
        typer.echo(f"❌ Database initialization failed: {e}")
        raise typer.Exit(1) from e  # Exit code 1 indicates failure


# ========== NFL DATA COLLECTION COMMANDS ==========


@app.command()
def collect_teams(
    # typer.Option creates command-line options with short/long forms
    # [] default means empty list if no options provided
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """
    Collect NFL team data from nfl_data_py.

    Teams are foundational data - they must be collected first since
    players, games, and other data reference team records.

    Examples:
        python -m src.cli.collect_data collect-teams
        python -m src.cli.collect_data collect-teams -s 2023 -s 2024

    Args:
        seasons: List of specific seasons to collect (defaults to all available)
    """
    # Enable logging for this operation
    setup_logging()

    # Convert empty list to None for collector API
    seasons_list = seasons if seasons else None

    typer.echo(f"Collecting NFL team data for seasons: {seasons_list or 'all available'}")

    try:
        # Create collector instance and run team collection
        collector = NFLDataCollector()
        teams_added = collector.collect_teams(seasons=seasons_list)

        # Report success with count of new records
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
def collect_injuries(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
    weeks: list[int] = typer.Option([], "--week", "-w", help="Weeks to collect (e.g., -w 1 -w 2)"),
):
    """
    Collect NFL injury report data from nfl_data_py.

    Injury reports provide crucial information for fantasy predictions including:
    - Player availability status (Out, Doubtful, Questionable, Probable)
    - Practice participation levels (DNP, Limited, Full)
    - Specific injury types and affected body parts
    - Historical injury patterns for trend analysis

    The system uses nfl_data_py to collect comprehensive injury data that includes
    both official injury reports and practice participation status.

    Examples:
        python -m src.cli.collect_data collect-injuries
        python -m src.cli.collect_data collect-injuries -s 2024
        python -m src.cli.collect_data collect-injuries -s 2024 -w 1 -w 2

    Args:
        seasons: List of specific seasons to collect (defaults to current season)
        weeks: List of specific weeks to collect (defaults to all weeks)
    """
    setup_logging()
    seasons_list = seasons if seasons else None
    weeks_list = weeks if weeks else None

    typer.echo(
        f"Collecting NFL injury data for seasons: {seasons_list or 'current season'}, weeks: {weeks_list or 'all weeks'}..."
    )

    try:
        collector = NFLDataCollector()
        injuries_added = collector.collect_injuries(seasons_list, weeks_list)
        typer.echo(f"✅ Injury collection complete! Added {injuries_added} new injury reports.")
    except Exception as e:
        typer.echo(f"❌ Injury collection failed: {e}")
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


# ========== BULK OPERATIONS ==========


@app.command()
def collect_all(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
):
    """
    Collect all NFL data in the correct order (teams, players, schedules, stats).

    This is the most convenient command for initial data setup or bulk updates.
    It automatically handles dependencies between data types:
    1. Teams (foundational)
    2. Players (references teams)
    3. Schedules (references teams)
    4. Statistics (references players and games)

    Examples:
        python -m src.cli.collect_data collect-all
        python -m src.cli.collect_data collect-all -s 2023 -s 2024

    Args:
        seasons: List of specific seasons to collect (defaults to current season)
    """
    setup_logging()
    seasons_list = seasons if seasons else None

    typer.echo(f"Starting full data collection for seasons: {seasons_list or 'current season'}...")

    try:
        collector = NFLDataCollector()
        # collect_all_data() handles the proper sequence and dependencies
        results = collector.collect_all_data(seasons_list)

        typer.echo("✅ Full data collection complete!")
        typer.echo("Results:")
        # Display summary of what was collected
        for data_type, count in results.items():
            typer.echo(f"  - {data_type}: {count} new records")

    except Exception as e:
        typer.echo(f"❌ Data collection failed: {e}")
        raise typer.Exit(1) from e


# ========== STATUS AND MONITORING COMMANDS ==========


@app.command()
def status():
    """
    Show database status and record counts for all data types.

    This command provides a quick overview of what data has been collected
    and is useful for:
    - Verifying data collection success
    - Monitoring database growth over time
    - Troubleshooting missing data issues
    - Planning data collection strategies

    Example:
        python -m src.cli.collect_data status
    """
    try:
        # Import database components
        from src.database.connection import SessionLocal
        from src.database.models import DraftKingsContest  # DraftKings contest information
        from src.database.models import DraftKingsSalary  # Player salary data
        from src.database.models import Game  # NFL game/schedule data
        from src.database.models import InjuryReport  # NFL injury reports
        from src.database.models import PlayByPlay  # Detailed play-by-play data
        from src.database.models import Player  # Player information
        from src.database.models import PlayerStats  # Player performance statistics
        from src.database.models import Team  # NFL team data

        # Create database session
        session = SessionLocal()
        try:
            # Query record counts for each data type
            teams_count = session.query(Team).count()
            players_count = session.query(Player).count()
            games_count = session.query(Game).count()
            stats_count = session.query(PlayerStats).count()
            pbp_count = session.query(PlayByPlay).count()
            injuries_count = session.query(InjuryReport).count()
            contests_count = session.query(DraftKingsContest).count()
            salaries_count = session.query(DraftKingsSalary).count()

            # Display formatted status report
            typer.echo("📊 Database Status:")
            typer.echo(f"  Teams: {teams_count:,}")  # NFL teams (should be ~32)
            typer.echo(f"  Players: {players_count:,}")  # Active/historical players
            typer.echo(f"  Games: {games_count:,}")  # Scheduled/completed games
            typer.echo(f"  Player Stats: {stats_count:,}")  # Individual game performances
            typer.echo(f"  Play-by-Play: {pbp_count:,}")  # Detailed play data
            typer.echo(f"  Injury Reports: {injuries_count:,}")  # Player injury data
            typer.echo(f"  DK Contests: {contests_count:,}")  # DraftKings contests
            typer.echo(f"  DK Salaries: {salaries_count:,}")  # Player pricing data

        finally:
            # Always close database session to prevent connection leaks
            session.close()

    except Exception as e:
        typer.echo(f"❌ Failed to get database status: {e}")
        raise typer.Exit(1) from e


# ========== CLI EXECUTION ==========

if __name__ == "__main__":
    # Run the Typer CLI application
    # This allows running the script directly: python src/cli/collect_data.py
    # Or as a module: python -m src.cli.collect_data
    app()
