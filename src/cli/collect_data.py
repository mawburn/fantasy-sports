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
def collect_weather(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
    weeks: list[int] = typer.Option([], "--week", "-w", help="Weeks to collect (e.g., -w 1 -w 2)"),
    upcoming_only: bool = typer.Option(
        False, "--upcoming", "-u", help="Only collect weather for upcoming games"
    ),
    days_ahead: int = typer.Option(
        7, "--days", "-d", help="Days ahead to check for upcoming games (default: 7)"
    ),
):
    """
    Collect weather data for NFL games using OpenWeatherMap API.

    Weather significantly impacts fantasy football performance:
    - Wind Speed: Affects passing accuracy and field goal attempts (>15 MPH critical)
    - Temperature: Cold weather reduces player performance and ball handling
    - Precipitation: Rain/snow increases fumbles, reduces passing efficiency
    - Game Script: Bad weather leads to more conservative, run-heavy approaches

    The system collects real-time weather for upcoming games and historical weather
    for completed games to enhance prediction accuracy and feature engineering.

    API Key Required:
    Set WEATHER_API_KEY environment variable with your OpenWeatherMap API key.
    Get a free key at https://openweathermap.org/api

    Examples:
        python -m src.cli.collect_data collect-weather
        python -m src.cli.collect_data collect-weather -s 2024
        python -m src.cli.collect_data collect-weather --upcoming
        python -m src.cli.collect_data collect-weather -s 2024 -w 1 -w 2

    Args:
        seasons: List of specific seasons to collect (defaults to current season)
        weeks: List of specific weeks to collect (defaults to all weeks)
        upcoming_only: Only collect weather for games in the next 7 days
        days_ahead: Number of days ahead to check for upcoming games
    """
    setup_logging()

    try:
        from src.data.collection.weather_collector import WeatherCollector

        # Initialize weather collector (will validate API key)
        collector = WeatherCollector()

        if upcoming_only:
            # Collect weather for upcoming games only
            typer.echo(f"Collecting weather for upcoming games (next {days_ahead} days)...")
            results = collector.collect_upcoming_games_weather(days_ahead)
            typer.echo("✅ Upcoming games weather collection complete!")
            typer.echo(f"  Games processed: {results.get('total_games', 0)}")
            typer.echo(f"  Weather updated: {results.get('weather_updated', 0)}")
            if results.get("failed_updates", 0) > 0:
                typer.echo(f"  Failed updates: {results.get('failed_updates', 0)}")
        else:
            # Collect weather for specific seasons/weeks
            seasons_list = seasons if seasons else None
            weeks_list = weeks if weeks else None

            if not seasons_list:
                # Default to current season
                from datetime import datetime

                current_season = datetime.now().year
                if datetime.now().month < 9:  # Before September
                    current_season -= 1
                seasons_list = [current_season]

            typer.echo(
                f"Collecting weather data for seasons: {seasons_list}, weeks: {weeks_list or 'all weeks'}..."
            )

            total_results = {
                "total_games": 0,
                "weather_collected": 0,
                "already_had_weather": 0,
                "failed_collection": 0,
                "api_requests": 0,
            }

            # Process each season
            for season in seasons_list:
                season_results = collector.collect_weather_for_season(season, weeks_list)

                # Aggregate results
                for key in total_results:
                    total_results[key] += season_results.get(key, 0)

            typer.echo("✅ Weather collection complete!")
            typer.echo(f"  Total games: {total_results.get('total_games', 0)}")
            typer.echo(f"  Weather collected: {total_results.get('weather_collected', 0)}")
            typer.echo(f"  Already had weather: {total_results.get('already_had_weather', 0)}")
            if total_results.get("failed_collection", 0) > 0:
                typer.echo(f"  Failed collections: {total_results.get('failed_collection', 0)}")
            typer.echo(f"  API requests made: {total_results.get('api_requests', 0)}")

    except ValueError as e:
        if "API key" in str(e):
            typer.echo("❌ Weather API key not configured!")
            typer.echo("Set WEATHER_API_KEY environment variable or get a free key from:")
            typer.echo("https://openweathermap.org/api")
        else:
            typer.echo(f"❌ Configuration error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Weather collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_dk(
    file: str = typer.Option(None, "--file", "-f", help="Path to DraftKings salary CSV file"),
    directory: str = typer.Option(
        "data/draftkings/salaries",
        "--directory",
        "-d",
        help="Directory containing DraftKings CSV files",
    ),
    contest_name: str = typer.Option(
        None, "--contest-name", "-c", help="Contest name (derived from filename if not provided)"
    ),
):
    """Process DraftKings salary data from CSV files."""
    setup_logging()

    if file and directory != "data/draftkings/salaries":
        typer.echo("❌ Cannot specify both --file and --directory")
        raise typer.Exit(1) from None

    if not file and not Path(directory).exists():
        typer.echo(
            f"❌ Directory {directory} does not exist. Use --file or create the directory first."
        )
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
def collect_stadiums():
    """
    Collect NFL stadium data including venue characteristics and performance factors.

    Stadium data significantly impacts fantasy football performance through:
    - Playing Surface: Turf vs grass affects speed and injury rates
    - Roof Type: Domes eliminate weather variables, typically increase scoring
    - Altitude: Denver's elevation affects kicking distance and ball flight
    - Climate: Regional weather patterns influence season-long performance
    - Acoustics: Crowd noise affects road team communication and false starts

    This command collects comprehensive stadium information for all 32 NFL venues
    including physical characteristics, performance factors, and team associations.

    Example:
        python -m src.cli.collect_data collect-stadiums
    """
    setup_logging()

    try:
        from src.data.collection.stadium_collector import StadiumDataCollector

        typer.echo("Collecting NFL stadium data...")
        collector = StadiumDataCollector()
        results = collector.collect_all_stadiums()

        typer.echo("✅ Stadium data collection complete!")
        typer.echo(f"  Total stadiums: {results.get('total_stadiums', 0)}")
        typer.echo(f"  Stadiums added: {results.get('stadiums_added', 0)}")
        typer.echo(f"  Stadiums updated: {results.get('stadiums_updated', 0)}")
        typer.echo(f"  Team relationships: {results.get('relationships_created', 0)}")

    except Exception as e:
        typer.echo(f"❌ Stadium data collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_odds(
    season: int = typer.Option(None, "--season", "-s", help="Season to collect (default: current)"),
    week: int = typer.Option(None, "--week", "-w", help="Specific week to collect"),
    upcoming_only: bool = typer.Option(
        False, "--upcoming", "-u", help="Only collect odds for upcoming games"
    ),
    days_ahead: int = typer.Option(
        7, "--days", "-d", help="Days ahead to check for upcoming games (default: 7)"
    ),
):
    """
    Collect Vegas betting odds for NFL games using The Odds API.

    Betting odds are critical for fantasy football because they provide:
    - Game totals (over/under): Predict high/low scoring games
    - Point spreads: Identify favored teams with positive game script
    - Moneyline odds: Probability of each team winning
    - Line movement: Market sentiment and injury/weather impacts

    Vegas odds incorporate all available information including injury reports,
    weather forecasts, and professional handicapper analysis, making them
    extremely valuable for fantasy predictions.

    API Key Required:
    Set ODDS_API_KEY environment variable with your The Odds API key.
    Get a free key (500 requests/month) at https://the-odds-api.com/

    Examples:
        python -m src.cli.collect_data collect-odds
        python -m src.cli.collect_data collect-odds -s 2024 -w 5
        python -m src.cli.collect_data collect-odds --upcoming

    Args:
        season: NFL season year (defaults to current season)
        week: Specific week number (1-18 regular season, 19+ playoffs)
        upcoming_only: Only collect odds for games in the next 7 days
        days_ahead: Number of days ahead to check for upcoming games
    """
    setup_logging()

    try:
        from src.data.collection.vegas_odds_collector import VegasOddsCollector

        collector = VegasOddsCollector()

        if upcoming_only:
            # Collect odds for upcoming games only
            typer.echo(f"Collecting odds for upcoming games (next {days_ahead} days)...")
            results = collector.collect_upcoming_games_odds(days_ahead)
            typer.echo("✅ Upcoming games odds collection complete!")
            typer.echo(f"  Games processed: {results.get('total_games', 0)}")
            typer.echo(f"  Odds collected: {results.get('odds_collected', 0)}")
            if results.get("failed_collection", 0) > 0:
                typer.echo(f"  Failed collections: {results.get('failed_collection', 0)}")
        else:
            # Collect odds for specific season/week
            if not season:
                from datetime import datetime

                current_season = datetime.now().year
                if datetime.now().month < 9:  # Before September
                    current_season -= 1
                season = current_season

            if not week:
                # Collect for current week (approximate)
                from datetime import datetime

                current_week = min(
                    18, max(1, (datetime.now() - datetime(season, 9, 1)).days // 7 + 1)
                )
                week = current_week

            typer.echo(f"Collecting odds for {season} season, week {week}...")
            results = collector.collect_odds_for_week(season, week)

            typer.echo("✅ Odds collection complete!")
            typer.echo(f"  Total games: {results.get('total_games', 0)}")
            typer.echo(f"  Odds collected: {results.get('odds_collected', 0)}")
            if results.get("failed_collection", 0) > 0:
                typer.echo(f"  Failed collections: {results.get('failed_collection', 0)}")

    except ValueError as e:
        if "API key" in str(e):
            typer.echo("❌ Odds API key not configured!")
            typer.echo("Set ODDS_API_KEY environment variable or get a free key from:")
            typer.echo("https://the-odds-api.com/")
        else:
            typer.echo(f"❌ Configuration error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Odds collection failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def collect_enhanced(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
    _include_weather: bool = typer.Option(
        True, "--weather/--no-weather", help="Include weather data collection"
    ),
    _include_odds: bool = typer.Option(
        True, "--odds/--no-odds", help="Include Vegas odds collection"
    ),
    _include_stadiums: bool = typer.Option(
        True, "--stadiums/--no-stadiums", help="Include stadium data collection"
    ),
):
    """
    Enhanced data collection including all NFL data plus weather, odds, and stadium data.

    This command provides comprehensive data collection using the new orchestration system.
    It collects all available data types in the correct dependency order and provides
    enhanced error handling and progress reporting.

    Collection Order:
    1. Teams (foundational data)
    2. Stadium data (venue characteristics)
    3. Players (roster information)
    4. Games/Schedules (matchup data)
    5. Player statistics (performance data)
    6. Weather data (environmental factors)
    7. Vegas odds (market intelligence)

    Examples:
        python -m src.cli.collect_data collect-enhanced
        python -m src.cli.collect_data collect-enhanced -s 2023 -s 2024
        python -m src.cli.collect_data collect-enhanced --no-weather --no-odds

    Args:
        seasons: List of specific seasons to collect (defaults to recent seasons)
        include_weather: Whether to collect weather data (requires API key)
        include_odds: Whether to collect Vegas odds (requires API key)
        include_stadiums: Whether to collect stadium data
    """
    setup_logging()

    try:
        from src.data.collection.data_orchestrator import DataCollectionOrchestrator

        orchestrator = DataCollectionOrchestrator()

        # Show collector status
        status = orchestrator.get_collection_status()
        typer.echo("🔍 Data Collector Status:")
        for name, info in status["collectors"].items():
            status_icon = "✅" if info["available"] else "❌"
            typer.echo(f"  {status_icon} {name}: {info['description']}")

        typer.echo("\n🚀 Starting enhanced data collection...")

        # Use orchestrator for comprehensive data collection
        if not seasons:
            results = orchestrator.collect_initial_setup()
        else:
            results = orchestrator.collect_initial_setup(seasons)

        typer.echo("✅ Enhanced data collection complete!")
        typer.echo("\nCollection Summary:")

        # Display results for each collector used
        for collector in results.get("collectors_used", []):
            if collector in results:
                count = results[collector]
                if isinstance(count, dict):
                    # Weather/odds results are dictionaries
                    total = count.get("total_games", count.get("total_stadiums", 0))
                    collected = count.get(
                        "weather_collected",
                        count.get("odds_collected", count.get("stadiums_added", 0)),
                    )
                    typer.echo(f"  {collector}: {collected}/{total} records")
                else:
                    # NFL data results are integers
                    typer.echo(f"  {collector}: {count} records")

        duration = results.get("duration", 0)
        typer.echo(f"\nTotal duration: {duration:.1f} seconds")

        if results.get("total_errors", 0) > 0:
            typer.echo(f"⚠️  Total errors: {results['total_errors']}")

    except Exception as e:
        typer.echo(f"❌ Enhanced data collection failed: {e}")
        raise typer.Exit(1) from e


# ========== BULK OPERATIONS ==========


@app.command()
def collect_all(
    seasons: list[int] = typer.Option(
        [], "--season", "-s", help="Seasons to collect (e.g., -s 2023 -s 2024)"
    ),
    include_weather: bool = typer.Option(
        True, "--weather/--no-weather", help="Include weather data collection (default: True)"
    ),
):
    """
    Collect all NFL data in the correct order (teams, players, schedules, stats, weather).

    This is the most convenient command for initial data setup or bulk updates.
    It automatically handles dependencies between data types:
    1. Teams (foundational)
    2. Players (references teams)
    3. Schedules (references teams)
    4. Statistics (references players and games)
    5. Weather (enhances game context)

    Examples:
        python -m src.cli.collect_data collect-all
        python -m src.cli.collect_data collect-all -s 2023 -s 2024
        python -m src.cli.collect_data collect-all --no-weather

    Args:
        seasons: List of specific seasons to collect (defaults to current season)
        include_weather: Whether to collect weather data (requires API key)
    """
    setup_logging()
    seasons_list = seasons if seasons else None

    typer.echo(f"Starting full data collection for seasons: {seasons_list or 'current season'}...")

    try:
        # Core NFL data collection
        collector = NFLDataCollector()
        results = collector.collect_all_data(seasons_list)

        typer.echo("✅ Core NFL data collection complete!")
        typer.echo("Core Results:")
        for data_type, count in results.items():
            typer.echo(f"  - {data_type}: {count} new records")

        # Weather data collection (optional)
        if include_weather:
            try:
                from src.data.collection.weather_collector import WeatherCollector

                typer.echo("\n🌤️  Starting weather data collection...")
                weather_collector = WeatherCollector()

                # Use same seasons as core data
                if not seasons_list:
                    from datetime import datetime

                    current_season = datetime.now().year
                    if datetime.now().month < 9:  # Before September
                        current_season -= 1
                    seasons_list = [current_season]

                weather_results = {
                    "total_games": 0,
                    "weather_collected": 0,
                    "already_had_weather": 0,
                    "failed_collection": 0,
                }

                for season in seasons_list:
                    season_weather = weather_collector.collect_weather_for_season(season)
                    for key in weather_results:
                        weather_results[key] += season_weather.get(key, 0)

                typer.echo("✅ Weather data collection complete!")
                typer.echo("Weather Results:")
                typer.echo(f"  - total_games: {weather_results['total_games']}")
                typer.echo(f"  - weather_collected: {weather_results['weather_collected']}")
                typer.echo(f"  - already_had_weather: {weather_results['already_had_weather']}")
                if weather_results["failed_collection"] > 0:
                    typer.echo(f"  - failed_collection: {weather_results['failed_collection']}")

            except ValueError as e:
                if "API key" in str(e):
                    typer.echo("⚠️  Skipping weather collection: API key not configured")
                    typer.echo("   Set WEATHER_API_KEY environment variable to enable weather data")
                else:
                    typer.echo(f"⚠️  Skipping weather collection: {e}")
            except Exception as e:
                typer.echo(f"⚠️  Weather collection failed: {e}")
                typer.echo("   Continuing with core data collection results")

        typer.echo("\n🎉 Full data collection pipeline complete!")

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
        from src.database.models import Stadium  # Stadium information
        from src.database.models import Team  # NFL team data
        from src.database.models import VegasOdds  # Betting odds data

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
            stadiums_count = session.query(Stadium).count()
            odds_count = session.query(VegasOdds).count()
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
            typer.echo(f"  Stadiums: {stadiums_count:,}")  # Stadium characteristics
            typer.echo(f"  Vegas Odds: {odds_count:,}")  # Betting odds records
            typer.echo(f"  DK Contests: {contests_count:,}")  # DraftKings contests
            typer.echo(f"  DK Salaries: {salaries_count:,}")  # Player pricing data

            # Weather data statistics
            games_with_weather = (
                session.query(Game).filter(Game.weather_temperature.isnot(None)).count()
            )
            weather_percentage = (games_with_weather / games_count * 100) if games_count > 0 else 0
            typer.echo(
                f"  Weather Data: {games_with_weather:,}/{games_count:,} games ({weather_percentage:.1f}%)"
            )

            # Odds data statistics
            if odds_count > 0:
                games_with_odds = session.query(VegasOdds.game_id).distinct().count()
                odds_percentage = (games_with_odds / games_count * 100) if games_count > 0 else 0
                typer.echo(
                    f"  Odds Coverage: {games_with_odds:,}/{games_count:,} games ({odds_percentage:.1f}%)"
                )

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
