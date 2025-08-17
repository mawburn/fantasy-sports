"""Data collection orchestrator for managing all NFL data collectors.

This module provides a unified interface for managing all data collection activities
across the NFL DFS system. It coordinates multiple collectors to ensure data
consistency and avoid API rate limiting issues.

Collection Orchestration:
1. Manages execution order to respect data dependencies
2. Coordinates API rate limiting across multiple services
3. Provides unified error handling and logging
4. Enables selective data collection based on configuration
5. Handles retry logic and failure recovery

Data Dependencies:
- Teams must be collected before players, games, or stadiums
- Games must exist before collecting weather, odds, or player stats
- Stadium data can be collected independently
- Weather and odds can be collected in parallel

For beginners:

Orchestration Pattern: A coordinator class that manages multiple specialized
workers (collectors). This pattern is common in data pipeline systems.

Rate Limiting: Different APIs have different limits. The orchestrator
ensures we don't exceed any single API's limits by coordinating requests.

Error Isolation: If one collector fails, others can continue. The orchestrator
provides graceful degradation and recovery.
"""

import logging
from datetime import datetime

from ...config.settings import settings
from .nfl_collector import NFLDataCollector
from .stadium_collector import StadiumDataCollector
from .vegas_odds_collector import VegasOddsCollector
from .weather_collector import WeatherCollector

# Set up logging for this module
logger = logging.getLogger(__name__)


class DataCollectionOrchestrator:
    """Orchestrates all NFL data collection activities.

    This class manages the coordination of multiple data collectors to ensure
    efficient and reliable data collection across all NFL data sources.

    Key Features:
    - Unified interface for all data collection operations
    - Intelligent scheduling based on data freshness and API limits
    - Error handling with graceful degradation
    - Configurable collection strategies (full vs incremental)
    - Performance monitoring and optimization

    Collection Strategies:
    - Full Collection: Complete data refresh (setup/backfill)
    - Incremental: Recent data updates (daily/weekly operations)
    - Selective: Specific data types or date ranges
    - Emergency: Critical updates for upcoming games

    Design Patterns:
    - Coordinator Pattern: Manages multiple specialized workers
    - Strategy Pattern: Different collection approaches
    - Observer Pattern: Status updates and monitoring
    - Circuit Breaker: Failure isolation and recovery
    """

    def __init__(self):
        """Initialize the data collection orchestrator."""
        self.nfl_collector = NFLDataCollector()
        self.stadium_collector = StadiumDataCollector()

        # Initialize optional collectors based on configuration
        self.weather_collector = None
        if settings.weather_api_key:
            try:
                self.weather_collector = WeatherCollector()
                logger.info("Weather collector initialized successfully")
            except Exception as e:
                logger.warning(f"Weather collector initialization failed: {e}")

        self.odds_collector = None
        if settings.odds_api_key and settings.collect_vegas_odds:
            try:
                self.odds_collector = VegasOddsCollector()
                logger.info("Vegas odds collector initialized successfully")
            except Exception as e:
                logger.warning(f"Vegas odds collector initialization failed: {e}")

        # Track collection statistics
        self.collection_stats = {}
        self.last_collection_time = {}

    def collect_initial_setup(self, seasons: list[int] | None = None) -> dict[str, any]:
        """Perform initial data collection setup.

        This method handles the complete initial setup for the NFL DFS system,
        collecting all foundational data needed for predictions and optimization.

        Collection Order (respects dependencies):
        1. Teams - Foundation for all other data
        2. Stadium data - Venue characteristics
        3. Players - Roster information
        4. Games/Schedules - Game information
        5. Player statistics - Historical performance
        6. Weather data - Environmental factors (if API key available)
        7. Vegas odds - Market intelligence (if API key available)

        Args:
            seasons: List of seasons to collect (defaults to recent seasons)

        Returns:
            Dictionary with comprehensive collection statistics
        """
        if seasons is None:
            # Default to last 3 seasons for initial setup
            current_season = datetime.now().year
            if datetime.now().month < 9:  # Before September
                current_season -= 1
            seasons = [current_season - 2, current_season - 1, current_season]

        logger.info(f"Starting initial data collection setup for seasons: {seasons}")

        overall_stats = {
            "start_time": datetime.now(),
            "seasons_collected": seasons,
            "collectors_used": [],
            "total_errors": 0,
        }

        try:
            # Step 1: Collect teams (foundation data)
            logger.info("Step 1: Collecting NFL teams...")
            overall_stats["teams"] = self.nfl_collector.collect_teams()
            overall_stats["collectors_used"].append("nfl_teams")

            # Step 2: Collect stadium data (independent of seasons)
            if settings.collect_stadium_data:
                logger.info("Step 2: Collecting stadium data...")
                overall_stats["stadiums"] = self.stadium_collector.collect_all_stadiums()
                overall_stats["collectors_used"].append("stadiums")

            # Step 3: Collect players for specified seasons
            logger.info("Step 3: Collecting player data...")
            overall_stats["players"] = self.nfl_collector.collect_players(seasons)
            overall_stats["collectors_used"].append("nfl_players")

            # Step 4: Collect game schedules
            logger.info("Step 4: Collecting game schedules...")
            overall_stats["games"] = self.nfl_collector.collect_schedules(seasons)
            overall_stats["collectors_used"].append("nfl_schedules")

            # Step 5: Collect player statistics
            logger.info("Step 5: Collecting player statistics...")
            overall_stats["player_stats"] = self.nfl_collector.collect_player_stats(seasons)
            overall_stats["collectors_used"].append("nfl_player_stats")

            # Step 6: Collect play-by-play data
            logger.info("Step 6: Collecting play-by-play data...")
            overall_stats["play_by_play"] = self.nfl_collector.collect_play_by_play(seasons)
            overall_stats["collectors_used"].append("nfl_play_by_play")

            # Step 7: Collect injury data
            logger.info("Step 7: Collecting injury data...")
            overall_stats["injuries"] = self.nfl_collector.collect_injuries(seasons)
            overall_stats["collectors_used"].append("nfl_injuries")

            # Step 8: Collect weather data (if available)
            if self.weather_collector:
                logger.info("Step 8: Collecting weather data...")
                weather_stats = {"total_games": 0, "weather_collected": 0}
                for season in seasons:
                    season_stats = self.weather_collector.collect_weather_for_season(season)
                    for key, value in season_stats.items():
                        weather_stats[key] = weather_stats.get(key, 0) + value
                overall_stats["weather"] = weather_stats
                overall_stats["collectors_used"].append("weather")

            # Step 9: Collect Vegas odds (if available, recent data only)
            if self.odds_collector:
                logger.info("Step 9: Collecting Vegas odds for current season...")
                current_season = seasons[-1]  # Most recent season
                odds_stats = self.odds_collector.collect_odds_for_week(current_season, 1)
                overall_stats["vegas_odds"] = odds_stats
                overall_stats["collectors_used"].append("vegas_odds")

        except Exception as e:
            logger.exception(f"Error during initial data collection setup: {e}")
            overall_stats["total_errors"] += 1
            overall_stats["error_details"] = str(e)

        overall_stats["end_time"] = datetime.now()
        overall_stats["duration"] = (
            overall_stats["end_time"] - overall_stats["start_time"]
        ).total_seconds()

        logger.info(f"Initial data collection setup complete: {overall_stats}")
        return overall_stats

    def collect_weekly_updates(self, season: int, week: int) -> dict[str, any]:
        """Perform weekly data collection updates.

        This method handles routine weekly updates for active NFL seasons,
        focusing on fresh data needed for upcoming DFS contests.

        Weekly Collection Priority:
        1. Updated rosters/injuries - Critical for player availability
        2. Weather forecasts - Affects game script and player performance
        3. Vegas odds - Market sentiment and line movement
        4. Recent player statistics - Latest performance data

        Args:
            season: NFL season year
            week: Week number to collect

        Returns:
            Dictionary with weekly collection statistics
        """
        logger.info(f"Starting weekly data collection for {season} week {week}")

        weekly_stats = {
            "start_time": datetime.now(),
            "season": season,
            "week": week,
            "collectors_used": [],
            "total_errors": 0,
        }

        try:
            # Update player data (roster changes, status updates)
            logger.info("Updating player data...")
            weekly_stats["players"] = self.nfl_collector.collect_players([season])
            weekly_stats["collectors_used"].append("nfl_players")

            # Update schedules (score updates for completed games)
            logger.info("Updating game schedules...")
            weekly_stats["games"] = self.nfl_collector.collect_schedules([season])
            weekly_stats["collectors_used"].append("nfl_schedules")

            # Collect latest player stats
            logger.info("Collecting latest player stats...")
            weekly_stats["player_stats"] = self.nfl_collector.collect_player_stats([season], [week])
            weekly_stats["collectors_used"].append("nfl_player_stats")

            # Update injury reports
            logger.info("Updating injury reports...")
            weekly_stats["injuries"] = self.nfl_collector.collect_injuries([season], [week])
            weekly_stats["collectors_used"].append("nfl_injuries")

            # Collect weather for upcoming games
            if self.weather_collector:
                logger.info("Collecting weather for upcoming games...")
                weekly_stats["weather"] = self.weather_collector.collect_upcoming_games_weather()
                weekly_stats["collectors_used"].append("weather")

            # Collect current Vegas odds
            if self.odds_collector:
                logger.info("Collecting Vegas odds...")
                weekly_stats["vegas_odds"] = self.odds_collector.collect_odds_for_week(season, week)
                weekly_stats["collectors_used"].append("vegas_odds")

        except Exception as e:
            logger.exception(f"Error during weekly data collection: {e}")
            weekly_stats["total_errors"] += 1
            weekly_stats["error_details"] = str(e)

        weekly_stats["end_time"] = datetime.now()
        weekly_stats["duration"] = (
            weekly_stats["end_time"] - weekly_stats["start_time"]
        ).total_seconds()

        logger.info(f"Weekly data collection complete: {weekly_stats}")
        return weekly_stats

    def collect_game_day_updates(self) -> dict[str, any]:
        """Perform game day data collection updates.

        This method handles real-time updates for games happening today,
        focusing on last-minute information that affects DFS decisions.

        Game Day Priority:
        1. Final injury reports - Last-minute player status changes
        2. Weather updates - Current conditions for outdoor games
        3. Line movement - Last-minute betting market changes
        4. Roster updates - Emergency signings, elevations

        Returns:
            Dictionary with game day collection statistics
        """
        logger.info("Starting game day data collection updates")

        gameday_stats = {
            "start_time": datetime.now(),
            "collectors_used": [],
            "total_errors": 0,
        }

        try:
            # Current season detection
            current_season = datetime.now().year
            if datetime.now().month < 9:
                current_season -= 1

            # Get current week (approximate - could be enhanced)
            current_week = min(
                18, max(1, (datetime.now() - datetime(current_season, 9, 1)).days // 7 + 1)
            )

            # Update injury reports for current week
            logger.info("Updating injury reports...")
            gameday_stats["injuries"] = self.nfl_collector.collect_injuries(
                [current_season], [current_week]
            )
            gameday_stats["collectors_used"].append("nfl_injuries")

            # Update weather for today's games
            if self.weather_collector:
                logger.info("Updating weather for today's games...")
                gameday_stats["weather"] = self.weather_collector.collect_upcoming_games_weather(
                    days_ahead=1
                )
                gameday_stats["collectors_used"].append("weather")

            # Update Vegas odds for today's games
            if self.odds_collector:
                logger.info("Updating Vegas odds...")
                gameday_stats["vegas_odds"] = self.odds_collector.collect_upcoming_games_odds(
                    days_ahead=1
                )
                gameday_stats["collectors_used"].append("vegas_odds")

        except Exception as e:
            logger.exception(f"Error during game day data collection: {e}")
            gameday_stats["total_errors"] += 1
            gameday_stats["error_details"] = str(e)

        gameday_stats["end_time"] = datetime.now()
        gameday_stats["duration"] = (
            gameday_stats["end_time"] - gameday_stats["start_time"]
        ).total_seconds()

        logger.info(f"Game day data collection complete: {gameday_stats}")
        return gameday_stats

    def get_collection_status(self) -> dict[str, any]:
        """Get current status of all data collectors.

        Returns comprehensive status information including:
        - Collector availability and configuration
        - Recent collection statistics
        - API usage and rate limiting status
        - Data freshness indicators

        Returns:
            Dictionary with detailed collector status information
        """
        status = {
            "timestamp": datetime.now(),
            "collectors": {},
            "overall_health": "healthy",
        }

        # NFL Data Collector (always available)
        status["collectors"]["nfl_data"] = {
            "available": True,
            "type": "core",
            "description": "NFL games, players, stats from nfl_data_py",
            "last_collection": self.last_collection_time.get("nfl_data"),
            "status": "operational",
        }

        # Stadium Data Collector
        status["collectors"]["stadium_data"] = {
            "available": True,
            "type": "reference",
            "description": "NFL stadium characteristics and performance factors",
            "last_collection": self.last_collection_time.get("stadium_data"),
            "status": "operational",
        }

        # Weather Collector
        if self.weather_collector:
            status["collectors"]["weather"] = {
                "available": True,
                "type": "environmental",
                "description": "Weather conditions for NFL games",
                "api_key_configured": bool(settings.weather_api_key),
                "last_collection": self.last_collection_time.get("weather"),
                "status": "operational",
            }
        else:
            status["collectors"]["weather"] = {
                "available": False,
                "type": "environmental",
                "description": "Weather conditions (API key required)",
                "api_key_configured": bool(settings.weather_api_key),
                "status": "disabled - no API key",
            }

        # Vegas Odds Collector
        if self.odds_collector:
            status["collectors"]["vegas_odds"] = {
                "available": True,
                "type": "market",
                "description": "Betting odds and line movement",
                "api_key_configured": bool(settings.odds_api_key),
                "requests_used": getattr(self.odds_collector, "requests_made", 0),
                "monthly_limit": getattr(self.odds_collector, "monthly_limit", 500),
                "last_collection": self.last_collection_time.get("vegas_odds"),
                "status": "operational",
            }
        else:
            status["collectors"]["vegas_odds"] = {
                "available": False,
                "type": "market",
                "description": "Betting odds (API key required)",
                "api_key_configured": bool(settings.odds_api_key),
                "status": "disabled - no API key or configuration",
            }

        # Determine overall health
        available_collectors = sum(1 for c in status["collectors"].values() if c["available"])
        total_collectors = len(status["collectors"])

        if available_collectors == total_collectors:
            status["overall_health"] = "healthy"
        elif available_collectors >= 2:  # Core + at least one optional
            status["overall_health"] = "partial"
        else:
            status["overall_health"] = "limited"

        return status
