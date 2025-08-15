"""NFL data collection using nfl_data_py.

This module handles downloading and storing NFL data from the nfl_data_py library,
which provides free access to NFL statistics, play-by-play data, and roster information.

Data Collection Pipeline:
1. Teams: Static team information (names, divisions, conferences)
2. Players: Roster data with physical attributes and experience
3. Schedules: Game information with dates, scores, and matchups
4. Player Stats: Weekly performance statistics for fantasy scoring
5. Play-by-Play: Detailed play-level data for advanced analytics

For beginners:

nfl_data_py: A Python library that provides free access to NFL data.
It's built on top of the nflfastR R package and provides historical
NFL data going back to 1999.

Database Sessions: SQLAlchemy uses sessions to manage database connections.
Sessions group database operations together and handle transactions.

ETL Pattern: Extract, Transform, Load - a common data processing pattern:
- Extract: Get data from nfl_data_py API
- Transform: Convert to our database schema format
- Load: Insert/update records in our database

Data Freshness: This collector handles both initial data loading and
incremental updates, checking for existing records to avoid duplicates.
"""

import logging
from datetime import datetime

import nfl_data_py as nfl  # Free NFL data API library
import pandas as pd  # Data manipulation and analysis library

from ...config.settings import settings  # Application configuration
from ...database.connection import SessionLocal  # Database session factory
from ...database.models import Game, PlayByPlay, Player, PlayerStats, Team  # SQLAlchemy models

# Set up logging for this module - helps track data collection progress and errors
logger = logging.getLogger(__name__)


class NFLDataCollector:
    """Collects and stores NFL data using nfl_data_py.

    This class orchestrates the collection of various NFL data types from the
    nfl_data_py library and stores them in our local database. It handles:
    - Season detection based on current date
    - Incremental updates (avoiding duplicate data)
    - Error handling and logging
    - Database session management

    Design Patterns Used:
    - Repository Pattern: Encapsulates data access logic
    - Error Handling: Graceful degradation when data sources fail
    - Idempotency: Safe to run multiple times without duplicating data

    For beginners:

    Class-based Design: Groups related functionality together.
    Each method handles a specific type of data (teams, players, etc.)

    Season Logic: NFL seasons run from September to February, so we need
    special logic to determine which season we're currently in.

    Database Transactions: Each method manages its own database session
    to ensure data consistency and proper cleanup.
    """

    def __init__(self):
        """Initialize the collector with current season detection.

        Season Detection Logic:
        - If current month is September or later: use current year
        - If current month is before September: use previous year

        Example: In March 2024, we're still in the 2023 NFL season
        because the season runs Sep 2023 - Feb 2024.
        """
        self.current_season = datetime.now().year
        if datetime.now().month < 9:  # Before September, use previous year
            self.current_season -= 1

    def collect_teams(self, seasons: list[int] | None = None) -> int:
        """Collect NFL team data and store in database.

        Team data collection is unique because team information is mostly static
        (unlike player stats which change weekly). This method:
        1. Downloads current team information from nfl_data_py
        2. Checks for existing teams to avoid duplicates
        3. Updates any changed information (team relocations, name changes)
        4. Returns count of newly added teams

        Upsert Pattern: "Update or Insert" - check if record exists,
        update if it does, insert if it doesn't. Ensures data freshness
        without duplicates.

        Database Session Management:
        - Create session at start of operation
        - Use try/finally to ensure session is always closed
        - Commit all changes at once for atomicity

        Args:
            seasons: Ignored for team data (teams don't change by season)

        Returns:
            Number of new teams added to database
        """
        # Team data doesn't vary by season, so seasons parameter is acknowledged but not used
        _ = seasons  # Acknowledge parameter to avoid unused warning
        logger.info("Collecting NFL team data...")

        try:
            # Extract: Get team data from nfl_data_py API
            # This returns a pandas DataFrame with team information
            teams_df = nfl.import_team_desc()

            teams_added = 0  # Counter for new teams added
            session = SessionLocal()  # Create database session
            try:
                # Transform & Load: Process each team record
                for _, row in teams_df.iterrows():  # Iterate through DataFrame rows
                    # Check if team already exists using team abbreviation as unique key
                    existing_team = (
                        session.query(Team).filter_by(team_abbr=row["team_abbr"]).first()
                    )

                    if not existing_team:
                        # Create new team record if it doesn't exist
                        team = Team(
                            team_abbr=row["team_abbr"],  # e.g., "KC", "TB", "NE"
                            team_name=row["team_name"],  # e.g., "Kansas City Chiefs"
                            conference=row["team_conf"],  # "AFC" or "NFC"
                            division=row["team_division"],  # "North", "South", "East", "West"
                        )
                        session.add(team)  # Add to session (not yet committed)
                        teams_added += 1
                    else:
                        # Update existing team info (handles team relocations, name changes)
                        existing_team.team_name = row["team_name"]
                        existing_team.conference = row["team_conf"]
                        existing_team.division = row["team_division"]
                        existing_team.updated_at = datetime.now()  # Track when updated

                # Commit all changes atomically - either all succeed or all fail
                session.commit()
            finally:
                # Always close session, even if error occurs (cleanup)
                session.close()

            logger.info(f"Teams processed: {len(teams_df)}, new teams added: {teams_added}")
            return teams_added

        except Exception:
            # Log full error details including stack trace
            logger.exception("Error collecting teams")
            raise  # Re-raise exception for caller to handle

    def collect_players(self, seasons: list[int] | None = None) -> int:
        """Collect NFL player data and store in database.

        Player data collection is more complex than teams because:
        1. Players change teams (trades, free agency, cuts)
        2. Rookies enter the league each season
        3. Players retire or become inactive
        4. Physical attributes and experience change over time

        This method handles:
        - Multiple seasons of roster data
        - Player movement between teams
        - Updating existing player information
        - Foreign key relationships (linking players to teams)

        Foreign Key Pattern: Players reference teams via team_id.
        We create a mapping dict for efficient lookups rather than
        querying the database for each player.

        Defensive Programming: Use .get() with defaults for optional
        fields to handle missing data gracefully.

        Args:
            seasons: List of seasons to collect (defaults to current season)

        Returns:
            Number of new players added to database
        """
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting player data for seasons: {seasons}")

        try:
            players_added = 0

            for season in seasons:
                # Get roster data
                rosters_df = nfl.import_seasonal_rosters([season])

                session = SessionLocal()  # Create database session for this season
                try:
                    # Get team mapping for foreign keys - creates dict for fast lookups
                    # This avoids querying teams table for every player
                    teams = {team.team_abbr: team.id for team in session.query(Team).all()}

                    # Process each player in the roster data
                    for _, row in rosters_df.iterrows():
                        # Check if player already exists using unique player_id
                        # player_id is a consistent identifier across seasons
                        existing_player = (
                            session.query(Player).filter_by(player_id=row["player_id"]).first()
                        )

                        # Look up team ID using team abbreviation
                        # Use .get() to handle cases where team not found (returns None)
                        team_id = teams.get(row.get("team"))

                        if not existing_player:
                            # Create new player record
                            player = Player(
                                player_id=row["player_id"],  # Unique NFL player identifier
                                display_name=row.get("player_name", ""),  # Full display name
                                first_name=row.get("first_name", ""),  # First name
                                last_name=row.get("last_name", ""),  # Last name
                                position=row.get("position", ""),  # e.g., "QB", "RB", "WR"
                                jersey_number=row.get("jersey_number"),  # Jersey #
                                team_id=team_id,  # Foreign key to teams table
                                height=row.get("height"),  # Height in inches
                                weight=row.get("weight"),  # Weight in pounds
                                age=row.get("age"),  # Current age
                                years_exp=row.get("years_exp"),  # Years of NFL experience
                                college=row.get("college"),  # College attended
                                rookie_year=row.get("rookie_year"),  # Year entered NFL
                                status=row.get("status", "Active"),  # Active, Inactive, etc.
                            )
                            session.add(player)
                            players_added += 1
                        else:
                            # Update existing player info (handles team changes, status updates)
                            # Only update fields that can change; preserve historical data
                            existing_player.display_name = row.get(
                                "player_name", existing_player.display_name
                            )
                            existing_player.team_id = team_id  # Important: track team changes
                            existing_player.position = row.get("position", existing_player.position)
                            existing_player.jersey_number = row.get(
                                "jersey_number", existing_player.jersey_number
                            )
                            existing_player.status = row.get("status", existing_player.status)
                            existing_player.updated_at = datetime.now()  # Track when updated

                    # Commit all changes for this season atomically
                    session.commit()
                finally:
                    # Always close session to free database connections
                    session.close()

            logger.info(f"Players processed, new players added: {players_added}")
            return players_added

        except Exception:
            # Log full error details for debugging
            logger.exception("Error collecting players")
            raise  # Re-raise for caller to handle

    def collect_schedules(self, seasons: list[int] | None = None) -> int:
        """Collect NFL schedule data and store in database.

        Schedule data is crucial for fantasy predictions because it provides:
        1. Game dates for temporal feature engineering
        2. Home/away matchups for venue advantages
        3. Final scores for completed games (game script analysis)
        4. Stadium and weather conditions (roof type)

        Game States:
        - Future games: No scores, used for predictions
        - Completed games: Has scores, used for training data

        Date Handling: Uses pandas.to_datetime() to parse various date
        formats from the API into consistent datetime objects.

        Validation Logic: Only creates games if both teams exist in
        our database (prevents foreign key errors).

        Args:
            seasons: List of seasons to collect schedules for

        Returns:
            Number of new games added to database
        """
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting schedule data for seasons: {seasons}")

        try:
            games_added = 0

            for season in seasons:
                # Get schedule data
                schedule_df = nfl.import_schedules([season])

                session = SessionLocal()  # Create database session
                try:
                    # Get team mapping for foreign key lookups
                    teams = {team.team_abbr: team.id for team in session.query(Team).all()}

                    # Process each game in the schedule
                    for _, row in schedule_df.iterrows():
                        # Check if game already exists using unique game_id
                        existing_game = (
                            session.query(Game).filter_by(game_id=row["game_id"]).first()
                        )

                        # Look up team IDs - both must exist to create valid game
                        home_team_id = teams.get(row["home_team"])
                        away_team_id = teams.get(row["away_team"])

                        # Create new game only if it doesn't exist and both teams are valid
                        if not existing_game and home_team_id and away_team_id:
                            game = Game(
                                game_id=row["game_id"],  # Unique NFL game identifier
                                season=season,  # Year of the season
                                week=row["week"],  # Week number (1-18 regular season)
                                game_date=pd.to_datetime(row["gameday"]),  # Parse date string
                                home_team_id=home_team_id,  # Foreign key to home team
                                away_team_id=away_team_id,  # Foreign key to away team
                                game_type=row[
                                    "game_type"
                                ],  # REG, POST, PRE (Regular, Playoff, Preseason)
                                home_score=row.get(
                                    "home_score"
                                ),  # Final score (None if not played)
                                away_score=row.get(
                                    "away_score"
                                ),  # Final score (None if not played)
                                stadium=row.get("stadium"),  # Stadium name
                                roof_type=row.get("roof"),  # "dome", "outdoors", "retractable"
                                # Game is finished if home_score is not NaN (scores are populated)
                                game_finished=not pd.isna(row.get("home_score")),
                            )
                            session.add(game)
                            games_added += 1
                        elif existing_game:
                            # Update existing game info (important for live score updates)
                            # Preserve existing values if new data is missing
                            existing_game.home_score = row.get(
                                "home_score", existing_game.home_score
                            )
                            existing_game.away_score = row.get(
                                "away_score", existing_game.away_score
                            )
                            existing_game.game_finished = not pd.isna(row.get("home_score"))
                            existing_game.updated_at = datetime.now()  # Track update time

                    # Commit all game changes for this season
                    session.commit()
                finally:
                    # Always close session to prevent connection leaks
                    session.close()

            logger.info(f"Games processed, new games added: {games_added}")
            return games_added

        except Exception:
            # Log full exception details for troubleshooting
            logger.exception("Error collecting schedules")
            raise  # Re-raise for caller handling

    def collect_player_stats(
        self, seasons: list[int] | None = None, weeks: list[int] | None = None
    ) -> int:
        """Collect NFL player statistics and store in database.

        Player statistics are the core data for fantasy predictions. This method:
        1. Downloads weekly performance data from nfl_data_py
        2. Stores comprehensive statistics for all fantasy-relevant categories
        3. Handles both seasonal data loads and incremental weekly updates

        Statistical Categories:
        - Passing: yards, touchdowns, interceptions, attempts, completions
        - Rushing: yards, touchdowns, attempts (carries)
        - Receiving: yards, touchdowns, receptions, targets
        - Scoring: fantasy points (standard and PPR), two-point conversions
        - Turnovers: fumbles lost, interceptions thrown

        Data Relationships:
        Uses composite foreign keys (player_id + game_id) to link statistics
        to specific player-game combinations. This enables temporal analysis.

        Null Handling: Uses 'or 0' pattern to convert None/NaN values to 0,
        since statistical absence typically means zero performance.

        Args:
            seasons: Seasons to collect (defaults to current season)
            weeks: Specific weeks to collect (None = all weeks)

        Returns:
            Number of new statistical records added
        """
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting player stats for seasons: {seasons}")

        try:
            stats_added = 0

            for season in seasons:
                # Get weekly stats
                weekly_df = nfl.import_weekly_data([season])

                # Filter by weeks if specified
                if weeks is not None:
                    weekly_df = weekly_df[weekly_df["week"].isin(weeks)]

                session = SessionLocal()  # Create database session
                try:
                    # Create lookup dictionaries for foreign key mapping
                    # This avoids database queries for each statistical record
                    players = {p.player_id: p.id for p in session.query(Player).all()}
                    games = {
                        g.game_id: g.id for g in session.query(Game).filter_by(season=season).all()
                    }

                    for _, row in weekly_df.iterrows():
                        player_id = players.get(row["player_id"])
                        game_id = games.get(row["game_id"])

                        if not player_id or not game_id:
                            continue

                        # Check if stats already exist
                        existing_stats = (
                            session.query(PlayerStats)
                            .filter_by(player_id=player_id, game_id=game_id)
                            .first()
                        )

                        if not existing_stats:
                            stats = PlayerStats(
                                player_id=player_id,
                                game_id=game_id,
                                passing_yards=row.get("passing_yards", 0) or 0,
                                passing_tds=row.get("passing_tds", 0) or 0,
                                passing_interceptions=row.get("interceptions", 0) or 0,
                                passing_attempts=row.get("attempts", 0) or 0,
                                passing_completions=row.get("completions", 0) or 0,
                                rushing_yards=row.get("rushing_yards", 0) or 0,
                                rushing_tds=row.get("rushing_tds", 0) or 0,
                                rushing_attempts=row.get("carries", 0) or 0,
                                receiving_yards=row.get("receiving_yards", 0) or 0,
                                receiving_tds=row.get("receiving_tds", 0) or 0,
                                receptions=row.get("receptions", 0) or 0,
                                targets=row.get("targets", 0) or 0,
                                fumbles_lost=row.get("fumbles_lost", 0) or 0,
                                two_point_conversions=row.get("two_point_conversions", 0) or 0,
                                fantasy_points=row.get("fantasy_points", 0) or 0,
                                fantasy_points_ppr=row.get("fantasy_points_ppr", 0) or 0,
                            )
                            session.add(stats)
                            stats_added += 1
                        else:
                            # Update existing stats
                            for field in [
                                "passing_yards",
                                "passing_tds",
                                "passing_interceptions",
                                "passing_attempts",
                                "passing_completions",
                                "rushing_yards",
                                "rushing_tds",
                                "rushing_attempts",
                                "receiving_yards",
                                "receiving_tds",
                                "receptions",
                                "targets",
                                "fumbles_lost",
                                "two_point_conversions",
                                "fantasy_points",
                                "fantasy_points_ppr",
                            ]:
                                if field in ["attempts", "completions", "carries"]:
                                    # Map field names
                                    db_field = (
                                        f"passing_{field}"
                                        if field in ["attempts", "completions"]
                                        else f"rushing_{field}"
                                    )
                                    setattr(existing_stats, db_field, row.get(field, 0) or 0)
                                else:
                                    setattr(existing_stats, field, row.get(field, 0) or 0)
                            existing_stats.updated_at = datetime.now()

                    session.commit()
                finally:
                    session.close()

            logger.info(f"Player stats processed, new stats added: {stats_added}")
            return stats_added

        except Exception:
            logger.exception("Error collecting player stats")
            raise

    def collect_play_by_play(
        self, seasons: list[int] | None = None, weeks: list[int] | None = None
    ) -> int:
        """Collect NFL play-by-play data and store in database."""
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting play-by-play data for seasons: {seasons}")

        try:
            plays_added = 0

            for season in seasons:
                # Get play-by-play data
                pbp_df = nfl.import_pbp_data([season])

                # Filter by weeks if specified
                if weeks is not None:
                    pbp_df = pbp_df[pbp_df["week"].isin(weeks)]

                session = SessionLocal()
                try:
                    # Get game mapping
                    games = {
                        g.game_id: g.id for g in session.query(Game).filter_by(season=season).all()
                    }

                    # Process in batches for memory efficiency
                    batch_size = 1000
                    for i in range(0, len(pbp_df), batch_size):
                        batch = pbp_df.iloc[i : i + batch_size]

                        for _, row in batch.iterrows():
                            game_id = games.get(row["game_id"])

                            if not game_id:
                                continue

                            # Check if play already exists
                            existing_play = (
                                session.query(PlayByPlay).filter_by(play_id=row["play_id"]).first()
                            )

                            if not existing_play:
                                play = PlayByPlay(
                                    play_id=row["play_id"],
                                    game_id=game_id,
                                    season=season,
                                    week=row.get("week"),
                                    posteam=row.get("posteam"),
                                    defteam=row.get("defteam"),
                                    quarter=row.get("qtr"),
                                    time=row.get("time"),
                                    down=row.get("down"),
                                    ydstogo=row.get("ydstogo"),
                                    yardline_100=row.get("yardline_100"),
                                    play_type=row.get("play_type"),
                                    desc=row.get("desc", "")[:500],  # Truncate to fit column
                                    yards_gained=row.get("yards_gained"),
                                    first_down=bool(row.get("first_down", 0)),
                                    touchdown=bool(row.get("touchdown", 0)),
                                    interception=bool(row.get("interception", 0)),
                                    fumble=bool(row.get("fumble", 0)),
                                    sack=bool(row.get("sack", 0)),
                                    penalty=bool(row.get("penalty", 0)),
                                    epa=row.get("epa"),
                                    wp=row.get("wp"),
                                    wpa=row.get("wpa"),
                                    home_score=row.get("total_home_score"),
                                    away_score=row.get("total_away_score"),
                                )
                                session.add(play)
                                plays_added += 1

                        # Commit in batches to avoid memory issues
                        session.commit()
                        logger.debug(
                            f"Processed batch {i // batch_size + 1}, plays added: {plays_added}"
                        )

                finally:
                    session.close()

            logger.info(f"Play-by-play data processed, new plays added: {plays_added}")
            return plays_added

        except Exception:
            logger.exception("Error collecting play-by-play data")
            raise

    def collect_all_data(self, seasons: list[int] | None = None) -> dict[str, int]:
        """Collect all NFL data (teams, players, schedules, stats)."""
        if seasons is None:
            seasons = list(
                range(
                    self.current_season - settings.nfl_seasons_to_load + 1, self.current_season + 1
                )
            )

        logger.info(f"Starting full data collection for seasons: {seasons}")

        results = {}

        try:
            # Collect in order of dependencies
            results["teams"] = self.collect_teams()
            results["players"] = self.collect_players(seasons)
            results["games"] = self.collect_schedules(seasons)
            results["stats"] = self.collect_player_stats(seasons)
            results["play_by_play"] = self.collect_play_by_play(seasons)

            logger.info(f"Data collection complete: {results}")
            return results

        except Exception:
            logger.exception("Error in full data collection")
            raise
