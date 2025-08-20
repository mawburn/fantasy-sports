"""Simplified data handling for NFL fantasy sports.

This module consolidates all data operations into a single file:
1. Direct SQLite database operations (no ORM)
2. NFL data collection using nfl_data_py
3. DraftKings CSV parsing
4. Feature engineering
5. Data validation and cleaning

No abstractions or complex classes - just functions that work.
"""

import csv
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

try:
    import nfl_data_py as nfl
except ImportError:
    print("Warning: nfl_data_py not available. Install with: pip install nfl_data_py")
    nfl = None

logger = logging.getLogger(__name__)


class ProgressDisplay:
    """Simple progress display that updates in place."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.last_percentage = -1
        self._finished = False

    def update(self, current: int, total: int):
        """Update progress display with current/total."""
        if self._finished or total == 0:
            return

        percentage = int((current / total) * 100)

        # Only update if percentage changed to reduce output
        if percentage != self.last_percentage:
            print(f"\r{self.description}: {percentage}%", end="", flush=True)
            self.last_percentage = percentage

    def finish(self, message: str = None):
        """Clear the progress line and optionally print a completion message."""
        if self._finished:
            return

        self._finished = True
        if message:
            print(f"\r{message}")
        else:
            print()  # Just add newline to clear the line


def parse_date_flexible(date_str: str) -> datetime:
    """Parse date string in either YYYY-MM-DD or M/D/YYYY format.

    Args:
        date_str: Date string in either format

    Returns:
        datetime object

    Raises:
        ValueError: If neither format can be parsed
    """
    if not date_str:
        raise ValueError("Date string is empty")

    # Try YYYY-MM-DD format first (ISO format)
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        pass

    # Try M/D/YYYY format (spreadspoke format)
    try:
        return datetime.strptime(date_str, '%m/%d/%Y')
    except ValueError:
        pass

    # Try MM/DD/YYYY format (padded version)
    try:
        return datetime.strptime(date_str, '%m/%d/%Y')
    except ValueError:
        pass

    raise ValueError(f"Unable to parse date string: '{date_str}'. Expected format: YYYY-MM-DD or M/D/YYYY")


# Database schema - simplified tables
DB_SCHEMA = {
    'games': '''
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            game_date TEXT,
            season INTEGER,
            week INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            game_finished INTEGER DEFAULT 0,
            stadium TEXT,
            stadium_neutral INTEGER DEFAULT 0,
            weather_temperature INTEGER,
            weather_wind_mph INTEGER,
            weather_humidity INTEGER,
            weather_detail TEXT
        )
    ''',
    'teams': '''
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY,
            team_abbr TEXT UNIQUE,
            team_name TEXT,
            division TEXT,
            conference TEXT
        )
    ''',
    'players': '''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            player_name TEXT,
            display_name TEXT,
            position TEXT,
            team_id INTEGER,
            gsis_id TEXT,
            status TEXT DEFAULT 'Active',
            injury_status TEXT DEFAULT NULL,
            FOREIGN KEY (team_id) REFERENCES teams (id)
        )
    ''',
    'player_stats': '''
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            game_id TEXT,
            passing_yards REAL DEFAULT 0,
            passing_tds INTEGER DEFAULT 0,
            passing_interceptions INTEGER DEFAULT 0,
            rushing_yards REAL DEFAULT 0,
            rushing_attempts INTEGER DEFAULT 0,
            rushing_tds INTEGER DEFAULT 0,
            receiving_yards REAL DEFAULT 0,
            targets INTEGER DEFAULT 0,
            receptions INTEGER DEFAULT 0,
            receiving_tds INTEGER DEFAULT 0,
            fumbles_lost INTEGER DEFAULT 0,
            fantasy_points REAL DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (game_id) REFERENCES games (id),
            UNIQUE(player_id, game_id)
        )
    ''',
    'draftkings_salaries': '''
        CREATE TABLE IF NOT EXISTS draftkings_salaries (
            id INTEGER PRIMARY KEY,
            contest_id TEXT,
            player_id INTEGER,
            salary INTEGER,
            roster_position TEXT,
            game_info TEXT,
            team_abbr TEXT,
            opponent TEXT,
            FOREIGN KEY (player_id) REFERENCES players (id)
        )
    ''',
    'dst_stats': '''
        CREATE TABLE IF NOT EXISTS dst_stats (
            id INTEGER PRIMARY KEY,
            game_id TEXT,
            team_abbr TEXT,
            season INTEGER,
            week INTEGER,
            points_allowed INTEGER DEFAULT 0,
            sacks INTEGER DEFAULT 0,
            interceptions INTEGER DEFAULT 0,
            fumbles_recovered INTEGER DEFAULT 0,
            fumbles_forced INTEGER DEFAULT 0,
            safeties INTEGER DEFAULT 0,
            defensive_tds INTEGER DEFAULT 0,
            return_tds INTEGER DEFAULT 0,
            special_teams_tds INTEGER DEFAULT 0,
            fantasy_points REAL DEFAULT 0.0,
            UNIQUE(team_abbr, game_id)
        )
    ''',
    'play_by_play': '''
        CREATE TABLE IF NOT EXISTS play_by_play (
            id INTEGER PRIMARY KEY,
            play_id TEXT UNIQUE,
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            posteam TEXT,
            defteam TEXT,
            play_type TEXT,
            description TEXT,
            down INTEGER,
            ydstogo INTEGER,
            yardline_100 INTEGER,
            quarter_seconds_remaining INTEGER,
            yards_gained INTEGER DEFAULT 0,
            touchdown INTEGER DEFAULT 0,
            pass_attempt INTEGER DEFAULT 0,
            rush_attempt INTEGER DEFAULT 0,
            complete_pass INTEGER DEFAULT 0,
            incomplete_pass INTEGER DEFAULT 0,
            interception INTEGER DEFAULT 0,
            fumble INTEGER DEFAULT 0,
            fumble_lost INTEGER DEFAULT 0,
            sack INTEGER DEFAULT 0,
            safety INTEGER DEFAULT 0,
            penalty INTEGER DEFAULT 0,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    ''',
    'weather': '''
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY,
            game_id TEXT,
            stadium_name TEXT,
            latitude REAL,
            longitude REAL,
            temperature INTEGER,
            feels_like INTEGER,
            humidity INTEGER,
            wind_speed INTEGER,
            wind_direction TEXT,
            precipitation_chance INTEGER,
            conditions TEXT,
            visibility INTEGER,
            pressure REAL,
            collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id),
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    ''',
    'betting_odds': '''
        CREATE TABLE IF NOT EXISTS betting_odds (
            id INTEGER PRIMARY KEY,
            game_id TEXT,
            favorite_team TEXT,
            spread_favorite REAL,
            over_under_line REAL,
            home_team_spread REAL,
            away_team_spread REAL,
            source TEXT DEFAULT 'spreadspoke',
            UNIQUE(game_id),
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    '''
}

def get_db_connection(db_path: str = "data/nfl_dfs.db") -> sqlite3.Connection:
    """Get database connection and ensure directory exists."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    return conn

def init_database(db_path: str = "data/nfl_dfs.db") -> None:
    """Initialize database with required tables."""
    conn = get_db_connection(db_path)

    try:
        for table_name, schema in DB_SCHEMA.items():
            conn.execute(schema)
            logger.info(f"Created/verified table: {table_name}")

        conn.commit()
        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        conn.close()

def load_teams(db_path: str = "data/nfl_dfs.db") -> None:
    """Load NFL teams into database."""
    # Standard NFL teams
    teams_data = [
        (1, 'ARI', 'Arizona Cardinals', 'NFC West', 'NFC'),
        (2, 'ATL', 'Atlanta Falcons', 'NFC South', 'NFC'),
        (3, 'BAL', 'Baltimore Ravens', 'AFC North', 'AFC'),
        (4, 'BUF', 'Buffalo Bills', 'AFC East', 'AFC'),
        (5, 'CAR', 'Carolina Panthers', 'NFC South', 'NFC'),
        (6, 'CHI', 'Chicago Bears', 'NFC North', 'NFC'),
        (7, 'CIN', 'Cincinnati Bengals', 'AFC North', 'AFC'),
        (8, 'CLE', 'Cleveland Browns', 'AFC North', 'AFC'),
        (9, 'DAL', 'Dallas Cowboys', 'NFC East', 'NFC'),
        (10, 'DEN', 'Denver Broncos', 'AFC West', 'AFC'),
        (11, 'DET', 'Detroit Lions', 'NFC North', 'NFC'),
        (12, 'GB', 'Green Bay Packers', 'NFC North', 'NFC'),
        (13, 'HOU', 'Houston Texans', 'AFC South', 'AFC'),
        (14, 'IND', 'Indianapolis Colts', 'AFC South', 'AFC'),
        (15, 'JAX', 'Jacksonville Jaguars', 'AFC South', 'AFC'),
        (16, 'KC', 'Kansas City Chiefs', 'AFC West', 'AFC'),
        (17, 'LV', 'Las Vegas Raiders', 'AFC West', 'AFC'),
        (18, 'LAC', 'Los Angeles Chargers', 'AFC West', 'AFC'),
        (19, 'LAR', 'Los Angeles Rams', 'NFC West', 'NFC'),
        (20, 'MIA', 'Miami Dolphins', 'AFC East', 'AFC'),
        (21, 'MIN', 'Minnesota Vikings', 'NFC North', 'NFC'),
        (22, 'NE', 'New England Patriots', 'AFC East', 'AFC'),
        (23, 'NO', 'New Orleans Saints', 'NFC South', 'NFC'),
        (24, 'NYG', 'New York Giants', 'NFC East', 'NFC'),
        (25, 'NYJ', 'New York Jets', 'AFC East', 'AFC'),
        (26, 'PHI', 'Philadelphia Eagles', 'NFC East', 'NFC'),
        (27, 'PIT', 'Pittsburgh Steelers', 'AFC North', 'AFC'),
        (28, 'SF', 'San Francisco 49ers', 'NFC West', 'NFC'),
        (29, 'SEA', 'Seattle Seahawks', 'NFC West', 'NFC'),
        (30, 'TB', 'Tampa Bay Buccaneers', 'NFC South', 'NFC'),
        (31, 'TEN', 'Tennessee Titans', 'AFC South', 'AFC'),
        (32, 'WAS', 'Washington Commanders', 'NFC East', 'NFC'),
    ]

    conn = get_db_connection(db_path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO teams (id, team_abbr, team_name, division, conference) VALUES (?, ?, ?, ?, ?)",
            teams_data
        )
        conn.commit()
        logger.info(f"Loaded {len(teams_data)} teams")
    finally:
        conn.close()

def collect_nfl_data(seasons: List[int], db_path: str = "data/nfl_dfs.db") -> None:
    """Collect NFL data using nfl_data_py."""
    if nfl is None:
        logger.error("nfl_data_py not available")
        return

    conn = get_db_connection(db_path)

    try:
        for season in seasons:
            logger.info(f"Collecting data for {season} season...")

            try:
                # Collect schedule data
                schedule = nfl.import_schedules([season])
            except Exception as e:
                logger.warning(f"Could not collect {season} season data: {e}")
                continue

            for _, game in schedule.iterrows():
                try:
                    # Convert and validate data types
                    game_id = str(game.get('game_id', ''))[:50] if game.get('game_id') else None
                    game_date = str(game.get('gameday', ''))[:10] if game.get('gameday') else None
                    week = int(game.get('week', 0)) if pd.notna(game.get('week')) else 0
                    home_score = int(game.get('home_score', 0)) if pd.notna(game.get('home_score')) else 0
                    away_score = int(game.get('away_score', 0)) if pd.notna(game.get('away_score')) else 0

                    home_team_id = get_team_id_by_abbr(game.get('home_team'), conn)
                    away_team_id = get_team_id_by_abbr(game.get('away_team'), conn)

                    if not home_team_id or not away_team_id:
                        continue  # Skip if teams not found

                    game_data = (
                        game_id,
                        game_date,
                        season,
                        week,
                        home_team_id,
                        away_team_id,
                        home_score,
                        away_score,
                        1 if home_score > 0 or away_score > 0 else 0
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO games
                           (id, game_date, season, week, home_team_id, away_team_id, home_score, away_score, game_finished)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        game_data
                    )
                except Exception as e:
                    logger.warning(f"Error processing game data: {e}")
                    continue

            # Collect weekly data
            try:
                weekly_data = nfl.import_weekly_data([season])
            except Exception as e:
                logger.warning(f"Could not collect {season} weekly data: {e}")
                continue

            for _, player_week in weekly_data.iterrows():
                # Get or create player
                player_id = get_or_create_player(
                    player_week.get('player_display_name', ''),
                    player_week.get('position', ''),
                    player_week.get('recent_team', ''),
                    player_week.get('player_id', ''),
                    conn
                )

                # Get game ID
                game_id = get_game_id(
                    season,
                    player_week.get('week'),
                    player_week.get('recent_team'),
                    conn
                )

                if player_id and game_id:
                    try:
                        # Calculate DraftKings fantasy points
                        fantasy_points = calculate_dk_fantasy_points(player_week)

                        # Convert and validate all numeric fields
                        def safe_float(val, default=0.0):
                            try:
                                return float(val) if pd.notna(val) else default
                            except (ValueError, TypeError):
                                return default

                        def safe_int(val, default=0):
                            try:
                                return int(val) if pd.notna(val) else default
                            except (ValueError, TypeError):
                                return default

                        stats_data = (
                            player_id,
                            game_id,
                            safe_float(player_week.get('passing_yards', 0)),
                            safe_int(player_week.get('passing_tds', 0)),
                            safe_int(player_week.get('interceptions', 0)),
                            safe_float(player_week.get('rushing_yards', 0)),
                            safe_int(player_week.get('carries', 0)),
                            safe_int(player_week.get('rushing_tds', 0)),
                            safe_float(player_week.get('receiving_yards', 0)),
                            safe_int(player_week.get('targets', 0)),
                            safe_int(player_week.get('receptions', 0)),
                            safe_int(player_week.get('receiving_tds', 0)),
                            safe_int(player_week.get('fumbles_lost', 0)),
                            fantasy_points
                        )

                        conn.execute(
                            """INSERT OR REPLACE INTO player_stats
                               (player_id, game_id, passing_yards, passing_tds, passing_interceptions,
                                rushing_yards, rushing_attempts, rushing_tds, receiving_yards, targets,
                                receptions, receiving_tds, fumbles_lost, fantasy_points)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            stats_data
                        )
                    except Exception as e:
                        logger.warning(f"Error processing player stats: {e}")
                        continue

            # Collect play-by-play data
            try:
                logger.info(f"Collecting play-by-play data for {season} season...")
                collect_pbp_data(season, conn)
            except Exception as e:
                logger.warning(f"Could not collect play-by-play data for {season}: {e}")

            # Collect DST data from play-by-play
            try:
                logger.info(f"Collecting DST data for {season} season...")
                collect_dst_data(season, conn)
            except Exception as e:
                logger.warning(f"Could not collect DST data for {season}: {e}")

            conn.commit()
            logger.info(f"Completed {season} season")

    except Exception as e:
        logger.error(f"Error collecting NFL data: {e}")
        raise
    finally:
        conn.close()

def get_team_id_by_abbr(team_abbr: str, conn: sqlite3.Connection) -> Optional[int]:
    """Get team ID by abbreviation."""
    if not team_abbr:
        return None

    cursor = conn.execute("SELECT id FROM teams WHERE team_abbr = ?", (team_abbr,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_or_create_player(
    name: str,
    position: str,
    team_abbr: str,
    gsis_id: str,
    conn: sqlite3.Connection
) -> Optional[int]:
    """Get existing player or create new one."""
    if not name:
        return None

    # Try to find existing player
    cursor = conn.execute(
        "SELECT id FROM players WHERE player_name = ? AND gsis_id = ?",
        (name, gsis_id)
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Create new player
    team_id = get_team_id_by_abbr(team_abbr, conn)
    cursor = conn.execute(
        """INSERT INTO players (player_name, display_name, position, team_id, gsis_id)
           VALUES (?, ?, ?, ?, ?)""",
        (name, name, position, team_id, gsis_id)
    )

    return cursor.lastrowid

def get_game_id(season: int, week: int, team_abbr: str, conn: sqlite3.Connection) -> Optional[str]:
    """Get game ID for a team in a specific week."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    cursor = conn.execute(
        """SELECT id FROM games
           WHERE season = ? AND week = ?
           AND (home_team_id = ? OR away_team_id = ?)
           LIMIT 1""",
        (season, week, team_id, team_id)
    )
    result = cursor.fetchone()
    return result[0] if result else None

def calculate_dk_fantasy_points(player_data: pd.Series) -> float:
    """Calculate DraftKings fantasy points."""
    points = 0.0

    # Passing
    points += (player_data.get('passing_yards', 0) or 0) * 0.04  # 1 pt per 25 yards
    points += (player_data.get('passing_tds', 0) or 0) * 4
    points += (player_data.get('interceptions', 0) or 0) * -1

    # Rushing
    points += (player_data.get('rushing_yards', 0) or 0) * 0.1  # 1 pt per 10 yards
    points += (player_data.get('rushing_tds', 0) or 0) * 6

    # Receiving
    points += (player_data.get('receiving_yards', 0) or 0) * 0.1  # 1 pt per 10 yards
    points += (player_data.get('receptions', 0) or 0) * 1  # 1 pt per reception
    points += (player_data.get('receiving_tds', 0) or 0) * 6

    # Fumbles
    points += (player_data.get('fumbles_lost', 0) or 0) * -1

    return round(points, 2)

def calculate_dst_fantasy_points(stats: Dict[str, int]) -> float:
    """Calculate DraftKings DST fantasy points."""
    points = 0.0

    # Points allowed (tiered system)
    points_allowed = stats.get('points_allowed', 0)
    if points_allowed == 0:
        points += 10
    elif points_allowed <= 6:
        points += 7
    elif points_allowed <= 13:
        points += 4
    elif points_allowed <= 20:
        points += 1
    elif points_allowed <= 27:
        points += 0
    elif points_allowed <= 34:
        points += -1
    else:
        points += -4

    # Sacks (1 pt each)
    points += stats.get('sacks', 0) * 1

    # Interceptions (2 pts each)
    points += stats.get('interceptions', 0) * 2

    # Fumbles recovered (2 pts each)
    points += stats.get('fumbles_recovered', 0) * 2

    # Safeties (2 pts each)
    points += stats.get('safeties', 0) * 2

    # Defensive TDs (6 pts each)
    points += stats.get('defensive_tds', 0) * 6

    # Return TDs (6 pts each)
    points += stats.get('return_tds', 0) * 6

    # Special teams TDs (6 pts each)
    points += stats.get('special_teams_tds', 0) * 6

    return round(points, 2)

def collect_pbp_data(season: int, conn: sqlite3.Connection) -> None:
    """Collect and store play-by-play data."""
    if nfl is None:
        logger.warning("nfl_data_py not available for play-by-play data")
        return

    try:
        logger.info(f"Loading play-by-play data for season {season}...")
        pbp_data = nfl.import_pbp_data([season], downcast=True)
    except (NameError, Exception) as e:
        if "Error" in str(e) and "not defined" in str(e):
            logger.warning(f"nfl_data_py library bug for season {season}: {e}")
        elif "404" in str(e) or "Not Found" in str(e):
            logger.warning(f"No play-by-play data available for season {season}")
        else:
            logger.warning(f"Could not load PBP data for season {season}: {e}")
        return

    # Store play-by-play data
    play_count = 0
    for _, play in pbp_data.iterrows():
        try:
            # Skip invalid plays
            if pd.isna(play.get('play_id')) or not play.get('game_id'):
                continue

            game_id = str(play.get('game_id', ''))

            # Check if the referenced game exists in the games table
            existing_game = conn.execute(
                "SELECT id FROM games WHERE id = ?", (game_id,)
            ).fetchone()

            if not existing_game:
                # Skip plays for games that don't exist in our games table
                continue

            def safe_int(val, default=0):
                try:
                    return int(val) if pd.notna(val) and val != '' else default
                except (ValueError, TypeError):
                    return default

            play_data = (
                str(play.get('play_id', '')),
                game_id,
                season,
                safe_int(play.get('week')),
                str(play.get('home_team', '') or ''),
                str(play.get('away_team', '') or ''),
                str(play.get('posteam', '') or ''),
                str(play.get('defteam', '') or ''),
                str(play.get('play_type', '') or ''),
                str(play.get('desc', '') or '')[:500],  # Limit description length
                safe_int(play.get('down')),
                safe_int(play.get('ydstogo')),
                safe_int(play.get('yardline_100')),
                safe_int(play.get('quarter_seconds_remaining')),
                safe_int(play.get('yards_gained')),
                safe_int(play.get('touchdown')),
                safe_int(play.get('pass_attempt')),
                safe_int(play.get('rush_attempt')),
                safe_int(play.get('complete_pass')),
                safe_int(play.get('incomplete_pass')),
                safe_int(play.get('interception')),
                safe_int(play.get('fumble')),
                safe_int(play.get('fumble_lost')),
                safe_int(play.get('sack')),
                safe_int(play.get('safety')),
                safe_int(play.get('penalty'))
            )

            conn.execute(
                """INSERT OR REPLACE INTO play_by_play
                   (play_id, game_id, season, week, home_team, away_team, posteam, defteam,
                    play_type, description, down, ydstogo, yardline_100, quarter_seconds_remaining,
                    yards_gained, touchdown, pass_attempt, rush_attempt, complete_pass,
                    incomplete_pass, interception, fumble, fumble_lost, sack, safety, penalty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                play_data
            )
            play_count += 1

        except Exception as e:
            logger.warning(f"Error processing play {play.get('play_id', 'unknown')}: {e}")
            continue

    logger.info(f"Stored {play_count} plays for season {season}")

def load_env_file():
    """Load .env file manually if it exists."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load .env file
load_env_file()

def collect_weather_data_optimized(db_path: str = "data/nfl_dfs.db", limit: int = None, rate_limit_delay: float = 1.5, max_days_per_batch: int = 30) -> None:
    """Collect weather data using batch API calls for date ranges to minimize requests."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2781, 'lon': -76.6227},
        'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870},
        'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8533},
        'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167},
        'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5160},
        'CLE': {'name': 'FirstEnergy Stadium', 'lat': 41.5061, 'lon': -81.6995},
        'DEN': {'name': 'Empower Field at Mile High', 'lat': 39.7439, 'lon': -105.0201},
        'GB': {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622},
        'JAX': {'name': 'TIAA Bank Field', 'lat': 32.0815, 'lon': -81.6370},
        'KC': {'name': 'Arrowhead Stadium', 'lat': 39.0489, 'lon': -94.4839},
        'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9581, 'lon': -80.2389},
        'NE': {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643},
        'NYG': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9008, 'lon': -75.1675},
        'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4468, 'lon': -80.0158},
        'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316},
        'TB': {'name': 'Raymond James Stadium', 'lat': 27.9756, 'lon': -82.5034},
        'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713},
        'WAS': {'name': 'FedExField', 'lat': 38.9077, 'lon': -76.8645}
    }

    try:
        # Get ALL games needing weather data (historical only)
        games_for_weather = conn.execute(
            """SELECT g.id, ht.team_abbr as home_team, g.game_date
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               WHERE g.id NOT IN (SELECT game_id FROM weather WHERE game_id IS NOT NULL)
               AND g.game_date < ?
               AND ht.team_abbr IN ('BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DEN', 'GB', 'JAX', 'KC', 'MIA', 'NE', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'TB', 'TEN', 'WAS')
               ORDER BY ht.team_abbr, g.game_date""",
            (datetime.now().date().strftime('%Y-%m-%d'),)
        ).fetchall()

        # Group games by stadium for batch processing
        stadium_games = {}
        for game_id, home_team, game_date in games_for_weather:
            if home_team not in stadium_games:
                stadium_games[home_team] = []
            stadium_games[home_team].append((game_id, game_date))

        weather_count = 0
        api_calls_made = 0

        logger.info(f"Found {len(games_for_weather)} games needing weather data across {len(stadium_games)} stadiums")
        if limit:
            logger.info(f"Processing limited to {limit} API calls")

        for home_team, games in stadium_games.items():
            if limit and api_calls_made >= limit:
                logger.info(f"Reached API call limit ({limit}), stopping collection")
                break

            stadium = outdoor_stadiums[home_team]
            logger.info(f"Processing {len(games)} games for {stadium['name']}")

            # Sort games by date
            games.sort(key=lambda x: x[1])

            # Group games into date ranges
            i = 0
            while i < len(games):
                batch_games = []
                batch_start_date = games[i][1]
                current_date = batch_start_date

                # Collect consecutive games within max_days_per_batch
                while (i < len(games) and
                       len(batch_games) < max_days_per_batch and
                       (parse_date_flexible(games[i][1]) -
                        parse_date_flexible(batch_start_date)).days <= max_days_per_batch):
                    batch_games.append(games[i])
                    current_date = games[i][1]
                    i += 1

                if limit and api_calls_made >= limit:
                    break

                # Make batch API call
                batch_end_date = current_date
                try:
                    logger.info(f"Fetching weather batch for {stadium['name']}: {batch_start_date} to {batch_end_date} ({len(batch_games)} games)")

                    batch_data = get_historical_weather_batch(
                        stadium['lat'], stadium['lon'],
                        batch_start_date, batch_end_date,
                        rate_limit_delay
                    )
                    api_calls_made += 1

                    if not batch_data:
                        logger.warning(f"Failed to get batch weather for {stadium['name']} - may have hit API limit")
                        break

                    # Process each day in the batch response
                    if batch_data.get('days'):
                        days_data = {day['datetime']: day for day in batch_data['days']}

                        for game_id, game_date in batch_games:
                            if game_date in days_data:
                                day_data = days_data[game_date]

                                weather_record = (
                                    game_id,
                                    stadium['name'],
                                    stadium['lat'],
                                    stadium['lon'],
                                    day_data.get('temp'),
                                    day_data.get('feelslike'),
                                    day_data.get('humidity'),
                                    day_data.get('windspeed'),
                                    day_data.get('winddir'),
                                    day_data.get('precipprob'),
                                    day_data.get('conditions'),
                                    day_data.get('visibility'),
                                    day_data.get('pressure')
                                )

                                conn.execute(
                                    """INSERT OR REPLACE INTO weather
                                       (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                                        humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                                        visibility, pressure, collected_at)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                                    weather_record
                                )
                                weather_count += 1
                            else:
                                logger.warning(f"No weather data for {game_date} in batch response")

                    if weather_count % 20 == 0:
                        logger.info(f"Collected weather for {weather_count} games ({api_calls_made} API calls)")

                except Exception as e:
                    logger.warning(f"Error collecting batch weather for {stadium['name']}: {e}")
                    continue

        conn.commit()
        logger.info(f"Stored weather data for {weather_count} games using {api_calls_made} API calls")

    except Exception as e:
        logger.error(f"Error collecting weather data: {e}")
    finally:
        conn.close()


def collect_weather_data_with_limits(db_path: str = "data/nfl_dfs.db", limit: int = None, rate_limit_delay: float = 1.5) -> None:
    """Collect weather data with API rate limiting and request limits."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2781, 'lon': -76.6227},
        'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870},
        'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8533},
        'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167},
        'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5160},
        'CLE': {'name': 'FirstEnergy Stadium', 'lat': 41.5061, 'lon': -81.6995},
        'DEN': {'name': 'Empower Field at Mile High', 'lat': 39.7439, 'lon': -105.0201},
        'GB': {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622},
        'JAX': {'name': 'TIAA Bank Field', 'lat': 32.0815, 'lon': -81.6370},
        'KC': {'name': 'Arrowhead Stadium', 'lat': 39.0489, 'lon': -94.4839},
        'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9581, 'lon': -80.2389},
        'NE': {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643},
        'NYG': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9008, 'lon': -75.1675},
        'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4468, 'lon': -80.0158},
        'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316},
        'TB': {'name': 'Raymond James Stadium', 'lat': 27.9756, 'lon': -82.5034},
        'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713},
        'WAS': {'name': 'FedExField', 'lat': 38.9077, 'lon': -76.8645}
        # Domes/covered stadiums excluded: ARI, ATL, DAL, DET, HOU, IND, LAC, LAR, LV, MIN, NO, SF
    }

    try:
        # Get ALL games needing weather data (all historical + upcoming)
        games_for_weather = conn.execute(
            """SELECT g.id, ht.team_abbr as home_team, g.game_date
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               WHERE g.id NOT IN (SELECT game_id FROM weather WHERE game_id IS NOT NULL)
               ORDER BY g.game_date DESC"""
        ).fetchall()

        weather_count = 0
        api_calls_made = 0

        logger.info(f"Found {len(games_for_weather)} games needing weather data")
        if limit:
            logger.info(f"Processing limited to {limit} API calls")

        for game_id, home_team, game_date in games_for_weather:
            # Check API call limit
            if limit and api_calls_made >= limit:
                logger.info(f"Reached API call limit ({limit}), stopping collection")
                break

            # Only collect weather for outdoor stadiums
            stadium = outdoor_stadiums.get(home_team)
            if not stadium:
                logger.debug(f"Skipping weather for {home_team} - dome/covered stadium")
                continue

            try:
                # For historical games, use Visual Crossing API for historical data
                # For future games, get forecast from weather.gov
                if game_date < datetime.now().date().strftime('%Y-%m-%d'):
                    # Historical weather from Visual Crossing API with rate limiting
                    weather_data = get_historical_weather(stadium['lat'], stadium['lon'], game_date, rate_limit_delay=rate_limit_delay)
                    api_calls_made += 1

                    if not weather_data:
                        # Check if we hit daily limit
                        logger.warning(f"Failed to get weather for {game_id} - may have hit API limit")
                        break
                else:
                    # Future games - get forecast (free)
                    weather_data = get_weather_forecast(stadium['lat'], stadium['lon'])

                if weather_data:
                    weather_record = (
                        game_id,
                        stadium['name'],
                        stadium['lat'],
                        stadium['lon'],
                        weather_data.get('temperature'),
                        weather_data.get('feels_like'),
                        weather_data.get('humidity'),
                        weather_data.get('wind_speed'),
                        weather_data.get('wind_direction'),
                        weather_data.get('precipitation_chance'),
                        weather_data.get('conditions'),
                        weather_data.get('visibility'),
                        weather_data.get('pressure')
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO weather
                           (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                            humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                            visibility, pressure, collected_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                        weather_record
                    )
                    weather_count += 1
                    if weather_count % 10 == 0:
                        logger.info(f"Collected weather for {weather_count} games ({api_calls_made} API calls)")

            except Exception as e:
                logger.warning(f"Error collecting weather for {game_id}: {e}")
                continue

        conn.commit()
        logger.info(f"Stored weather data for {weather_count} games (made {api_calls_made} API calls)")

    except Exception as e:
        logger.error(f"Error collecting weather data: {e}")
    finally:
        conn.close()


def collect_weather_data(db_path: str = "data/nfl_dfs.db") -> None:
    """Collect weather data for upcoming games using weather.gov API."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2781, 'lon': -76.6227},
        'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870},
        'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8533},
        'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167},
        'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5160},
        'CLE': {'name': 'FirstEnergy Stadium', 'lat': 41.5061, 'lon': -81.6995},
        'DEN': {'name': 'Empower Field at Mile High', 'lat': 39.7439, 'lon': -105.0201},
        'GB': {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622},
        'JAX': {'name': 'TIAA Bank Field', 'lat': 32.0815, 'lon': -81.6370},
        'KC': {'name': 'Arrowhead Stadium', 'lat': 39.0489, 'lon': -94.4839},
        'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9581, 'lon': -80.2389},
        'NE': {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643},
        'NYG': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745},
        'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9008, 'lon': -75.1675},
        'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4468, 'lon': -80.0158},
        'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316},
        'TB': {'name': 'Raymond James Stadium', 'lat': 27.9756, 'lon': -82.5034},
        'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713},
        'WAS': {'name': 'FedExField', 'lat': 38.9077, 'lon': -76.8645}
        # Domes/covered stadiums excluded: ARI, ATL, DAL, DET, HOU, IND, LAC, LAR, LV, MIN, NO, SF
    }

    try:
        # Get ALL games needing weather data (all historical + upcoming)
        games_for_weather = conn.execute(
            """SELECT g.id, ht.team_abbr as home_team, g.game_date
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               WHERE g.id NOT IN (SELECT game_id FROM weather WHERE game_id IS NOT NULL)
               ORDER BY g.game_date DESC"""
        ).fetchall()

        weather_count = 0
        for game_id, home_team, game_date in games_for_weather:
            # Only collect weather for outdoor stadiums
            stadium = outdoor_stadiums.get(home_team)
            if not stadium:
                logger.debug(f"Skipping weather for {home_team} - dome/covered stadium")
                continue

            try:
                # For historical games, use Visual Crossing API for historical data
                # For future games, get forecast from weather.gov
                if game_date < datetime.now().date().strftime('%Y-%m-%d'):
                    # Historical weather from Visual Crossing API with rate limiting
                    weather_data = get_historical_weather(stadium['lat'], stadium['lon'], game_date, rate_limit_delay=1.5)
                    if not weather_data:
                        # Fallback to placeholder if API fails
                        weather_data = {
                            'temperature': None,
                            'feels_like': None,
                            'humidity': None,
                            'wind_speed': None,
                            'wind_direction': None,
                            'precipitation_chance': None,
                            'conditions': 'Historical data unavailable',
                            'visibility': None,
                            'pressure': None
                        }
                else:
                    # Future games - get forecast
                    weather_data = get_weather_forecast(stadium['lat'], stadium['lon'])

                if weather_data:
                    weather_record = (
                        game_id,
                        stadium['name'],
                        stadium['lat'],
                        stadium['lon'],
                        weather_data.get('temperature'),
                        weather_data.get('feels_like'),
                        weather_data.get('humidity'),
                        weather_data.get('wind_speed'),
                        weather_data.get('wind_direction'),
                        weather_data.get('precipitation_chance'),
                        weather_data.get('conditions'),
                        weather_data.get('visibility'),
                        weather_data.get('pressure')
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO weather
                           (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                            humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                            visibility, pressure)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        weather_record
                    )
                    weather_count += 1

            except Exception as e:
                logger.warning(f"Error collecting weather for {home_team}: {e}")
                continue

        conn.commit()
        logger.info(f"Stored weather data for {weather_count} games")

    except Exception as e:
        logger.error(f"Error collecting weather data: {e}")
    finally:
        conn.close()

def get_historical_weather_batch(lat: float, lon: float, start_date: str, end_date: str, rate_limit_delay: float = 1.0) -> Optional[Dict]:
    """Get historical weather data for a date range from Visual Crossing API."""
    visual_crossing_key = os.environ.get('VISUAL_CROSSING_API_KEY')
    if not visual_crossing_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    try:
        # Rate limiting - wait between requests
        time.sleep(rate_limit_delay)

        # Visual Crossing Timeline API for date range: location/start_date/end_date
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
        params = {
            'key': visual_crossing_key,
            'unitGroup': 'us',  # Fahrenheit, mph, inches
            'include': 'days',  # Only daily data
            'elements': 'datetime,temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure'
        }

        response = requests.get(url, params=params, timeout=30)  # Longer timeout for batch requests

        # Log response headers for rate limit debugging
        logger.debug(f"API headers for {start_date} to {end_date}: {dict(response.headers)}")

        if response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {start_date} to {end_date} at {lat},{lon}")
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After', '60')
            try:
                wait_time = int(retry_after)
            except ValueError:
                wait_time = 60
            logger.warning(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            return None
        elif response.status_code == 400 and "Maximum daily cost exceeded" in response.text:
            logger.error("Daily API limit exceeded - stopping weather collection")
            return None
        elif response.status_code != 200:
            logger.warning(f"Visual Crossing API error: {response.status_code} for {start_date} to {end_date} at {lat},{lon}")
            logger.debug(f"Response body: {response.text[:200]}")
            if response.status_code == 401:
                logger.warning("API authentication failed - check VISUAL_CROSSING_API_KEY")
            return None

        data = response.json()
        query_cost = data.get('queryCost', 0)
        logger.debug(f"API batch response for {start_date} to {end_date}: {query_cost} query cost")

        # Log remaining quota if available in response
        if 'remainingCost' in data:
            logger.debug(f"Remaining API cost: {data['remainingCost']}")

        return data

    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error fetching batch weather for {start_date} to {end_date}: {e}")
        return None
    except ValueError as e:
        logger.warning(f"JSON parsing error for batch weather {start_date} to {end_date}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching batch weather for {start_date} to {end_date}: {e}")
        return None


def get_historical_weather(lat: float, lon: float, game_date: str, rate_limit_delay: float = 1.0) -> Optional[Dict]:
    """Get historical weather data from Visual Crossing API with rate limiting."""
    visual_crossing_key = os.environ.get('VISUAL_CROSSING_API_KEY')
    if not visual_crossing_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    try:
        # Rate limiting - wait between requests
        time.sleep(rate_limit_delay)

        # Visual Crossing Timeline API for historical weather
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{game_date}"
        params = {
            'key': visual_crossing_key,
            'unitGroup': 'us',  # Fahrenheit, mph, inches
            'include': 'days',  # Only daily data
            'elements': 'temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure'
        }

        response = requests.get(url, params=params, timeout=15)

        # Log response headers for rate limit debugging
        logger.debug(f"API headers for {game_date}: {dict(response.headers)}")

        if response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {game_date} at {lat},{lon}")
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After', '60')
            try:
                wait_time = int(retry_after)
            except ValueError:
                wait_time = 60
            logger.warning(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            return None
        elif response.status_code == 400 and "Maximum daily cost exceeded" in response.text:
            logger.error("Daily API limit exceeded - stopping weather collection")
            return None
        elif response.status_code != 200:
            logger.warning(f"Visual Crossing API error: {response.status_code} for {game_date} at {lat},{lon}")
            logger.debug(f"Response body: {response.text[:200]}")
            if response.status_code == 401:
                logger.warning("API authentication failed - check VISUAL_CROSSING_API_KEY")
            return None

        data = response.json()
        query_cost = data.get('queryCost', 0)
        logger.debug(f"API response for {game_date}: {query_cost} query cost")

        # Log remaining quota if available in response
        if 'remainingCost' in data:
            logger.debug(f"Remaining API cost: {data['remainingCost']}")

        # Extract daily weather data
        if data.get('days') and len(data['days']) > 0:
            day_data = data['days'][0]

            return {
                'temperature': day_data.get('temp'),
                'feels_like': day_data.get('feelslike'),
                'humidity': day_data.get('humidity'),
                'wind_speed': day_data.get('windspeed'),
                'wind_direction': day_data.get('winddir'),
                'precipitation_chance': day_data.get('precipprob'),
                'conditions': day_data.get('conditions', ''),
                'visibility': day_data.get('visibility'),
                'pressure': day_data.get('pressure')
            }

    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error fetching historical weather for {game_date}: {e}")
        return None
    except ValueError as e:
        logger.warning(f"JSON parsing error for historical weather {game_date}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching historical weather for {game_date}: {e}")
        return None

def get_weather_forecast(lat: float, lon: float) -> Optional[Dict]:
    """Get weather forecast from weather.gov API."""
    try:
        # First get the grid coordinates
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        headers = {'User-Agent': 'NFL-DFS-Weather-Collector (contact@example.com)'}

        response = requests.get(points_url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Weather.gov points API error: {response.status_code}")
            return None

        points_data = response.json()
        forecast_url = points_data['properties']['forecast']

        # Get the forecast
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        if forecast_response.status_code != 200:
            logger.warning(f"Weather.gov forecast API error: {forecast_response.status_code}")
            return None

        forecast_data = forecast_response.json()

        # Extract current conditions (first period)
        if forecast_data.get('properties', {}).get('periods'):
            current = forecast_data['properties']['periods'][0]

            return {
                'temperature': current.get('temperature'),
                'feels_like': current.get('temperature'),  # weather.gov doesn't provide feels like
                'humidity': None,  # Not in basic forecast
                'wind_speed': parse_wind_speed(current.get('windSpeed', '')),
                'wind_direction': current.get('windDirection', ''),
                'precipitation_chance': None,  # Would need detailed forecast
                'conditions': current.get('shortForecast', ''),
                'visibility': None,  # Not in basic forecast
                'pressure': None   # Not in basic forecast
            }

    except Exception as e:
        logger.warning(f"Error fetching weather forecast: {e}")
        return None

def parse_wind_speed(wind_str: str) -> Optional[int]:
    """Parse wind speed from string like '10 mph'."""
    try:
        if wind_str and 'mph' in wind_str:
            return int(wind_str.split()[0])
    except:
        pass
    return None


def collect_dst_data(season: int, conn: sqlite3.Connection) -> None:
    """Collect DST data from play-by-play data."""
    if nfl is None:
        logger.warning("nfl_data_py not available for DST data")
        return

    try:
        # Get play-by-play data
        logger.info(f"Loading PBP data for DST stats (season {season})...")
        pbp_data = nfl.import_pbp_data([season])
    except (NameError, Exception) as e:
        # Handle nfl_data_py bugs like "name 'Error' is not defined" and HTTP 404s
        if "Error" in str(e) and "not defined" in str(e):
            logger.warning(f"nfl_data_py library bug for season {season}: {e}")
        elif "404" in str(e) or "Not Found" in str(e):
            logger.warning(f"No play-by-play data available for season {season}")
        else:
            logger.warning(f"Could not load PBP data for season {season}: {e}")
        return

    try:

        # Get schedule for points allowed calculation
        schedule = nfl.import_schedules([season])

        # Process each game for DST stats
        for _, game in schedule.iterrows():
            game_id = str(game.get('game_id', ''))
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            home_score = int(game.get('home_score', 0)) if pd.notna(game.get('home_score')) else 0
            away_score = int(game.get('away_score', 0)) if pd.notna(game.get('away_score')) else 0
            week = int(game.get('week', 0)) if pd.notna(game.get('week')) else 0

            # Skip games without scores (not played yet)
            if home_score == 0 and away_score == 0:
                continue

            # Get plays for this game
            game_plays = pbp_data[pbp_data['game_id'] == game_id]

            if len(game_plays) == 0:
                continue

            # Calculate stats for each team
            for team, opponent, points_allowed in [(home_team, away_team, away_score), (away_team, home_team, home_score)]:
                if not team:
                    continue

                # Aggregate defensive stats when this team was defending
                team_defense_plays = game_plays[game_plays['defteam'] == team]

                dst_stats = {
                    'points_allowed': points_allowed,
                    'sacks': int(team_defense_plays['sack'].sum() or 0),
                    'interceptions': int(team_defense_plays['interception'].sum() or 0),
                    'fumbles_recovered': int(team_defense_plays['fumble_lost'].sum() or 0),  # fumble_lost by offense = recovered by defense
                    'fumbles_forced': int(team_defense_plays['fumble_lost'].sum() or 0),  # Same as recovered for now
                    'safeties': int(team_defense_plays['safety'].sum() or 0),
                    'defensive_tds': 0,  # Would need more complex logic to identify defensive TDs
                    'return_tds': 0,     # Would need return TD logic
                    'special_teams_tds': 0  # Would need special teams logic
                }

                # Calculate fantasy points
                fantasy_points = calculate_dst_fantasy_points(dst_stats)

                # Insert into database
                conn.execute(
                    """INSERT OR REPLACE INTO dst_stats
                       (game_id, team_abbr, season, week, points_allowed, sacks, interceptions,
                        fumbles_recovered, fumbles_forced, safeties, defensive_tds, return_tds,
                        special_teams_tds, fantasy_points)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (game_id, team, season, week, dst_stats['points_allowed'], dst_stats['sacks'],
                     dst_stats['interceptions'], dst_stats['fumbles_recovered'], dst_stats['fumbles_forced'],
                     dst_stats['safeties'], dst_stats['defensive_tds'], dst_stats['return_tds'],
                     dst_stats['special_teams_tds'], fantasy_points)
                )

        logger.info(f"Completed DST data collection for {season} season")

    except Exception as e:
        logger.error(f"Error collecting DST data for {season}: {e}")
        raise

def load_draftkings_csv(csv_path: str, contest_id: str = None, db_path: str = "data/nfl_dfs.db") -> None:
    """Load DraftKings salary CSV file."""
    conn = get_db_connection(db_path)

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                player_name = row.get('Name', '').strip()
                salary = int(row.get('Salary', 0))
                roster_position = row.get('Roster Position', '').strip()
                game_info = row.get('Game Info', '').strip()
                team_abbr = row.get('TeamAbbrev', '').strip()

                # Use provided contest_id or generate one from game date in CSV
                if contest_id is None:
                    from pathlib import Path
                    import re
                    # Extract date from game_info (e.g., "CIN@CLE 09/07/2025 01:00PM ET")
                    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', game_info)
                    if date_match:
                        game_date = date_match.group(1).replace('/', '')
                        contest_id = f"DK_{game_date}"
                    else:
                        filename = Path(csv_path).stem
                        contest_id = f"{filename}_unknown"

                # Handle DST/Defense teams specially
                if roster_position == 'DST':
                    # Create or find defense "player" entry
                    player_id = get_or_create_defense_player(team_abbr, conn)
                else:
                    # Try to find matching individual player
                    player_id = find_player_by_name_and_team(player_name, team_abbr, conn)

                if player_id:
                    # Check if this player already exists for this contest
                    existing = conn.execute(
                        "SELECT id FROM draftkings_salaries WHERE contest_id = ? AND player_id = ?",
                        (contest_id, player_id)
                    ).fetchone()

                    if existing:
                        # Update existing entry
                        conn.execute(
                            """UPDATE draftkings_salaries
                               SET salary = ?, roster_position = ?, game_info = ?, team_abbr = ?, opponent = ?
                               WHERE contest_id = ? AND player_id = ?""",
                            (salary, roster_position, game_info, team_abbr, '', contest_id, player_id)
                        )
                    else:
                        # Insert new entry
                        conn.execute(
                            """INSERT INTO draftkings_salaries
                               (contest_id, player_id, salary, roster_position, game_info, team_abbr, opponent)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (contest_id, player_id, salary, roster_position, game_info, team_abbr, '')
                        )
                else:
                    logger.warning(f"Could not find player: {player_name} ({team_abbr})")

        conn.commit()
        logger.info(f"Loaded DraftKings salaries from {csv_path}")

    except Exception as e:
        logger.error(f"Error loading DraftKings CSV: {e}")
        raise
    finally:
        conn.close()

def get_or_create_defense_player(team_abbr: str, conn: sqlite3.Connection) -> Optional[int]:
    """Get or create defense player entry for a team."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    defense_name = f"{team_abbr} Defense"

    # Try to find existing defense player
    cursor = conn.execute(
        "SELECT id FROM players WHERE player_name = ? AND team_id = ? AND position = ?",
        (defense_name, team_id, 'DST')
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Create new defense player
    cursor = conn.execute(
        """INSERT INTO players (player_name, display_name, position, team_id, gsis_id)
           VALUES (?, ?, ?, ?, ?)""",
        (defense_name, defense_name, 'DST', team_id, f"DST_{team_abbr}")
    )

    return cursor.lastrowid

def find_player_by_name_and_team(name: str, team_abbr: str, conn: sqlite3.Connection) -> Optional[int]:
    """Find player by name and team."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    # Try exact match first
    cursor = conn.execute(
        "SELECT id FROM players WHERE display_name = ? AND team_id = ?",
        (name, team_id)
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Try partial match (for name variations)
    cursor = conn.execute(
        "SELECT id FROM players WHERE display_name LIKE ? AND team_id = ?",
        (f"%{name}%", team_id)
    )
    result = cursor.fetchone()
    return result[0] if result else None

def get_defensive_matchup_features(
    team_abbr: str,
    opponent_abbr: str,
    season: int,
    week: int,
    lookback_weeks: int = 4,
    db_path: str = "data/nfl_dfs.db"
) -> Dict[str, float]:
    """Extract defensive matchup features from play-by-play data."""
    conn = get_db_connection(db_path)
    features = {}

    try:
        # Get defensive performance vs specific opponent types
        def_vs_qb = conn.execute(
            """SELECT
                AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END) as sack_rate,
                AVG(CASE WHEN interception = 1 THEN 1.0 ELSE 0.0 END) as int_rate,
                AVG(CASE WHEN pass_attempt = 1 AND complete_pass = 0 AND interception = 0 THEN 1.0 ELSE 0.0 END) as pressure_rate
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND pass_attempt = 1
               GROUP BY defteam""",
            (team_abbr, season, week)
        ).fetchone()

        if def_vs_qb:
            features.update({
                'def_sack_rate': def_vs_qb[0] or 0,
                'def_int_rate': def_vs_qb[1] or 0,
                'def_pressure_rate': def_vs_qb[2] or 0
            })

        # Run defense efficiency
        def_vs_run = conn.execute(
            """SELECT
                AVG(yards_gained) as avg_yards_allowed,
                AVG(CASE WHEN yards_gained <= 3 THEN 1.0 ELSE 0.0 END) as stuff_rate,
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as td_rate
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND rush_attempt = 1""",
            (team_abbr, season, week)
        ).fetchone()

        if def_vs_run:
            features.update({
                'def_run_yards_allowed': def_vs_run[0] or 0,
                'def_stuff_rate': def_vs_run[1] or 0,
                'def_rush_td_rate': def_vs_run[2] or 0
            })

        # Red zone defense
        red_zone_def = conn.execute(
            """SELECT
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as rz_td_rate,
                COUNT(*) as rz_plays
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND yardline_100 <= 20 AND yardline_100 > 0""",
            (team_abbr, season, week)
        ).fetchone()

        if red_zone_def:
            features.update({
                'def_rz_td_rate': red_zone_def[0] or 0,
                'def_rz_plays': red_zone_def[1] or 0
            })

        # 3rd down defense
        third_down_def = conn.execute(
            """SELECT
                AVG(CASE WHEN yards_gained >= ydstogo THEN 1.0 ELSE 0.0 END) as third_down_conv_rate
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND down = 3""",
            (team_abbr, season, week)
        ).fetchone()

        if third_down_def:
            features['def_3rd_down_rate'] = third_down_def[0] or 0

        # Opponent offensive tendencies vs this defense
        opp_vs_def = conn.execute(
            """SELECT
                AVG(CASE WHEN pass_attempt = 1 THEN 1.0 ELSE 0.0 END) as pass_rate,
                AVG(CASE WHEN rush_attempt = 1 THEN 1.0 ELSE 0.0 END) as rush_rate,
                AVG(yards_gained) as avg_yards_gained
               FROM play_by_play
               WHERE posteam = ? AND defteam = ? AND season = ?
               AND (pass_attempt = 1 OR rush_attempt = 1)""",
            (opponent_abbr, team_abbr, season)
        ).fetchone()

        if opp_vs_def:
            features.update({
                'opp_pass_rate_vs_def': opp_vs_def[0] or 0.5,
                'opp_rush_rate_vs_def': opp_vs_def[1] or 0.5,
                'opp_avg_yards_vs_def': opp_vs_def[2] or 0
            })

    except Exception as e:
        logger.error(f"Error extracting defensive matchup features: {e}")
    finally:
        conn.close()

    return features

def get_player_vs_defense_features(
    player_id: int,
    team_abbr: str,
    opponent_abbr: str,
    season: int,
    week: int,
    position: str,
    lookback_weeks: int = 4,
    db_path: str = "data/nfl_dfs.db"
) -> Dict[str, float]:
    """Extract player-specific PbP features vs this defense."""
    conn = get_db_connection(db_path)
    features = {}

    try:
        # Get player name for PbP matching (if available in description)
        player_name = conn.execute(
            "SELECT display_name FROM players WHERE id = ?",
            (player_id,)
        ).fetchone()

        if not player_name:
            return features

        player_name = player_name[0]

        if position == 'QB':
            # QB pressure/sack rate vs this specific defense
            qb_vs_def = conn.execute(
                """SELECT
                    AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END) as sack_rate_vs_def,
                    AVG(CASE WHEN interception = 1 THEN 1.0 ELSE 0.0 END) as int_rate_vs_def,
                    AVG(yards_gained) as avg_yards_vs_def,
                    COUNT(*) as plays_vs_def
                   FROM play_by_play
                   WHERE posteam = ? AND defteam = ? AND season >= ?
                   AND pass_attempt = 1
                   AND (description LIKE ? OR description LIKE ?)""",
                (team_abbr, opponent_abbr, season - 2, f"%{player_name}%", f"%{player_name.split()[0]}%")
            ).fetchone()

            if qb_vs_def and qb_vs_def[3] > 0:  # Has plays vs this defense
                features.update({
                    'qb_sack_rate_vs_def': qb_vs_def[0] or 0,
                    'qb_int_rate_vs_def': qb_vs_def[1] or 0,
                    'qb_avg_yards_vs_def': qb_vs_def[2] or 0,
                    'qb_experience_vs_def': min(qb_vs_def[3] / 10.0, 1.0)  # Normalize experience
                })

        elif position == 'RB':
            # RB performance vs this specific defense
            rb_vs_def = conn.execute(
                """SELECT
                    AVG(yards_gained) as avg_rush_yards_vs_def,
                    AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as td_rate_vs_def,
                    AVG(CASE WHEN yards_gained <= 2 THEN 1.0 ELSE 0.0 END) as stuff_rate_vs_def,
                    COUNT(*) as carries_vs_def
                   FROM play_by_play
                   WHERE posteam = ? AND defteam = ? AND season >= ?
                   AND rush_attempt = 1
                   AND (description LIKE ? OR description LIKE ?)""",
                (team_abbr, opponent_abbr, season - 2, f"%{player_name}%", f"%{player_name.split()[0]}%")
            ).fetchone()

            if rb_vs_def and rb_vs_def[3] > 0:
                features.update({
                    'rb_avg_yards_vs_def': rb_vs_def[0] or 0,
                    'rb_td_rate_vs_def': rb_vs_def[1] or 0,
                    'rb_stuff_rate_vs_def': rb_vs_def[2] or 0,
                    'rb_carries_vs_def': min(rb_vs_def[3] / 20.0, 1.0)
                })

        elif position in ['WR', 'TE']:
            # Receiver performance vs this defense
            rec_vs_def = conn.execute(
                """SELECT
                    COUNT(CASE WHEN complete_pass = 1 THEN 1 END) as catches_vs_def,
                    COUNT(CASE WHEN pass_attempt = 1 THEN 1 END) as targets_vs_def,
                    AVG(yards_gained) as avg_rec_yards_vs_def,
                    AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as rec_td_rate_vs_def
                   FROM play_by_play
                   WHERE posteam = ? AND defteam = ? AND season >= ?
                   AND pass_attempt = 1
                   AND (description LIKE ? OR description LIKE ?)""",
                (team_abbr, opponent_abbr, season - 2, f"%{player_name}%", f"%{player_name.split()[0]}%")
            ).fetchone()

            if rec_vs_def and rec_vs_def[1] > 0:  # Has targets vs this defense
                catch_rate = rec_vs_def[0] / rec_vs_def[1] if rec_vs_def[1] > 0 else 0
                features.update({
                    'rec_catch_rate_vs_def': catch_rate,
                    'rec_avg_yards_vs_def': rec_vs_def[2] or 0,
                    'rec_td_rate_vs_def': rec_vs_def[3] or 0,
                    'rec_targets_vs_def': min(rec_vs_def[1] / 15.0, 1.0)
                })

        # Get defensive rank vs this position type
        def_rank_vs_pos = conn.execute(
            """SELECT
                RANK() OVER (ORDER BY AVG(yards_gained) ASC) as def_rank_yards,
                RANK() OVER (ORDER BY AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) ASC) as def_rank_tds
               FROM play_by_play
               WHERE season = ? AND week < ?
               AND ((? = 'QB' AND pass_attempt = 1) OR
                    (? = 'RB' AND rush_attempt = 1) OR
                    (? IN ('WR', 'TE') AND pass_attempt = 1 AND complete_pass = 1))
               GROUP BY defteam""",
            (season, week, position, position, position)
        ).fetchone()

        if def_rank_vs_pos:
            features.update({
                f'def_rank_vs_{position.lower()}_yards': def_rank_vs_pos[0] or 16,
                f'def_rank_vs_{position.lower()}_tds': def_rank_vs_pos[1] or 16
            })

    except Exception as e:
        logger.error(f"Error extracting player vs defense features: {e}")
    finally:
        conn.close()

    return features

def compute_weekly_odds_z_scores(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Compute weekly z-scores for odds features."""
    weekly_mask = (df['season'] == season) & (df['week'] == week)
    weekly_df = df[weekly_mask].copy()

    if len(weekly_df) == 0:
        return df

    # Compute z-scores for this week
    if 'total_line' in weekly_df.columns and weekly_df['total_line'].std() > 0:
        weekly_mean = weekly_df['total_line'].mean()
        weekly_std = weekly_df['total_line'].std()
        df.loc[weekly_mask, 'game_tot_z'] = (df.loc[weekly_mask, 'total_line'] - weekly_mean) / weekly_std

    if 'team_itt' in weekly_df.columns and weekly_df['team_itt'].std() > 0:
        weekly_mean = weekly_df['team_itt'].mean()
        weekly_std = weekly_df['team_itt'].std()
        df.loc[weekly_mask, 'team_itt_z'] = (df.loc[weekly_mask, 'team_itt'] - weekly_mean) / weekly_std

    return df


def get_player_features(
    player_id: int,
    game_id: int,
    lookback_weeks: int = 4,
    db_path: str = "data/nfl_dfs.db"
) -> Dict[str, float]:
    """Extract features for a player for model training/prediction."""
    conn = get_db_connection(db_path)
    features = {}

    try:
        # Get player info
        player_info = conn.execute(
            """SELECT p.position, p.team_id, t.team_abbr
               FROM players p
               JOIN teams t ON p.team_id = t.id
               WHERE p.id = ?""",
            (player_id,)
        ).fetchone()

        if not player_info:
            return features

        position, team_id, team_abbr = player_info

        # Get game info and opponent
        game_info = conn.execute(
            """SELECT g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                      ht.team_abbr as home_abbr, at.team_abbr as away_abbr
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               JOIN teams at ON g.away_team_id = at.id
               WHERE g.id = ?""",
            (game_id,)
        ).fetchone()

        if not game_info:
            return features

        game_date, season, week, home_team_id, away_team_id, home_abbr, away_abbr = game_info

        # Determine opponent
        is_home = team_id == home_team_id
        opponent_abbr = away_abbr if is_home else home_abbr

        # Parse game date
        game_date = parse_date_flexible(game_date).date()
        start_date = game_date - timedelta(weeks=lookback_weeks)

        # Get recent stats
        recent_stats = conn.execute(
            """SELECT
                AVG(ps.fantasy_points) as avg_points,
                AVG(ps.passing_yards) as avg_pass_yards,
                AVG(ps.rushing_yards) as avg_rush_yards,
                AVG(ps.receiving_yards) as avg_rec_yards,
                AVG(ps.targets) as avg_targets,
                COUNT(*) as games_played,
                MAX(ps.fantasy_points) as max_points,
                MIN(ps.fantasy_points) as min_points
               FROM player_stats ps
               JOIN games g ON ps.game_id = g.id
               WHERE ps.player_id = ?
               AND g.game_date >= ? AND g.game_date < ?""",
            (player_id, start_date, game_date)
        ).fetchone()

        if recent_stats:
            features.update({
                'avg_fantasy_points': recent_stats[0] or 0,
                'avg_passing_yards': recent_stats[1] or 0,
                'avg_rushing_yards': recent_stats[2] or 0,
                'avg_receiving_yards': recent_stats[3] or 0,
                'avg_targets': recent_stats[4] or 0,
                'games_played': recent_stats[5] or 0,
                'max_points': recent_stats[6] or 0,
                'min_points': recent_stats[7] or 0,
                'consistency': 1 - ((recent_stats[6] or 0) - (recent_stats[7] or 0)) / max(recent_stats[0] or 1, 1),
            })

        # Position-specific features
        if position == 'QB':
            qb_stats = conn.execute(
                """SELECT AVG(ps.passing_tds), AVG(ps.passing_interceptions)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date)
            ).fetchone()
            if qb_stats:
                features['avg_pass_tds'] = qb_stats[0] or 0
                features['avg_interceptions'] = qb_stats[1] or 0

        elif position in ['RB']:
            rb_stats = conn.execute(
                """SELECT AVG(ps.rushing_attempts), AVG(ps.rushing_tds)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date)
            ).fetchone()
            if rb_stats:
                features['avg_carries'] = rb_stats[0] or 0
                features['avg_rush_tds'] = rb_stats[1] or 0

        elif position in ['WR', 'TE']:
            rec_stats = conn.execute(
                """SELECT AVG(ps.receptions), AVG(ps.receiving_tds)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date)
            ).fetchone()
            if rec_stats:
                features['avg_receptions'] = rec_stats[0] or 0
                features['avg_rec_tds'] = rec_stats[1] or 0

        # Add defensive matchup features from PbP data
        defensive_features = get_defensive_matchup_features(
            opponent_abbr, team_abbr, season, week, lookback_weeks, db_path
        )
        features.update(defensive_features)

        # Add situational PbP features for the player vs this defense
        pbp_matchup = get_player_vs_defense_features(
            player_id, team_abbr, opponent_abbr, season, week, position, lookback_weeks, db_path
        )
        features.update(pbp_matchup)

        # Add weather features
        weather_data = conn.execute(
            """SELECT weather_temperature, weather_wind_mph, weather_humidity,
                      weather_detail, stadium_neutral
               FROM games WHERE id = ?""",
            (game_id,)
        ).fetchone()

        if weather_data:
            temp, wind, humidity, conditions, neutral = weather_data

            # Raw weather features
            features['temperature_f'] = temp or 72
            features['wind_mph'] = wind or 0
            features['humidity_pct'] = humidity or 50

            # Weather threshold features
            features['cold_lt40'] = 1 if (temp or 72) < 40 else 0
            features['hot_gt85'] = 1 if (temp or 72) > 85 else 0
            features['wind_gt15'] = 1 if (wind or 0) > 15 else 0
            features['dome'] = 1 if conditions and 'indoor' in conditions.lower() else 0
        else:
            # Default weather values if no data
            features.update({
                'temperature_f': 72, 'wind_mph': 0, 'humidity_pct': 50,
                'cold_lt40': 0, 'hot_gt85': 0, 'wind_gt15': 0, 'dome': 0
            })

        # Add comprehensive injury features
        # Get player injury status
        player_injury = conn.execute(
            """SELECT p.injury_status FROM players p WHERE p.id = ?""",
            (player_id,)
        ).fetchone()

        injury_status = (player_injury[0] if player_injury else None) or 'Healthy'

        # One-hot encode injury status
        features['injury_status_Out'] = 1 if injury_status == 'Out' else 0
        features['injury_status_Doubtful'] = 1 if injury_status == 'Doubtful' else 0
        features['injury_status_Questionable'] = 1 if injury_status == 'Questionable' else 0
        features['injury_status_Probable'] = 1 if injury_status in ['Probable', 'Healthy'] else 0

        # Count games missed in last 4 weeks
        games_missed = conn.execute(
            """SELECT COUNT(*) FROM games g
               WHERE g.game_date >= ? AND g.game_date < ?
               AND NOT EXISTS (
                   SELECT 1 FROM player_stats ps
                   WHERE ps.player_id = ? AND ps.game_id = g.id
               )""",
            (start_date, game_date, player_id)
        ).fetchone()

        features['games_missed_last4'] = games_missed[0] if games_missed else 0

        # Practice trend (simplified - assume stable for now)
        features['practice_trend'] = 0  # 0=stable, 1=improving, -1=regressing

        # Returning from injury flag
        features['returning_from_injury'] = 1 if features['games_missed_last4'] > 0 and injury_status == 'Healthy' else 0

        # Team injury aggregates - count injured starters
        team_injured = conn.execute(
            """SELECT COUNT(*) FROM players p
               WHERE p.team_id = ? AND p.injury_status IN ('Out', 'Doubtful', 'Questionable')""",
            (team_id,)
        ).fetchone()

        opponent_team_id = away_team_id if is_home else home_team_id
        opp_injured = conn.execute(
            """SELECT COUNT(*) FROM players p
               WHERE p.team_id = ? AND p.injury_status IN ('Out', 'Doubtful', 'Questionable')""",
            (opponent_team_id,)
        ).fetchone()

        features['team_injured_starters'] = min(team_injured[0] if team_injured else 0, 11)  # Cap at 11
        features['opp_injured_starters'] = min(opp_injured[0] if opp_injured else 0, 11)  # Cap at 11

        # Add enhanced betting odds features (prioritize live odds from odds_api)
        betting_data = conn.execute(
            """SELECT spread_favorite, over_under_line, home_team_spread, away_team_spread, source
               FROM betting_odds WHERE game_id = ?
               ORDER BY CASE WHEN source = 'odds_api' THEN 1 ELSE 2 END
               LIMIT 1""",
            (game_id,)
        ).fetchone()

        if betting_data:
            spread_fav, over_under, home_spread, away_spread, source = betting_data
            team_spread = home_spread if is_home else away_spread
            over_under_line = over_under or 45

            # Core odds features
            features['team_spread'] = team_spread or 0
            features['team_spread_abs'] = abs(team_spread or 0)
            features['total_line'] = over_under_line
            features['is_favorite'] = 1 if (team_spread or 0) < 0 else 0

            # Derived features
            features['team_itt'] = over_under_line / 2.0 - (team_spread or 0) / 2.0  # Implied team total

            # Z-scores will be computed in batch processing
            features['game_tot_z'] = 0.0  # Placeholder
            features['team_itt_z'] = 0.0  # Placeholder
        else:
            # Default betting values if no data
            features.update({
                'team_spread': 0, 'team_spread_abs': 0, 'total_line': 45, 'is_favorite': 0,
                'team_itt': 22.5, 'game_tot_z': 0.0, 'team_itt_z': 0.0
            })

        # Add contextual features to match schema
        features['salary'] = 5000  # Default salary - should be populated from DK data
        features['home'] = 1 if is_home else 0
        features['rest_days'] = 7  # Default NFL week rest
        features['travel'] = 0  # Travel distance - placeholder
        features['season_week'] = week  # Normalized week

        # Add placeholder usage/opportunity features (to be computed from historical data)
        features['targets_ema'] = features.get('avg_targets', 0)
        features['routes_run_ema'] = 0  # Placeholder
        features['rush_att_ema'] = features.get('avg_carries', 0)
        features['snap_share_ema'] = 0.5  # Placeholder
        features['redzone_opps_ema'] = 0  # Placeholder
        features['air_yards_ema'] = 0  # Placeholder
        features['adot_ema'] = 0  # Placeholder - Average Depth of Target
        features['yprr_ema'] = 0  # Placeholder - Yards Per Route Run

        # Add efficiency features (placeholders)
        features['yards_after_contact'] = 0  # Placeholder
        features['missed_tackles_forced'] = 0  # Placeholder
        features['pressure_rate'] = 0  # Placeholder - for QBs
        features['opp_dvp_pos_allowed'] = 0  # Opponent defense vs position

    except Exception as e:
        logger.error(f"Error extracting features for player {player_id}: {e}")
    finally:
        conn.close()

    return features

def is_home_game(team_id: int, game_id: int, conn: sqlite3.Connection) -> bool:
    """Check if team is playing at home."""
    result = conn.execute(
        "SELECT home_team_id FROM games WHERE id = ?",
        (game_id,)
    ).fetchone()
    return result and result[0] == team_id

def get_dst_training_data(
    seasons: List[int],
    db_path: str = "data/nfl_dfs.db"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get DST training data from historical team defense stats."""
    conn = get_db_connection(db_path)

    try:
        # Get DST stats for specified seasons with window functions
        data_query = """
            SELECT
                team_abbr,
                season,
                week,
                points_allowed,
                sacks,
                interceptions,
                fumbles_recovered,
                safeties,
                defensive_tds,
                fantasy_points,
                -- Add recent performance features using window functions
                AVG(fantasy_points) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_points,
                AVG(points_allowed) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_points_allowed,
                AVG(sacks) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_sacks
            FROM dst_stats
            WHERE season IN ({})
            AND week >= 4  -- Only include games where we have historical data
            ORDER BY season, week, team_abbr
        """.format(','.join('?' * len(seasons)))

        cursor = conn.execute(data_query, seasons)
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No DST training data found - dst_stats table may be empty")
            logger.info("Consider running data collection to populate dst_stats table")
            return np.array([]), np.array([]), []

        # Extract features and targets
        X_list = []
        y_list = []

        feature_names = [
            'points_allowed', 'sacks', 'interceptions', 'fumbles_recovered',
            'safeties', 'defensive_tds', 'avg_recent_points', 'avg_recent_points_allowed',
            'avg_recent_sacks', 'week', 'season'
        ]

        for row in rows:
            # Skip rows where we don't have recent averages (early in season)
            if row[10] is None:  # avg_recent_points
                continue

            features = [
                row[3],   # points_allowed
                row[4],   # sacks
                row[5],   # interceptions
                row[6],   # fumbles_recovered
                row[7],   # safeties
                row[8],   # defensive_tds
                row[10],  # avg_recent_points
                row[11],  # avg_recent_points_allowed
                row[12],  # avg_recent_sacks
                row[2],   # week
                row[1] - 2000  # season (normalize to reduce magnitude)
            ]

            target = row[9]  # fantasy_points

            X_list.append(features)
            y_list.append(target)

        if not X_list:
            logger.warning("No valid DST features extracted")
            return np.array([]), np.array([]), []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(f"Extracted {len(X)} DST training samples")
        return X, y, feature_names

    except Exception as e:
        logger.error(f"Error getting DST training data: {e}")
        return np.array([]), np.array([]), []
    finally:
        conn.close()


def batch_get_defensive_features(
    opponent_abbr: str,
    season: int,
    week: int,
    conn: sqlite3.Connection
) -> Dict[str, float]:
    """Batch compute defensive matchup features."""
    features = {}

    try:
        # Get basic defensive stats from PbP data
        def_stats = conn.execute(
            """SELECT
                AVG(yards_gained) as avg_yards_allowed,
                AVG(CASE WHEN rush_attempt = 1 THEN yards_gained ELSE NULL END) as avg_rush_yards,
                AVG(CASE WHEN pass_attempt = 1 THEN yards_gained ELSE NULL END) as avg_pass_yards,
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as td_rate_allowed,
                COUNT(*) as total_plays
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND week >= ?""",
            (opponent_abbr, season, week, max(1, week - 4))
        ).fetchone()

        if def_stats:
            features.update({
                'def_avg_yards_allowed': def_stats[0] or 0,
                'def_rush_yards_allowed': def_stats[1] or 0,
                'def_pass_yards_allowed': def_stats[2] or 0,
                'def_td_rate_allowed': def_stats[3] or 0,
                'def_total_plays': def_stats[4] or 0
            })

        # Get red zone defense
        rz_defense = conn.execute(
            """SELECT
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as rz_td_rate,
                COUNT(*) as rz_plays
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ? AND week >= ?
               AND yardline_100 <= 20""",
            (opponent_abbr, season, max(1, week - 4), week)
        ).fetchone()

        if rz_defense:
            features.update({
                'def_rz_td_rate_allowed': rz_defense[0] or 0,
                'def_rz_plays_allowed': rz_defense[1] or 0
            })

    except Exception as e:
        logger.warning(f"Error getting defensive features for {opponent_abbr}: {e}")

    return features


def get_training_data(
    position: str,
    seasons: List[int],
    db_path: str = "data/nfl_dfs.db"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get training data for a specific position using optimized batch queries."""
    # Handle DST position specially
    if position in ['DST', 'DEF']:
        return get_dst_training_data(seasons, db_path)

    conn = get_db_connection(db_path)

    try:
        logger.info(f"Loading training data for {position} position...")

        # Get all player-game combinations for the position
        data_query = """
            SELECT ps.player_id, ps.game_id, ps.fantasy_points,
                   p.position, p.team_id, t.team_abbr,
                   g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                   ht.team_abbr as home_abbr, at.team_abbr as away_abbr
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN teams t ON p.team_id = t.id
            JOIN games g ON ps.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE p.position = ? AND g.season IN ({})
            AND g.game_finished = 1
            ORDER BY g.game_date
        """.format(','.join('?' * len(seasons)))

        cursor = conn.execute(data_query, [position] + seasons)
        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"No training data found for position {position}")
            return np.array([]), np.array([]), []

        progress = ProgressDisplay("Loading training data")
        progress.finish(f"Found {len(rows)} player-game combinations")

        # Pre-compute all player stats in batch to avoid N+1 queries
        player_ids = list(set(row[0] for row in rows))

        # Batch load recent stats for all players with ALL columns
        player_stats_query = """
            SELECT ps.player_id, ps.game_id, ps.fantasy_points, ps.passing_yards,
                   ps.rushing_yards, ps.receiving_yards, ps.targets, ps.passing_tds,
                   ps.rushing_tds, ps.receiving_tds, ps.passing_interceptions, ps.fumbles_lost,
                   ps.rushing_attempts, ps.receptions,
                   g.game_date, g.season, g.week
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.id
            WHERE ps.player_id IN ({})
            AND g.season IN ({})
            ORDER BY ps.player_id, g.game_date DESC
        """.format(','.join('?' * len(player_ids)), ','.join('?' * len(seasons)))

        all_stats = conn.execute(player_stats_query, player_ids + seasons).fetchall()

        # Group stats by player for fast lookup
        player_stats = {}
        for stat_row in all_stats:
            pid = stat_row[0]
            if pid not in player_stats:
                player_stats[pid] = []
            player_stats[pid].append(stat_row)

        # Pre-compute defensive matchup features for all unique team-opponent combinations
        unique_matchups = set()
        for row in rows:
            team_id, home_team_id, away_team_id = row[4], row[9], row[10]
            team_abbr, home_abbr, away_abbr = row[5], row[11], row[12]
            season, week = row[7], row[8]

            # Determine opponent
            opponent_abbr = away_abbr if team_id == home_team_id else home_abbr
            unique_matchups.add((opponent_abbr, season, week))

        defensive_progress = ProgressDisplay("Computing defensive features")
        defensive_features_cache = {}
        for idx, (opponent_abbr, season, week) in enumerate(unique_matchups):
            defensive_progress.update(idx, len(unique_matchups))
            def_features = batch_get_defensive_features(opponent_abbr, season, week, conn)
            defensive_features_cache[(opponent_abbr, season, week)] = def_features
        defensive_progress.finish(f"Computed features for {len(unique_matchups)} matchups")

        # First pass: collect all possible features to ensure consistency
        all_features_dict = {}

        # Pre-define expected statistical features that all positions should have
        expected_stat_features = [
            'avg_fantasy_points', 'avg_passing_yards', 'avg_rushing_yards', 'avg_receiving_yards',
            'avg_targets', 'avg_pass_tds', 'avg_rush_tds', 'avg_rec_tds', 'avg_interceptions',
            'avg_fumbles', 'avg_rush_attempts', 'avg_receptions', 'yards_per_carry',
            'yards_per_reception', 'catch_rate', 'games_played', 'max_points', 'min_points', 'consistency'
        ]

        # Pre-define weather, betting and other contextual features that all positions should have
        weather_betting_features = [
            'weather_temp', 'weather_wind', 'weather_humidity', 'weather_is_indoor',
            'weather_is_rain', 'weather_is_snow', 'stadium_neutral', 'cold_weather',
            'hot_weather', 'high_wind', 'team_spread', 'team_spread_abs', 'total_line',
            'is_favorite', 'is_big_favorite', 'is_big_underdog', 'expected_pace',
            'team_itt', 'game_tot_z', 'team_itt_z', 'temperature_f', 'wind_mph',
            'humidity_pct', 'cold_lt40', 'hot_gt85', 'wind_gt15', 'dome',
            'injury_status_Out', 'injury_status_Doubtful', 'injury_status_Questionable',
            'injury_status_Probable', 'games_missed_last4', 'practice_trend',
            'returning_from_injury', 'team_injured_starters', 'opp_injured_starters',
            'targets_ema', 'routes_run_ema', 'rush_att_ema', 'snap_share_ema',
            'redzone_opps_ema', 'air_yards_ema', 'adot_ema', 'yprr_ema',
            'yards_after_contact', 'missed_tackles_forced', 'pressure_rate',
            'opp_dvp_pos_allowed', 'salary', 'home', 'rest_days', 'travel', 'season_week',
            'completion_pct_trend', 'yds_per_attempt_trend', 'td_int_ratio_trend', 
            'passer_rating_est', 'passing_volume_trend', 'dual_threat_factor',
            'red_zone_efficiency_est', 'game_script_favorability', 'pressure_situation',
            'ceiling_indicator'
        ]

        # Combine all expected features
        expected_features = expected_stat_features + weather_betting_features

        # Always include all expected features
        for feat in expected_features:
            all_features_dict[feat] = 0.0

        # Sample more comprehensively to get all feature types
        sample_indices = []
        # Sample from beginning, middle, and end to get variety
        sample_indices.extend(range(0, min(50, len(rows)), 5))  # Every 5th in first 50
        if len(rows) > 100:
            mid_start = len(rows) // 2
            sample_indices.extend(range(mid_start, min(mid_start + 50, len(rows)), 5))
        if len(rows) > 200:
            end_start = len(rows) - 50
            sample_indices.extend(range(end_start, len(rows), 5))

        # Remove duplicates and limit sample size
        sample_indices = sorted(list(set(sample_indices)))[:100]

        sampling_progress = ProgressDisplay("Identifying features")

        for idx, i in enumerate(sample_indices):
            sampling_progress.update(idx, len(sample_indices))
            row = rows[i]
            player_id, game_id = row[0], row[1]
            game_date = parse_date_flexible(row[6]).date()
            player_recent_stats = player_stats.get(player_id, [])

            # Get all feature types
            stat_features = compute_features_from_stats(player_recent_stats, game_date, lookback_weeks=4)
            team_id, home_team_id = row[4], row[9]
            home_abbr, away_abbr = row[11], row[12]
            season, week = row[7], row[8]
            opponent_abbr = away_abbr if team_id == home_team_id else home_abbr

            context_features = {'season': season, 'week': week, 'is_home': 1 if team_id == home_team_id else 0}
            def_features = defensive_features_cache.get((opponent_abbr, season, week), {})

            # Add correlation features if available
            try:
                import importlib
                models_module = importlib.import_module('models')
                correlation_extractor = models_module.CorrelationFeatureExtractor(db_path)
                correlation_features = correlation_extractor.extract_correlation_features(
                    player_id, game_id, position
                )
            except:
                correlation_features = {}

            # Collect all unique feature names
            all_features_dict.update(stat_features)
            all_features_dict.update(context_features)
            all_features_dict.update(def_features)
            all_features_dict.update(correlation_features)

        feature_names = sorted(list(all_features_dict.keys()))
        sampling_progress.finish(f"Found {len(feature_names)} total features")

        # Second pass: extract features for each player-game using consistent feature space
        X_list = []
        y_list = []

        feature_progress = ProgressDisplay("Processing samples")

        for row_idx, row in enumerate(rows):
            if row_idx % 100 == 0:
                feature_progress.update(row_idx, len(rows))

            player_id, game_id, fantasy_points = row[0], row[1], row[2]
            game_date = parse_date_flexible(row[6]).date()

            # Extract all feature types
            features = {}

            # Player statistical features
            player_recent_stats = player_stats.get(player_id, [])
            stat_features = compute_features_from_stats(player_recent_stats, game_date, lookback_weeks=4)
            features.update(stat_features)

            # Game context features
            team_id, home_team_id = row[4], row[9]
            home_abbr, away_abbr = row[11], row[12]
            season, week = row[7], row[8]
            opponent_abbr = away_abbr if team_id == home_team_id else home_abbr

            features.update({
                'season': season,
                'week': week,
                'is_home': 1 if team_id == home_team_id else 0
            })

            # Defensive matchup features
            def_features = defensive_features_cache.get((opponent_abbr, season, week), {})
            features.update(def_features)

            # Weather and betting features (using production pipeline logic)
            weather_betting_features = {}

            # Add weather features
            weather_data = conn.execute(
                """SELECT weather_temperature, weather_wind_mph, weather_humidity,
                          weather_detail, stadium_neutral
                   FROM games WHERE id = ?""",
                (game_id,)
            ).fetchone()

            if weather_data:
                temp, wind, humidity, conditions, neutral = weather_data
                weather_betting_features.update({
                    'weather_temp': temp or 72,
                    'weather_wind': wind or 0,
                    'weather_humidity': humidity or 50,
                    'weather_is_indoor': 1 if conditions and 'indoor' in conditions.lower() else 0,
                    'weather_is_rain': 1 if conditions and 'rain' in conditions.lower() else 0,
                    'weather_is_snow': 1 if conditions and 'snow' in conditions.lower() else 0,
                    'stadium_neutral': neutral or 0,
                    'cold_weather': 1 if (temp or 72) < 40 else 0,
                    'hot_weather': 1 if (temp or 72) > 85 else 0,
                    'high_wind': 1 if (wind or 0) > 15 else 0,
                    # Additional weather features matching expected schema
                    'temperature_f': temp or 72,
                    'wind_mph': wind or 0,
                    'humidity_pct': humidity or 50,
                    'cold_lt40': 1 if (temp or 72) < 40 else 0,
                    'hot_gt85': 1 if (temp or 72) > 85 else 0,
                    'wind_gt15': 1 if (wind or 0) > 15 else 0,
                    'dome': 1 if conditions and 'indoor' in conditions.lower() else 0
                })
            else:
                weather_betting_features.update({
                    'weather_temp': 72, 'weather_wind': 0, 'weather_humidity': 50,
                    'weather_is_indoor': 0, 'weather_is_rain': 0, 'weather_is_snow': 0,
                    'stadium_neutral': 0, 'cold_weather': 0, 'hot_weather': 0, 'high_wind': 0,
                    # Additional weather defaults
                    'temperature_f': 72, 'wind_mph': 0, 'humidity_pct': 50,
                    'cold_lt40': 0, 'hot_gt85': 0, 'wind_gt15': 0, 'dome': 0
                })

            # Add betting odds features (prioritize live odds)
            is_home = team_id == home_team_id
            betting_data = conn.execute(
                """SELECT spread_favorite, over_under_line, home_team_spread, away_team_spread
                   FROM betting_odds WHERE game_id = ? ORDER BY
                   CASE WHEN source = 'odds_api' THEN 1 ELSE 2 END
                   LIMIT 1""",
                (game_id,)
            ).fetchone()

            if betting_data:
                spread_fav, over_under, home_spread, away_spread = betting_data
                team_spread = home_spread if is_home else away_spread
                over_under_line = over_under or 45
                weather_betting_features.update({
                    'team_spread': team_spread or 0,
                    'team_spread_abs': abs(team_spread or 0),
                    'total_line': over_under_line,
                    'is_favorite': 1 if (team_spread or 0) < 0 else 0,
                    'is_big_favorite': 1 if (team_spread or 0) < -7 else 0,
                    'is_big_underdog': 1 if (team_spread or 0) > 7 else 0,
                    'expected_pace': over_under_line / 45.0,
                    'team_itt': over_under_line / 2.0 - (team_spread or 0) / 2.0,
                    'game_tot_z': 0.0,  # Will be computed in post-processing
                    'team_itt_z': 0.0   # Will be computed in post-processing
                })
            else:
                weather_betting_features.update({
                    'team_spread': 0, 'team_spread_abs': 0, 'total_line': 45, 'is_favorite': 0,
                    'is_big_favorite': 0, 'is_big_underdog': 0, 'expected_pace': 1.0,
                    'team_itt': 22.5, 'game_tot_z': 0.0, 'team_itt_z': 0.0
                })

            features.update(weather_betting_features)

            # Add injury and contextual features
            try:
                # Get player injury status
                player_injury = conn.execute(
                    """SELECT injury_status FROM players WHERE id = ?""",
                    (player_id,)
                ).fetchone()

                injury_status = (player_injury[0] if player_injury else None) or 'Healthy'

                # One-hot encode injury status
                features['injury_status_Out'] = 1 if injury_status == 'Out' else 0
                features['injury_status_Doubtful'] = 1 if injury_status == 'Doubtful' else 0
                features['injury_status_Questionable'] = 1 if injury_status == 'Questionable' else 0
                features['injury_status_Probable'] = 1 if injury_status in ['Probable', 'Healthy'] else 0

                # Add advanced QB-specific features for better prediction
                qb_features = {}
                
                # Enhanced passing efficiency metrics for QBs
                if position == 'QB':
                    # Use recent averages to compute advanced metrics
                    avg_pass_att = features.get('avg_pass_attempts', 30)
                    avg_pass_comp = features.get('avg_completions', 18)
                    avg_pass_yds = features.get('avg_pass_yards', 250)
                    avg_pass_tds = features.get('avg_pass_tds', 1.5)
                    avg_ints = features.get('avg_interceptions', 0.7)
                    avg_rush_yds = features.get('avg_rush_yards', 15)
                    
                    # Advanced efficiency metrics
                    completion_pct = avg_pass_comp / max(avg_pass_att, 1) 
                    yds_per_att = avg_pass_yds / max(avg_pass_att, 1)
                    td_to_int_ratio = avg_pass_tds / max(avg_ints, 0.1)
                    passer_rating_est = min((completion_pct - 0.3) * 5 + (yds_per_att - 3) * 0.25 + avg_pass_tds * 0.2 - avg_ints * 0.25, 4.0)
                    
                    # Game flow and situational features
                    over_under_line = features.get('total_line', 45)
                    team_spread = features.get('team_spread', 0)
                    expected_pace = over_under_line / 45.0
                    
                    qb_features.update({
                        'completion_pct_trend': completion_pct,
                        'yds_per_attempt_trend': yds_per_att,
                        'td_int_ratio_trend': td_to_int_ratio,
                        'passer_rating_est': max(passer_rating_est, 0.0),
                        'passing_volume_trend': min(avg_pass_att / 35.0, 2.0),
                        'dual_threat_factor': min(avg_rush_yds / 20.0, 1.5),  # Rushing upside
                        'red_zone_efficiency_est': avg_pass_tds / max(avg_pass_att * 0.15, 1),
                        'game_script_favorability': expected_pace * (1.0 if team_spread > -3 else 0.8),
                        'pressure_situation': 1.0 if team_spread < -7 else (0.7 if team_spread > 7 else 1.0),
                        'ceiling_indicator': min(avg_pass_yds + avg_rush_yds + avg_pass_tds * 20, 400) / 400.0
                    })
                
                # Add all contextual features including new QB features
                all_features = {
                    'games_missed_last4': 0, 'practice_trend': 0, 'returning_from_injury': 0,
                    'team_injured_starters': 0, 'opp_injured_starters': 0,
                    'targets_ema': features.get('avg_targets', 0), 'routes_run_ema': 0,
                    'rush_att_ema': features.get('avg_rush_attempts', 0), 'snap_share_ema': 0.7,
                    'redzone_opps_ema': 0, 'air_yards_ema': 0, 'adot_ema': 0, 'yprr_ema': 0,
                    'yards_after_contact': 0, 'missed_tackles_forced': 0, 'pressure_rate': 0,
                    'opp_dvp_pos_allowed': 0, 'salary': 5000, 'home': 1 if is_home else 0,
                    'rest_days': 7, 'travel': 0, 'season_week': week
                }
                all_features.update(qb_features)
                features.update(all_features)

            except Exception as e:
                logger.debug(f"Error adding contextual features for player {player_id}: {e}")
                # Add defaults if query fails
                features.update({
                    'injury_status_Out': 0, 'injury_status_Doubtful': 0, 'injury_status_Questionable': 0,
                    'injury_status_Probable': 1, 'games_missed_last4': 0, 'practice_trend': 0,
                    'returning_from_injury': 0, 'team_injured_starters': 0, 'opp_injured_starters': 0,
                    'targets_ema': 0, 'routes_run_ema': 0, 'rush_att_ema': 0, 'snap_share_ema': 0.7,
                    'redzone_opps_ema': 0, 'air_yards_ema': 0, 'adot_ema': 0, 'yprr_ema': 0,
                    'yards_after_contact': 0, 'missed_tackles_forced': 0, 'pressure_rate': 0,
                    'opp_dvp_pos_allowed': 0, 'salary': 5000, 'home': 1 if is_home else 0,
                    'rest_days': 7, 'travel': 0, 'season_week': week
                })

            # Correlation features
            try:
                import importlib
                models_module = importlib.import_module('models')
                correlation_extractor = models_module.CorrelationFeatureExtractor(db_path)
                correlation_features = correlation_extractor.extract_correlation_features(
                    player_id, game_id, position
                )
                features.update(correlation_features)
            except (ImportError, ModuleNotFoundError, Exception) as e:
                logger.debug(f"Correlation features not available: {e}")
                pass

            if fantasy_points is not None:
                # Use consistent feature order - missing features default to 0
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                X_list.append(feature_vector)
                y_list.append(fantasy_points)

        if not X_list:
            logger.warning(f"No valid features extracted for position {position}")
            return np.array([]), np.array([]), []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        feature_progress.finish(f"Processed {len(rows)} samples")

        # Compute z-scores for betting features by season/week
        logger.info("Computing betting feature z-scores...")

        # Check if we have total_line and team_itt features
        total_line_idx = -1
        team_itt_idx = -1
        game_tot_z_idx = -1
        team_itt_z_idx = -1

        for i, fname in enumerate(feature_names):
            if fname == 'total_line':
                total_line_idx = i
            elif fname == 'team_itt':
                team_itt_idx = i
            elif fname == 'game_tot_z':
                game_tot_z_idx = i
            elif fname == 'team_itt_z':
                team_itt_z_idx = i

        if total_line_idx >= 0 and game_tot_z_idx >= 0:
            # Group samples by season/week and compute z-scores
            week_groups = {}
            for idx, row in enumerate(rows):
                season, week = row[7], row[8]
                key = (season, week)
                if key not in week_groups:
                    week_groups[key] = []
                week_groups[key].append(idx)

            # Compute z-scores for each week
            for (season, week), indices in week_groups.items():
                if len(indices) < 2:
                    continue

                # Total line z-scores
                total_values = X[indices, total_line_idx]
                if np.std(total_values) > 0:
                    mean_val = np.mean(total_values)
                    std_val = np.std(total_values)
                    X[indices, game_tot_z_idx] = (total_values - mean_val) / std_val

                # Team ITT z-scores
                if team_itt_idx >= 0 and team_itt_z_idx >= 0:
                    itt_values = X[indices, team_itt_idx]
                    if np.std(itt_values) > 0:
                        mean_val = np.mean(itt_values)
                        std_val = np.std(itt_values)
                        X[indices, team_itt_z_idx] = (itt_values - mean_val) / std_val

        # Clean data: handle NaNs and remove constant features
        cleaning_progress = ProgressDisplay("Cleaning data")
        cleaning_progress.update(1, 3)  # Step 1 of 3

        # Replace NaNs with 0 (since missing features should be 0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        cleaning_progress.update(2, 3)  # Step 2 of 3

        # Remove constant features (zero variance) but preserve important weather/betting features
        if len(X) > 1:
            feature_variance = np.var(X, axis=0)
            non_constant_mask = feature_variance > 1e-8  # Very small threshold

            # Always preserve weather and betting features even if they appear "constant"
            important_features = ['weather_', 'spread', 'total', 'favorite', 'underdog', 'pace',
                                'cold_weather', 'hot_weather', 'high_wind', 'stadium']

            for i, fname in enumerate(feature_names):
                if any(keyword in fname for keyword in important_features):
                    non_constant_mask[i] = True  # Force keep important features

            if not np.all(non_constant_mask):
                constant_features = [feature_names[i] for i, keep in enumerate(non_constant_mask) if not keep]
                cleaning_progress.finish(f"Removed {len(constant_features)} constant features")

                X = X[:, non_constant_mask]
                feature_names = [feature_names[i] for i, keep in enumerate(non_constant_mask) if keep]
            else:
                cleaning_progress.update(3, 3)  # Final step

        # Apply feature validation if available
        try:
            from utils_feature_validation import validate_and_prepare_features, load_expected_schema
            import pandas as pd

            # Convert to DataFrame for validation
            df = pd.DataFrame(X, columns=feature_names)

            # Load expected schema and validate
            try:
                expected_schema = load_expected_schema("feature_names.json")
                df = validate_and_prepare_features(df, expected_schema, allow_extra=True)

                # Update arrays after validation
                X = df.values.astype(np.float32)
                feature_names = df.columns.tolist()
                pass  # Validation successful
            except Exception as ve:
                logger.warning(f"Feature validation failed, continuing without validation: {ve}")

        except ImportError:
            pass  # Validation not available

        if not cleaning_progress._finished:
            cleaning_progress.finish()

        logger.info(f"Final dataset: {len(X)} training samples for {position} with {len(feature_names)} features")
        return X, y, feature_names

    except Exception as e:
        logger.error(f"Error getting training data for {position}: {e}")
        return np.array([]), np.array([]), []
    finally:
        conn.close()


def compute_features_from_stats(
    player_stats: List[Tuple],
    target_game_date,
    lookback_weeks: int = 4
) -> Dict[str, float]:
    """Compute features from pre-loaded player stats."""
    features = {}

    if not player_stats:
        return features

    # Filter to recent games before target date
    cutoff_date = target_game_date - timedelta(weeks=lookback_weeks)
    recent_stats = [
        stat for stat in player_stats
        if parse_date_flexible(stat[14]).date() >= cutoff_date
        and parse_date_flexible(stat[14]).date() < target_game_date
    ]

    # If no recent stats, expand the window to get any historical data
    if not recent_stats:
        # Try with all available data before the target date
        recent_stats = [
            stat for stat in player_stats
            if parse_date_flexible(stat[14]).date() < target_game_date
        ]
        # Take the most recent games if we have too many
        if len(recent_stats) > 8:
            recent_stats = recent_stats[-8:]  # Take last 8 games

    if not recent_stats:
        return features

    # Extract all stats from enhanced query
    # Schema: player_id(0), game_id(1), fantasy_points(2), passing_yards(3),
    #         rushing_yards(4), receiving_yards(5), targets(6), passing_tds(7),
    #         rushing_tds(8), receiving_tds(9), passing_interceptions(10), fumbles_lost(11),
    #         rushing_attempts(12), receptions(13), game_date(14), season(15), week(16)

    recent_points = [stat[2] for stat in recent_stats if stat[2] is not None]
    recent_pass_yards = [stat[3] for stat in recent_stats if stat[3] is not None]
    recent_rush_yards = [stat[4] for stat in recent_stats if stat[4] is not None]
    recent_rec_yards = [stat[5] for stat in recent_stats if stat[5] is not None]
    recent_targets = [stat[6] for stat in recent_stats if stat[6] is not None]
    recent_pass_tds = [stat[7] for stat in recent_stats if stat[7] is not None]
    recent_rush_tds = [stat[8] for stat in recent_stats if stat[8] is not None]
    recent_rec_tds = [stat[9] for stat in recent_stats if stat[9] is not None]
    recent_interceptions = [stat[10] for stat in recent_stats if stat[10] is not None]
    recent_fumbles = [stat[11] for stat in recent_stats if stat[11] is not None]
    recent_rush_attempts = [stat[12] for stat in recent_stats if stat[12] is not None]
    recent_receptions = [stat[13] for stat in recent_stats if stat[13] is not None]

    # Compute basic averages
    features['avg_fantasy_points'] = np.mean(recent_points) if recent_points else 0
    features['avg_passing_yards'] = np.mean(recent_pass_yards) if recent_pass_yards else 0
    features['avg_rushing_yards'] = np.mean(recent_rush_yards) if recent_rush_yards else 0
    features['avg_receiving_yards'] = np.mean(recent_rec_yards) if recent_rec_yards else 0
    features['avg_targets'] = np.mean(recent_targets) if recent_targets else 0
    features['avg_pass_tds'] = np.mean(recent_pass_tds) if recent_pass_tds else 0
    features['avg_rush_tds'] = np.mean(recent_rush_tds) if recent_rush_tds else 0
    features['avg_rec_tds'] = np.mean(recent_rec_tds) if recent_rec_tds else 0
    features['avg_interceptions'] = np.mean(recent_interceptions) if recent_interceptions else 0
    features['avg_fumbles'] = np.mean(recent_fumbles) if recent_fumbles else 0
    features['avg_rush_attempts'] = np.mean(recent_rush_attempts) if recent_rush_attempts else 0
    features['avg_receptions'] = np.mean(recent_receptions) if recent_receptions else 0

    # Advanced metrics
    if recent_rush_attempts and recent_rush_yards:
        features['yards_per_carry'] = np.mean([y/max(a, 1) for y, a in zip(recent_rush_yards, recent_rush_attempts)])
    else:
        features['yards_per_carry'] = 0

    if recent_receptions and recent_rec_yards:
        features['yards_per_reception'] = np.mean([y/max(r, 1) for y, r in zip(recent_rec_yards, recent_receptions)])
    else:
        features['yards_per_reception'] = 0

    if recent_targets and recent_receptions:
        features['catch_rate'] = np.mean([r/max(t, 1) for r, t in zip(recent_receptions, recent_targets)])
    else:
        features['catch_rate'] = 0

    # Games played and consistency metrics
    features['games_played'] = len(recent_stats)
    if recent_points:
        features['max_points'] = max(recent_points)
        features['min_points'] = min(recent_points)
        features['consistency'] = 1 - (np.std(recent_points) / (np.mean(recent_points) + 1e-6))
    else:
        features['max_points'] = 0
        features['min_points'] = 0
        features['consistency'] = 0

    return features

def get_latest_contest_id(db_path: str = "data/nfl_dfs.db") -> Optional[str]:
    """Get the most recently loaded contest ID."""
    conn = get_db_connection(db_path)
    try:
        result = conn.execute(
            "SELECT contest_id FROM draftkings_salaries ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def get_current_week_players(
    contest_id: str = None,
    db_path: str = "data/nfl_dfs.db"
) -> List[Dict[str, Any]]:
    """Get players available for current week contest."""
    conn = get_db_connection(db_path)

    try:
        # If no contest_id provided, get the latest one
        if contest_id is None:
            contest_id = get_latest_contest_id(db_path)
            if not contest_id:
                logger.error("No DraftKings salary data found")
                return []

        players = conn.execute(
            """SELECT
                dk.player_id,
                p.display_name,
                p.position,
                dk.salary,
                dk.roster_position,
                dk.team_abbr,
                t.team_name
               FROM draftkings_salaries dk
               JOIN players p ON dk.player_id = p.id
               JOIN teams t ON p.team_id = t.id
               WHERE dk.contest_id = ?
               ORDER BY p.position, dk.salary DESC""",
            (contest_id,)
        ).fetchall()

        return [
            {
                'player_id': row[0],
                'name': row[1],
                'position': row[2],
                'salary': row[3],
                'roster_position': row[4],
                'team_abbr': row[5],
                'team_name': row[6]
            }
            for row in players
        ]

    except Exception as e:
        logger.error(f"Error getting current week players: {e}")
        return []
    finally:
        conn.close()

def cleanup_database(db_path: str = "data/nfl_dfs.db") -> None:
    """Clean up database by removing old data."""
    conn = get_db_connection(db_path)

    try:
        # Remove old games (keep last 3 seasons)
        current_year = datetime.now().year
        cutoff_season = current_year - 3

        conn.execute("DELETE FROM games WHERE season < ?", (cutoff_season,))
        conn.execute("DELETE FROM player_stats WHERE game_id NOT IN (SELECT id FROM games)")

        conn.commit()
        logger.info(f"Cleaned up data older than {cutoff_season} season")

    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")
    finally:
        conn.close()

# Simple data validation functions
def validate_data_quality(db_path: str = "data/nfl_dfs.db") -> Dict[str, Any]:
    """Basic data quality checks."""
    conn = get_db_connection(db_path)
    issues = {}

    try:
        # Check for missing fantasy points
        missing_points = conn.execute(
            "SELECT COUNT(*) FROM player_stats WHERE fantasy_points IS NULL"
        ).fetchone()[0]

        if missing_points > 0:
            issues['missing_fantasy_points'] = missing_points

        # Check for players without team
        orphan_players = conn.execute(
            "SELECT COUNT(*) FROM players WHERE team_id IS NULL"
        ).fetchone()[0]

        if orphan_players > 0:
            issues['orphan_players'] = orphan_players

        # Check data freshness
        latest_game = conn.execute(
            "SELECT MAX(game_date) FROM games WHERE game_finished = 1"
        ).fetchone()[0]

        if latest_game:
            latest_date = parse_date_flexible(latest_game).date()
            days_old = (datetime.now().date() - latest_date).days
            # Only flag as stale during active season (Sep-Feb)
            current_month = datetime.now().month
            in_season = current_month >= 9 or current_month <= 2
            threshold = 14 if in_season else 365  # More lenient in off-season
            if days_old > threshold:
                issues['stale_data_days'] = days_old

    except Exception as e:
        issues['validation_error'] = str(e)
    finally:
        conn.close()

    return issues

def import_spreadspoke_data(csv_path: str, db_path: str = "data/nfl_dfs.db") -> None:
    """Import weather and betting data from spreadspoke CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    conn = get_db_connection(db_path)

    # Add weather columns to games table if they don't exist
    try:
        # Check what columns already exist
        columns = conn.execute('PRAGMA table_info(games)').fetchall()
        existing_cols = {col[1] for col in columns}

        weather_columns_to_add = [
            ('stadium', 'TEXT'),
            ('stadium_neutral', 'INTEGER DEFAULT 0'),
            ('weather_temperature', 'INTEGER'),
            ('weather_wind_mph', 'INTEGER'),
            ('weather_humidity', 'INTEGER'),
            ('weather_detail', 'TEXT')
        ]

        for col_name, col_type in weather_columns_to_add:
            if col_name not in existing_cols:
                conn.execute(f'ALTER TABLE games ADD COLUMN {col_name} {col_type}')
                logger.info(f"Added column {col_name} to games table")

        # Create betting_odds table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS betting_odds (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                favorite_team TEXT,
                spread_favorite REAL,
                over_under_line REAL,
                home_team_spread REAL,
                away_team_spread REAL,
                source TEXT DEFAULT 'spreadspoke',
                UNIQUE(game_id),
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')

        conn.commit()

    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        raise

    # Create team abbreviation mapping for different naming conventions
    team_mapping = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Oakland Raiders': 'LV', 'Philadelphia Eagles': 'PHI',
        'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA',
        'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
        'Washington Redskins': 'WAS', 'Washington Football Team': 'WAS'
    }

    # Get team IDs from database
    team_ids = {}
    for row in conn.execute("SELECT id, team_abbr FROM teams").fetchall():
        team_ids[row[1]] = row[0]

    games_inserted = 0
    odds_inserted = 0
    games_updated = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Parse basic game info
                raw_game_date = row['schedule_date']
                # Convert date from M/D/YYYY to YYYY-MM-DD format
                try:
                    parsed_date = parse_date_flexible(raw_game_date)
                    game_date = parsed_date.strftime('%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Could not parse date {raw_game_date}: {e}")
                    continue
                season = int(row['schedule_season'])
                is_playoff = row['schedule_playoff'].upper() == 'TRUE'

                # Handle playoff weeks (Wildcard, Division, Conference, Superbowl)
                week_str = row['schedule_week']
                if is_playoff:
                    week_mapping = {'Wildcard': 18, 'Division': 19, 'Conference': 20, 'Superbowl': 21}
                    week = week_mapping.get(week_str, 18)  # Default to 18 if unknown
                else:
                    week = int(week_str)

                # Map team names to abbreviations
                home_team = team_mapping.get(row['team_home'], row['team_home'])
                away_team = team_mapping.get(row['team_away'], row['team_away'])

                if home_team not in team_ids or away_team not in team_ids:
                    logger.warning(f"Unknown team: {home_team} vs {away_team}")
                    continue

                home_team_id = team_ids[home_team]
                away_team_id = team_ids[away_team]

                # Create game ID
                game_id = f"{season}_{week:02d}_{away_team}_{home_team}"
                if is_playoff:
                    game_id = f"{season}_PO_{week:02d}_{away_team}_{home_team}"

                # Parse scores (empty string if not played yet)
                home_score = int(row['score_home']) if row['score_home'] else None
                away_score = int(row['score_away']) if row['score_away'] else None
                game_finished = 1 if home_score is not None and away_score is not None else 0

                # Parse weather data
                weather_temp = int(row['weather_temperature']) if row['weather_temperature'] else None
                weather_wind = int(row['weather_wind_mph']) if row['weather_wind_mph'] else None
                weather_humidity = int(row['weather_humidity']) if row['weather_humidity'] else None
                weather_detail = row['weather_detail'] if row['weather_detail'] else None

                # Parse betting data
                favorite_team = team_mapping.get(row['team_favorite_id'], row['team_favorite_id']) if row['team_favorite_id'] else None
                spread_favorite = float(row['spread_favorite']) if row['spread_favorite'] else None
                over_under = float(row['over_under_line']) if row['over_under_line'] else None

                # Calculate individual team spreads
                home_spread = None
                away_spread = None
                if spread_favorite and favorite_team:
                    if favorite_team == home_team:
                        home_spread = spread_favorite  # negative for favorite
                        away_spread = -spread_favorite  # positive for underdog
                    elif favorite_team == away_team:
                        away_spread = spread_favorite  # negative for favorite
                        home_spread = -spread_favorite  # positive for underdog

                stadium = row['stadium'] if row['stadium'] else None
                stadium_neutral = 1 if row['stadium_neutral'].upper() == 'TRUE' else 0

                try:
                    # Check if game already exists
                    existing_game = conn.execute('SELECT id FROM games WHERE id = ?', (game_id,)).fetchone()

                    if existing_game:
                        # Update existing game with weather/stadium data only
                        conn.execute('''
                            UPDATE games SET
                                stadium = ?, stadium_neutral = ?,
                                weather_temperature = ?, weather_wind_mph = ?,
                                weather_humidity = ?, weather_detail = ?
                            WHERE id = ?
                        ''', (
                            stadium, stadium_neutral,
                            weather_temp, weather_wind, weather_humidity, weather_detail,
                            game_id
                        ))
                        games_updated += 1
                    else:
                        # Insert new game (if it doesn't exist in your data)
                        conn.execute('''
                            INSERT INTO games (
                                id, game_date, season, week, home_team_id, away_team_id,
                                home_score, away_score, game_finished, stadium, stadium_neutral,
                                weather_temperature, weather_wind_mph, weather_humidity, weather_detail
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_id, game_date, season, week, home_team_id, away_team_id,
                            home_score, away_score, game_finished, stadium, stadium_neutral,
                            weather_temp, weather_wind, weather_humidity, weather_detail
                        ))
                        games_inserted += 1

                    # Insert betting odds if available
                    if spread_favorite or over_under:
                        conn.execute('''
                            INSERT OR REPLACE INTO betting_odds (
                                game_id, favorite_team, spread_favorite, over_under_line,
                                home_team_spread, away_team_spread, source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_id, favorite_team, spread_favorite, over_under,
                            home_spread, away_spread, 'spreadspoke'
                        ))
                        odds_inserted += 1

                except Exception as e:
                    logger.error(f"Error inserting game {game_id}: {e}")
                    continue

        conn.commit()
        logger.info(f"Spreadspoke import complete: {games_inserted} games inserted, {games_updated} games updated, {odds_inserted} betting records inserted")

    except Exception as e:
        logger.error(f"Error importing spreadspoke data: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def collect_odds_data(target_date: str = None, db_path: str = "data/nfl_dfs.db") -> None:
    """Collect NFL betting odds from The Odds API for upcoming games.

    Args:
        target_date: Date in YYYY-MM-DD format. If None, collects all upcoming games.
        db_path: Path to the SQLite database
    """
    odds_api_key = os.getenv('ODDS_API_KEY')
    if not odds_api_key or odds_api_key == 'your_key_here':
        raise ValueError("ODDS_API_KEY environment variable not set. Please set it in your .env file.")

    conn = get_db_connection(db_path)

    try:
        # API endpoint for NFL odds
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            'apiKey': odds_api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        logger.info("Fetching NFL odds from The Odds API...")
        response = requests.get(url, params=params)
        response.raise_for_status()

        odds_data = response.json()
        logger.info(f"Retrieved {len(odds_data)} games with odds data")

        odds_inserted = 0

        for game in odds_data:
            game_id = game['id']
            commence_time = game['commence_time']
            home_team = game['home_team']
            away_team = game['away_team']

            # Parse the commence time to check if it matches target date
            game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00')).date()
            if target_date:
                target_date_obj = parse_date_flexible(target_date).date()
                if game_date != target_date_obj:
                    continue

            # Find matching game in our database (check both games table and draftkings_salaries)
            db_game = conn.execute("""
                SELECT id FROM games
                WHERE (home_team_id = (SELECT id FROM teams WHERE team_abbr = ?)
                       OR home_team_id = (SELECT id FROM teams WHERE team_name = ?))
                  AND (away_team_id = (SELECT id FROM teams WHERE team_abbr = ?)
                       OR away_team_id = (SELECT id FROM teams WHERE team_name = ?))
                  AND date(game_date) = date(?)
            """, (home_team, home_team, away_team, away_team, commence_time[:10])).fetchone()

            # If not found in games table, check draftkings_salaries table for upcoming games
            if not db_game:
                # Convert team names to abbreviations for DraftKings format matching
                away_abbr = conn.execute("SELECT team_abbr FROM teams WHERE team_name = ? OR team_abbr = ?", (away_team, away_team)).fetchone()
                home_abbr = conn.execute("SELECT team_abbr FROM teams WHERE team_name = ? OR team_abbr = ?", (home_team, home_team)).fetchone()

                if away_abbr and home_abbr:
                    away_abbr = away_abbr[0]
                    home_abbr = home_abbr[0]

                    # Format date to match DraftKings format (MM/DD/YYYY)
                    dk_date = game_date.strftime("%m/%d/%Y")

                    # Check if we have DraftKings data for this matchup - format: "AWAY@HOME MM/DD/YYYY"
                    dk_game = conn.execute("""
                        SELECT DISTINCT game_info FROM draftkings_salaries
                        WHERE game_info LIKE ?
                    """, (f"{away_abbr}@{home_abbr} {dk_date}%",)).fetchone()

                    if dk_game:
                        # Use a meaningful game_id for upcoming games
                        db_game_id = f"{game_date}_{away_abbr}@{home_abbr}"
                        logger.info(f"Found upcoming game in DraftKings data: {away_abbr}@{home_abbr} on {dk_date}")

                        # Create minimal game record for foreign key constraint
                        away_team_id = conn.execute("SELECT id FROM teams WHERE team_abbr = ?", (away_abbr,)).fetchone()
                        home_team_id = conn.execute("SELECT id FROM teams WHERE team_abbr = ?", (home_abbr,)).fetchone()

                        if away_team_id and home_team_id:
                            away_team_id = away_team_id[0]
                            home_team_id = home_team_id[0]

                            # Insert minimal game record if it doesn't exist
                            conn.execute("""
                                INSERT OR IGNORE INTO games (
                                    id, game_date, season, week, home_team_id, away_team_id,
                                    home_score, away_score, game_finished
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (db_game_id, game_date.strftime('%Y-%m-%d'), game_date.year, 1,
                                  home_team_id, away_team_id, 0, 0, 0))
                        else:
                            logger.warning(f"Could not find team IDs for {away_abbr}/{home_abbr}")
                            continue
                    else:
                        logger.warning(f"No matching DraftKings game found for {away_abbr}@{home_abbr} on {dk_date}")
                        continue
                else:
                    logger.warning(f"Could not find team abbreviations for {away_team} / {home_team}")
                    continue
            else:
                db_game_id = db_game[0]

            # Extract odds data
            spread_favorite = None
            spread_fav_team = None
            over_under = None
            home_spread = None
            away_spread = None

            # Process bookmaker odds (use consensus or first available)
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]  # Use first bookmaker

                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                home_spread = float(outcome['point'])
                            elif outcome['name'] == away_team:
                                away_spread = float(outcome['point'])

                        # Determine favorite
                        if home_spread is not None and away_spread is not None:
                            if home_spread < away_spread:
                                spread_favorite = abs(home_spread)
                                spread_fav_team = home_team
                            else:
                                spread_favorite = abs(away_spread)
                                spread_fav_team = away_team

                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                over_under = float(outcome['point'])
                                break

            # Insert or update betting odds
            conn.execute("""
                INSERT OR REPLACE INTO betting_odds (
                    game_id, favorite_team, spread_favorite, over_under_line,
                    home_team_spread, away_team_spread, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (db_game_id, spread_fav_team, spread_favorite, over_under,
                  home_spread, away_spread, 'odds_api'))

            odds_inserted += 1
            logger.info(f"Processed odds for {away_team} @ {home_team}: spread={spread_favorite}, o/u={over_under}")

        conn.commit()
        logger.info(f"Odds collection complete: {odds_inserted} records processed")

        # Check remaining API quota
        remaining_requests = response.headers.get('x-requests-remaining')
        if remaining_requests:
            logger.info(f"Remaining API requests: {remaining_requests}")

    except requests.RequestException as e:
        logger.error(f"Error fetching odds from API: {e}")
        conn.rollback()
        raise
    except Exception as e:
        logger.error(f"Error processing odds data: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def collect_injury_data(seasons: List[int] = None, db_path: str = "data/nfl_dfs.db") -> None:
    """Collect NFL injury data from nfl_data_py.

    Args:
        seasons: List of seasons to collect injury data for. If None, collects current season.
        db_path: Path to the SQLite database
    """
    if nfl is None:
        logger.error("nfl_data_py not available. Cannot collect injury data.")
        return

    if seasons is None:
        current_year = datetime.now().year
        # NFL season spans two calendar years, so check if we're in NFL season
        if datetime.now().month >= 9:  # September onwards is current NFL season
            seasons = [current_year]
        else:  # January-August is previous NFL season
            seasons = [current_year - 1]

    conn = get_db_connection(db_path)

    try:
        # Ensure injury_status column exists in players table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'injury_status' not in columns:
            logger.info("Adding injury_status column to players table...")
            cursor.execute("ALTER TABLE players ADD COLUMN injury_status TEXT DEFAULT NULL")
            conn.commit()

        total_updates = 0

        for season in seasons:
            try:
                logger.info(f"Fetching injury data for season {season}...")

                # Import injury data from nfl_data_py
                injury_df = nfl.import_injuries([season])

                if injury_df is None or injury_df.empty:
                    logger.warning(f"No injury data found for season {season}")
                    continue

                injury_progress = ProgressDisplay(f"Processing injury data {season}")
                season_updates = 0

                for idx, injury_row in injury_df.iterrows():
                    if idx % 100 == 0:
                        injury_progress.update(idx, len(injury_df))
                    try:
                        # Get injury status and handle None values
                        nfl_status = injury_row.get('report_status')
                        if not nfl_status or pd.isna(nfl_status):
                            continue

                        nfl_status = str(nfl_status).upper()

                        # Map NFL injury statuses to our standardized codes
                        status_mapping = {
                            'OUT': 'OUT',
                            'DOUBTFUL': 'D',
                            'QUESTIONABLE': 'Q',
                            'PROBABLE': 'P',
                            'NOTE': None,  # Skip these
                            'INJURED_RESERVE': 'IR',
                            'PHYSICALLY_UNABLE_TO_PERFORM': 'PUP',
                            'NON_FOOTBALL_INJURY': 'NFI',
                            'SUSPENSION': 'SUSP',
                            'RESERVE_COVID_19': 'COV',
                            'PRACTICE_SQUAD_INJURED': 'PS-INJ'
                        }

                        injury_status = status_mapping.get(nfl_status)
                        if injury_status is None:
                            if nfl_status == 'NOTE':
                                continue  # Skip notes
                            # If status not in mapping, use original if it's short enough
                            injury_status = nfl_status[:10] if len(nfl_status) <= 10 else None

                        if not injury_status:
                            continue

                        # Get player identifiers
                        gsis_id = injury_row.get('gsis_id')
                        player_name = injury_row.get('full_name', '')
                        team_abbr = injury_row.get('team')

                        if not gsis_id:
                            continue

                        # Try to find player by GSIS ID first (most reliable)
                        player_record = conn.execute(
                            "SELECT id, player_name FROM players WHERE gsis_id = ?",
                            (gsis_id,)
                        ).fetchone()

                        # If not found by GSIS ID, try name and team
                        if not player_record and player_name and team_abbr:
                            # Get team ID
                            team_record = conn.execute(
                                "SELECT id FROM teams WHERE team_abbr = ?",
                                (team_abbr,)
                            ).fetchone()

                            if team_record:
                                team_id = team_record[0]
                                # Try both player_name and display_name columns
                                player_record = conn.execute(
                                    """SELECT id, player_name FROM players
                                       WHERE (player_name LIKE ? OR display_name LIKE ?) AND team_id = ?""",
                                    (f"%{player_name}%", f"%{player_name}%", team_id)
                                ).fetchone()

                        if player_record:
                            player_id = player_record[0]

                            # Update injury status
                            conn.execute(
                                "UPDATE players SET injury_status = ? WHERE id = ?",
                                (injury_status, player_id)
                            )
                            season_updates += 1

                    except Exception as e:
                        logger.warning(f"Error processing injury record: {e}")
                        continue

                injury_progress.finish(f"Completed season {season}: {season_updates} updates")
                conn.commit()
                total_updates += season_updates

            except Exception as e:
                logger.error(f"Error collecting injury data for season {season}: {e}")
                continue

        logger.info(f"Injury data collection complete. Total updates: {total_updates}")

    except Exception as e:
        logger.error(f"Error collecting injury data: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def backtest_production_pipeline(
    position: str,
    test_season: int = 2023,
    db_path: str = "data/nfl_dfs.db"
) -> Dict[str, float]:
    """Backtest model using exact production pipeline for accurate performance metrics."""
    from models import create_model
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score

    conn = get_db_connection(db_path)

    try:
        # Get test data (last season's QB performances)
        test_query = """
            SELECT ps.player_id, ps.game_id, ps.fantasy_points, p.player_name
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE p.position = ? AND g.season = ? AND ps.fantasy_points > 0
            ORDER BY g.game_date
        """

        test_data = conn.execute(test_query, (position, test_season)).fetchall()
        logger.info(f"Backtesting {len(test_data)} {position} performances from {test_season}")

        if not test_data:
            return {"error": "No test data found"}

        # Load trained model
        model_path = f"models/{position.lower()}_model.pth"
        if not Path(model_path).exists():
            return {"error": f"No trained model found: {model_path}"}

        model = create_model(position)

        # Get expected feature count from training data
        X_train, _, feature_names = get_training_data(position, [2022, 2023], db_path)
        expected_features = len(feature_names)

        model.load_model(model_path, expected_features)

        predictions = []
        actuals = []
        failed_predictions = 0

        pred_progress = ProgressDisplay("Generating predictions")
        for i, (player_id, game_id, actual_points, player_name) in enumerate(test_data):
            if i % 50 == 0:
                pred_progress.update(i, len(test_data))

            try:
                # Use EXACT production pipeline
                features = get_player_features(player_id, game_id, db_path=db_path)

                if not features:
                    failed_predictions += 1
                    continue

                # Convert features to model input (same as production)
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                X = np.array([feature_vector], dtype=np.float32)

                # Handle NaN values (same as training)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                if np.isnan(X).any():
                    failed_predictions += 1
                    continue

                # Make prediction using production model
                pred_result = model.predict(X)
                pred_points = pred_result.point_estimate[0]

                if not np.isnan(pred_points) and not np.isinf(pred_points):
                    predictions.append(pred_points)
                    actuals.append(actual_points)
                else:
                    failed_predictions += 1

            except Exception as e:
                failed_predictions += 1
                if failed_predictions <= 5:  # Log first few failures
                    logger.debug(f"Prediction failed for {player_name} in {game_id}: {e}")

        pred_progress.finish(f"Generated {len(predictions)} predictions")

        if len(predictions) < 10:
            return {"error": f"Too few valid predictions: {len(predictions)}"}

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # Additional metrics
        mean_error = np.mean(predictions - actuals)  # Bias
        std_error = np.std(predictions - actuals)    # Consistency

        logger.info(f"Production backtesting complete: {len(predictions)} valid predictions")

        return {
            "mae": mae,
            "r2": r2,
            "mean_error": mean_error,
            "std_error": std_error,
            "valid_predictions": len(predictions),
            "failed_predictions": failed_predictions,
            "success_rate": len(predictions) / len(test_data)
        }

    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return {"error": str(e)}
    finally:
        conn.close()
