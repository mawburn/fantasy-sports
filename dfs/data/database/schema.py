"""Database schema definitions."""

import sqlite3
from dfs.core.logging import get_logger
from dfs.core.exceptions import DataError

logger = get_logger("data.database.schema")

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
            passing_2pt_conversions INTEGER DEFAULT 0,
            rushing_2pt_conversions INTEGER DEFAULT 0,
            receiving_2pt_conversions INTEGER DEFAULT 0,
            special_teams_tds INTEGER DEFAULT 0,
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
    'historical_ownership': '''
        CREATE TABLE IF NOT EXISTS historical_ownership (
            id INTEGER PRIMARY KEY,
            contest_date TEXT,
            slate_type TEXT,
            player_id INTEGER,
            actual_ownership REAL,
            projected_ownership REAL,
            salary INTEGER,
            projected_points REAL,
            actual_points REAL,
            contest_entries INTEGER,
            FOREIGN KEY (player_id) REFERENCES players (id),
            UNIQUE(contest_date, slate_type, player_id)
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
    ''',
    'dfs_scores': '''
        CREATE TABLE IF NOT EXISTS dfs_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            position TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            game_id TEXT,
            dfs_points REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (team_id) REFERENCES teams (id),
            FOREIGN KEY (opponent_id) REFERENCES teams (id),
            FOREIGN KEY (game_id) REFERENCES games (id),
            UNIQUE(player_id, season, week)
        )
    '''
}


def create_tables(connection: sqlite3.Connection) -> None:
    """Create all database tables."""
    try:
        for table_name, schema in DB_SCHEMA.items():
            connection.execute(schema)
            logger.debug(f"Created table: {table_name}")
        
        connection.commit()
        logger.info("Database schema created successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Failed to create database schema: {e}")
        raise DataError(f"Schema creation failed: {e}")


def get_table_info(connection: sqlite3.Connection, table_name: str) -> list:
    """Get table schema information."""
    try:
        cursor = connection.execute(f"PRAGMA table_info({table_name})")
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Failed to get table info for {table_name}: {e}")
        raise DataError(f"Table info query failed: {e}")


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if table exists in database."""
    try:
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Failed to check table existence: {e}")
        return False