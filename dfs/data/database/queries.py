"""Common database queries."""

import sqlite3
from typing import List, Dict, Optional, Any
from dfs.core.logging import get_logger
from dfs.core.exceptions import DataError

logger = get_logger("data.database.queries")


def execute_query(
    connection: sqlite3.Connection,
    query: str, 
    params: Optional[tuple] = None
) -> sqlite3.Cursor:
    """Execute a query and return cursor."""
    try:
        return connection.execute(query, params or ())
    except sqlite3.Error as e:
        logger.error(f"Query failed: {query[:100]}... Error: {e}")
        raise DataError(f"Database query failed: {e}")


def fetch_one(
    connection: sqlite3.Connection,
    query: str, 
    params: Optional[tuple] = None
) -> Optional[sqlite3.Row]:
    """Execute query and fetch one result."""
    cursor = execute_query(connection, query, params)
    return cursor.fetchone()


def fetch_all(
    connection: sqlite3.Connection,
    query: str, 
    params: Optional[tuple] = None
) -> List[sqlite3.Row]:
    """Execute query and fetch all results."""
    cursor = execute_query(connection, query, params)
    return cursor.fetchall()


def get_available_seasons(connection: sqlite3.Connection) -> List[int]:
    """Get all seasons available in the database."""
    query = "SELECT DISTINCT season FROM games ORDER BY season"
    rows = fetch_all(connection, query)
    return [row[0] for row in rows]


def get_player_by_name(
    connection: sqlite3.Connection,
    player_name: str,
    team_abbr: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """Find player by name, optionally filtered by team."""
    if team_abbr:
        query = """
            SELECT p.*, t.team_abbr 
            FROM players p
            JOIN teams t ON p.team_id = t.id
            WHERE (p.player_name = ? OR p.display_name = ?)
            AND t.team_abbr = ?
        """
        params = (player_name, player_name, team_abbr)
    else:
        query = """
            SELECT p.*, t.team_abbr 
            FROM players p
            JOIN teams t ON p.team_id = t.id
            WHERE p.player_name = ? OR p.display_name = ?
        """
        params = (player_name, player_name)
    
    return fetch_one(connection, query, params)


def get_team_by_abbr(
    connection: sqlite3.Connection,
    team_abbr: str
) -> Optional[sqlite3.Row]:
    """Get team by abbreviation."""
    query = "SELECT * FROM teams WHERE team_abbr = ?"
    return fetch_one(connection, query, (team_abbr,))


def get_recent_games(
    connection: sqlite3.Connection,
    player_id: int,
    limit: int = 10
) -> List[sqlite3.Row]:
    """Get recent games for a player."""
    query = """
        SELECT ps.*, g.game_date, g.season, g.week
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        WHERE ps.player_id = ?
        ORDER BY g.game_date DESC
        LIMIT ?
    """
    return fetch_all(connection, query, (player_id, limit))


def get_player_season_stats(
    connection: sqlite3.Connection,
    player_id: int,
    season: int
) -> List[sqlite3.Row]:
    """Get all stats for a player in a specific season."""
    query = """
        SELECT ps.*, g.week, g.game_date
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        WHERE ps.player_id = ? AND g.season = ?
        ORDER BY g.week
    """
    return fetch_all(connection, query, (player_id, season))


def get_team_schedule(
    connection: sqlite3.Connection,
    team_id: int,
    season: int
) -> List[sqlite3.Row]:
    """Get team's schedule for a season."""
    query = """
        SELECT g.*, 
               ht.team_abbr as home_team,
               at.team_abbr as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE (g.home_team_id = ? OR g.away_team_id = ?)
        AND g.season = ?
        ORDER BY g.week
    """
    return fetch_all(connection, query, (team_id, team_id, season))


def get_upcoming_games(
    connection: sqlite3.Connection,
    limit: int = 20
) -> List[sqlite3.Row]:
    """Get upcoming games that haven't finished."""
    query = """
        SELECT g.*,
               ht.team_abbr as home_team,
               at.team_abbr as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.game_finished = 0
        ORDER BY g.game_date
        LIMIT ?
    """
    return fetch_all(connection, query, (limit,))


def get_dst_stats(
    connection: sqlite3.Connection,
    team_abbr: str,
    season: int,
    weeks: Optional[List[int]] = None
) -> List[sqlite3.Row]:
    """Get DST stats for a team."""
    if weeks:
        placeholders = ','.join('?' * len(weeks))
        query = f"""
            SELECT * FROM dst_stats
            WHERE team_abbr = ? AND season = ? AND week IN ({placeholders})
            ORDER BY week
        """
        params = [team_abbr, season] + weeks
    else:
        query = """
            SELECT * FROM dst_stats
            WHERE team_abbr = ? AND season = ?
            ORDER BY week
        """
        params = [team_abbr, season]
    
    return fetch_all(connection, query, tuple(params))


def get_betting_odds(
    connection: sqlite3.Connection,
    game_id: str
) -> Optional[sqlite3.Row]:
    """Get betting odds for a game."""
    query = "SELECT * FROM betting_odds WHERE game_id = ?"
    return fetch_one(connection, query, (game_id,))