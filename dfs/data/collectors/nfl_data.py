"""NFL data collection using nfl_data_py."""

import logging
from typing import List
import pandas as pd

from dfs.core.logging import get_logger
from dfs.core.exceptions import DataError
from dfs.data.database.manager import get_db_connection
from dfs.data.database.schema import create_tables

logger = get_logger("data.collectors.nfl")

try:
    import nfl_data_py as nfl
except ImportError:
    logger.error("nfl_data_py not available. Install with: pip install nfl_data_py")
    nfl = None


def collect_nfl_data(seasons: List[int], db_path: str = None):
    """Collect NFL data for specified seasons.
    
    Args:
        seasons: List of seasons to collect (e.g., [2022, 2023])
        db_path: Optional database path override
    """
    if nfl is None:
        raise DataError("nfl_data_py is required for data collection")
    
    logger.info(f"Starting NFL data collection for seasons: {seasons}")
    conn = get_db_connection(db_path)
    
    try:
        # Create tables
        create_tables(conn)
        
        # Collect teams data
        _collect_teams(conn)
        
        # Collect games and stats for each season
        for season in seasons:
            logger.info(f"Collecting data for {season} season...")
            _collect_games(conn, season)
            _collect_player_stats(conn, season)
            _collect_dst_stats(conn, season)
            _collect_play_by_play(conn, season)
        
        conn.commit()
        logger.info(f"Successfully collected NFL data for seasons: {seasons}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to collect NFL data: {e}")
        raise DataError(f"NFL data collection failed: {e}")
    finally:
        conn.close()


def _collect_teams(conn):
    """Collect and store team data."""
    logger.info("Collecting team data...")
    
    try:
        teams_df = nfl.import_team_desc()
        
        for _, team in teams_df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO teams (id, team_abbr, team_name, division, conference)
                VALUES (?, ?, ?, ?, ?)
            """, (
                team.get('team_id', 0),
                team.get('team_abbr', ''),
                team.get('team_name', ''),
                team.get('team_division', ''),
                team.get('team_conf', '')
            ))
        
        logger.info(f"Stored {len(teams_df)} teams")
        
    except Exception as e:
        logger.error(f"Failed to collect teams: {e}")
        raise


def _collect_games(conn, season: int):
    """Collect and store games data for a season."""
    logger.info(f"Collecting games for {season}...")
    
    try:
        games_df = nfl.import_schedules([season])
        
        for _, game in games_df.iterrows():
            # Get team IDs
            home_team_id = _get_team_id(conn, game.get('home_team', ''))
            away_team_id = _get_team_id(conn, game.get('away_team', ''))
            
            conn.execute("""
                INSERT OR REPLACE INTO games (
                    id, game_date, season, week, home_team_id, away_team_id,
                    home_score, away_score, game_finished, stadium
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game.get('game_id', ''),
                game.get('gameday', ''),
                season,
                game.get('week', 0),
                home_team_id,
                away_team_id,
                game.get('home_score', 0),
                game.get('away_score', 0),
                1 if game.get('home_score') is not None else 0,
                game.get('stadium', '')
            ))
        
        logger.info(f"Stored {len(games_df)} games for {season}")
        
    except Exception as e:
        logger.error(f"Failed to collect games for {season}: {e}")
        raise


def _collect_player_stats(conn, season: int):
    """Collect and store player stats for a season."""
    logger.info(f"Collecting player stats for {season}...")
    
    try:
        # Collect weekly stats
        weekly_df = nfl.import_weekly_data([season])
        
        # Process each player's stats
        for _, stat in weekly_df.iterrows():
            player_id = _get_or_create_player(conn, stat)
            
            if player_id:
                conn.execute("""
                    INSERT OR REPLACE INTO player_stats (
                        player_id, game_id, passing_yards, passing_tds, passing_interceptions,
                        rushing_yards, rushing_attempts, rushing_tds,
                        receiving_yards, targets, receptions, receiving_tds,
                        fumbles_lost, fantasy_points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_id,
                    stat.get('game_id', ''),
                    stat.get('passing_yards', 0),
                    stat.get('passing_tds', 0),
                    stat.get('interceptions', 0),
                    stat.get('rushing_yards', 0),
                    stat.get('rushing_attempts', 0),
                    stat.get('rushing_tds', 0),
                    stat.get('receiving_yards', 0),
                    stat.get('targets', 0),
                    stat.get('receptions', 0),
                    stat.get('receiving_tds', 0),
                    stat.get('fumbles_lost', 0),
                    stat.get('fantasy_points', 0)
                ))
        
        logger.info(f"Stored player stats for {season}")
        
    except Exception as e:
        logger.error(f"Failed to collect player stats for {season}: {e}")
        raise


def _collect_dst_stats(conn, season: int):
    """Collect and store DST stats for a season."""
    logger.info(f"Collecting DST stats for {season}...")
    
    try:
        # This would use defensive stats from nfl_data_py
        # Implementation depends on available data structure
        pass
        
    except Exception as e:
        logger.error(f"Failed to collect DST stats for {season}: {e}")
        raise


def _collect_play_by_play(conn, season: int):
    """Collect and store play-by-play data for a season."""
    logger.info(f"Collecting play-by-play for {season}...")
    
    try:
        pbp_df = nfl.import_pbp_data([season])
        
        # Sample and store key plays (not all plays due to volume)
        key_plays = pbp_df[
            (pbp_df['play_type'].isin(['pass', 'run'])) &
            (pbp_df['down'].isin([1, 2, 3, 4]))
        ].sample(n=min(50000, len(pbp_df)), random_state=42)
        
        for _, play in key_plays.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO play_by_play (
                    play_id, game_id, season, week, home_team, away_team,
                    posteam, defteam, play_type, description, down, ydstogo,
                    yardline_100, yards_gained, touchdown, pass_attempt,
                    rush_attempt, complete_pass, interception, fumble, sack
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                play.get('play_id', ''),
                play.get('game_id', ''),
                season,
                play.get('week', 0),
                play.get('home_team', ''),
                play.get('away_team', ''),
                play.get('posteam', ''),
                play.get('defteam', ''),
                play.get('play_type', ''),
                play.get('desc', ''),
                play.get('down', 0),
                play.get('ydstogo', 0),
                play.get('yardline_100', 0),
                play.get('yards_gained', 0),
                play.get('touchdown', 0),
                play.get('pass_attempt', 0),
                play.get('rush_attempt', 0),
                play.get('complete_pass', 0),
                play.get('interception', 0),
                play.get('fumble', 0),
                play.get('sack', 0)
            ))
        
        logger.info(f"Stored {len(key_plays)} key plays for {season}")
        
    except Exception as e:
        logger.error(f"Failed to collect play-by-play for {season}: {e}")
        raise


def _get_team_id(conn, team_abbr: str) -> int:
    """Get team ID by abbreviation."""
    result = conn.execute(
        "SELECT id FROM teams WHERE team_abbr = ?", 
        (team_abbr,)
    ).fetchone()
    return result[0] if result else 0


def _get_or_create_player(conn, stat_row) -> int:
    """Get or create player and return ID."""
    player_name = stat_row.get('player_name', '')
    display_name = stat_row.get('player_display_name', player_name)
    position = stat_row.get('position', '')
    team_abbr = stat_row.get('recent_team', '')
    
    if not player_name:
        return None
    
    # Try to find existing player
    existing = conn.execute("""
        SELECT id FROM players WHERE player_name = ? OR display_name = ?
    """, (player_name, display_name)).fetchone()
    
    if existing:
        return existing[0]
    
    # Create new player
    team_id = _get_team_id(conn, team_abbr)
    
    cursor = conn.execute("""
        INSERT INTO players (player_name, display_name, position, team_id, gsis_id)
        VALUES (?, ?, ?, ?, ?)
    """, (
        player_name,
        display_name,
        position,
        team_id,
        stat_row.get('player_id', '')
    ))
    
    return cursor.lastrowid