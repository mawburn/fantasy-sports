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
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError:
    print("Warning: nfl_data_py not available. Install with: pip install nfl_data_py")
    nfl = None

logger = logging.getLogger(__name__)

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
            game_finished INTEGER DEFAULT 0
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
            FOREIGN KEY (team_id) REFERENCES teams (id)
        )
    ''',
    'player_stats': '''
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            game_id INTEGER,
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
            FOREIGN KEY (game_id) REFERENCES games (id)
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

                # Use provided contest_id or generate one from filename/date
                if contest_id is None:
                    from pathlib import Path
                    from datetime import datetime
                    filename = Path(csv_path).stem
                    date_str = datetime.now().strftime('%Y%m%d')
                    contest_id = f"{filename}_{date_str}"

                # Handle DST/Defense teams specially
                if roster_position == 'DST':
                    # Create or find defense "player" entry
                    player_id = get_or_create_defense_player(team_abbr, conn)
                else:
                    # Try to find matching individual player
                    player_id = find_player_by_name_and_team(player_name, team_abbr, conn)

                if player_id:
                    conn.execute(
                        """INSERT OR REPLACE INTO draftkings_salaries
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

        # Get game info
        game_info = conn.execute(
            "SELECT game_date, season, week FROM games WHERE id = ?",
            (game_id,)
        ).fetchone()

        if not game_info:
            return features

        game_date, season, week = game_info

        # Parse game date
        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
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

        # Add team and opponent features
        features['home_game'] = 1 if is_home_game(team_id, game_id, conn) else 0
        features['week'] = week
        features['season'] = season

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

def get_training_data(
    position: str,
    seasons: List[int],
    db_path: str = "data/nfl_dfs.db"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get training data for a specific position."""
    conn = get_db_connection(db_path)

    try:
        # Get all player-game combinations for the position
        data_query = """
            SELECT ps.player_id, ps.game_id, ps.fantasy_points
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE p.position = ? AND g.season IN ({})
            AND g.game_finished = 1
            ORDER BY g.game_date
        """.format(','.join('?' * len(seasons)))

        cursor = conn.execute(data_query, [position] + seasons)
        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"No training data found for position {position}")
            return np.array([]), np.array([]), []

        # Extract features for each player-game
        X_list = []
        y_list = []
        feature_names = None

        for player_id, game_id, fantasy_points in rows:
            features = get_player_features(player_id, game_id, db_path=db_path)

            if features and fantasy_points is not None:
                if feature_names is None:
                    feature_names = list(features.keys())

                # Ensure consistent feature order
                feature_vector = [features.get(name, 0) for name in feature_names]
                X_list.append(feature_vector)
                y_list.append(fantasy_points)

        if not X_list:
            logger.warning(f"No valid features extracted for position {position}")
            return np.array([]), np.array([]), []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(f"Extracted {len(X)} training samples for {position}")
        return X, y, feature_names

    except Exception as e:
        logger.error(f"Error getting training data for {position}: {e}")
        return np.array([]), np.array([]), []
    finally:
        conn.close()

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
            latest_date = datetime.strptime(latest_game, '%Y-%m-%d').date()
            days_old = (datetime.now().date() - latest_date).days
            if days_old > 14:
                issues['stale_data_days'] = days_old

    except Exception as e:
        issues['validation_error'] = str(e)
    finally:
        conn.close()

    return issues
