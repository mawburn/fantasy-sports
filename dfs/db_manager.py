"""Optimized database access layer for NFL DFS system.

Provides:
1. Connection pooling with SQLite optimizations
2. Bulk loading methods for training data
3. Cached queries using LRU cache
4. Batch operations for inserts/updates
5. Index management for query optimization
6. Database schema definitions and migrations

Designed to replace direct SQL queries in data.py with optimized batch operations.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from helpers import ProgressDisplay

logger = logging.getLogger(__name__)

# Database schema definitions
DB_SCHEMA = {
    "games": """
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
    """,
    "teams": """
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY,
            team_abbr TEXT UNIQUE,
            team_name TEXT,
            division TEXT,
            conference TEXT
        )
    """,
    "players": """
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
    """,
    "injury_reports": """
        CREATE TABLE IF NOT EXISTS injury_reports (
            id INTEGER PRIMARY KEY,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT,
            report_date TEXT NOT NULL,
            injury_status TEXT,
            injury_designation TEXT,
            injury_body_part TEXT,
            practice_status TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (game_id) REFERENCES games (id),
            UNIQUE(player_id, season, week, report_date)
        )
    """,
    "player_stats": """
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            game_id TEXT,
            passing_yards REAL DEFAULT 0,
            passing_tds INTEGER DEFAULT 0,
            passing_interceptions INTEGER DEFAULT 0,
            passing_attempts INTEGER DEFAULT 0,
            passing_completions INTEGER DEFAULT 0,
            sack_yards REAL DEFAULT 0,
            rushing_yards REAL DEFAULT 0,
            rushing_attempts INTEGER DEFAULT 0,
            rushing_tds INTEGER DEFAULT 0,
            receiving_yards REAL DEFAULT 0,
            targets INTEGER DEFAULT 0,
            receptions INTEGER DEFAULT 0,
            receiving_tds INTEGER DEFAULT 0,
            receiving_air_yards REAL DEFAULT 0,
            receiving_yac REAL DEFAULT 0,
            fumbles INTEGER DEFAULT 0,
            fumbles_lost INTEGER DEFAULT 0,
            passing_2pt_conversions INTEGER DEFAULT 0,
            rushing_2pt_conversions INTEGER DEFAULT 0,
            receiving_2pt_conversions INTEGER DEFAULT 0,
            special_teams_tds INTEGER DEFAULT 0,
            return_yards REAL DEFAULT 0,
            snap_count INTEGER DEFAULT 0,
            snap_percentage REAL DEFAULT 0,
            route_participation REAL DEFAULT 0,
            target_share REAL DEFAULT 0,
            rush_attempt_share REAL DEFAULT 0,
            red_zone_targets INTEGER DEFAULT 0,
            red_zone_touches INTEGER DEFAULT 0,
            opportunity_share REAL DEFAULT 0,
            fantasy_points REAL DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (game_id) REFERENCES games (id),
            UNIQUE(player_id, game_id)
        )
    """,
    "draftkings_salaries": """
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
    """,
    "dst_stats": """
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
    """,
    "historical_ownership": """
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
    """,
    "play_by_play": """
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
    """,
    "weather": """
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
    """,
    "betting_odds": """
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
    """,
    "dfs_scores": """
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
            UNIQUE(player_id, game_id)
        )
    """,
    "stat_corrections": """
        CREATE TABLE IF NOT EXISTS stat_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            original_value REAL,
            corrected_value REAL NOT NULL,
            source TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (game_id) REFERENCES games (id),
            UNIQUE(player_id, game_id, stat_type)
        )
    """,
}


def get_db_connection(db_path: str = "data/nfl_dfs.db") -> sqlite3.Connection:
    """Get database connection with fast timeout for concurrent access."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Use short timeout to fail fast on locks
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
    conn.execute("PRAGMA synchronous = NORMAL")  # Balance safety and performance
    conn.execute("PRAGMA busy_timeout = 5000")  # 5 second busy timeout - fail fast
    return conn


@contextmanager
def db_connection(db_path: str = "data/nfl_dfs.db"):
    """Context manager for database connections.

    Ensures connections are properly closed even if errors occur.

    Args:
        db_path: Path to SQLite database

    Yields:
        SQLite connection object

    Example:
        with db_connection() as conn:
            result = conn.execute("SELECT * FROM players").fetchall()
    """
    from data import get_db_connection  # Import here to avoid circular imports

    conn = get_db_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


class DatabaseManager:
    """Optimized database access layer with bulk loading and caching."""

    def __init__(self, db_path: str = "data/nfl_dfs.db"):
        self.db_path = db_path
        self._connection = None
        self._cache = {}

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection with optimizations."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            # Enable query optimizations
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA cache_size = -64000")  # 64MB cache
            self._connection.execute("PRAGMA temp_store = MEMORY")
            self._connection.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
            logger.info("Database connection optimized with WAL mode and 64MB cache")
        return self._connection

    def create_indexes(self) -> None:
        """Create indexes for common query patterns."""
        indexes = [
            # Player stats queries (most critical)
            "CREATE INDEX IF NOT EXISTS idx_player_stats_player_game ON player_stats(player_id, game_id)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_game_id ON player_stats(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_player_id ON player_stats(player_id)",
            # Game queries
            "CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week)",
            "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_games_home_away ON games(home_team_id, away_team_id)",
            # Injury reports queries (new temporal injury tracking)
            "CREATE INDEX IF NOT EXISTS idx_injury_player_season_week ON injury_reports(player_id, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_injury_game_id ON injury_reports(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_injury_report_date ON injury_reports(report_date)",
            "CREATE INDEX IF NOT EXISTS idx_injury_status ON injury_reports(injury_status)",
            "CREATE INDEX IF NOT EXISTS idx_injury_season_week ON injury_reports(season, week)",
            # Play by play queries
            "CREATE INDEX IF NOT EXISTS idx_pbp_game ON play_by_play(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_pbp_season_week ON play_by_play(season, week)",
            "CREATE INDEX IF NOT EXISTS idx_pbp_posteam ON play_by_play(posteam, play_type)",
            # DraftKings queries
            "CREATE INDEX IF NOT EXISTS idx_dk_contest ON draftkings_salaries(contest_id)",
            "CREATE INDEX IF NOT EXISTS idx_dk_player_contest ON draftkings_salaries(player_id, contest_id)",
            # DST queries
            "CREATE INDEX IF NOT EXISTS idx_dst_stats_team_game ON dst_stats(team_abbr, game_id)",
            "CREATE INDEX IF NOT EXISTS idx_dst_stats_season_week ON dst_stats(season, week)",
            # Weather queries
            "CREATE INDEX IF NOT EXISTS idx_weather_game_id ON weather(game_id)",
            # Betting odds queries
            "CREATE INDEX IF NOT EXISTS idx_betting_odds_game_id ON betting_odds(game_id)",
            # Player queries
            "CREATE INDEX IF NOT EXISTS idx_players_name_team ON players(player_name, team_id)",
            "CREATE INDEX IF NOT EXISTS idx_players_display_name ON players(display_name)",
            "CREATE INDEX IF NOT EXISTS idx_players_gsis_id ON players(gsis_id)",
            # Team lookup optimization
            "CREATE INDEX IF NOT EXISTS idx_teams_abbr ON teams(team_abbr)",
        ]

        index_progress = ProgressDisplay("Creating indexes")
        created_count = 0

        for i, idx in enumerate(indexes):
            index_progress.update(i, len(indexes))
            try:
                self.conn.execute(idx)
                created_count += 1
            except sqlite3.Error as e:
                if "already exists" not in str(e):
                    logger.warning(f"Error creating index: {e}")

        self.conn.commit()

        # Update query planner statistics
        self.conn.execute("ANALYZE")
        self.conn.commit()

        index_progress.finish(f"Created {created_count} indexes")

    # ==================== BULK LOADING METHODS ====================

    def bulk_load_player_stats(
        self, season: int, weeks: List[int] = None
    ) -> pd.DataFrame:
        """Load all player stats for a season in one query."""
        if weeks:
            week_filter = f"AND g.week IN ({','.join(map(str, weeks))})"
        else:
            week_filter = ""

        query = f"""
        SELECT
            ps.*,
            p.player_name,
            p.display_name,
            p.position,
            p.team_id,
            t.team_abbr,
            g.game_date,
            g.season,
            g.week,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            ht.team_abbr as home_abbr,
            at.team_abbr as away_abbr
        FROM player_stats ps
        JOIN players p ON ps.player_id = p.id
        LEFT JOIN teams t ON p.team_id = t.id
        JOIN games g ON ps.game_id = g.id
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.season = ? {week_filter}
        AND g.game_finished = 1
        ORDER BY g.game_date, ps.player_id
        """

        return pd.read_sql(query, self.conn, params=[season])

    def bulk_load_training_data(
        self, position: str, seasons: List[int]
    ) -> pd.DataFrame:
        """Load all training data for a position in one optimized query."""

        # Position-specific optimized queries
        if position == "QB":
            query = """
            SELECT
                ps.player_id, ps.game_id, ps.fantasy_points,
                ps.passing_yards, ps.passing_tds, ps.passing_interceptions,
                ps.rushing_yards, ps.rushing_tds, ps.fumbles_lost,
                p.player_name, p.display_name, p.position, p.team_id,
                t.team_abbr,
                g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                ht.team_abbr as home_abbr, at.team_abbr as away_abbr,
                -- Precompute recent performance in the query
                (SELECT AVG(ps2.fantasy_points)
                 FROM player_stats ps2
                 JOIN games g2 ON ps2.game_id = g2.id
                 WHERE ps2.player_id = ps.player_id
                 AND g2.game_date < g.game_date
                 AND g2.game_date >= date(g.game_date, '-28 days')) as recent_avg_points
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN teams t ON p.team_id = t.id
            JOIN games g ON ps.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE p.position = ?
            AND g.season IN ({})
            AND g.game_finished = 1
            AND ps.fantasy_points IS NOT NULL
            ORDER BY g.game_date
            """.format(",".join("?" * len(seasons)))

        elif position in ["RB", "WR", "TE"]:
            query = """
            SELECT
                ps.player_id, ps.game_id, ps.fantasy_points,
                ps.rushing_yards, ps.rushing_attempts, ps.rushing_tds,
                ps.receiving_yards, ps.targets, ps.receptions, ps.receiving_tds,
                ps.fumbles_lost,
                p.player_name, p.display_name, p.position, p.team_id,
                t.team_abbr,
                g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                ht.team_abbr as home_abbr, at.team_abbr as away_abbr,
                -- Target share calculation
                CAST(ps.targets AS FLOAT) / NULLIF(
                    (SELECT SUM(ps2.targets)
                     FROM player_stats ps2
                     JOIN players p2 ON ps2.player_id = p2.id
                     WHERE p2.team_id = p.team_id
                     AND ps2.game_id = ps.game_id
                     AND p2.position IN ('WR', 'TE', 'RB')), 0) as target_share,
                -- Recent performance
                (SELECT AVG(ps3.fantasy_points)
                 FROM player_stats ps3
                 JOIN games g3 ON ps3.game_id = g3.id
                 WHERE ps3.player_id = ps.player_id
                 AND g3.game_date < g.game_date
                 AND g3.game_date >= date(g.game_date, '-28 days')) as recent_avg_points
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN teams t ON p.team_id = t.id
            JOIN games g ON ps.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE p.position = ?
            AND g.season IN ({})
            AND g.game_finished = 1
            AND ps.fantasy_points IS NOT NULL
            ORDER BY g.game_date
            """.format(",".join("?" * len(seasons)))

        else:  # DST
            query = """
            SELECT
                ds.team_abbr as player_id, ds.game_id, ds.fantasy_points,
                ds.points_allowed, ds.sacks, ds.interceptions, ds.fumbles_recovered,
                ds.defensive_tds, ds.return_tds, ds.special_teams_tds, ds.safeties,
                ds.team_abbr, ds.season, ds.week,
                t.team_name,
                g.game_date, g.home_team_id, g.away_team_id,
                ht.team_abbr as home_abbr, at.team_abbr as away_abbr,
                -- Recent defensive performance
                (SELECT AVG(ds2.fantasy_points)
                 FROM dst_stats ds2
                 WHERE ds2.team_abbr = ds.team_abbr
                 AND ds2.season = ds.season
                 AND ds2.week < ds.week
                 AND ds2.week >= ds.week - 4) as recent_avg_points
            FROM dst_stats ds
            JOIN teams t ON ds.team_abbr = t.team_abbr
            JOIN games g ON ds.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.season IN ({})
            AND ds.fantasy_points IS NOT NULL
            ORDER BY g.game_date
            """.format(",".join("?" * len(seasons)))

        params = [position] + seasons if position != "DST" else seasons

        df = pd.read_sql(query, self.conn, params=params)
        logger.info(f"Bulk loaded {len(df)} rows for {position} training data")
        return df

    def bulk_load_contest_data(self, contest_id: str) -> pd.DataFrame:
        """Load all contest data in one optimized query."""
        query = """
        SELECT
            dk.*,
            p.id as player_id,
            p.display_name,
            p.position as db_position,
            t.team_abbr as db_team,
            -- Latest performance metrics
            (SELECT ps.fantasy_points
             FROM player_stats ps
             JOIN games g ON ps.game_id = g.id
             WHERE ps.player_id = p.id
             ORDER BY g.game_date DESC
             LIMIT 1) as last_game_points,
            -- Season average (current season)
            (SELECT AVG(ps.fantasy_points)
             FROM player_stats ps
             JOIN games g ON ps.game_id = g.id
             WHERE ps.player_id = p.id
             AND g.season = (SELECT MAX(season) FROM games)) as season_avg_points,
            -- Recent 5 game average
            (SELECT AVG(ps.fantasy_points)
             FROM player_stats ps
             JOIN games g ON ps.game_id = g.id
             WHERE ps.player_id = p.id
             ORDER BY g.game_date DESC
             LIMIT 5) as recent_5_avg_points
        FROM draftkings_salaries dk
        LEFT JOIN players p ON dk.player_id = p.id
        LEFT JOIN teams t ON p.team_id = t.id
        WHERE dk.contest_id = ?
        ORDER BY dk.roster_position, dk.salary DESC
        """

        return pd.read_sql(query, self.conn, params=[contest_id])

    def bulk_load_weather_data(self, game_ids: List[str]) -> pd.DataFrame:
        """Load weather data for multiple games."""
        if not game_ids:
            return pd.DataFrame()

        placeholders = ",".join("?" * len(game_ids))
        query = f"""
        SELECT w.*, g.game_date, g.stadium
        FROM weather w
        JOIN games g ON w.game_id = g.id
        WHERE w.game_id IN ({placeholders})
        """

        return pd.read_sql(query, self.conn, params=game_ids)

    def bulk_load_betting_odds(self, game_ids: List[str]) -> pd.DataFrame:
        """Load betting odds for multiple games."""
        if not game_ids:
            return pd.DataFrame()

        placeholders = ",".join("?" * len(game_ids))
        query = f"""
        SELECT bo.*, g.game_date
        FROM betting_odds bo
        JOIN games g ON bo.game_id = g.id
        WHERE bo.game_id IN ({placeholders})
        """

        return pd.read_sql(query, self.conn, params=game_ids)

    # ==================== CACHED QUERIES ====================

    def get_team_stats_cached(self, team_abbr: str, season: int) -> Dict:
        """Cached team statistics."""
        cache_key = f"team_stats_{team_abbr}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        query = """
        SELECT
            AVG(CASE WHEN home_team_id = t.id THEN home_score
                     ELSE away_score END) as avg_points_for,
            AVG(CASE WHEN home_team_id = t.id THEN away_score
                     ELSE home_score END) as avg_points_against,
            COUNT(*) as games_played
        FROM games g
        JOIN teams t ON t.team_abbr = ?
        WHERE (g.home_team_id = t.id OR g.away_team_id = t.id)
        AND g.season = ?
        AND g.game_finished = 1
        """

        result = self.conn.execute(query, [team_abbr, season]).fetchone()
        data = {
            "avg_points_for": result[0] or 0,
            "avg_points_against": result[1] or 0,
            "games_played": result[2] or 0,
        }
        self._cache[cache_key] = data
        return data

    def get_player_recent_performance(
        self, player_id: int, num_games: int = 5
    ) -> pd.DataFrame:
        """Cached recent player performance."""
        cache_key = f"player_performance_{player_id}_{num_games}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        query = """
        SELECT ps.*, g.game_date, g.season, g.week
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        WHERE ps.player_id = ?
        ORDER BY g.game_date DESC
        LIMIT ?
        """

        data = pd.read_sql(query, self.conn, params=[player_id, num_games])
        self._cache[cache_key] = data
        return data

    def get_defensive_matchup_cached(
        self, opponent_abbr: str, season: int, week: int
    ) -> Dict:
        """Cached defensive statistics for matchup analysis."""
        cache_key = f"defensive_matchup_{opponent_abbr}_{season}_{week}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        query = """
        SELECT
            AVG(ds.points_allowed) as avg_points_allowed,
            AVG(ds.sacks) as avg_sacks,
            AVG(ds.interceptions) as avg_interceptions,
            AVG(ds.fumbles_recovered) as avg_fumbles_recovered,
            AVG(ds.fantasy_points) as avg_fantasy_points
        FROM dst_stats ds
        WHERE ds.team_abbr = ?
        AND ds.season = ?
        AND ds.week < ?
        AND ds.week >= ?
        """

        result = self.conn.execute(
            query, [opponent_abbr, season, week, max(1, week - 4)]
        ).fetchone()
        data = {
            "avg_points_allowed": result[0] or 24.0,
            "avg_sacks": result[1] or 2.0,
            "avg_interceptions": result[2] or 1.0,
            "avg_fumbles_recovered": result[3] or 1.0,
            "avg_fantasy_points": result[4] or 5.0,
        }
        self._cache[cache_key] = data
        return data

    # ==================== BATCH OPERATIONS ====================

    def batch_insert_player_stats(self, stats_df: pd.DataFrame) -> None:
        """Efficiently insert multiple player stats records."""
        stats_df.to_sql(
            "player_stats", self.conn, if_exists="append", index=False, method="multi"
        )
        logger.info(f"Batch inserted {len(stats_df)} player stats records")

    def batch_update_fantasy_points(
        self, updates: List[Tuple[float, int, str]]
    ) -> None:
        """Batch update fantasy points."""
        query = """
        UPDATE player_stats
        SET fantasy_points = ?
        WHERE player_id = ? AND game_id = ?
        """

        self.conn.executemany(query, updates)
        self.conn.commit()
        logger.info(f"Batch updated {len(updates)} fantasy points")

    def batch_insert_injury_reports(self, injury_reports: pd.DataFrame) -> None:
        """Batch insert injury reports for temporal tracking."""
        injury_reports.to_sql(
            "injury_reports", self.conn, if_exists="append", index=False, method="multi"
        )
        logger.info(f"Batch inserted {len(injury_reports)} injury reports")

    def get_latest_injury_status(self, player_id: int, season: int, week: int) -> Dict:
        """Get the most recent injury report for a player before a given week."""
        query = """
        SELECT injury_status, injury_designation, injury_body_part, practice_status, report_date
        FROM injury_reports
        WHERE player_id = ? AND season = ? AND week <= ?
        ORDER BY week DESC, report_date DESC
        LIMIT 1
        """
        result = self.conn.execute(query, [player_id, season, week]).fetchone()
        if result:
            return {
                "injury_status": result[0],
                "injury_designation": result[1],
                "injury_body_part": result[2],
                "practice_status": result[3],
                "report_date": result[4],
            }
        return {}

    # ==================== UTILITY METHODS ====================

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database."""
        tables = [
            "games",
            "teams",
            "players",
            "player_stats",
            "draftkings_salaries",
            "dst_stats",
            "play_by_play",
            "weather",
            "betting_odds",
        ]

        stats = {}
        for table in tables:
            try:
                stats[table] = self.get_table_row_count(table)
            except sqlite3.Error:
                stats[table] = 0  # Table doesn't exist

        return stats

    # ==================== CLEANUP ====================

    def close(self) -> None:
        """Close database connection and clean up WAL files."""
        if self._connection:
            # Checkpoint WAL file back to main database
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.close()
            self._connection = None

    def vacuum(self) -> None:
        """Optimize database file size and update statistics."""
        logger.info("Running database maintenance...")
        self.conn.execute("VACUUM")
        self.conn.execute("ANALYZE")
        logger.info("Database maintenance completed")

    def clear_cache(self) -> None:
        """Clear LRU caches."""
        self.get_team_stats_cached.cache_clear()
        self.get_player_recent_performance.cache_clear()
        self.get_defensive_matchup_cached.cache_clear()
        logger.info("Database caches cleared")
