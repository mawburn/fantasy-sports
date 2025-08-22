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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import nfl_data_py as nfl
except ImportError:
    print("Warning: nfl_data_py not available. Install with: pip install nfl_data_py")
    nfl = None

logger = logging.getLogger(__name__)


def _filter_for_top_performers(
    X: np.ndarray, y: np.ndarray, position: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter training data to focus on top performers and meaningful games.

    Removes low-scoring games that add noise when trying to identify
    top performers for DFS lineup construction.

    Args:
        X: Feature matrix
        y: Target fantasy points
        position: Player position (QB, RB, WR, TE, DST)

    Returns:
        Tuple of filtered (X, y)
    """
    position = position.upper()

    # Position-specific minimum thresholds for meaningful performances
    min_thresholds = {
        "QB": 5.0,  # QB should have at least 5 fantasy points (allows backup performances)
        "RB": 6.0,  # RB should have at least 6 fantasy points (60 rush yds OR 1 TD)
        "WR": 5.0,  # WR should have at least 5 fantasy points (50 rec yds OR 1 TD)
        "TE": 4.0,  # TE should have at least 4 fantasy points (40 rec yds)
        "DST": 2.0,  # DST should have at least 2 fantasy points
        "DEF": 2.0,  # DEF should have at least 2 fantasy points
    }

    # Additional filtering: remove bottom 10% of performances to focus on upside
    percentile_cutoff = 10  # Remove bottom 10%

    min_threshold = min_thresholds.get(position, 3.0)
    percentile_threshold = np.percentile(y, percentile_cutoff)

    # Use the higher of the two thresholds
    final_threshold = max(min_threshold, percentile_threshold)

    # Filter out low performances
    keep_mask = y >= final_threshold
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]

    removed_count = len(y) - len(y_filtered)
    logger.info(
        f"Filtered {removed_count} low-performing {position} games (threshold: {final_threshold:.1f})"
    )
    logger.info(
        f"Kept {len(y_filtered)}/{len(y)} samples ({100 * len(y_filtered) / len(y):.1f}%)"
    )

    return X_filtered, y_filtered


# Feature extraction functions for QB model optimization
def extract_vegas_features(
    db_path: str, game_id: str, team_id: int, is_home: bool
) -> Dict[str, float]:
    """Extract critical Vegas-based features for QB predictions."""
    features = {}

    with sqlite3.connect(db_path) as conn:
        # Get betting odds for the game
        odds_query = """
            SELECT over_under_line, home_team_spread, away_team_spread
            FROM betting_odds
            WHERE game_id = ?
        """
        odds_result = conn.execute(odds_query, (game_id,)).fetchone()

        if odds_result:
            over_under, home_spread, away_spread = odds_result
            team_spread = home_spread if is_home else away_spread

            # Team Implied Total (most predictive single feature)
            implied_total = (
                (over_under / 2) - (team_spread / 2)
                if over_under and team_spread
                else 0
            )

            # Game Environment Features
            features.update(
                {
                    "team_implied_total": implied_total,
                    "game_total": over_under or 0,
                    "spread": team_spread or 0,
                    "is_favorite": 1 if team_spread and team_spread < 0 else 0,
                    "favorite_margin": abs(team_spread)
                    if team_spread and team_spread < 0
                    else 0,
                    "expected_pass_rate": min(0.7, 0.55 + abs(team_spread or 0) * 0.01),
                    "shootout_probability": 1 if over_under and over_under > 50 else 0,
                    "blowout_risk": 1 if team_spread and abs(team_spread) > 10 else 0,
                }
            )
        else:
            # Use NaN for missing betting data to distinguish from actual values
            features.update(
                {
                    "team_implied_total": float("nan"),
                    "game_total": float("nan"),
                    "spread": float("nan"),
                    "is_favorite": float("nan"),
                    "favorite_margin": float("nan"),
                    "expected_pass_rate": 0.55,  # Keep reasonable default for pass rate
                    "shootout_probability": float("nan"),
                    "blowout_risk": float("nan"),
                }
            )

    return features


def extract_volume_features(
    db_path: str, player_id: int, game_id: str, lookback_weeks: int = 4
) -> Dict[str, float]:
    """Extract passing volume and opportunity metrics."""
    features = {}

    with sqlite3.connect(db_path) as conn:
        # Get game date for lookback window
        game_date_query = "SELECT game_date FROM games WHERE id = ?"
        game_date_result = conn.execute(game_date_query, (game_id,)).fetchone()

        if not game_date_result:
            return features

        game_date_str = game_date_result[0]
        try:
            game_date = datetime.strptime(game_date_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return features

        start_date = (game_date - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")

        # Core volume metrics from play-by-play data
        volume_query = """
            SELECT
                COUNT(CASE WHEN pass_attempt = 1 THEN 1 END) as pass_attempts,
                COUNT(CASE WHEN pass_attempt = 1 AND yardline_100 <= 20 THEN 1 END) as rz_pass_attempts,
                COUNT(CASE WHEN pass_attempt = 1 AND yardline_100 <= 10 THEN 1 END) as inside_10_passes,
                COUNT(CASE WHEN touchdown = 1 AND pass_attempt = 1 AND yardline_100 <= 20 THEN 1 END) as rz_pass_tds,
                COUNT(CASE WHEN yardline_100 <= 20 THEN 1 END) as rz_plays,
                COUNT(CASE WHEN yardline_100 <= 10 THEN 1 END) as inside_10_plays,
                COUNT(*) as total_plays
            FROM play_by_play pbp
            JOIN games g ON pbp.game_id = g.id
            WHERE pbp.posteam = (
                SELECT t.team_abbr
                FROM players p
                JOIN teams t ON p.team_id = t.id
                WHERE p.id = ?
            )
            AND g.game_date >= ? AND g.game_date < ?
            AND g.game_finished = 1
        """

        volume_result = conn.execute(
            volume_query, (player_id, start_date, game_date_str)
        ).fetchone()

        if volume_result:
            (
                pass_attempts,
                rz_pass_attempts,
                inside_10_passes,
                rz_pass_tds,
                rz_plays,
                inside_10_plays,
                total_plays,
            ) = volume_result
            games_count = max(1, lookback_weeks)  # Assume roughly 1 game per week

            features.update(
                {
                    "avg_pass_attempts": pass_attempts / games_count,
                    "rz_pass_attempts_pg": rz_pass_attempts / games_count,
                    "inside_10_pass_rate": inside_10_passes / max(inside_10_plays, 1),
                    "td_rate_rz": rz_pass_tds / max(rz_pass_attempts, 1),
                    "pass_rate_overall": pass_attempts / max(total_plays, 1),
                }
            )
        else:
            features.update(
                {
                    "avg_pass_attempts": 0,
                    "rz_pass_attempts_pg": 0,
                    "inside_10_pass_rate": 0,
                    "td_rate_rz": 0,
                    "pass_rate_overall": 0,
                }
            )

    return features


def extract_qb_rushing_features(
    db_path: str, player_id: int, game_id: str, lookback_weeks: int = 4
) -> Dict[str, float]:
    """QB rushing is the stickiest fantasy advantage."""
    features = {}

    with sqlite3.connect(db_path) as conn:
        # Get game date for lookback window
        game_date_query = "SELECT game_date FROM games WHERE id = ?"
        game_date_result = conn.execute(game_date_query, (game_id,)).fetchone()

        if not game_date_result:
            return features

        game_date_str = game_date_result[0]
        try:
            game_date = datetime.strptime(game_date_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return features

        start_date = (game_date - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")

        # Get QB rushing stats from player_stats table
        rushing_query = """
            SELECT
                AVG(ps.rushing_attempts) as avg_rush_attempts,
                AVG(ps.rushing_yards) as avg_rush_yards,
                AVG(ps.rushing_tds) as avg_rush_tds,
                SUM(ps.rushing_tds) as total_rush_tds,
                COUNT(*) as games_played
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.id
            WHERE ps.player_id = ?
            AND g.game_date >= ? AND g.game_date < ?
            AND g.game_finished = 1
        """

        rushing_result = conn.execute(
            rushing_query, (player_id, start_date, game_date_str)
        ).fetchone()

        if rushing_result and rushing_result[4] > 0:  # games_played > 0
            avg_attempts, avg_yards, avg_tds, total_tds, games_played = rushing_result

            features.update(
                {
                    "avg_rush_attempts": avg_attempts or 0,
                    "rush_yards_per_attempt": (avg_yards / max(avg_attempts, 1))
                    if avg_attempts
                    else 0,
                    "rush_td_rate": (avg_tds / max(avg_attempts, 1))
                    if avg_attempts
                    else 0,
                    "recent_rush_tds": total_tds or 0,
                    "rush_upside": min(
                        1, (avg_attempts or 0) * 0.1
                    ),  # Rushing attempt rate as upside indicator
                }
            )
        else:
            features.update(
                {
                    "avg_rush_attempts": 0,
                    "rush_yards_per_attempt": 0,
                    "rush_td_rate": 0,
                    "recent_rush_tds": 0,
                    "rush_upside": 0,
                }
            )

    return features


def extract_opponent_features(
    db_path: str, opponent_team_id: int, position: str = "QB", lookback_weeks: int = 4
) -> Dict[str, float]:
    """Opponent defensive efficiency metrics."""
    features = {}

    with sqlite3.connect(db_path) as conn:
        # Get opponent team abbreviation
        team_query = "SELECT team_abbr FROM teams WHERE id = ?"
        team_result = conn.execute(team_query, (opponent_team_id,)).fetchone()

        if not team_result:
            return features

        opponent_abbr = team_result[0]

        # Get current season/week for lookback
        current_season_query = "SELECT MAX(season) FROM games WHERE game_finished = 1"
        current_season = conn.execute(current_season_query).fetchone()[0] or 2024

        # Defensive stats from play-by-play data
        def_query = """
            SELECT
                AVG(CASE WHEN complete_pass = 1 THEN yards_gained ELSE 0 END) as avg_completion_yards,
                AVG(CASE WHEN complete_pass = 1 THEN 1.0 ELSE 0.0 END) as completion_rate_allowed,
                AVG(CASE WHEN touchdown = 1 AND pass_attempt = 1 THEN 1.0 ELSE 0.0 END) as pass_td_rate_allowed,
                COUNT(CASE WHEN sack = 1 THEN 1 END) as sacks,
                COUNT(CASE WHEN pass_attempt = 1 THEN 1 END) as pass_attempts_faced,
                COUNT(CASE WHEN interception = 1 THEN 1 END) as interceptions
            FROM play_by_play
            WHERE defteam = ?
            AND season = ?
            AND pass_attempt = 1
        """

        def_result = conn.execute(def_query, (opponent_abbr, current_season)).fetchone()

        if def_result:
            avg_comp_yards, comp_rate, td_rate, sacks, attempts_faced, ints = def_result

            features.update(
                {
                    "def_completion_yards_allowed": avg_comp_yards or 0,
                    "def_completion_rate_allowed": comp_rate or 0,
                    "def_pass_td_rate_allowed": td_rate or 0,
                    "def_pressure_rate": (sacks or 0) / max(attempts_faced or 1, 1),
                    "def_int_rate": (ints or 0) / max(attempts_faced or 1, 1),
                }
            )
        else:
            features.update(
                {
                    "def_completion_yards_allowed": 0,
                    "def_completion_rate_allowed": 0,
                    "def_pass_td_rate_allowed": 0,
                    "def_pressure_rate": 0,
                    "def_int_rate": 0,
                }
            )

        # Fantasy points allowed to position
        fps_query = """
            SELECT AVG(ds.dfs_points) as avg_fps_allowed
            FROM dfs_scores ds
            JOIN players p ON ds.player_id = p.id
            WHERE ds.opponent_id = ?
            AND p.position = ?
            AND ds.season = ?
        """

        fps_result = conn.execute(
            fps_query, (opponent_team_id, position, current_season)
        ).fetchone()

        if fps_result:
            features["def_qb_fps_allowed_avg"] = fps_result[0] or 0
        else:
            features["def_qb_fps_allowed_avg"] = 0

    return features


def extract_pace_features(
    db_path: str, team_id: int, opponent_id: int, lookback_weeks: int = 4
) -> Dict[str, float]:
    """Game pace and neutral situation metrics."""
    features = {}

    with sqlite3.connect(db_path) as conn:
        # Get team abbreviations
        team_query = "SELECT team_abbr FROM teams WHERE id = ?"
        team_result = conn.execute(team_query, (team_id,)).fetchone()
        opp_result = conn.execute(team_query, (opponent_id,)).fetchone()

        if not team_result or not opp_result:
            return features

        team_abbr, opp_abbr = team_result[0], opp_result[0]

        # Get current season
        current_season_query = "SELECT MAX(season) FROM games WHERE game_finished = 1"
        current_season = conn.execute(current_season_query).fetchone()[0] or 2024

        # Team pace metrics
        pace_query = """
            SELECT
                COUNT(*) / COUNT(DISTINCT game_id) as plays_per_game,
                AVG(CASE WHEN pass_attempt = 1 THEN 1.0 ELSE 0.0 END) as pass_rate,
                COUNT(CASE WHEN down >= 3 THEN 1 END) / COUNT(*) as third_down_rate
            FROM play_by_play
            WHERE posteam = ?
            AND season = ?
            AND (pass_attempt = 1 OR rush_attempt = 1)
        """

        team_pace = conn.execute(pace_query, (team_abbr, current_season)).fetchone()
        opp_pace = conn.execute(pace_query, (opp_abbr, current_season)).fetchone()

        if team_pace and opp_pace:
            team_plays, team_pass_rate, team_3d_rate = team_pace
            opp_plays, opp_pass_rate, opp_3d_rate = opp_pace

            features.update(
                {
                    "team_plays_per_game": team_plays or 0,
                    "opponent_plays_per_game": opp_plays or 0,
                    "combined_pace": (team_plays + opp_plays) / 2
                    if team_plays and opp_plays
                    else 0,
                    "team_pass_rate": team_pass_rate or 0,
                    "opponent_pass_rate": opp_pass_rate or 0,
                    "team_third_down_freq": team_3d_rate or 0,
                }
            )
        else:
            features.update(
                {
                    "team_plays_per_game": 0,
                    "opponent_plays_per_game": 0,
                    "combined_pace": 0,
                    "team_pass_rate": 0,
                    "opponent_pass_rate": 0,
                    "team_third_down_freq": 0,
                }
            )

    return features


class FeatureProcessor:
    """Enhanced feature processing pipeline with proper scaling and normalization."""

    def __init__(self):
        self.scalers = {}
        self.feature_groups = {
            "volume": [
                "avg_pass_attempts",
                "rz_pass_attempts_pg",
                "avg_rush_attempts",
                "team_plays_per_game",
            ],
            "efficiency": [
                "rush_yards_per_attempt",
                "td_rate_rz",
                "def_completion_rate_allowed",
                "inside_10_pass_rate",
            ],
            "vegas": ["team_implied_total", "spread", "game_total", "favorite_margin"],
            "defensive": [
                "def_pressure_rate",
                "def_int_rate",
                "def_completion_yards_allowed",
                "def_qb_fps_allowed_avg",
            ],
            "pace": [
                "combined_pace",
                "team_pass_rate",
                "opponent_pass_rate",
                "team_third_down_freq",
            ],
        }

    def process_features(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """Process features with position-specific enhancements."""

        # Handle missing values with position-specific defaults
        features_dict = self._impute_missing_values(features_dict)

        # Create ratio and interaction features
        features_dict = self._create_interaction_features(features_dict)

        # Apply feature transformations
        features_dict = self._apply_transformations(features_dict)

        return features_dict

    def _impute_missing_values(self, features: Dict[str, float]) -> Dict[str, float]:
        """Handle missing values with intelligent defaults."""
        defaults = {
            "team_implied_total": 21.0,  # Average NFL team score
            "game_total": 42.0,  # Average total
            "spread": 0.0,  # Pick'em game
            "avg_pass_attempts": 35.0,  # League average
            "avg_rush_attempts": 4.0,  # QB rushing attempts
            "def_completion_rate_allowed": 0.65,  # League average
            "def_pressure_rate": 0.25,  # League average pressure rate
            "combined_pace": 65.0,  # Average plays per game
            "team_pass_rate": 0.60,  # Average pass rate
        }

        for key, default_value in defaults.items():
            if key not in features or features[key] is None or np.isnan(features[key]):
                features[key] = default_value

        return features

    def _create_interaction_features(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """Create interaction features that capture QB-specific relationships."""

        # Vegas x Volume interactions
        features["implied_total_x_attempts"] = features.get(
            "team_implied_total", 0
        ) * features.get("avg_pass_attempts", 0)
        features["spread_x_rush_attempts"] = abs(
            features.get("spread", 0)
        ) * features.get("avg_rush_attempts", 0)

        # Efficiency ratios
        if features.get("avg_pass_attempts", 0) > 0:
            features["rz_pass_rate"] = features.get(
                "rz_pass_attempts_pg", 0
            ) / features.get("avg_pass_attempts", 1)
        else:
            features["rz_pass_rate"] = 0

        # Defensive matchup interactions
        features["def_weakness_score"] = (
            features.get("def_completion_rate_allowed", 0)
            * features.get("def_completion_yards_allowed", 0)
            * (1 - features.get("def_pressure_rate", 0))
        )

        # Game script features
        if features.get("spread", 0) < -3:  # Big favorite
            features["positive_game_script"] = 1
            features["rush_game_script_boost"] = (
                features.get("avg_rush_attempts", 0) * 0.2
            )
        elif features.get("spread", 0) > 7:  # Big underdog
            features["positive_game_script"] = -1
            features["rush_game_script_boost"] = (
                features.get("avg_rush_attempts", 0) * -0.1
            )
        else:
            features["positive_game_script"] = 0
            features["rush_game_script_boost"] = 0

        # Pace matchups
        pace_diff = features.get("team_plays_per_game", 65) - features.get(
            "opponent_plays_per_game", 65
        )
        features["pace_advantage"] = pace_diff / 65.0  # Normalized pace advantage

        return features

    def _apply_transformations(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply mathematical transformations to improve feature distributions."""

        # Log transform for right-skewed features (add 1 to handle zeros)
        log_features = [
            "team_implied_total",
            "game_total",
            "avg_pass_attempts",
            "combined_pace",
        ]
        for feature in log_features:
            if feature in features and features[feature] > 0:
                features[f"{feature}_log"] = np.log1p(features[feature])

        # Square root transform for moderately skewed features
        sqrt_features = ["favorite_margin", "avg_rush_attempts"]
        for feature in sqrt_features:
            if feature in features and features[feature] >= 0:
                features[f"{feature}_sqrt"] = np.sqrt(features[feature])

        # Polynomial features for key predictors
        if "team_implied_total" in features:
            itt = features["team_implied_total"]
            features["team_implied_total_squared"] = itt**2
            features["team_implied_total_cubed"] = itt**3

        # Boolean transformations
        features["is_high_total"] = 1 if features.get("game_total", 0) > 47 else 0
        features["is_road_favorite"] = (
            1
            if features.get("spread", 0) < -3 and not features.get("is_home", 0)
            else 0
        )
        features["has_rush_upside"] = (
            1 if features.get("avg_rush_attempts", 0) > 6 else 0
        )

        return features


def create_comprehensive_qb_features(
    db_path: str, player_id: int, game_id: str
) -> Dict[str, float]:
    """Create comprehensive QB feature set using all optimization guide recommendations."""

    # Get basic game info
    with sqlite3.connect(db_path) as conn:
        game_info_query = """
            SELECT g.home_team_id, g.away_team_id, p.team_id
            FROM games g, players p
            WHERE g.id = ? AND p.id = ?
        """
        result = conn.execute(game_info_query, (game_id, player_id)).fetchone()

        if not result:
            return {}

        home_team_id, away_team_id, team_id = result
        is_home = team_id == home_team_id
        opponent_id = away_team_id if is_home else home_team_id

    # Extract all feature categories
    vegas_features = extract_vegas_features(db_path, game_id, team_id, is_home)
    volume_features = extract_volume_features(db_path, player_id, game_id)
    rushing_features = extract_qb_rushing_features(db_path, player_id, game_id)
    opponent_features = extract_opponent_features(db_path, opponent_id, "QB")
    pace_features = extract_pace_features(db_path, team_id, opponent_id)

    # Combine all features
    all_features = {
        **vegas_features,
        **volume_features,
        **rushing_features,
        **opponent_features,
        **pace_features,
    }

    # Add contextual features
    all_features["is_home"] = 1 if is_home else 0

    # Process features through enhanced pipeline
    processor = FeatureProcessor()
    processed_features = processor.process_features(all_features)

    return processed_features


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
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pass

    # Try M/D/YYYY format (spreadspoke format)
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        pass

    # Try MM/DD/YYYY format (padded version)
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        pass

    raise ValueError(
        f"Unable to parse date string: '{date_str}'. Expected format: YYYY-MM-DD or M/D/YYYY"
    )


# Database schema - simplified tables
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
            injury_status TEXT DEFAULT NULL,
            FOREIGN KEY (team_id) REFERENCES teams (id)
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

        # Create indexes for dfs_scores table
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_dfs_scores_player_week ON dfs_scores (player_id, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_dfs_scores_team ON dfs_scores (team_id)",
            "CREATE INDEX IF NOT EXISTS idx_dfs_scores_opponent ON dfs_scores (opponent_id)",
            "CREATE INDEX IF NOT EXISTS idx_dfs_scores_position ON dfs_scores (position)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        logger.info("Created/verified indexes for dfs_scores table")
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
        (1, "ARI", "Arizona Cardinals", "NFC West", "NFC"),
        (2, "ATL", "Atlanta Falcons", "NFC South", "NFC"),
        (3, "BAL", "Baltimore Ravens", "AFC North", "AFC"),
        (4, "BUF", "Buffalo Bills", "AFC East", "AFC"),
        (5, "CAR", "Carolina Panthers", "NFC South", "NFC"),
        (6, "CHI", "Chicago Bears", "NFC North", "NFC"),
        (7, "CIN", "Cincinnati Bengals", "AFC North", "AFC"),
        (8, "CLE", "Cleveland Browns", "AFC North", "AFC"),
        (9, "DAL", "Dallas Cowboys", "NFC East", "NFC"),
        (10, "DEN", "Denver Broncos", "AFC West", "AFC"),
        (11, "DET", "Detroit Lions", "NFC North", "NFC"),
        (12, "GB", "Green Bay Packers", "NFC North", "NFC"),
        (13, "HOU", "Houston Texans", "AFC South", "AFC"),
        (14, "IND", "Indianapolis Colts", "AFC South", "AFC"),
        (15, "JAX", "Jacksonville Jaguars", "AFC South", "AFC"),
        (16, "KC", "Kansas City Chiefs", "AFC West", "AFC"),
        (17, "LV", "Las Vegas Raiders", "AFC West", "AFC"),
        (18, "LAC", "Los Angeles Chargers", "AFC West", "AFC"),
        (19, "LAR", "Los Angeles Rams", "NFC West", "NFC"),
        (20, "MIA", "Miami Dolphins", "AFC East", "AFC"),
        (21, "MIN", "Minnesota Vikings", "NFC North", "NFC"),
        (22, "NE", "New England Patriots", "AFC East", "AFC"),
        (23, "NO", "New Orleans Saints", "NFC South", "NFC"),
        (24, "NYG", "New York Giants", "NFC East", "NFC"),
        (25, "NYJ", "New York Jets", "AFC East", "AFC"),
        (26, "PHI", "Philadelphia Eagles", "NFC East", "NFC"),
        (27, "PIT", "Pittsburgh Steelers", "AFC North", "AFC"),
        (28, "SF", "San Francisco 49ers", "NFC West", "NFC"),
        (29, "SEA", "Seattle Seahawks", "NFC West", "NFC"),
        (30, "TB", "Tampa Bay Buccaneers", "NFC South", "NFC"),
        (31, "TEN", "Tennessee Titans", "AFC South", "AFC"),
        (32, "WAS", "Washington Commanders", "NFC East", "NFC"),
    ]

    conn = get_db_connection(db_path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO teams (id, team_abbr, team_name, division, conference) VALUES (?, ?, ?, ?, ?)",
            teams_data,
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
                    game_id = (
                        str(game.get("game_id", ""))[:50]
                        if game.get("game_id")
                        else None
                    )
                    game_date = (
                        str(game.get("gameday", ""))[:10]
                        if game.get("gameday")
                        else None
                    )
                    week = int(game.get("week", 0)) if pd.notna(game.get("week")) else 0
                    home_score = (
                        int(game.get("home_score", 0))
                        if pd.notna(game.get("home_score"))
                        else 0
                    )
                    away_score = (
                        int(game.get("away_score", 0))
                        if pd.notna(game.get("away_score"))
                        else 0
                    )

                    home_team_id = get_team_id_by_abbr(game.get("home_team"), conn)
                    away_team_id = get_team_id_by_abbr(game.get("away_team"), conn)

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
                        1 if home_score > 0 or away_score > 0 else 0,
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO games
                           (id, game_date, season, week, home_team_id, away_team_id, home_score, away_score, game_finished)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        game_data,
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
                    player_week.get("player_display_name", ""),
                    player_week.get("position", ""),
                    player_week.get("recent_team", ""),
                    player_week.get("player_id", ""),
                    conn,
                )

                # Get game ID
                game_id = get_game_id(
                    season,
                    player_week.get("week"),
                    player_week.get("recent_team"),
                    conn,
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
                            safe_float(player_week.get("passing_yards", 0)),
                            safe_int(player_week.get("passing_tds", 0)),
                            safe_int(player_week.get("interceptions", 0)),
                            safe_float(player_week.get("rushing_yards", 0)),
                            safe_int(player_week.get("carries", 0)),
                            safe_int(player_week.get("rushing_tds", 0)),
                            safe_float(player_week.get("receiving_yards", 0)),
                            safe_int(player_week.get("targets", 0)),
                            safe_int(player_week.get("receptions", 0)),
                            safe_int(player_week.get("receiving_tds", 0)),
                            safe_int(player_week.get("fumbles_lost", 0)),
                            safe_int(player_week.get("passing_2pt_conversions", 0)),
                            safe_int(player_week.get("rushing_2pt_conversions", 0)),
                            safe_int(player_week.get("receiving_2pt_conversions", 0)),
                            safe_int(player_week.get("special_teams_tds", 0)),
                            fantasy_points,
                        )

                        conn.execute(
                            """INSERT OR REPLACE INTO player_stats
                               (player_id, game_id, passing_yards, passing_tds, passing_interceptions,
                                rushing_yards, rushing_attempts, rushing_tds, receiving_yards, targets,
                                receptions, receiving_tds, fumbles_lost, passing_2pt_conversions,
                                rushing_2pt_conversions, receiving_2pt_conversions, special_teams_tds, fantasy_points)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            stats_data,
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

            # Populate DFS scores table for the completed season
            try:
                logger.info(f"Populating DFS scores for {season} season...")
                populate_dfs_scores_for_season(season, db_path)
            except Exception as e:
                logger.warning(f"Could not populate DFS scores for {season}: {e}")

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
    name: str, position: str, team_abbr: str, gsis_id: str, conn: sqlite3.Connection
) -> Optional[int]:
    """Get existing player or create new one."""
    if not name:
        return None

    # Try to find existing player
    cursor = conn.execute(
        "SELECT id FROM players WHERE player_name = ? AND gsis_id = ?", (name, gsis_id)
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Create new player
    team_id = get_team_id_by_abbr(team_abbr, conn)
    cursor = conn.execute(
        """INSERT INTO players (player_name, display_name, position, team_id, gsis_id)
           VALUES (?, ?, ?, ?, ?)""",
        (name, name, position, team_id, gsis_id),
    )

    return cursor.lastrowid


def get_game_id(
    season: int, week: int, team_abbr: str, conn: sqlite3.Connection
) -> Optional[str]:
    """Get game ID for a team in a specific week."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    cursor = conn.execute(
        """SELECT id FROM games
           WHERE season = ? AND week = ?
           AND (home_team_id = ? OR away_team_id = ?)
           LIMIT 1""",
        (season, week, team_id, team_id),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def calculate_dk_fantasy_points(player_data: pd.Series) -> float:
    """Calculate DraftKings fantasy points."""
    points = 0.0

    # Passing
    passing_yards = player_data.get("passing_yards", 0) or 0
    points += passing_yards * 0.04  # 1 pt per 25 yards
    points += (player_data.get("passing_tds", 0) or 0) * 4
    points += (player_data.get("interceptions", 0) or 0) * -1
    points += (
        player_data.get("passing_interceptions", 0) or 0
    ) * -1  # Alternative column name
    if passing_yards >= 300:
        points += 3  # 300+ yard bonus

    # Rushing
    rushing_yards = player_data.get("rushing_yards", 0) or 0
    points += rushing_yards * 0.1  # 1 pt per 10 yards
    points += (player_data.get("rushing_tds", 0) or 0) * 6
    if rushing_yards >= 100:
        points += 3  # 100+ yard bonus

    # Receiving
    receiving_yards = player_data.get("receiving_yards", 0) or 0
    points += receiving_yards * 0.1  # 1 pt per 10 yards
    points += (player_data.get("receptions", 0) or 0) * 1  # 1 pt per reception
    points += (player_data.get("receiving_tds", 0) or 0) * 6
    if receiving_yards >= 100:
        points += 3  # 100+ yard bonus

    # Fumbles
    points += (player_data.get("fumbles_lost", 0) or 0) * -1

    # Two-point conversions (all types)
    points += (player_data.get("passing_2pt_conversions", 0) or 0) * 2
    points += (player_data.get("rushing_2pt_conversions", 0) or 0) * 2
    points += (player_data.get("receiving_2pt_conversions", 0) or 0) * 2

    # Special teams TDs (punt/kickoff/FG return TDs, offensive fumble recovery TDs)
    points += (player_data.get("special_teams_tds", 0) or 0) * 6

    return round(points, 2)


def calculate_dst_fantasy_points(stats: Dict[str, int]) -> float:
    """Calculate DraftKings DST fantasy points."""
    points = 0.0

    # Points allowed (tiered system)
    points_allowed = stats.get("points_allowed", 0)
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
    points += stats.get("sacks", 0) * 1

    # Interceptions (2 pts each)
    points += stats.get("interceptions", 0) * 2

    # Fumbles recovered (2 pts each)
    points += stats.get("fumbles_recovered", 0) * 2

    # Safeties (2 pts each)
    points += stats.get("safeties", 0) * 2

    # Defensive TDs (6 pts each)
    points += stats.get("defensive_tds", 0) * 6

    # Return TDs (6 pts each)
    points += stats.get("return_tds", 0) * 6

    # Special teams TDs (6 pts each)
    points += stats.get("special_teams_tds", 0) * 6

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
            if pd.isna(play.get("play_id")) or not play.get("game_id"):
                continue

            game_id = str(play.get("game_id", ""))

            # Check if the referenced game exists in the games table
            existing_game = conn.execute(
                "SELECT id FROM games WHERE id = ?", (game_id,)
            ).fetchone()

            if not existing_game:
                # Skip plays for games that don't exist in our games table
                continue

            def safe_int(val, default=0):
                try:
                    return int(val) if pd.notna(val) and val != "" else default
                except (ValueError, TypeError):
                    return default

            play_data = (
                str(play.get("play_id", "")),
                game_id,
                season,
                safe_int(play.get("week")),
                str(play.get("home_team", "") or ""),
                str(play.get("away_team", "") or ""),
                str(play.get("posteam", "") or ""),
                str(play.get("defteam", "") or ""),
                str(play.get("play_type", "") or ""),
                str(play.get("desc", "") or "")[:500],  # Limit description length
                safe_int(play.get("down")),
                safe_int(play.get("ydstogo")),
                safe_int(play.get("yardline_100")),
                safe_int(play.get("quarter_seconds_remaining")),
                safe_int(play.get("yards_gained")),
                safe_int(play.get("touchdown")),
                safe_int(play.get("pass_attempt")),
                safe_int(play.get("rush_attempt")),
                safe_int(play.get("complete_pass")),
                safe_int(play.get("incomplete_pass")),
                safe_int(play.get("interception")),
                safe_int(play.get("fumble")),
                safe_int(play.get("fumble_lost")),
                safe_int(play.get("sack")),
                safe_int(play.get("safety")),
                safe_int(play.get("penalty")),
            )

            conn.execute(
                """INSERT OR REPLACE INTO play_by_play
                   (play_id, game_id, season, week, home_team, away_team, posteam, defteam,
                    play_type, description, down, ydstogo, yardline_100, quarter_seconds_remaining,
                    yards_gained, touchdown, pass_attempt, rush_attempt, complete_pass,
                    incomplete_pass, interception, fumble, fumble_lost, sack, safety, penalty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                play_data,
            )
            play_count += 1

        except Exception as e:
            logger.warning(
                f"Error processing play {play.get('play_id', 'unknown')}: {e}"
            )
            continue

    logger.info(f"Stored {play_count} plays for season {season}")


def load_env_file():
    """Load .env file manually if it exists."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value


# Load .env file
load_env_file()


def collect_weather_data_optimized(
    db_path: str = "data/nfl_dfs.db",
    limit: int = None,
    rate_limit_delay: float = 1.5,
    max_days_per_batch: int = 30,
) -> None:
    """Collect weather data using batch API calls for date ranges to minimize requests."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        "BAL": {"name": "M&T Bank Stadium", "lat": 39.2781, "lon": -76.6227},
        "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870},
        "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8533},
        "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167},
        "CIN": {"name": "Paycor Stadium", "lat": 39.0955, "lon": -84.5160},
        "CLE": {"name": "FirstEnergy Stadium", "lat": 41.5061, "lon": -81.6995},
        "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201},
        "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622},
        "JAX": {"name": "TIAA Bank Field", "lat": 32.0815, "lon": -81.6370},
        "KC": {"name": "Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839},
        "MIA": {"name": "Hard Rock Stadium", "lat": 25.9581, "lon": -80.2389},
        "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643},
        "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675},
        "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158},
        "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316},
        "TB": {"name": "Raymond James Stadium", "lat": 27.9756, "lon": -82.5034},
        "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713},
        "WAS": {"name": "FedExField", "lat": 38.9077, "lon": -76.8645},
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
            (datetime.now().date().strftime("%Y-%m-%d"),),
        ).fetchall()

        # Group games by stadium for batch processing
        stadium_games = {}
        for game_id, home_team, game_date in games_for_weather:
            if home_team not in stadium_games:
                stadium_games[home_team] = []
            stadium_games[home_team].append((game_id, game_date))

        weather_count = 0
        api_calls_made = 0

        logger.info(
            f"Found {len(games_for_weather)} games needing weather data across {len(stadium_games)} stadiums"
        )
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
                while (
                    i < len(games)
                    and len(batch_games) < max_days_per_batch
                    and (
                        parse_date_flexible(games[i][1])
                        - parse_date_flexible(batch_start_date)
                    ).days
                    <= max_days_per_batch
                ):
                    batch_games.append(games[i])
                    current_date = games[i][1]
                    i += 1

                if limit and api_calls_made >= limit:
                    break

                # Make batch API call
                batch_end_date = current_date
                try:
                    logger.info(
                        f"Fetching weather batch for {stadium['name']}: {batch_start_date} to {batch_end_date} ({len(batch_games)} games)"
                    )

                    batch_data = get_historical_weather_batch(
                        stadium["lat"],
                        stadium["lon"],
                        batch_start_date,
                        batch_end_date,
                        rate_limit_delay,
                    )
                    api_calls_made += 1

                    if not batch_data:
                        logger.warning(
                            f"Failed to get batch weather for {stadium['name']} - may have hit API limit"
                        )
                        break

                    # Process each day in the batch response
                    if batch_data.get("days"):
                        days_data = {day["datetime"]: day for day in batch_data["days"]}

                        for game_id, game_date in batch_games:
                            if game_date in days_data:
                                day_data = days_data[game_date]

                                weather_record = (
                                    game_id,
                                    stadium["name"],
                                    stadium["lat"],
                                    stadium["lon"],
                                    day_data.get("temp"),
                                    day_data.get("feelslike"),
                                    day_data.get("humidity"),
                                    day_data.get("windspeed"),
                                    day_data.get("winddir"),
                                    day_data.get("precipprob"),
                                    day_data.get("conditions"),
                                    day_data.get("visibility"),
                                    day_data.get("pressure"),
                                )

                                conn.execute(
                                    """INSERT OR REPLACE INTO weather
                                       (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                                        humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                                        visibility, pressure, collected_at)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                                    weather_record,
                                )
                                weather_count += 1
                            else:
                                logger.warning(
                                    f"No weather data for {game_date} in batch response"
                                )

                    if weather_count % 20 == 0:
                        logger.info(
                            f"Collected weather for {weather_count} games ({api_calls_made} API calls)"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error collecting batch weather for {stadium['name']}: {e}"
                    )
                    continue

        conn.commit()
        logger.info(
            f"Stored weather data for {weather_count} games using {api_calls_made} API calls"
        )

    except Exception as e:
        logger.error(f"Error collecting weather data: {e}")
    finally:
        conn.close()


def collect_weather_data_with_limits(
    db_path: str = "data/nfl_dfs.db", limit: int = None, rate_limit_delay: float = 1.5
) -> None:
    """Collect weather data with API rate limiting and request limits."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        "BAL": {"name": "M&T Bank Stadium", "lat": 39.2781, "lon": -76.6227},
        "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870},
        "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8533},
        "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167},
        "CIN": {"name": "Paycor Stadium", "lat": 39.0955, "lon": -84.5160},
        "CLE": {"name": "FirstEnergy Stadium", "lat": 41.5061, "lon": -81.6995},
        "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201},
        "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622},
        "JAX": {"name": "TIAA Bank Field", "lat": 32.0815, "lon": -81.6370},
        "KC": {"name": "Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839},
        "MIA": {"name": "Hard Rock Stadium", "lat": 25.9581, "lon": -80.2389},
        "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643},
        "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675},
        "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158},
        "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316},
        "TB": {"name": "Raymond James Stadium", "lat": 27.9756, "lon": -82.5034},
        "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713},
        "WAS": {"name": "FedExField", "lat": 38.9077, "lon": -76.8645},
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
                if game_date < datetime.now().date().strftime("%Y-%m-%d"):
                    # Historical weather from Visual Crossing API with rate limiting
                    weather_data = get_historical_weather(
                        stadium["lat"],
                        stadium["lon"],
                        game_date,
                        rate_limit_delay=rate_limit_delay,
                    )
                    api_calls_made += 1

                    if not weather_data:
                        # Check if we hit daily limit
                        logger.warning(
                            f"Failed to get weather for {game_id} - may have hit API limit"
                        )
                        break
                else:
                    # Future games - get forecast (free)
                    weather_data = get_weather_forecast(stadium["lat"], stadium["lon"])

                if weather_data:
                    weather_record = (
                        game_id,
                        stadium["name"],
                        stadium["lat"],
                        stadium["lon"],
                        weather_data.get("temperature"),
                        weather_data.get("feels_like"),
                        weather_data.get("humidity"),
                        weather_data.get("wind_speed"),
                        weather_data.get("wind_direction"),
                        weather_data.get("precipitation_chance"),
                        weather_data.get("conditions"),
                        weather_data.get("visibility"),
                        weather_data.get("pressure"),
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO weather
                           (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                            humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                            visibility, pressure, collected_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                        weather_record,
                    )
                    weather_count += 1
                    if weather_count % 10 == 0:
                        logger.info(
                            f"Collected weather for {weather_count} games ({api_calls_made} API calls)"
                        )

            except Exception as e:
                logger.warning(f"Error collecting weather for {game_id}: {e}")
                continue

        conn.commit()
        logger.info(
            f"Stored weather data for {weather_count} games (made {api_calls_made} API calls)"
        )

    except Exception as e:
        logger.error(f"Error collecting weather data: {e}")
    finally:
        conn.close()


def collect_weather_data(db_path: str = "data/nfl_dfs.db") -> None:
    """Collect weather data for upcoming games using weather.gov API."""
    conn = get_db_connection(db_path)

    # NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
    outdoor_stadiums = {
        "BAL": {"name": "M&T Bank Stadium", "lat": 39.2781, "lon": -76.6227},
        "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870},
        "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8533},
        "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167},
        "CIN": {"name": "Paycor Stadium", "lat": 39.0955, "lon": -84.5160},
        "CLE": {"name": "FirstEnergy Stadium", "lat": 41.5061, "lon": -81.6995},
        "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201},
        "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622},
        "JAX": {"name": "TIAA Bank Field", "lat": 32.0815, "lon": -81.6370},
        "KC": {"name": "Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839},
        "MIA": {"name": "Hard Rock Stadium", "lat": 25.9581, "lon": -80.2389},
        "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643},
        "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
        "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675},
        "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158},
        "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316},
        "TB": {"name": "Raymond James Stadium", "lat": 27.9756, "lon": -82.5034},
        "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713},
        "WAS": {"name": "FedExField", "lat": 38.9077, "lon": -76.8645},
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
                if game_date < datetime.now().date().strftime("%Y-%m-%d"):
                    # Historical weather from Visual Crossing API with rate limiting
                    weather_data = get_historical_weather(
                        stadium["lat"], stadium["lon"], game_date, rate_limit_delay=1.5
                    )
                    if not weather_data:
                        # Fallback to placeholder if API fails
                        weather_data = {
                            "temperature": None,
                            "feels_like": None,
                            "humidity": None,
                            "wind_speed": None,
                            "wind_direction": None,
                            "precipitation_chance": None,
                            "conditions": "Historical data unavailable",
                            "visibility": None,
                            "pressure": None,
                        }
                else:
                    # Future games - get forecast
                    weather_data = get_weather_forecast(stadium["lat"], stadium["lon"])

                if weather_data:
                    weather_record = (
                        game_id,
                        stadium["name"],
                        stadium["lat"],
                        stadium["lon"],
                        weather_data.get("temperature"),
                        weather_data.get("feels_like"),
                        weather_data.get("humidity"),
                        weather_data.get("wind_speed"),
                        weather_data.get("wind_direction"),
                        weather_data.get("precipitation_chance"),
                        weather_data.get("conditions"),
                        weather_data.get("visibility"),
                        weather_data.get("pressure"),
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO weather
                           (game_id, stadium_name, latitude, longitude, temperature, feels_like,
                            humidity, wind_speed, wind_direction, precipitation_chance, conditions,
                            visibility, pressure)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        weather_record,
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


def get_historical_weather_batch(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    rate_limit_delay: float = 1.0,
) -> Optional[Dict]:
    """Get historical weather data for a date range from Visual Crossing API."""
    visual_crossing_key = os.environ.get("VISUAL_CROSSING_API_KEY")
    if not visual_crossing_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    try:
        # Rate limiting - wait between requests
        time.sleep(rate_limit_delay)

        # Visual Crossing Timeline API for date range: location/start_date/end_date
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
        params = {
            "key": visual_crossing_key,
            "unitGroup": "us",  # Fahrenheit, mph, inches
            "include": "days",  # Only daily data
            "elements": "datetime,temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure",
        }

        response = requests.get(
            url, params=params, timeout=30
        )  # Longer timeout for batch requests

        # Log response headers for rate limit debugging
        logger.debug(
            f"API headers for {start_date} to {end_date}: {dict(response.headers)}"
        )

        if response.status_code == 429:
            logger.warning(
                f"Rate limit exceeded for {start_date} to {end_date} at {lat},{lon}"
            )
            # Check for Retry-After header
            retry_after = response.headers.get("Retry-After", "60")
            try:
                wait_time = int(retry_after)
            except ValueError:
                wait_time = 60
            logger.warning(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            return None
        elif (
            response.status_code == 400
            and "Maximum daily cost exceeded" in response.text
        ):
            logger.error("Daily API limit exceeded - stopping weather collection")
            return None
        elif response.status_code != 200:
            logger.warning(
                f"Visual Crossing API error: {response.status_code} for {start_date} to {end_date} at {lat},{lon}"
            )
            logger.debug(f"Response body: {response.text[:200]}")
            if response.status_code == 401:
                logger.warning(
                    "API authentication failed - check VISUAL_CROSSING_API_KEY"
                )
            return None

        data = response.json()
        query_cost = data.get("queryCost", 0)
        logger.debug(
            f"API batch response for {start_date} to {end_date}: {query_cost} query cost"
        )

        # Log remaining quota if available in response
        if "remainingCost" in data:
            logger.debug(f"Remaining API cost: {data['remainingCost']}")

        return data

    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Network error fetching batch weather for {start_date} to {end_date}: {e}"
        )
        return None
    except ValueError as e:
        logger.warning(
            f"JSON parsing error for batch weather {start_date} to {end_date}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error fetching batch weather for {start_date} to {end_date}: {e}"
        )
        return None


def get_historical_weather(
    lat: float, lon: float, game_date: str, rate_limit_delay: float = 1.0
) -> Optional[Dict]:
    """Get historical weather data from Visual Crossing API with rate limiting."""
    visual_crossing_key = os.environ.get("VISUAL_CROSSING_API_KEY")
    if not visual_crossing_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    try:
        # Rate limiting - wait between requests
        time.sleep(rate_limit_delay)

        # Visual Crossing Timeline API for historical weather
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{game_date}"
        params = {
            "key": visual_crossing_key,
            "unitGroup": "us",  # Fahrenheit, mph, inches
            "include": "days",  # Only daily data
            "elements": "temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure",
        }

        response = requests.get(url, params=params, timeout=15)

        # Log response headers for rate limit debugging
        logger.debug(f"API headers for {game_date}: {dict(response.headers)}")

        if response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {game_date} at {lat},{lon}")
            # Check for Retry-After header
            retry_after = response.headers.get("Retry-After", "60")
            try:
                wait_time = int(retry_after)
            except ValueError:
                wait_time = 60
            logger.warning(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            return None
        elif (
            response.status_code == 400
            and "Maximum daily cost exceeded" in response.text
        ):
            logger.error("Daily API limit exceeded - stopping weather collection")
            return None
        elif response.status_code != 200:
            logger.warning(
                f"Visual Crossing API error: {response.status_code} for {game_date} at {lat},{lon}"
            )
            logger.debug(f"Response body: {response.text[:200]}")
            if response.status_code == 401:
                logger.warning(
                    "API authentication failed - check VISUAL_CROSSING_API_KEY"
                )
            return None

        data = response.json()
        query_cost = data.get("queryCost", 0)
        logger.debug(f"API response for {game_date}: {query_cost} query cost")

        # Log remaining quota if available in response
        if "remainingCost" in data:
            logger.debug(f"Remaining API cost: {data['remainingCost']}")

        # Extract daily weather data
        if data.get("days") and len(data["days"]) > 0:
            day_data = data["days"][0]

            return {
                "temperature": day_data.get("temp"),
                "feels_like": day_data.get("feelslike"),
                "humidity": day_data.get("humidity"),
                "wind_speed": day_data.get("windspeed"),
                "wind_direction": day_data.get("winddir"),
                "precipitation_chance": day_data.get("precipprob"),
                "conditions": day_data.get("conditions", ""),
                "visibility": day_data.get("visibility"),
                "pressure": day_data.get("pressure"),
            }

    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Network error fetching historical weather for {game_date}: {e}"
        )
        return None
    except ValueError as e:
        logger.warning(f"JSON parsing error for historical weather {game_date}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error fetching historical weather for {game_date}: {e}"
        )
        return None


def get_weather_forecast(lat: float, lon: float) -> Optional[Dict]:
    """Get weather forecast from weather.gov API."""
    try:
        # First get the grid coordinates
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        headers = {"User-Agent": "NFL-DFS-Weather-Collector (contact@example.com)"}

        response = requests.get(points_url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Weather.gov points API error: {response.status_code}")
            return None

        points_data = response.json()
        forecast_url = points_data["properties"]["forecast"]

        # Get the forecast
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        if forecast_response.status_code != 200:
            logger.warning(
                f"Weather.gov forecast API error: {forecast_response.status_code}"
            )
            return None

        forecast_data = forecast_response.json()

        # Extract current conditions (first period)
        if forecast_data.get("properties", {}).get("periods"):
            current = forecast_data["properties"]["periods"][0]

            return {
                "temperature": current.get("temperature"),
                "feels_like": current.get(
                    "temperature"
                ),  # weather.gov doesn't provide feels like
                "humidity": None,  # Not in basic forecast
                "wind_speed": parse_wind_speed(current.get("windSpeed", "")),
                "wind_direction": current.get("windDirection", ""),
                "precipitation_chance": None,  # Would need detailed forecast
                "conditions": current.get("shortForecast", ""),
                "visibility": None,  # Not in basic forecast
                "pressure": None,  # Not in basic forecast
            }

    except Exception as e:
        logger.warning(f"Error fetching weather forecast: {e}")
        return None


def parse_wind_speed(wind_str: str) -> Optional[int]:
    """Parse wind speed from string like '10 mph'."""
    try:
        if wind_str and "mph" in wind_str:
            return int(wind_str.split()[0])
    except (ValueError, AttributeError, IndexError):
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
            game_id = str(game.get("game_id", ""))
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            home_score = (
                int(game.get("home_score", 0))
                if pd.notna(game.get("home_score"))
                else 0
            )
            away_score = (
                int(game.get("away_score", 0))
                if pd.notna(game.get("away_score"))
                else 0
            )
            week = int(game.get("week", 0)) if pd.notna(game.get("week")) else 0

            # Skip games without scores (not played yet)
            if home_score == 0 and away_score == 0:
                continue

            # Get plays for this game
            game_plays = pbp_data[pbp_data["game_id"] == game_id]

            if len(game_plays) == 0:
                continue

            # Calculate stats for each team
            for team, _opponent, points_allowed in [
                (home_team, away_team, away_score),
                (away_team, home_team, home_score),
            ]:
                if not team:
                    continue

                # Aggregate defensive stats when this team was defending
                team_defense_plays = game_plays[game_plays["defteam"] == team]

                dst_stats = {
                    "points_allowed": points_allowed,
                    "sacks": int(team_defense_plays["sack"].sum() or 0),
                    "interceptions": int(team_defense_plays["interception"].sum() or 0),
                    "fumbles_recovered": int(
                        team_defense_plays["fumble_lost"].sum() or 0
                    ),  # fumble_lost by offense = recovered by defense
                    "fumbles_forced": int(
                        team_defense_plays["fumble_lost"].sum() or 0
                    ),  # Same as recovered for now
                    "safeties": int(team_defense_plays["safety"].sum() or 0),
                    "defensive_tds": 0,  # Would need more complex logic to identify defensive TDs
                    "return_tds": 0,  # Would need return TD logic
                    "special_teams_tds": 0,  # Would need special teams logic
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
                    (
                        game_id,
                        team,
                        season,
                        week,
                        dst_stats["points_allowed"],
                        dst_stats["sacks"],
                        dst_stats["interceptions"],
                        dst_stats["fumbles_recovered"],
                        dst_stats["fumbles_forced"],
                        dst_stats["safeties"],
                        dst_stats["defensive_tds"],
                        dst_stats["return_tds"],
                        dst_stats["special_teams_tds"],
                        fantasy_points,
                    ),
                )

        logger.info(f"Completed DST data collection for {season} season")

    except Exception as e:
        logger.error(f"Error collecting DST data for {season}: {e}")
        raise


def load_draftkings_csv(
    csv_path: str, contest_id: str = None, db_path: str = "data/nfl_dfs.db"
) -> None:
    """Load DraftKings salary CSV file."""
    conn = get_db_connection(db_path)

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                player_name = row.get("Name", "").strip()
                salary = int(row.get("Salary", 0))
                roster_position = row.get("Roster Position", "").strip()
                game_info = row.get("Game Info", "").strip()
                team_abbr = row.get("TeamAbbrev", "").strip()

                # Use provided contest_id or generate one from game date in CSV
                if contest_id is None:
                    import re
                    from pathlib import Path

                    # Extract date from game_info (e.g., "CIN@CLE 09/07/2025 01:00PM ET")
                    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", game_info)
                    if date_match:
                        game_date = date_match.group(1).replace("/", "")
                        contest_id = f"DK_{game_date}"
                    else:
                        filename = Path(csv_path).stem
                        contest_id = f"{filename}_unknown"

                # Handle DST/Defense teams specially
                if roster_position == "DST":
                    # Create or find defense "player" entry
                    player_id = get_or_create_defense_player(team_abbr, conn)
                else:
                    # Try to find matching individual player
                    player_id = find_player_by_name_and_team(
                        player_name, team_abbr, conn
                    )

                if player_id:
                    # Check if this player already exists for this contest
                    existing = conn.execute(
                        "SELECT id FROM draftkings_salaries WHERE contest_id = ? AND player_id = ?",
                        (contest_id, player_id),
                    ).fetchone()

                    if existing:
                        # Update existing entry
                        conn.execute(
                            """UPDATE draftkings_salaries
                               SET salary = ?, roster_position = ?, game_info = ?, team_abbr = ?, opponent = ?
                               WHERE contest_id = ? AND player_id = ?""",
                            (
                                salary,
                                roster_position,
                                game_info,
                                team_abbr,
                                "",
                                contest_id,
                                player_id,
                            ),
                        )
                    else:
                        # Insert new entry
                        conn.execute(
                            """INSERT INTO draftkings_salaries
                               (contest_id, player_id, salary, roster_position, game_info, team_abbr, opponent)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                contest_id,
                                player_id,
                                salary,
                                roster_position,
                                game_info,
                                team_abbr,
                                "",
                            ),
                        )
                else:
                    logger.warning(
                        f"Could not find player: {player_name} ({team_abbr})"
                    )

        conn.commit()
        logger.info(f"Loaded DraftKings salaries from {csv_path}")

    except Exception as e:
        logger.error(f"Error loading DraftKings CSV: {e}")
        raise
    finally:
        conn.close()


def get_or_create_defense_player(
    team_abbr: str, conn: sqlite3.Connection
) -> Optional[int]:
    """Get or create defense player entry for a team."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    defense_name = f"{team_abbr} Defense"

    # Try to find existing defense player
    cursor = conn.execute(
        "SELECT id FROM players WHERE player_name = ? AND team_id = ? AND position = ?",
        (defense_name, team_id, "DST"),
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Create new defense player
    cursor = conn.execute(
        """INSERT INTO players (player_name, display_name, position, team_id, gsis_id)
           VALUES (?, ?, ?, ?, ?)""",
        (defense_name, defense_name, "DST", team_id, f"DST_{team_abbr}"),
    )

    return cursor.lastrowid


def find_player_by_name_and_team(
    name: str, team_abbr: str, conn: sqlite3.Connection
) -> Optional[int]:
    """Find player by name and team."""
    team_id = get_team_id_by_abbr(team_abbr, conn)
    if not team_id:
        return None

    # Try exact match first
    cursor = conn.execute(
        "SELECT id FROM players WHERE display_name = ? AND team_id = ?", (name, team_id)
    )
    result = cursor.fetchone()
    if result:
        return result[0]

    # Try partial match (for name variations)
    cursor = conn.execute(
        "SELECT id FROM players WHERE display_name LIKE ? AND team_id = ?",
        (f"%{name}%", team_id),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def get_defensive_matchup_features(
    team_abbr: str,
    opponent_abbr: str,
    season: int,
    week: int,
    lookback_weeks: int = 4,
    db_path: str = "data/nfl_dfs.db",
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
            (team_abbr, season, week),
        ).fetchone()

        if def_vs_qb:
            features.update(
                {
                    "def_sack_rate": def_vs_qb[0] or 0,
                    "def_int_rate": def_vs_qb[1] or 0,
                    "def_pressure_rate": def_vs_qb[2] or 0,
                }
            )

        # Run defense efficiency
        def_vs_run = conn.execute(
            """SELECT
                AVG(yards_gained) as avg_yards_allowed,
                AVG(CASE WHEN yards_gained <= 3 THEN 1.0 ELSE 0.0 END) as stuff_rate,
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as td_rate
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND rush_attempt = 1""",
            (team_abbr, season, week),
        ).fetchone()

        if def_vs_run:
            features.update(
                {
                    "def_run_yards_allowed": def_vs_run[0] or 0,
                    "def_stuff_rate": def_vs_run[1] or 0,
                    "def_rush_td_rate": def_vs_run[2] or 0,
                }
            )

        # Red zone defense
        red_zone_def = conn.execute(
            """SELECT
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as rz_td_rate,
                COUNT(*) as rz_plays
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND yardline_100 <= 20 AND yardline_100 > 0""",
            (team_abbr, season, week),
        ).fetchone()

        if red_zone_def:
            features.update(
                {
                    "def_rz_td_rate": red_zone_def[0] or 0,
                    "def_rz_plays": red_zone_def[1] or 0,
                }
            )

        # 3rd down defense
        third_down_def = conn.execute(
            """SELECT
                AVG(CASE WHEN yards_gained >= ydstogo THEN 1.0 ELSE 0.0 END) as third_down_conv_rate
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ?
               AND down = 3""",
            (team_abbr, season, week),
        ).fetchone()

        if third_down_def:
            features["def_3rd_down_rate"] = third_down_def[0] or 0

        # Opponent offensive tendencies vs this defense
        opp_vs_def = conn.execute(
            """SELECT
                AVG(CASE WHEN pass_attempt = 1 THEN 1.0 ELSE 0.0 END) as pass_rate,
                AVG(CASE WHEN rush_attempt = 1 THEN 1.0 ELSE 0.0 END) as rush_rate,
                AVG(yards_gained) as avg_yards_gained
               FROM play_by_play
               WHERE posteam = ? AND defteam = ? AND season = ?
               AND (pass_attempt = 1 OR rush_attempt = 1)""",
            (opponent_abbr, team_abbr, season),
        ).fetchone()

        if opp_vs_def:
            features.update(
                {
                    "opp_pass_rate_vs_def": opp_vs_def[0] or 0.5,
                    "opp_rush_rate_vs_def": opp_vs_def[1] or 0.5,
                    "opp_avg_yards_vs_def": opp_vs_def[2] or 0,
                }
            )

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
    db_path: str = "data/nfl_dfs.db",
) -> Dict[str, float]:
    """Extract player-specific PbP features vs this defense."""
    conn = get_db_connection(db_path)
    features = {}

    try:
        # Get player name for PbP matching (if available in description)
        player_name = conn.execute(
            "SELECT display_name FROM players WHERE id = ?", (player_id,)
        ).fetchone()

        if not player_name:
            return features

        player_name = player_name[0]

        if position == "QB":
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
                (
                    team_abbr,
                    opponent_abbr,
                    season - 2,
                    f"%{player_name}%",
                    f"%{player_name.split()[0]}%",
                ),
            ).fetchone()

            if qb_vs_def and qb_vs_def[3] > 0:  # Has plays vs this defense
                features.update(
                    {
                        "qb_sack_rate_vs_def": qb_vs_def[0] or 0,
                        "qb_int_rate_vs_def": qb_vs_def[1] or 0,
                        "qb_avg_yards_vs_def": qb_vs_def[2] or 0,
                        "qb_experience_vs_def": min(
                            qb_vs_def[3] / 10.0, 1.0
                        ),  # Normalize experience
                    }
                )

        elif position == "RB":
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
                (
                    team_abbr,
                    opponent_abbr,
                    season - 2,
                    f"%{player_name}%",
                    f"%{player_name.split()[0]}%",
                ),
            ).fetchone()

            if rb_vs_def and rb_vs_def[3] > 0:
                features.update(
                    {
                        "rb_avg_yards_vs_def": rb_vs_def[0] or 0,
                        "rb_td_rate_vs_def": rb_vs_def[1] or 0,
                        "rb_stuff_rate_vs_def": rb_vs_def[2] or 0,
                        "rb_carries_vs_def": min(rb_vs_def[3] / 20.0, 1.0),
                    }
                )

        elif position in ["WR", "TE"]:
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
                (
                    team_abbr,
                    opponent_abbr,
                    season - 2,
                    f"%{player_name}%",
                    f"%{player_name.split()[0]}%",
                ),
            ).fetchone()

            if rec_vs_def and rec_vs_def[1] > 0:  # Has targets vs this defense
                catch_rate = rec_vs_def[0] / rec_vs_def[1] if rec_vs_def[1] > 0 else 0
                features.update(
                    {
                        "rec_catch_rate_vs_def": catch_rate,
                        "rec_avg_yards_vs_def": rec_vs_def[2] or 0,
                        "rec_td_rate_vs_def": rec_vs_def[3] or 0,
                        "rec_targets_vs_def": min(rec_vs_def[1] / 15.0, 1.0),
                    }
                )

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
            (season, week, position, position, position),
        ).fetchone()

        if def_rank_vs_pos:
            features.update(
                {
                    f"def_rank_vs_{position.lower()}_yards": def_rank_vs_pos[0] or 16,
                    f"def_rank_vs_{position.lower()}_tds": def_rank_vs_pos[1] or 16,
                }
            )

    except Exception as e:
        logger.error(f"Error extracting player vs defense features: {e}")
    finally:
        conn.close()

    return features


def get_rb_specific_features(
    player_id, player_name, team_abbr, opponent_abbr, season, week, conn
):
    """Extract RB-specific features including red zone and receiving metrics."""
    features = {}

    try:
        # Get recent games for rolling windows
        recent_games = conn.execute(
            """
            SELECT DISTINCT g.id
            FROM games g
            JOIN player_stats ps ON g.id = ps.game_id
            WHERE ps.player_id = ?
            AND (g.season < ? OR (g.season = ? AND g.week < ?))
            ORDER BY g.season DESC, g.week DESC
            LIMIT 5
        """,
            (player_id, season, season, week),
        ).fetchall()

        if not recent_games:
            return features

        game_ids = [g[0] for g in recent_games]
        game_id_placeholders = ",".join(["?" for _ in game_ids])

        # Red zone and goal line stats from play_by_play
        rz_query = f"""
            SELECT
                COUNT(CASE WHEN yardline_100 <= 20 THEN 1 END) as rz_attempts,
                COUNT(CASE WHEN yardline_100 <= 10 THEN 1 END) as inside_10,
                COUNT(CASE WHEN yardline_100 <= 5 THEN 1 END) as inside_5,
                COUNT(CASE WHEN touchdown = 1 THEN 1 END) as tds,
                COUNT(*) as total_attempts,
                AVG(yards_gained) as avg_yards
            FROM play_by_play
            WHERE game_id IN ({game_id_placeholders})
            AND rush_attempt = 1
            AND description LIKE ?
        """

        rz_stats = conn.execute(
            rz_query, (*game_ids, f"%{player_name.split()[0]}%")
        ).fetchone()

        if rz_stats and rz_stats[4] > 0:  # Has attempts
            features["rb_rz_attempts_pg"] = rz_stats[0] / len(game_ids)
            features["rb_inside10_attempts_pg"] = rz_stats[1] / len(game_ids)
            features["rb_inside5_attempts_pg"] = rz_stats[2] / len(game_ids)
            features["rb_td_rate"] = rz_stats[3] / rz_stats[4] if rz_stats[4] > 0 else 0
            features["rb_ypc"] = rz_stats[5] or 0

        # Get team red zone attempts for share calculation
        team_rz_query = f"""
            SELECT COUNT(CASE WHEN yardline_100 <= 20 THEN 1 END) as team_rz
            FROM play_by_play
            WHERE game_id IN ({game_id_placeholders})
            AND posteam = ?
            AND rush_attempt = 1
        """

        team_rz = conn.execute(team_rz_query, (*game_ids, team_abbr)).fetchone()
        if team_rz and team_rz[0] > 0 and "rb_rz_attempts_pg" in features:
            features["rb_rz_share"] = (
                features["rb_rz_attempts_pg"] * len(game_ids)
            ) / team_rz[0]

        # Volume and receiving stats from player_stats
        volume_query = f"""
            SELECT
                AVG(rushing_attempts) as avg_rushes,
                AVG(targets) as avg_targets,
                AVG(receptions) as avg_rec,
                AVG(rushing_attempts + receptions) as avg_touches,
                MIN(rushing_attempts + receptions) as touch_floor,
                MAX(rushing_attempts + receptions) as touch_ceiling,
                AVG(rushing_yards) as avg_rush_yds,
                AVG(receiving_yards) as avg_rec_yds,
                AVG(rushing_tds + receiving_tds) as avg_tds
            FROM player_stats
            WHERE player_id = ?
            AND game_id IN ({game_id_placeholders})
        """

        volume_stats = conn.execute(volume_query, (player_id, *game_ids)).fetchone()

        if volume_stats:
            features.update(
                {
                    "rb_avg_rushes": volume_stats[0] or 0,
                    "rb_avg_targets": volume_stats[1] or 0,
                    "rb_avg_receptions": volume_stats[2] or 0,
                    "rb_avg_touches": volume_stats[3] or 0,
                    "rb_touch_floor": volume_stats[4] or 0,
                    "rb_touch_ceiling": volume_stats[5] or 0,
                    "rb_avg_rush_yards": volume_stats[6] or 0,
                    "rb_avg_rec_yards": volume_stats[7] or 0,
                    "rb_avg_total_tds": volume_stats[8] or 0,
                }
            )

        # Game script features from betting odds
        game_script = conn.execute(
            """
            SELECT
                b.spread_favorite,
                b.over_under_line,
                CASE
                    WHEN ht.abbreviation = ? THEN b.home_team_spread
                    ELSE b.away_team_spread
                END as team_spread
            FROM games g
            JOIN betting_odds b ON g.id = b.game_id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.season = ? AND g.week = ?
            AND (ht.abbreviation = ? OR at.abbreviation = ?)
        """,
            (team_abbr, season, week, team_abbr, team_abbr),
        ).fetchone()

        if game_script:
            features["rb_team_spread"] = game_script[2] or 0
            features["rb_game_total"] = game_script[1] or 47
            features["rb_implied_total"] = (
                (game_script[1] / 2.0) - (game_script[2] / 2.0)
                if game_script[1]
                else 23.5
            )
            features["rb_positive_script"] = (
                1 if game_script[2] and game_script[2] < 0 else 0
            )

        # Opponent defense vs RB
        opp_def = conn.execute(
            """
            SELECT
                AVG(ps.rushing_yards + ps.receiving_yards) as avg_yards_allowed,
                AVG(ps.fantasy_points) as avg_fp_allowed
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.id
            JOIN teams t ON ps.team_id = t.id
            JOIN players p ON ps.player_id = p.player_id
            WHERE p.position = 'RB'
            AND t.abbreviation != ?
            AND (
                (g.home_team_id = (SELECT id FROM teams WHERE abbreviation = ?) AND
                 g.away_team_id = t.id) OR
                (g.away_team_id = (SELECT id FROM teams WHERE abbreviation = ?) AND
                 g.home_team_id = t.id)
            )
            AND g.season = ? AND g.week < ?
            AND g.week >= ?
        """,
            (
                opponent_abbr,
                opponent_abbr,
                opponent_abbr,
                season,
                week,
                max(1, week - 5),
            ),
        ).fetchone()

        if opp_def:
            features["rb_opp_yards_allowed"] = opp_def[0] or 85
            features["rb_opp_fp_allowed"] = opp_def[1] or 12

    except Exception as e:
        logger.error(f"Error extracting RB features: {e}")

    return features


def get_wr_specific_features(
    player_id, player_name, team_abbr, opponent_abbr, season, week, conn
):
    """Extract WR-specific features including target share and route metrics."""
    features = {}

    try:
        # Get recent games for rolling windows
        recent_games = conn.execute(
            """
            SELECT DISTINCT g.id
            FROM games g
            JOIN player_stats ps ON g.id = ps.game_id
            WHERE ps.player_id = ?
            AND (g.season < ? OR (g.season = ? AND g.week < ?))
            ORDER BY g.season DESC, g.week DESC
            LIMIT 5
        """,
            (player_id, season, season, week),
        ).fetchall()

        if not recent_games:
            return features

        game_ids = [g[0] for g in recent_games]
        game_id_placeholders = ",".join(["?" for _ in game_ids])

        # Target share and volume from player_stats
        target_query = f"""
            SELECT
                AVG(targets) as avg_targets,
                AVG(receptions) as avg_receptions,
                AVG(receiving_yards) as avg_rec_yards,
                AVG(receiving_tds) as avg_rec_tds,
                MIN(targets) as target_floor,
                MAX(targets) as target_ceiling,
                AVG(CASE WHEN targets > 0 THEN receptions * 1.0 / targets ELSE 0.6 END) as catch_rate,
                AVG(CASE WHEN receptions > 0 THEN receiving_yards * 1.0 / receptions ELSE 10 END) as yards_per_rec,
                AVG(CASE WHEN targets > 0 THEN receiving_yards * 1.0 / targets ELSE 6 END) as yards_per_target
            FROM player_stats
            WHERE player_id = ?
            AND game_id IN ({game_id_placeholders})
        """

        target_stats = conn.execute(target_query, (player_id, *game_ids)).fetchone()

        if target_stats:
            features.update(
                {
                    "wr_avg_targets": target_stats[0] or 0,
                    "wr_avg_receptions": target_stats[1] or 0,
                    "wr_avg_rec_yards": target_stats[2] or 0,
                    "wr_avg_rec_tds": target_stats[3] or 0,
                    "wr_target_floor": target_stats[4] or 0,
                    "wr_target_ceiling": target_stats[5] or 0,
                    "wr_catch_rate": target_stats[6] or 0.6,
                    "wr_yards_per_rec": target_stats[7] or 10,
                    "wr_yards_per_target": target_stats[8] or 6,
                }
            )

        # Team target share calculation - simplified approach
        team_targets_query = f"""
            SELECT
                AVG(total_team_targets) as avg_team_targets
            FROM (
                SELECT
                    g.id,
                    SUM(ps.targets) as total_team_targets
                FROM games g
                JOIN player_stats ps ON g.id = ps.game_id
                JOIN players p ON ps.player_id = p.id
                WHERE g.id IN ({game_id_placeholders})
                AND p.position IN ('WR', 'TE', 'RB')
                GROUP BY g.id
            )
        """

        team_targets = conn.execute(team_targets_query, (*game_ids,)).fetchone()
        if team_targets and team_targets[0] > 0 and "wr_avg_targets" in features:
            features["wr_target_share"] = features["wr_avg_targets"] / team_targets[0]

        # Red zone targets from play_by_play
        rz_targets_query = f"""
            SELECT
                COUNT(CASE WHEN yardline_100 <= 20 THEN 1 END) as rz_targets,
                COUNT(CASE WHEN yardline_100 <= 10 THEN 1 END) as ez_targets,
                8 as avg_air_yards,
                4 as avg_yac
            FROM play_by_play
            WHERE game_id IN ({game_id_placeholders})
            AND pass_attempt = 1
            AND (receiver LIKE ? OR description LIKE ?)
        """

        rz_stats = conn.execute(
            rz_targets_query,
            (*game_ids, f"%{player_name.split()[0]}%", f"%{player_name.split()[0]}%"),
        ).fetchone()

        if rz_stats:
            features.update(
                {
                    "wr_rz_targets_pg": (rz_stats[0] or 0) / len(game_ids),
                    "wr_ez_targets_pg": (rz_stats[1] or 0) / len(game_ids),
                    "wr_avg_air_yards": rz_stats[2] or 8,
                    "wr_avg_yac": rz_stats[3] or 4,
                }
            )

        # Game script features from betting odds
        game_script = conn.execute(
            """
            SELECT
                b.over_under_line,
                CASE
                    WHEN ht.abbreviation = ? THEN b.home_team_spread
                    ELSE b.away_team_spread
                END as team_spread
            FROM games g
            JOIN betting_odds b ON g.id = b.game_id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.season = ? AND g.week = ?
            AND (ht.abbreviation = ? OR at.abbreviation = ?)
        """,
            (team_abbr, season, week, team_abbr, team_abbr),
        ).fetchone()

        if game_script:
            over_under = game_script[0] or 47
            spread = game_script[1] or 0
            implied_total = (over_under / 2.0) - (spread / 2.0)

            features.update(
                {
                    "wr_game_total": over_under,
                    "wr_team_spread": spread,
                    "wr_implied_total": implied_total,
                    "wr_shootout_game": 1 if over_under > 50 else 0,
                    "wr_pass_heavy_script": 1 if (spread > 7 or over_under > 50) else 0,
                    "wr_garbage_time_upside": 1
                    if (spread > 10 and over_under > 48)
                    else 0,
                }
            )

        # Opponent pass defense - simplified query
        opp_def = conn.execute(
            """
            SELECT
                AVG(ps.receiving_yards) as avg_yards_allowed,
                AVG(ps.fantasy_points) as avg_fp_allowed,
                COUNT(*) as games_played
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.id
            JOIN players p ON ps.player_id = p.id
            WHERE p.position = 'WR'
            AND g.season = ?
            AND g.week < ?
            AND g.week >= ?
        """,
            (season, week, max(1, week - 5)),
        ).fetchone()

        if opp_def and opp_def[2] > 0:  # Check we have data
            features["wr_opp_yards_allowed"] = opp_def[0] or 65
            features["wr_opp_fp_allowed"] = opp_def[1] or 10

    except Exception as e:
        logger.error(f"Error extracting WR features: {e}")

    return features


def get_te_specific_features(
    player_id: int,
    player_name: str,
    team_abbr: str,
    opponent_abbr: str,
    season: int,
    week: int,
    conn: sqlite3.Connection,
    lookback_weeks: int = 6,
) -> Dict[str, float]:
    """Extract TE-specific features for enhanced prediction accuracy."""
    features = {}

    try:
        # 1. Red Zone Target Share (most predictive for TEs)
        rz_query = """
        SELECT
            COALESCE(SUM(CASE WHEN ps.targets > 0 AND ps.red_zone_targets > 0 THEN ps.red_zone_targets ELSE 0 END), 0) as player_rz_targets,
            COALESCE(SUM(team_totals.total_rz_targets), 1) as team_rz_targets
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        LEFT JOIN (
            SELECT
                ps2.game_id,
                ps2.team_id,
                SUM(CASE WHEN ps2.red_zone_targets > 0 THEN ps2.red_zone_targets ELSE 0 END) as total_rz_targets
            FROM player_stats ps2
            GROUP BY ps2.game_id, ps2.team_id
        ) team_totals ON ps.game_id = team_totals.game_id AND ps.team_id = team_totals.team_id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        rz_params = (
            player_id,
            team_abbr,
            season,
            max(1, week - lookback_weeks),
            week - 1,
        )
        rz_result = conn.execute(rz_query, rz_params).fetchone()
        if rz_result:
            player_rz, team_rz = rz_result
            features["te_rz_target_share"] = player_rz / max(team_rz, 1)
        else:
            features["te_rz_target_share"] = 0.0

        # 2. Two-TE Set Usage (formation-based opportunities)
        formation_query = """
        SELECT
            AVG(CASE WHEN ps.snaps > 0 THEN ps.snaps ELSE 0 END) as avg_snaps,
            AVG(team_totals.te_snaps) as avg_team_te_snaps
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        LEFT JOIN (
            SELECT
                ps2.game_id,
                t2.id as team_id,
                SUM(CASE WHEN p2.position = 'TE' AND ps2.snaps > 0 THEN ps2.snaps ELSE 0 END) as te_snaps
            FROM player_stats ps2
            JOIN players p2 ON ps2.player_id = p2.id
            JOIN teams t2 ON p2.team_id = t2.id
            GROUP BY ps2.game_id, t2.id
        ) team_totals ON ps.game_id = team_totals.game_id AND ps.team_id = team_totals.team_id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        formation_result = conn.execute(formation_query, rz_params).fetchone()
        if formation_result and formation_result[0] is not None:
            player_snaps, team_te_snaps = formation_result
            features["te_snap_share"] = (
                player_snaps / max(team_te_snaps, 1) if team_te_snaps else 0
            )
            features["te_two_te_sets"] = (
                min(team_te_snaps / 65.0, 1.0) if team_te_snaps else 0
            )
        else:
            features["te_snap_share"] = 0.5
            features["te_two_te_sets"] = 0.3

        # 3. Route Concentration (slot vs wide usage)
        route_query = """
        SELECT
            AVG(CASE WHEN ps.targets > 0 THEN ps.targets ELSE 0 END) as avg_targets,
            AVG(CASE WHEN ps.receptions > 0 THEN ps.receptions ELSE 0 END) as avg_receptions,
            AVG(CASE WHEN ps.receiving_yards > 0 THEN ps.receiving_yards ELSE 0 END) as avg_yards,
            COUNT(*) as games_played
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        route_result = conn.execute(route_query, rz_params).fetchone()
        if route_result and route_result[3] > 0:
            avg_targets, avg_receptions, avg_yards, games = route_result
            features["te_target_efficiency"] = (
                avg_receptions / max(avg_targets, 1) if avg_targets else 0
            )
            features["te_yards_per_target"] = (
                avg_yards / max(avg_targets, 1) if avg_targets else 0
            )
            features["te_route_volume"] = avg_targets
        else:
            features["te_target_efficiency"] = 0.6
            features["te_yards_per_target"] = 8.5
            features["te_route_volume"] = 4.0

        # 4. Goal Line Opportunities
        gl_query = """
        SELECT
            COALESCE(SUM(CASE WHEN ps.red_zone_touches > 0 THEN ps.red_zone_touches ELSE 0 END), 0) as rz_touches,
            COALESCE(SUM(CASE WHEN ps.touchdowns > 0 THEN ps.touchdowns ELSE 0 END), 0) as tds,
            COUNT(*) as games
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        gl_result = conn.execute(gl_query, rz_params).fetchone()
        if gl_result:
            rz_touches, tds, games = gl_result
            features["te_rz_touch_rate"] = rz_touches / max(games, 1)
            features["te_td_rate"] = tds / max(games, 1)
        else:
            features["te_rz_touch_rate"] = 0.5
            features["te_td_rate"] = 0.15

        # 5. Game Script Dependency
        game_script_query = """
        SELECT
            AVG(CASE WHEN team_totals.team_score > opp_totals.opp_score THEN ps.dk_points ELSE 0 END) as avg_winning_points,
            AVG(CASE WHEN team_totals.team_score <= opp_totals.opp_score THEN ps.dk_points ELSE 0 END) as avg_losing_points,
            AVG(ps.dk_points) as avg_total_points
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        LEFT JOIN (
            SELECT game_id, team_id, SUM(dk_points) as team_score
            FROM player_stats GROUP BY game_id, team_id
        ) team_totals ON ps.game_id = team_totals.game_id AND ps.team_id = team_totals.team_id
        LEFT JOIN (
            SELECT g2.id as game_id,
                   CASE WHEN g2.home_team_id = ps2.team_id THEN g2.away_team_id ELSE g2.home_team_id END as opp_team_id,
                   SUM(ps2.dk_points) as opp_score
            FROM games g2
            JOIN player_stats ps2 ON g2.id = ps2.game_id
            WHERE ps2.team_id != t.id
            GROUP BY g2.id, opp_team_id
        ) opp_totals ON ps.game_id = opp_totals.game_id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        script_result = conn.execute(game_script_query, rz_params).fetchone()
        if script_result and script_result[2]:
            winning_avg, losing_avg, total_avg = script_result
            features["te_winning_game_boost"] = (winning_avg or 0) / max(total_avg, 1)
            features["te_losing_game_penalty"] = (losing_avg or 0) / max(total_avg, 1)
        else:
            features["te_winning_game_boost"] = 1.1
            features["te_losing_game_penalty"] = 0.85

        # 6. Opponent TE Defense Strength
        def_query = """
        SELECT
            AVG(CASE WHEN p.position = 'TE' THEN ps.dk_points ELSE 0 END) as avg_te_points_allowed,
            COUNT(CASE WHEN p.position = 'TE' THEN 1 END) as te_games
        FROM player_stats ps
        JOIN players p ON ps.player_id = p.id
        JOIN games g ON ps.game_id = g.id
        JOIN teams opp_t ON (
            CASE WHEN g.home_team_id = ps.team_id THEN g.away_team_id ELSE g.home_team_id END = opp_t.id
        )
        WHERE opp_t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
            AND p.position = 'TE'
        """
        def_params = (opponent_abbr, season, max(1, week - 4), week - 1)
        def_result = conn.execute(def_query, def_params).fetchone()
        if def_result and def_result[1] > 0:
            avg_allowed, games = def_result
            features["te_opp_def_strength"] = min(
                avg_allowed / 8.0, 2.0
            )  # Normalize around 8 points
        else:
            features["te_opp_def_strength"] = 1.0

        # 7. Receiving Yards After Catch (YAC) Efficiency
        yac_query = """
        SELECT
            AVG(CASE WHEN ps.receptions > 0 AND ps.receiving_yards > 0
                     THEN ps.receiving_yards / ps.receptions ELSE 0 END) as avg_yac,
            AVG(CASE WHEN ps.targets > 0 THEN ps.receptions / ps.targets ELSE 0 END) as catch_rate
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
            AND ps.receptions > 0
        """
        yac_result = conn.execute(yac_query, rz_params).fetchone()
        if yac_result:
            avg_yac, catch_rate = yac_result
            features["te_yac_efficiency"] = (
                avg_yac or 0
            ) / 12.0  # Normalize around 12 yards
            features["te_catch_rate"] = catch_rate or 0.65
        else:
            features["te_yac_efficiency"] = 0.7
            features["te_catch_rate"] = 0.65

        # 8. Blocking Role vs Receiving Role
        role_query = """
        SELECT
            AVG(ps.targets) as avg_targets,
            AVG(ps.snaps) as avg_snaps,
            AVG(CASE WHEN ps.targets >= 4 THEN 1 ELSE 0 END) as receiving_game_rate
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        role_result = conn.execute(role_query, rz_params).fetchone()
        if role_result:
            avg_targets, avg_snaps, receiving_rate = role_result
            features["te_receiving_role"] = min(
                (avg_targets or 0) / 6.0, 1.5
            )  # Normalize around 6 targets
            features["te_blocking_role"] = max(1.0 - (receiving_rate or 0), 0.1)
        else:
            features["te_receiving_role"] = 0.6
            features["te_blocking_role"] = 0.4

        # 9. Team Passing Volume Context
        team_pass_query = """
        SELECT
            AVG(team_totals.team_targets) as avg_team_targets,
            AVG(team_totals.team_pass_yards) as avg_team_pass_yards
        FROM (
            SELECT
                ps.game_id,
                SUM(ps.targets) as team_targets,
                SUM(ps.passing_yards + ps.receiving_yards) as team_pass_yards
            FROM player_stats ps
            JOIN teams t ON ps.team_id = t.id
            WHERE t.team_abbr = ?
                AND ps.season = ? AND ps.week BETWEEN ? AND ?
            GROUP BY ps.game_id
        ) team_totals
        """
        team_params = (team_abbr, season, max(1, week - lookback_weeks), week - 1)
        team_result = conn.execute(team_pass_query, team_params).fetchone()
        if team_result:
            team_targets, team_pass_yards = team_result
            features["te_team_pass_volume"] = (
                team_targets or 0
            ) / 35.0  # Normalize around 35 targets
            features["te_team_pass_efficiency"] = (
                team_pass_yards or 0
            ) / 300.0  # Normalize around 300 yards
        else:
            features["te_team_pass_volume"] = 1.0
            features["te_team_pass_efficiency"] = 1.0

        # 10. Weather Impact (TEs less affected than WRs)
        # Simplified weather resistance feature
        features["te_weather_resistance"] = (
            0.95  # TEs typically less affected by weather
        )

        # 11. Vegas Correlation Features
        vegas_query = """
        SELECT
            AVG(CASE WHEN g.total_line > 0 THEN g.total_line ELSE 45 END) as avg_total,
            AVG(CASE WHEN g.team_implied_total > 0 THEN g.team_implied_total ELSE 22.5 END) as avg_implied
        FROM games g
        JOIN teams t ON (g.home_team_id = t.id OR g.away_team_id = t.id)
        WHERE t.team_abbr = ?
            AND g.season = ? AND g.week BETWEEN ? AND ?
        """
        vegas_result = conn.execute(vegas_query, team_params).fetchone()
        if vegas_result:
            avg_total, avg_implied = vegas_result
            features["te_vegas_total_correlation"] = (avg_total or 45) / 50.0
            features["te_vegas_implied_correlation"] = (avg_implied or 22.5) / 25.0
        else:
            features["te_vegas_total_correlation"] = 0.9
            features["te_vegas_implied_correlation"] = 0.9

        # 12. Ceiling Game Indicators (high target games)
        ceiling_query = """
        SELECT
            MAX(ps.dk_points) as max_points,
            AVG(ps.dk_points) as avg_points,
            COUNT(CASE WHEN ps.dk_points >= 15 THEN 1 END) as ceiling_games,
            COUNT(*) as total_games
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.id
        JOIN teams t ON ps.team_id = t.id
        WHERE ps.player_id = ? AND t.team_abbr = ?
            AND ps.season = ? AND ps.week BETWEEN ? AND ?
        """
        ceiling_result = conn.execute(ceiling_query, rz_params).fetchone()
        if ceiling_result and ceiling_result[3] > 0:
            max_points, avg_points, ceiling_games, total_games = ceiling_result
            features["te_ceiling_potential"] = (
                max_points or 0
            ) / 25.0  # Normalize around 25 points
            features["te_ceiling_frequency"] = ceiling_games / max(total_games, 1)
            features["te_floor_consistency"] = min((avg_points or 0) / 6.0, 1.5)
        else:
            features["te_ceiling_potential"] = 0.6
            features["te_ceiling_frequency"] = 0.2
            features["te_floor_consistency"] = 0.7

    except Exception as e:
        logger.warning(f"Error computing TE features for {player_name}: {e}")
        # Provide safe defaults for all features
        default_features = {
            "te_rz_target_share": 0.15,
            "te_snap_share": 0.7,
            "te_two_te_sets": 0.3,
            "te_target_efficiency": 0.65,
            "te_yards_per_target": 8.5,
            "te_route_volume": 4.0,
            "te_rz_touch_rate": 0.3,
            "te_td_rate": 0.12,
            "te_winning_game_boost": 1.05,
            "te_losing_game_penalty": 0.9,
            "te_opp_def_strength": 1.0,
            "te_yac_efficiency": 0.7,
            "te_catch_rate": 0.65,
            "te_receiving_role": 0.6,
            "te_blocking_role": 0.4,
            "te_team_pass_volume": 1.0,
            "te_team_pass_efficiency": 1.0,
            "te_weather_resistance": 0.95,
            "te_vegas_total_correlation": 0.9,
            "te_vegas_implied_correlation": 0.9,
            "te_ceiling_potential": 0.6,
            "te_ceiling_frequency": 0.2,
            "te_floor_consistency": 0.7,
        }
        features.update(default_features)

    return features


def get_dst_specific_features(
    team_abbr: str,
    opponent_abbr: str,
    season: int,
    week: int,
    conn: sqlite3.Connection,
    lookback_weeks: int = 4,
) -> Dict[str, float]:
    """Extract DST-specific features for enhanced prediction accuracy.

    Based on research, DST prediction requires focusing on:
    1. Opponent offensive vulnerabilities (most predictive)
    2. Game script factors (spread, totals, pace)
    3. Weather impact on turnovers and scoring
    4. Defensive component prediction (sacks, INTs, points allowed)
    """
    features = {}

    try:
        # 1. Opponent Offensive Vulnerability Analysis (Most Important)
        opp_vuln_query = """
        SELECT
            AVG(CASE WHEN ds.position = 'QB' THEN ps.passing_interceptions ELSE 0 END) as avg_qb_ints,
            AVG(CASE WHEN ds.position = 'QB' THEN ps.fumbles_lost ELSE 0 END) as avg_qb_fumbles,
            AVG(CASE WHEN ds.position = 'QB' THEN 35.0 ELSE 0 END) as avg_pass_attempts,
            AVG(CASE WHEN ds.position = 'QB' THEN 2.5 ELSE 0 END) as avg_sacks_taken,
            COUNT(DISTINCT ds.game_id) as games_played
        FROM dfs_scores ds
        JOIN player_stats ps ON ds.player_id = ps.player_id AND ds.game_id = ps.game_id
        JOIN teams t ON ds.team_id = t.id
        WHERE t.team_abbr = ? AND ds.season = ?
            AND ds.week BETWEEN ? AND ?
        """
        opp_params = (opponent_abbr, season, max(1, week - lookback_weeks), week - 1)
        opp_result = conn.execute(opp_vuln_query, opp_params).fetchone()

        if opp_result and opp_result[4] > 0:  # games_played > 0
            avg_qb_ints, avg_qb_fumbles, avg_pass_att, avg_sacks_taken, games = (
                opp_result
            )
            # Opponent turnover rate (key predictor)
            features["dst_opp_turnover_rate"] = (avg_qb_ints + avg_qb_fumbles) / max(
                games, 1
            )
            features["dst_opp_pass_volume"] = avg_pass_att or 32.0
            features["dst_opp_sack_rate"] = (
                avg_sacks_taken / max(avg_pass_att, 1) if avg_pass_att else 0.08
            )
        else:
            features["dst_opp_turnover_rate"] = 1.2  # League average
            features["dst_opp_pass_volume"] = 32.0
            features["dst_opp_sack_rate"] = 0.08

        # 2. Own Defense Historical Performance
        def_history_query = """
        SELECT
            AVG(d.sacks) as avg_sacks,
            AVG(d.interceptions) as avg_ints,
            AVG(d.fumbles_recovered) as avg_fumbles,
            AVG(d.defensive_tds) as avg_def_tds,
            AVG(d.points_allowed) as avg_points_allowed,
            AVG(d.fantasy_points) as avg_fantasy_points,
            COUNT(*) as games
        FROM dst_stats d
        WHERE d.team_abbr = ? AND d.season = ?
            AND d.week BETWEEN ? AND ?
        """
        def_params = (team_abbr, season, max(1, week - lookback_weeks), week - 1)
        def_result = conn.execute(def_history_query, def_params).fetchone()

        if def_result and def_result[6] > 0:  # games > 0
            avg_sacks, avg_ints, avg_fumbles, avg_def_tds, avg_pa, avg_fp, games = (
                def_result
            )
            features["dst_def_sacks_rate"] = avg_sacks or 2.0
            features["dst_def_turnover_rate"] = (avg_ints + avg_fumbles) or 1.0
            features["dst_def_td_rate"] = avg_def_tds or 0.1
            features["dst_def_points_allowed"] = avg_pa or 22.0
            features["dst_recent_performance"] = avg_fp or 6.0
        else:
            features["dst_def_sacks_rate"] = 2.0
            features["dst_def_turnover_rate"] = 1.0
            features["dst_def_td_rate"] = 0.1
            features["dst_def_points_allowed"] = 22.0
            features["dst_recent_performance"] = 6.0

        # 3. Game Script and Vegas Analysis (Critical for DST)
        vegas_query = """
        SELECT
            AVG(CASE WHEN bo.over_under_line > 0 THEN bo.over_under_line ELSE 45 END) as avg_total,
            AVG(CASE WHEN t.id = g.home_team_id THEN bo.home_team_spread
                     ELSE bo.away_team_spread END) as avg_spread
        FROM games g
        JOIN betting_odds bo ON g.id = bo.game_id
        JOIN teams t ON (g.home_team_id = t.id OR g.away_team_id = t.id)
        WHERE t.team_abbr = ? AND g.season = ? AND g.week BETWEEN ? AND ?
        """
        vegas_params = (team_abbr, season, max(1, week - lookback_weeks), week - 1)
        vegas_result = conn.execute(vegas_query, vegas_params).fetchone()

        if vegas_result:
            avg_total, avg_spread = vegas_result
            features["dst_game_total"] = avg_total or 45.0
            features["dst_team_spread"] = avg_spread or 0.0
            # Opponent implied total (key predictor for points allowed)
            features["dst_opp_implied_total"] = (avg_total or 45.0) / 2.0 - (
                avg_spread or 0.0
            ) / 2.0
        else:
            features["dst_game_total"] = 45.0
            features["dst_team_spread"] = 0.0
            features["dst_opp_implied_total"] = 22.5

        # 4. Component-Based Predictions
        # Sacks prediction
        sack_prediction = (
            features["dst_def_sacks_rate"]
            * (features["dst_opp_pass_volume"] / 32.0)
            * (1 + features["dst_opp_sack_rate"])
        )
        features["dst_predicted_sacks"] = min(sack_prediction, 8.0)

        # Turnover prediction
        turnover_prediction = (
            features["dst_def_turnover_rate"]
            * features["dst_opp_turnover_rate"]
            * (features["dst_opp_pass_volume"] / 32.0)
        )
        features["dst_predicted_turnovers"] = min(turnover_prediction, 5.0)

        # Points allowed prediction (inverse relationship with DST points)
        pa_base = features["dst_opp_implied_total"]
        pa_adjustment = features["dst_def_points_allowed"] - 22.0  # League average
        features["dst_predicted_pa"] = max(pa_base + pa_adjustment * 0.3, 7.0)

        # 5. Weather Impact on Turnovers
        # Simplified weather impact (detailed weather data may not be available)
        features["dst_weather_turnover_boost"] = (
            1.0  # Baseline, can be enhanced with actual weather
        )

        # 6. Game Pace and Pass Volume Context
        pace_query = """
        SELECT AVG(team_totals.total_plays) as avg_plays
        FROM (
            SELECT ds.game_id, COUNT(*) as total_plays
            FROM dfs_scores ds
            JOIN teams t ON ds.team_id = t.id
            WHERE t.team_abbr = ? AND ds.season = ?
                AND ds.week BETWEEN ? AND ?
            GROUP BY ds.game_id
        ) team_totals
        """
        pace_result = conn.execute(pace_query, vegas_params).fetchone()
        if pace_result and pace_result[0]:
            features["dst_expected_pace"] = (
                pace_result[0] / 70.0
            )  # Normalize around 70 plays
        else:
            features["dst_expected_pace"] = 1.0

        # 7. Matchup-Specific Factors
        # Home/Away impact (home teams typically allow fewer points)
        features["dst_home_field_advantage"] = 0.8  # Default neutral, can be enhanced

        # Division rival familiarity (if data available)
        features["dst_division_game"] = (
            0.0  # Default, can be enhanced with schedule data
        )

        # 8. Ceiling/Floor Analysis
        ceiling_query = """
        SELECT
            MAX(d.fantasy_points) as max_fp,
            MIN(d.fantasy_points) as min_fp,
            AVG(d.fantasy_points) as avg_fp,
            COUNT(CASE WHEN d.fantasy_points >= 10 THEN 1 END) as ceiling_games,
            COUNT(*) as total_games
        FROM dst_stats d
        WHERE d.team_abbr = ? AND d.season = ?
            AND d.week BETWEEN ? AND ?
        """
        ceiling_result = conn.execute(ceiling_query, def_params).fetchone()

        if ceiling_result and ceiling_result[4] > 0:  # total_games > 0
            max_fp, min_fp, avg_fp, ceiling_games, total_games = ceiling_result
            features["dst_ceiling_potential"] = (
                max_fp or 15.0
            ) / 20.0  # Normalize around 20
            features["dst_floor_safety"] = (min_fp or 2.0) / 8.0  # Normalize around 8
            features["dst_ceiling_frequency"] = ceiling_games / max(total_games, 1)
            features["dst_consistency"] = (
                1.0 - ((max_fp or 15.0) - (min_fp or 2.0)) / 15.0
            )  # Consistency score
        else:
            features["dst_ceiling_potential"] = 0.75
            features["dst_floor_safety"] = 0.4
            features["dst_ceiling_frequency"] = 0.2
            features["dst_consistency"] = 0.6

        # 9. Advanced Component Scores
        # Pressure score (combine sacks + pass rush effectiveness)
        features["dst_pressure_score"] = (
            features["dst_predicted_sacks"] * 1.0 + features["dst_opp_sack_rate"] * 5.0
        )

        # Turnover generation score
        features["dst_turnover_score"] = (
            features["dst_predicted_turnovers"] * 2.0
            + features["dst_opp_turnover_rate"] * 3.0
        )

        # Points allowed tier score (DST scoring tiers)
        pa_predicted = features["dst_predicted_pa"]
        if pa_predicted <= 6:
            features["dst_pa_tier_score"] = 5.0  # Shutout/Elite
        elif pa_predicted <= 13:
            features["dst_pa_tier_score"] = 4.0  # Very good
        elif pa_predicted <= 20:
            features["dst_pa_tier_score"] = 3.0  # Average
        elif pa_predicted <= 27:
            features["dst_pa_tier_score"] = 2.0  # Below average
        elif pa_predicted <= 34:
            features["dst_pa_tier_score"] = 1.0  # Poor
        else:
            features["dst_pa_tier_score"] = 0.0  # Terrible

        # 10. Composite DST Score (combining all factors)
        features["dst_composite_score"] = (
            features["dst_pressure_score"] * 0.25
            + features["dst_turnover_score"] * 0.35  # Most important
            + features["dst_pa_tier_score"] * 0.25
            + features["dst_ceiling_potential"] * 0.15
        )

    except Exception as e:
        logger.warning(
            f"Error computing DST features for {team_abbr} vs {opponent_abbr}: {e}"
        )
        # Provide safe defaults
        default_features = {
            "dst_opp_turnover_rate": 1.2,
            "dst_opp_pass_volume": 32.0,
            "dst_opp_sack_rate": 0.08,
            "dst_def_sacks_rate": 2.0,
            "dst_def_turnover_rate": 1.0,
            "dst_def_td_rate": 0.1,
            "dst_def_points_allowed": 22.0,
            "dst_recent_performance": 6.0,
            "dst_game_total": 45.0,
            "dst_team_spread": 0.0,
            "dst_opp_implied_total": 22.5,
            "dst_predicted_sacks": 2.0,
            "dst_predicted_turnovers": 1.0,
            "dst_predicted_pa": 22.0,
            "dst_weather_turnover_boost": 1.0,
            "dst_expected_pace": 1.0,
            "dst_home_field_advantage": 0.8,
            "dst_division_game": 0.0,
            "dst_ceiling_potential": 0.75,
            "dst_floor_safety": 0.4,
            "dst_ceiling_frequency": 0.2,
            "dst_consistency": 0.6,
            "dst_pressure_score": 2.4,
            "dst_turnover_score": 5.6,
            "dst_pa_tier_score": 3.0,
            "dst_composite_score": 3.8,
        }
        features.update(default_features)

    return features


def compute_weekly_odds_z_scores(
    df: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """Compute weekly z-scores for odds features."""
    weekly_mask = (df["season"] == season) & (df["week"] == week)
    weekly_df = df[weekly_mask].copy()

    if len(weekly_df) == 0:
        return df

    # Compute z-scores for this week
    if "total_line" in weekly_df.columns and weekly_df["total_line"].std() > 0:
        weekly_mean = weekly_df["total_line"].mean()
        weekly_std = weekly_df["total_line"].std()
        df.loc[weekly_mask, "game_tot_z"] = (
            df.loc[weekly_mask, "total_line"] - weekly_mean
        ) / weekly_std

    if "team_itt" in weekly_df.columns and weekly_df["team_itt"].std() > 0:
        weekly_mean = weekly_df["team_itt"].mean()
        weekly_std = weekly_df["team_itt"].std()
        df.loc[weekly_mask, "team_itt_z"] = (
            df.loc[weekly_mask, "team_itt"] - weekly_mean
        ) / weekly_std

    return df


def get_player_features(
    player_id: int,
    game_id: int,
    lookback_weeks: int = 4,
    db_path: str = "data/nfl_dfs.db",
) -> Dict[str, float]:
    """Extract features for a player for model training/prediction."""
    conn = get_db_connection(db_path)
    features = {}

    try:
        # Get player info
        player_info = conn.execute(
            """SELECT p.position, p.team_id, t.team_abbr, p.player_name
               FROM players p
               JOIN teams t ON p.team_id = t.id
               WHERE p.id = ?""",
            (player_id,),
        ).fetchone()

        if not player_info:
            return features

        position, team_id, team_abbr, player_name = player_info

        # Get game info and opponent
        game_info = conn.execute(
            """SELECT g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                      ht.team_abbr as home_abbr, at.team_abbr as away_abbr
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               JOIN teams at ON g.away_team_id = at.id
               WHERE g.id = ?""",
            (game_id,),
        ).fetchone()

        if not game_info:
            return features

        game_date, season, week, home_team_id, away_team_id, home_abbr, away_abbr = (
            game_info
        )

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
            (player_id, start_date, game_date),
        ).fetchone()

        if recent_stats:
            features.update(
                {
                    "avg_fantasy_points": recent_stats[0] or 0,
                    "avg_passing_yards": recent_stats[1] or 0,
                    "avg_rushing_yards": recent_stats[2] or 0,
                    "avg_receiving_yards": recent_stats[3] or 0,
                    "avg_targets": recent_stats[4] or 0,
                    "games_played": recent_stats[5] or 0,
                    "max_points": recent_stats[6] or 0,
                    "min_points": recent_stats[7] or 0,
                    "consistency": 1
                    - ((recent_stats[6] or 0) - (recent_stats[7] or 0))
                    / max(recent_stats[0] or 1, 1),
                }
            )

        # Position-specific features
        if position == "QB":
            qb_stats = conn.execute(
                """SELECT AVG(ps.passing_tds), AVG(ps.passing_interceptions)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date),
            ).fetchone()
            if qb_stats:
                features["avg_pass_tds"] = qb_stats[0] or 0
                features["avg_interceptions"] = qb_stats[1] or 0

        elif position in ["RB"]:
            rb_stats = conn.execute(
                """SELECT AVG(ps.rushing_attempts), AVG(ps.rushing_tds)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date),
            ).fetchone()
            if rb_stats:
                features["avg_carries"] = rb_stats[0] or 0
                features["avg_rush_tds"] = rb_stats[1] or 0

        elif position in ["WR", "TE"]:
            rec_stats = conn.execute(
                """SELECT AVG(ps.receptions), AVG(ps.receiving_tds)
                   FROM player_stats ps
                   JOIN games g ON ps.game_id = g.id
                   WHERE ps.player_id = ? AND g.game_date >= ? AND g.game_date < ?""",
                (player_id, start_date, game_date),
            ).fetchone()
            if rec_stats:
                features["avg_receptions"] = rec_stats[0] or 0
                features["avg_rec_tds"] = rec_stats[1] or 0

        # Add defensive matchup features from PbP data
        defensive_features = get_defensive_matchup_features(
            opponent_abbr, team_abbr, season, week, lookback_weeks, db_path
        )
        features.update(defensive_features)

        # Add situational PbP features for the player vs this defense
        pbp_matchup = get_player_vs_defense_features(
            player_id,
            team_abbr,
            opponent_abbr,
            season,
            week,
            position,
            lookback_weeks,
            db_path,
        )
        features.update(pbp_matchup)

        # Add weather features
        weather_data = conn.execute(
            """SELECT weather_temperature, weather_wind_mph, weather_humidity,
                      weather_detail, stadium_neutral
               FROM games WHERE id = ?""",
            (game_id,),
        ).fetchone()

        if weather_data:
            temp, wind, humidity, conditions, neutral = weather_data

            # Raw weather features
            features["temperature_f"] = temp or 72
            features["wind_mph"] = wind or 0
            features["humidity_pct"] = humidity or 50

            # Weather threshold features
            features["cold_lt40"] = 1 if (temp or 72) < 40 else 0
            features["hot_gt85"] = 1 if (temp or 72) > 85 else 0
            features["wind_gt15"] = 1 if (wind or 0) > 15 else 0
            features["dome"] = 1 if conditions and "indoor" in conditions.lower() else 0
        else:
            # Default weather values if no data
            features.update(
                {
                    "temperature_f": 72,
                    "wind_mph": 0,
                    "humidity_pct": 50,
                    "cold_lt40": 0,
                    "hot_gt85": 0,
                    "wind_gt15": 0,
                    "dome": 0,
                }
            )

        # Add comprehensive injury features
        # Get player injury status
        player_injury = conn.execute(
            """SELECT p.injury_status FROM players p WHERE p.id = ?""", (player_id,)
        ).fetchone()

        injury_status = (player_injury[0] if player_injury else None) or "Healthy"

        # One-hot encode injury status
        features["injury_status_Out"] = 1 if injury_status == "Out" else 0
        features["injury_status_Doubtful"] = 1 if injury_status == "Doubtful" else 0
        features["injury_status_Questionable"] = (
            1 if injury_status == "Questionable" else 0
        )
        features["injury_status_Probable"] = (
            1 if injury_status in ["Probable", "Healthy"] else 0
        )

        # Count games missed in last 4 weeks
        games_missed = conn.execute(
            """SELECT COUNT(*) FROM games g
               WHERE g.game_date >= ? AND g.game_date < ?
               AND NOT EXISTS (
                   SELECT 1 FROM player_stats ps
                   WHERE ps.player_id = ? AND ps.game_id = g.id
               )""",
            (start_date, game_date, player_id),
        ).fetchone()

        features["games_missed_last4"] = games_missed[0] if games_missed else 0

        # Practice trend (simplified - assume stable for now)
        features["practice_trend"] = 0  # 0=stable, 1=improving, -1=regressing

        # Returning from injury flag
        features["returning_from_injury"] = (
            1
            if features["games_missed_last4"] > 0 and injury_status == "Healthy"
            else 0
        )

        # Team injury aggregates - count injured starters
        team_injured = conn.execute(
            """SELECT COUNT(*) FROM players p
               WHERE p.team_id = ? AND p.injury_status IN ('Out', 'Doubtful', 'Questionable')""",
            (team_id,),
        ).fetchone()

        opponent_team_id = away_team_id if is_home else home_team_id
        opp_injured = conn.execute(
            """SELECT COUNT(*) FROM players p
               WHERE p.team_id = ? AND p.injury_status IN ('Out', 'Doubtful', 'Questionable')""",
            (opponent_team_id,),
        ).fetchone()

        features["team_injured_starters"] = min(
            team_injured[0] if team_injured else 0, 11
        )  # Cap at 11
        features["opp_injured_starters"] = min(
            opp_injured[0] if opp_injured else 0, 11
        )  # Cap at 11

        # Add enhanced betting odds features (prioritize live odds from odds_api)
        betting_data = conn.execute(
            """SELECT spread_favorite, over_under_line, home_team_spread, away_team_spread, source
               FROM betting_odds WHERE game_id = ?
               ORDER BY CASE WHEN source = 'odds_api' THEN 1 ELSE 2 END
               LIMIT 1""",
            (game_id,),
        ).fetchone()

        if betting_data:
            spread_fav, over_under, home_spread, away_spread, source = betting_data
            team_spread = home_spread if is_home else away_spread
            over_under_line = over_under or 45

            # Core odds features
            features["team_spread"] = team_spread or 0
            features["team_spread_abs"] = abs(team_spread or 0)
            features["total_line"] = over_under_line
            features["is_favorite"] = 1 if (team_spread or 0) < 0 else 0

            # Derived features
            features["team_itt"] = (
                over_under_line / 2.0 - (team_spread or 0) / 2.0
            )  # Implied team total

            # Z-scores will be computed in batch processing
            features["game_tot_z"] = 0.0  # Placeholder
            features["team_itt_z"] = 0.0  # Placeholder
        else:
            # Default betting values if no data
            features.update(
                {
                    "team_spread": 0,
                    "team_spread_abs": 0,
                    "total_line": 45,
                    "is_favorite": 0,
                    "team_itt": 22.5,
                    "game_tot_z": 0.0,
                    "team_itt_z": 0.0,
                }
            )

        # Add contextual features to match schema
        features["salary"] = 5000  # Default salary - should be populated from DK data
        features["home"] = 1 if is_home else 0
        features["rest_days"] = 7  # Default NFL week rest
        features["travel"] = 0  # Travel distance - placeholder
        features["season_week"] = week  # Normalized week

        # Add placeholder usage/opportunity features (to be computed from historical data)
        features["targets_ema"] = features.get("avg_targets", 0)
        features["routes_run_ema"] = 0  # Placeholder
        features["rush_att_ema"] = features.get("avg_carries", 0)
        features["snap_share_ema"] = 0.5  # Placeholder
        features["redzone_opps_ema"] = 0  # Placeholder
        features["air_yards_ema"] = 0  # Placeholder
        features["adot_ema"] = 0  # Placeholder - Average Depth of Target
        features["yprr_ema"] = 0  # Placeholder - Yards Per Route Run

        # Add efficiency features (placeholders)
        features["yards_after_contact"] = 0  # Placeholder
        features["missed_tackles_forced"] = 0  # Placeholder
        features["pressure_rate"] = 0  # Placeholder - for QBs
        features["opp_dvp_pos_allowed"] = 0  # Opponent defense vs position

    except Exception as e:
        logger.error(f"Error extracting features for player {player_id}: {e}")
    finally:
        conn.close()

    return features


def is_home_game(team_id: int, game_id: int, conn: sqlite3.Connection) -> bool:
    """Check if team is playing at home."""
    result = conn.execute(
        "SELECT home_team_id FROM games WHERE id = ?", (game_id,)
    ).fetchone()
    return result and result[0] == team_id


def get_dst_training_data(
    seasons: List[int], db_path: str = "data/nfl_dfs.db"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get DST training data using ONLY historical features (no current-game data leakage)."""
    conn = get_db_connection(db_path)

    try:
        # FIXED: Get DST features without using current game stats (prevents data leakage)
        data_query = """
            SELECT
                d.team_abbr,
                d.season,
                d.week,
                d.fantasy_points,
                -- Historical averages (3-game rolling, EXCLUDING current game)
                AVG(h.fantasy_points) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_fantasy_points,
                AVG(h.points_allowed) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_points_allowed,
                AVG(h.sacks) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_sacks,
                AVG(h.interceptions) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_interceptions,
                AVG(h.fumbles_recovered) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_fumbles,
                AVG(h.defensive_tds) OVER (
                    PARTITION BY d.team_abbr
                    ORDER BY d.season, d.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_recent_def_tds,
                -- Season-long averages (up to but not including current week)
                AVG(s.fantasy_points) OVER (
                    PARTITION BY d.team_abbr, d.season
                    ORDER BY d.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_avg_fantasy_points,
                AVG(s.points_allowed) OVER (
                    PARTITION BY d.team_abbr, d.season
                    ORDER BY d.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_avg_points_allowed,
                -- Opponent strength features (would need game context)
                g.home_team_id,
                g.away_team_id,
                CASE WHEN d.team_abbr = ht.team_abbr THEN 1 ELSE 0 END as is_home
            FROM dst_stats d
            LEFT JOIN dst_stats h ON h.team_abbr = d.team_abbr
                AND h.season = d.season
                AND h.week < d.week
            LEFT JOIN dst_stats s ON s.team_abbr = d.team_abbr
                AND s.season = d.season
                AND s.week < d.week
            LEFT JOIN games g ON (
                (g.home_team_id = (SELECT id FROM teams WHERE team_abbr = d.team_abbr)
                 OR g.away_team_id = (SELECT id FROM teams WHERE team_abbr = d.team_abbr))
                AND g.season = d.season AND g.week = d.week
            )
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            WHERE d.season IN ({})
            AND d.week >= 4  -- Only include games where we have historical data
            GROUP BY d.team_abbr, d.season, d.week, d.fantasy_points,
                     g.home_team_id, g.away_team_id, ht.team_abbr
            ORDER BY d.season, d.week, d.team_abbr
        """.format(",".join("?" * len(seasons)))

        cursor = conn.execute(data_query, seasons)
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No DST training data found - dst_stats table may be empty")
            logger.info("Consider running data collection to populate dst_stats table")
            return np.array([]), np.array([]), []

        # Extract features and targets
        X_list = []
        y_list = []

        # Enhanced feature names (11 base + 31 DST-specific features)
        feature_names = [
            # Base features (11)
            "avg_recent_fantasy_points",
            "avg_recent_points_allowed",
            "avg_recent_sacks",
            "avg_recent_interceptions",
            "avg_recent_fumbles",
            "avg_recent_def_tds",
            "season_avg_fantasy_points",
            "season_avg_points_allowed",
            "is_home",
            "week",
            "season_normalized",
            # DST-specific features (31)
            "opp_implied_total",
            "game_total_ou",
            "spread_signed",
            "is_favorite",
            "spread_magnitude",
            "opp_pass_attempts_l3",
            "opp_pass_yards_l3",
            "opp_turnover_rate_l3",
            "opp_sack_rate_l3",
            "opp_scoring_rate_l3",
            "opp_explosive_play_rate_l3",
            "def_sacks_per_game_l3",
            "def_turnovers_per_game_l3",
            "def_points_allowed_l3",
            "def_fantasy_points_l3",
            "def_pressure_rate_l3",
            "wind_speed",
            "precipitation",
            "temperature",
            "is_dome_game",
            "qb_int_rate_roll",
            "ol_sack_rate_allowed_roll",
            "weather_turnover_boost",
            "game_script_dst_value",
            "sack_component",
            "turnover_component",
            "td_component",
            "sack_expectation",
            "turnover_expectation",
            "pa_tier_prediction",
            "td_probability",
        ]

        for row in rows:
            # Skip rows where we don't have sufficient historical data
            if (
                row[4] is None or row[11] is None
            ):  # avg_recent_fantasy_points or season_avg
                continue

            team_abbr = row[0]
            season = row[1]
            week = row[2]

            # Determine opponent team
            is_home = row[12] == team_abbr if row[12] else True
            if is_home:
                # If we're home, opponent is away team
                away_team_query = "SELECT t.team_abbr FROM teams t JOIN games g ON g.away_team_id = t.id WHERE g.home_team_id = (SELECT id FROM teams WHERE team_abbr = ?) AND g.season = ? AND g.week = ?"
                opp_result = conn.execute(
                    away_team_query, (team_abbr, season, week)
                ).fetchone()
                opponent_abbr = opp_result[0] if opp_result else "UNK"
            else:
                # If we're away, opponent is home team
                home_team_query = "SELECT t.team_abbr FROM teams t JOIN games g ON g.home_team_id = t.id WHERE g.away_team_id = (SELECT id FROM teams WHERE team_abbr = ?) AND g.season = ? AND g.week = ?"
                opp_result = conn.execute(
                    home_team_query, (team_abbr, season, week)
                ).fetchone()
                opponent_abbr = opp_result[0] if opp_result else "UNK"

            # Get DST-specific features
            try:
                dst_specific_features = get_dst_specific_features(
                    team_abbr, opponent_abbr, season, week, conn
                )
            except Exception as e:
                logger.warning(
                    f"Error getting DST features for {team_abbr} vs {opponent_abbr}: {e}"
                )
                dst_specific_features = {}

            # Base features
            features = [
                row[4] or 0,  # avg_recent_fantasy_points
                row[5] or 20,  # avg_recent_points_allowed (default to league avg)
                row[6] or 2,  # avg_recent_sacks (default to league avg)
                row[7] or 0.8,  # avg_recent_interceptions
                row[8] or 0.5,  # avg_recent_fumbles
                row[9] or 0.1,  # avg_recent_def_tds
                row[10] or 0,  # season_avg_fantasy_points
                row[11] or 20,  # season_avg_points_allowed
                row[14] or 0,  # is_home
                row[2],  # week
                (row[1] - 2022)
                / 5.0,  # season_normalized (center around 2022, scale by 5)
            ]

            # Add DST-specific features (27 features from enhancement)
            dst_feature_order = [
                "opp_implied_total",
                "game_total_ou",
                "spread_signed",
                "is_favorite",
                "spread_magnitude",
                "opp_pass_attempts_l3",
                "opp_pass_yards_l3",
                "opp_turnover_rate_l3",
                "opp_sack_rate_l3",
                "opp_scoring_rate_l3",
                "opp_explosive_play_rate_l3",
                "def_sacks_per_game_l3",
                "def_turnovers_per_game_l3",
                "def_points_allowed_l3",
                "def_fantasy_points_l3",
                "def_pressure_rate_l3",
                "wind_speed",
                "precipitation",
                "temperature",
                "is_dome_game",
                "qb_int_rate_roll",
                "ol_sack_rate_allowed_roll",
                "weather_turnover_boost",
                "game_script_dst_value",
                "sack_component",
                "turnover_component",
                "td_component",
            ]

            for feature_name in dst_feature_order:
                features.append(dst_specific_features.get(feature_name, 0.0))

            # Final component predictions - these are the most predictive
            if "sack_expectation" in dst_specific_features:
                features.append(dst_specific_features["sack_expectation"])
            else:
                features.append(2.0)  # League average sacks

            if "turnover_expectation" in dst_specific_features:
                features.append(dst_specific_features["turnover_expectation"])
            else:
                features.append(1.3)  # League average turnovers

            if "pa_tier_prediction" in dst_specific_features:
                features.append(dst_specific_features["pa_tier_prediction"])
            else:
                features.append(3.0)  # Middle tier PA

            if "td_probability" in dst_specific_features:
                features.append(dst_specific_features["td_probability"])
            else:
                features.append(0.15)  # 15% chance baseline

            target = row[
                3
            ]  # fantasy_points (current week - this is what we're predicting)

            X_list.append(features)
            y_list.append(target)

        if not X_list:
            logger.warning("No valid DST features extracted")
            return np.array([]), np.array([]), []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(
            f"Extracted {len(X)} DST training samples (FIXED - no data leakage)"
        )
        logger.info(f"Features: {feature_names}")
        return X, y, feature_names

    except Exception as e:
        logger.error(f"Error getting DST training data: {e}")
        return np.array([]), np.array([]), []
    finally:
        conn.close()


def batch_get_defensive_features(
    opponent_abbr: str, season: int, week: int, conn: sqlite3.Connection
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
            (opponent_abbr, season, week, max(1, week - 4)),
        ).fetchone()

        if def_stats:
            features.update(
                {
                    "def_avg_yards_allowed": def_stats[0] or 0,
                    "def_rush_yards_allowed": def_stats[1] or 0,
                    "def_pass_yards_allowed": def_stats[2] or 0,
                    "def_td_rate_allowed": def_stats[3] or 0,
                    "def_total_plays": def_stats[4] or 0,
                }
            )

        # Get red zone defense
        rz_defense = conn.execute(
            """SELECT
                AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END) as rz_td_rate,
                COUNT(*) as rz_plays
               FROM play_by_play
               WHERE defteam = ? AND season = ? AND week < ? AND week >= ?
               AND yardline_100 <= 20""",
            (opponent_abbr, season, max(1, week - 4), week),
        ).fetchone()

        if rz_defense:
            features.update(
                {
                    "def_rz_td_rate_allowed": rz_defense[0] or 0,
                    "def_rz_plays_allowed": rz_defense[1] or 0,
                }
            )

    except Exception as e:
        logger.warning(f"Error getting defensive features for {opponent_abbr}: {e}")

    return features


def get_training_data(
    position: str, seasons: List[int], db_path: str = "data/nfl_dfs.db"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get training data for a specific position using optimized batch queries."""
    # Handle DST position specially
    if position in ["DST", "DEF"]:
        return get_dst_training_data(seasons, db_path)

    conn = get_db_connection(db_path)

    try:
        logger.info(f"Loading training data for {position} position...")

        # Get all player-game combinations for the position from dfs_scores table
        data_query = """
            SELECT ds.player_id, ds.game_id, ds.dfs_points,
                   ds.position, ds.team_id, t.team_abbr,
                   g.game_date, ds.season, ds.week, g.home_team_id, g.away_team_id,
                   ht.team_abbr as home_abbr, at.team_abbr as away_abbr,
                   ds.opponent_id, ot.team_abbr as opponent_abbr, p.player_name
            FROM dfs_scores ds
            JOIN teams t ON ds.team_id = t.id
            JOIN teams ot ON ds.opponent_id = ot.id
            JOIN games g ON ds.game_id = g.id
            JOIN players p ON ds.player_id = p.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE ds.position = ? AND ds.season IN ({})
            AND g.game_finished = 1
            ORDER BY g.game_date
        """.format(",".join("?" * len(seasons)))

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
        """.format(",".join("?" * len(player_ids)), ",".join("?" * len(seasons)))

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
            # opponent_abbr is now at position 13
            season, week, opponent_abbr = row[7], row[8], row[13]
            unique_matchups.add((opponent_abbr, season, week))

        defensive_progress = ProgressDisplay("Computing defensive features")
        defensive_features_cache = {}
        for idx, (opponent_abbr, season, week) in enumerate(unique_matchups):
            defensive_progress.update(idx, len(unique_matchups))
            def_features = batch_get_defensive_features(
                opponent_abbr, season, week, conn
            )
            defensive_features_cache[(opponent_abbr, season, week)] = def_features
        defensive_progress.finish(
            f"Computed features for {len(unique_matchups)} matchups"
        )

        # First pass: collect all possible features to ensure consistency
        all_features_dict = {}

        # Pre-define expected statistical features that all positions should have
        expected_stat_features = [
            "avg_fantasy_points",
            "avg_passing_yards",
            "avg_rushing_yards",
            "avg_receiving_yards",
            "avg_targets",
            "avg_pass_tds",
            "avg_rush_tds",
            "avg_rec_tds",
            "avg_interceptions",
            "avg_fumbles",
            "avg_rush_attempts",
            "avg_receptions",
            "yards_per_carry",
            "yards_per_reception",
            "catch_rate",
            "games_played",
            "max_points",
            "min_points",
            "consistency",
            "vs_team_avg",  # Historical performance vs specific opponent
        ]

        # Pre-define weather, betting and other contextual features that all positions should have
        weather_betting_features = [
            "weather_temp",
            "weather_wind",
            "weather_humidity",
            "weather_is_indoor",
            "weather_is_rain",
            "weather_is_snow",
            "stadium_neutral",
            "cold_weather",
            "hot_weather",
            "high_wind",
            "team_spread",
            "team_spread_abs",
            "total_line",
            "is_favorite",
            "is_big_favorite",
            "is_big_underdog",
            "expected_pace",
            "team_itt",
            "game_tot_z",
            "team_itt_z",
            "temperature_f",
            "wind_mph",
            "humidity_pct",
            "cold_lt40",
            "hot_gt85",
            "wind_gt15",
            "dome",
            "injury_status_Out",
            "injury_status_Doubtful",
            "injury_status_Questionable",
            "injury_status_Probable",
            "games_missed_last4",
            "practice_trend",
            "returning_from_injury",
            "team_injured_starters",
            "opp_injured_starters",
            "targets_ema",
            "routes_run_ema",
            "rush_att_ema",
            "snap_share_ema",
            "redzone_opps_ema",
            "air_yards_ema",
            "adot_ema",
            "yprr_ema",
            "yards_after_contact",
            "missed_tackles_forced",
            "pressure_rate",
            "opp_dvp_pos_allowed",
            "salary",
            "home",
            "rest_days",
            "travel",
            "season_week",
            "completion_pct_trend",
            "yds_per_attempt_trend",
            "td_int_ratio_trend",
            "passer_rating_est",
            "passing_volume_trend",
            "dual_threat_factor",
            "red_zone_efficiency_est",
            "game_script_favorability",
            "pressure_situation",
            "ceiling_indicator",
            "implied_team_total",
            "shootout_potential",
            "defensive_game_script",
            "garbage_time_upside",
            "blowout_risk",
            # RB-specific features (Research-Based Enhanced Features)
            "yards_per_carry_trend",
            "rush_td_rate_trend",
            "receiving_involvement",
            "total_touches_trend",
            "workload_efficiency",
            "clock_management_upside",
            "snap_share_estimate",
            "workload_sustainability",
            "lead_back_role",
            "three_down_back_value",
            "goal_line_specialist",
            "goal_line_monopoly",
            "short_yardage_role",
            "red_zone_td_value",
            "third_down_involvement",
            "two_minute_drill_value",
            "receiving_value_multiplier",
            "committee_vs_bellcow",
            "weather_game_script_boost",
            "yards_after_contact_estimate",
            "breakaway_run_potential",
            "replacement_upside",
            "high_stakes_workload",
            # TE-specific features
            "te_target_rate",
            "red_zone_specialist",
            "short_area_role",
            "receiving_efficiency",
            "touchdown_dependency",
            "dual_role_value",
            "close_game_upside",
            "goal_line_opportunities",
            # WR-specific features
            "target_share_trend",
            "air_yards_per_target",
            "catch_rate_trend",
            "red_zone_involvement",
            "route_efficiency",
            "big_play_upside",
            "pass_heavy_script",
            "garbage_time_upside",
            # DST-specific features (Research-Based Enhanced Features)
            "pressure_rate_effectiveness",
            "turnover_generation_composite",
            "points_allowed_tier_bonus",
            "special_teams_upside",
            "opponent_offensive_strength",
            "opponent_turnover_prone",
            "opponent_line_protection",
            "opponent_implied_total",
            "game_script_multiplier",
            "high_volume_passing_game",
            "defensive_game_environment",
            "blowout_potential",
            "negative_game_script",
            "sack_rate_normalized",
            "interception_rate_boosted",
            "fumble_recovery_rate",
            "defensive_td_potential",
            "volatility_upside_factor",
            "defensive_floor_estimate",
            "defensive_ceiling_estimate",
            "overall_dst_favorability",
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
            stat_features = compute_features_from_stats(
                player_recent_stats, game_date, lookback_weeks=4
            )
            team_id, home_team_id = row[4], row[9]
            home_abbr, away_abbr = row[11], row[12]
            season, week = row[7], row[8]
            opponent_abbr = away_abbr if team_id == home_team_id else home_abbr

            context_features = {
                "season": season,
                "week": week,
                "is_home": 1 if team_id == home_team_id else 0,
            }
            def_features = defensive_features_cache.get(
                (opponent_abbr, season, week), {}
            )

            # Add correlation features if available
            try:
                import importlib

                models_module = importlib.import_module("models")
                correlation_extractor = models_module.CorrelationFeatureExtractor(
                    db_path
                )
                correlation_features = (
                    correlation_extractor.extract_correlation_features(
                        player_id, game_id, position
                    )
                )
            except (KeyError, TypeError, ValueError):
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
            stat_features = compute_features_from_stats(
                player_recent_stats, game_date, lookback_weeks=4
            )
            features.update(stat_features)

            # Game context features - now using dfs_scores structure
            team_id, home_team_id = row[4], row[9]
            season, week = row[7], row[8]
            team_abbr = row[5]
            opponent_id, opponent_abbr = row[13], row[14]
            player_name = row[15]

            # Calculate vs_team_avg on the fly
            vs_team_avg = calculate_vs_team_avg(
                player_id, opponent_id, season, week, conn
            )

            features.update(
                {
                    "season": season,
                    "week": week,
                    "is_home": 1 if team_id == home_team_id else 0,
                    "vs_team_avg": vs_team_avg,
                }
            )

            # Defensive matchup features
            def_features = defensive_features_cache.get(
                (opponent_abbr, season, week), {}
            )
            features.update(def_features)

            # Weather and betting features (using production pipeline logic)
            weather_betting_features = {}

            # Add weather features
            weather_data = conn.execute(
                """SELECT weather_temperature, weather_wind_mph, weather_humidity,
                          weather_detail, stadium_neutral
                   FROM games WHERE id = ?""",
                (game_id,),
            ).fetchone()

            if weather_data:
                temp, wind, humidity, conditions, neutral = weather_data
                weather_betting_features.update(
                    {
                        "weather_temp": temp or 72,
                        "weather_wind": wind or 0,
                        "weather_humidity": humidity or 50,
                        "weather_is_indoor": 1
                        if conditions and "indoor" in conditions.lower()
                        else 0,
                        "weather_is_rain": 1
                        if conditions and "rain" in conditions.lower()
                        else 0,
                        "weather_is_snow": 1
                        if conditions and "snow" in conditions.lower()
                        else 0,
                        "stadium_neutral": neutral or 0,
                        "cold_weather": 1 if (temp or 72) < 40 else 0,
                        "hot_weather": 1 if (temp or 72) > 85 else 0,
                        "high_wind": 1 if (wind or 0) > 15 else 0,
                        # Additional weather features matching expected schema
                        "temperature_f": temp or 72,
                        "wind_mph": wind or 0,
                        "humidity_pct": humidity or 50,
                        "cold_lt40": 1 if (temp or 72) < 40 else 0,
                        "hot_gt85": 1 if (temp or 72) > 85 else 0,
                        "wind_gt15": 1 if (wind or 0) > 15 else 0,
                        "dome": 1
                        if conditions and "indoor" in conditions.lower()
                        else 0,
                    }
                )
            else:
                weather_betting_features.update(
                    {
                        "weather_temp": 72,
                        "weather_wind": 0,
                        "weather_humidity": 50,
                        "weather_is_indoor": 0,
                        "weather_is_rain": 0,
                        "weather_is_snow": 0,
                        "stadium_neutral": 0,
                        "cold_weather": 0,
                        "hot_weather": 0,
                        "high_wind": 0,
                        # Additional weather defaults
                        "temperature_f": 72,
                        "wind_mph": 0,
                        "humidity_pct": 50,
                        "cold_lt40": 0,
                        "hot_gt85": 0,
                        "wind_gt15": 0,
                        "dome": 0,
                    }
                )

            # Add betting odds features (prioritize live odds)
            is_home = team_id == home_team_id
            betting_data = conn.execute(
                """SELECT spread_favorite, over_under_line, home_team_spread, away_team_spread
                   FROM betting_odds WHERE game_id = ? ORDER BY
                   CASE WHEN source = 'odds_api' THEN 1 ELSE 2 END
                   LIMIT 1""",
                (game_id,),
            ).fetchone()

            if betting_data:
                spread_fav, over_under, home_spread, away_spread = betting_data
                team_spread = home_spread if is_home else away_spread
                over_under_line = over_under or 45
                weather_betting_features.update(
                    {
                        "team_spread": team_spread or 0,
                        "team_spread_abs": abs(team_spread or 0),
                        "total_line": over_under_line,
                        "is_favorite": 1 if (team_spread or 0) < 0 else 0,
                        "is_big_favorite": 1 if (team_spread or 0) < -7 else 0,
                        "is_big_underdog": 1 if (team_spread or 0) > 7 else 0,
                        "expected_pace": over_under_line / 45.0,
                        "team_itt": over_under_line / 2.0 - (team_spread or 0) / 2.0,
                        "game_tot_z": 0.0,  # Will be computed in post-processing
                        "team_itt_z": 0.0,  # Will be computed in post-processing
                    }
                )
            else:
                weather_betting_features.update(
                    {
                        "team_spread": 0,
                        "team_spread_abs": 0,
                        "total_line": 45,
                        "is_favorite": 0,
                        "is_big_favorite": 0,
                        "is_big_underdog": 0,
                        "expected_pace": 1.0,
                        "team_itt": 22.5,
                        "game_tot_z": 0.0,
                        "team_itt_z": 0.0,
                    }
                )

            features.update(weather_betting_features)

            # Add injury and contextual features
            try:
                # Get player injury status
                player_injury = conn.execute(
                    """SELECT injury_status FROM players WHERE id = ?""", (player_id,)
                ).fetchone()

                injury_status = (
                    player_injury[0] if player_injury else None
                ) or "Healthy"

                # One-hot encode injury status
                features["injury_status_Out"] = 1 if injury_status == "Out" else 0
                features["injury_status_Doubtful"] = (
                    1 if injury_status == "Doubtful" else 0
                )
                features["injury_status_Questionable"] = (
                    1 if injury_status == "Questionable" else 0
                )
                features["injury_status_Probable"] = (
                    1 if injury_status in ["Probable", "Healthy"] else 0
                )

                # Add advanced QB-specific features for better prediction
                qb_features = {}

                # Enhanced passing efficiency metrics for QBs
                if position == "QB":
                    # Use recent averages to compute advanced metrics
                    avg_pass_att = features.get("avg_pass_attempts", 30)
                    avg_pass_comp = features.get("avg_completions", 18)
                    avg_pass_yds = features.get("avg_pass_yards", 250)
                    avg_pass_tds = features.get("avg_pass_tds", 1.5)
                    avg_ints = features.get("avg_interceptions", 0.7)
                    avg_rush_yds = features.get("avg_rush_yards", 15)

                    # Advanced efficiency metrics
                    completion_pct = avg_pass_comp / max(avg_pass_att, 1)
                    yds_per_att = avg_pass_yds / max(avg_pass_att, 1)
                    td_to_int_ratio = avg_pass_tds / max(avg_ints, 0.1)
                    passer_rating_est = min(
                        (completion_pct - 0.3) * 5
                        + (yds_per_att - 3) * 0.25
                        + avg_pass_tds * 0.2
                        - avg_ints * 0.25,
                        4.0,
                    )

                    # Enhanced game flow and situational features (Phase 4: Feature 1)
                    over_under_line = features.get("total_line", 45)
                    team_spread = features.get("team_spread", 0)
                    team_itt = (
                        over_under_line / 2.0 - team_spread / 2.0
                    )  # Implied team total
                    expected_pace = over_under_line / 45.0

                    # Phase 4: Advanced game script detection
                    is_big_favorite = team_spread < -7
                    is_big_underdog = team_spread > 7
                    is_shootout_game = over_under_line > 50  # High-scoring expected
                    is_low_total = over_under_line < 42  # Low-scoring expected

                    # Mismatch detection based on spread + total combination
                    pace_mismatch = 1.0  # Default to neutral (FIXED: was 0.0)
                    if (
                        is_shootout_game and abs(team_spread) < 3
                    ):  # High total, close spread = shootout
                        pace_mismatch = 1.5
                    elif (
                        is_low_total and is_big_favorite
                    ):  # Low total + big favorite = defensive game
                        pace_mismatch = 0.5
                    elif (
                        is_shootout_game and is_big_underdog
                    ):  # High total + big underdog = garbage time
                        pace_mismatch = 1.2

                    # Debug: Ensure safe values
                    if (
                        team_itt < 5 or team_itt > 60
                    ):  # Sanity check for implied team total
                        team_itt = 22.5  # Default to reasonable value

                    if (
                        pace_mismatch <= 0 or pace_mismatch > 3
                    ):  # Sanity check for pace mismatch
                        pace_mismatch = 1.0

                    qb_features.update(
                        {
                            "completion_pct_trend": completion_pct,
                            "yds_per_attempt_trend": yds_per_att,
                            "td_int_ratio_trend": td_to_int_ratio,
                            "passer_rating_est": max(passer_rating_est, 0.0),
                            "passing_volume_trend": min(avg_pass_att / 35.0, 2.0),
                            "dual_threat_factor": min(
                                avg_rush_yds / 20.0, 1.5
                            ),  # Rushing upside
                            "red_zone_efficiency_est": avg_pass_tds
                            / max(avg_pass_att * 0.15, 1),
                            # Phase 4: Enhanced game script features (with safe bounds)
                            "game_script_favorability": max(
                                0.2, min(expected_pace * pace_mismatch, 3.0)
                            ),
                            "implied_team_total": max(
                                0.3, min(team_itt / 30.0, 2.0)
                            ),  # Bounded normalized ITT
                            "shootout_potential": 1.0 if is_shootout_game else 0.0,
                            "defensive_game_script": 1.0 if is_low_total else 0.0,
                            "garbage_time_upside": 1.0
                            if (is_big_underdog and is_shootout_game)
                            else 0.0,
                            "blowout_risk": 1.0
                            if (is_big_favorite and is_low_total)
                            else 0.0,
                            "pressure_situation": 1.0
                            if team_spread < -7
                            else (0.7 if team_spread > 7 else 1.0),
                            "ceiling_indicator": min(
                                avg_pass_yds + avg_rush_yds + avg_pass_tds * 20, 400
                            )
                            / 400.0,
                        }
                    )

                # Enhanced RB-specific features for better prediction
                elif position == "RB":
                    # Get all RB-specific features
                    rb_features = get_rb_specific_features(
                        player_id,
                        player_name,
                        team_abbr,
                        opponent_abbr,
                        season,
                        week,
                        conn,
                    )
                    features.update(rb_features)

                    # Set default projections based on volume
                    if "rb_avg_touches" in features:
                        # More realistic baseline: 0.6 pts per touch + TD upside
                        base_projection = features["rb_avg_touches"] * 0.6
                        td_projection = features.get("rb_avg_total_tds", 0.5) * 6
                        features["baseline_projection"] = (
                            base_projection + td_projection
                        )

                    # Use recent averages to compute advanced RB metrics
                    avg_rush_att = features.get("avg_rush_attempts", 15)
                    avg_rush_yds = features.get("avg_rushing_yards", 75)
                    avg_rush_tds = features.get("avg_rush_tds", 0.5)
                    avg_rec_targs = features.get("avg_targets", 3)
                    avg_rec_yds = features.get("avg_receiving_yards", 25)
                    avg_rec_tds = features.get("avg_rec_tds", 0.2)

                    # PHASE 4B: RESEARCH-BASED ADVANCED RB FEATURES

                    # Core metrics
                    yards_per_carry = avg_rush_yds / max(avg_rush_att, 1)
                    rush_td_rate = avg_rush_tds / max(avg_rush_att, 1)
                    receiving_involvement = avg_rec_targs / max(
                        avg_rush_att + avg_rec_targs, 1
                    )
                    total_touches = avg_rush_att + avg_rec_targs

                    # FEATURE 1: Snap Share and Workload Analysis (Research Finding: #1 predictor)
                    estimated_snap_share = min(
                        total_touches / 25.0, 1.0
                    )  # Normalize to elite RB levels
                    workload_sustainability = (
                        1.0
                        if total_touches < 20
                        else (0.8 if total_touches < 25 else 0.6)
                    )
                    is_lead_back = (
                        1.0
                        if total_touches > 18
                        else (0.7 if total_touches > 15 else 0.4)
                    )
                    three_down_role = estimated_snap_share * (
                        1.5 if avg_rec_targs > 2 else 1.0
                    )

                    # FEATURE 2: Goal Line and Red Zone Specialization (Research Finding: Critical)
                    goal_line_back = 1.0 if avg_rush_tds > 0.8 else 0.0
                    short_yardage_specialist = (
                        1.0 if yards_per_carry < 3.8 and avg_rush_tds > 0.5 else 0.0
                    )
                    red_zone_value = min(
                        avg_rush_tds * 3.0, 2.0
                    )  # TDs are 3x more valuable
                    goal_line_monopoly = (
                        1.3
                        if avg_rush_tds > 1.0
                        else (1.1 if avg_rush_tds > 0.6 else 1.0)
                    )

                    # FEATURE 3: Receiving Role Depth (Research: Target value 2.5x rushing attempts)
                    third_down_back = receiving_involvement * (
                        1.5 if avg_rec_yds < 40 else 1.0
                    )
                    two_minute_drill_factor = receiving_involvement * 1.2
                    receiving_upside = min(
                        avg_rec_targs * 2.5, 15.0
                    )  # Research: 2.5x value multiplier

                    # FEATURE 4: Committee vs Bellcow Analysis (Research Finding)
                    rb_committee_factor = 1.0  # Default to bellcow
                    if total_touches < 15:  # Committee back
                        rb_committee_factor = 0.7
                    elif total_touches > 22:  # Workhorse (research: key indicator)
                        rb_committee_factor = 1.3
                    elif total_touches > 18:  # Volume back
                        rb_committee_factor = 1.2

                    # FEATURE 5: Weather and Game Environment (Research: RBs benefit from bad weather)
                    weather_boost = 1.0
                    if features.get("wind_gt15", 0) or features.get("cold_lt40", 0):
                        weather_boost = 1.2  # Bad weather = more rushing
                    elif features.get("dome", 0):
                        weather_boost = 0.95  # Dome = slightly less rushing emphasis
                    elif features.get("rain", 0):
                        weather_boost = 1.15  # Rain favors ground game

                    # Enhanced game flow features
                    over_under_line = features.get("total_line", 45)
                    team_spread = features.get("team_spread", 0)
                    team_itt = over_under_line / 2.0 - team_spread / 2.0
                    expected_pace = over_under_line / 45.0

                    # Game script detection
                    is_big_favorite = team_spread < -7
                    is_big_underdog = team_spread > 7
                    is_shootout_game = over_under_line > 50
                    is_low_total = over_under_line < 42

                    # Enhanced RB game script logic
                    rb_game_script = 1.0
                    if is_big_favorite and is_low_total:
                        rb_game_script = 1.8  # Clock control = RB paradise
                    elif is_big_favorite:
                        rb_game_script = 1.4  # Likely to run more in 2nd half
                    elif is_big_underdog and is_shootout_game:
                        rb_game_script = 0.6  # Pass-heavy game script
                    elif is_shootout_game:
                        rb_game_script = 1.1  # More plays overall
                    elif is_low_total:
                        rb_game_script = 1.3  # Defensive game = more rushing

                    # Advanced efficiency and opportunity metrics
                    workload_efficiency = min(
                        (avg_rush_yds + avg_rec_yds) / max(total_touches, 1), 20.0
                    )
                    dual_threat_factor = min(avg_rec_yds / max(avg_rush_yds, 1), 1.5)
                    yac_estimate = max(
                        0.0, yards_per_carry - 3.5
                    )  # Estimate YAC above average
                    breakaway_potential = (
                        1.0
                        if yards_per_carry > 5.0
                        else (0.5 if yards_per_carry > 4.5 else 0.0)
                    )
                    replacement_upside = (
                        1.3
                        if total_touches > 20 and workload_sustainability < 0.8
                        else 1.0
                    )
                    high_stakes_workload = 1.1 if estimated_snap_share > 0.7 else 1.0

                    # Sanity checks
                    if team_itt < 5 or team_itt > 60:
                        team_itt = 22.5
                    if rb_game_script <= 0 or rb_game_script > 3:
                        rb_game_script = 1.0

                    qb_features.update(
                        {
                            # Core enhanced metrics
                            "yards_per_carry_trend": yards_per_carry,
                            "rush_td_rate_trend": rush_td_rate * 100,
                            "receiving_involvement": receiving_involvement,
                            "total_touches_trend": min(total_touches / 20.0, 2.0),
                            "workload_efficiency": workload_efficiency,
                            "dual_threat_factor": dual_threat_factor,
                            "red_zone_efficiency_est": (avg_rush_tds + avg_rec_tds)
                            / max(avg_rush_att * 0.2, 1),
                            # PHASE 4B: Advanced RB features (Research-Based)
                            "snap_share_estimate": estimated_snap_share,
                            "workload_sustainability": workload_sustainability,
                            "lead_back_role": is_lead_back,
                            "three_down_back_value": three_down_role,
                            "goal_line_specialist": goal_line_back,
                            "goal_line_monopoly": goal_line_monopoly,
                            "short_yardage_role": short_yardage_specialist,
                            "red_zone_td_value": red_zone_value,
                            "third_down_involvement": third_down_back,
                            "two_minute_drill_value": two_minute_drill_factor,
                            "receiving_value_multiplier": min(
                                receiving_upside / 10.0, 2.0
                            ),
                            "committee_vs_bellcow": rb_committee_factor,
                            "weather_game_script_boost": weather_boost,
                            "yards_after_contact_estimate": min(yac_estimate, 3.0),
                            "breakaway_run_potential": breakaway_potential,
                            "replacement_upside": replacement_upside,
                            "high_stakes_workload": high_stakes_workload,
                            # Enhanced game script features (RB-optimized)
                            "game_script_favorability": max(
                                0.2, min(expected_pace * rb_game_script, 3.0)
                            ),
                            "implied_team_total": max(0.3, min(team_itt / 30.0, 2.0)),
                            "shootout_potential": 1.0 if is_shootout_game else 0.0,
                            "defensive_game_script": 1.0 if is_low_total else 0.0,
                            "garbage_time_upside": 0.3
                            if (is_big_underdog and is_shootout_game)
                            else 0.0,
                            "blowout_risk": 0.0,  # RBs benefit from blowouts (clock management)
                            "clock_management_upside": 1.5
                            if (is_big_favorite and is_low_total)
                            else (1.2 if is_big_favorite else 1.0),
                            "pressure_situation": 0.7
                            if team_spread < -7
                            else (1.3 if team_spread > 7 else 1.0),  # Inverse of QB
                            "ceiling_indicator": min(
                                avg_rush_yds
                                + avg_rec_yds
                                + (avg_rush_tds + avg_rec_tds) * 15,
                                300,
                            )
                            / 300.0,
                        }
                    )

                # Enhanced TE-specific features for better prediction
                elif position == "TE":
                    # First get TE-specific features
                    te_specific = get_te_specific_features(
                        player_id,
                        player_name=player_name,
                        team_abbr=team_abbr,
                        opponent_abbr=opponent_abbr,
                        season=season,
                        week=week,
                        conn=conn,
                    )
                    features.update(te_specific)

                    # TE dual-role metrics (keep existing for compatibility)
                    avg_targets = features.get("avg_targets", 3)
                    avg_rec_yds = features.get("avg_receiving_yards", 35)
                    avg_rec_tds = features.get("avg_rec_tds", 0.25)
                    avg_receptions = features.get("avg_receptions", 2.5)

                    # Enhanced with TE-specific projections
                    base_projection = (
                        te_specific.get("te_route_volume", 4.0)
                        * te_specific.get("te_catch_rate", 0.65)
                        * te_specific.get("te_yards_per_target", 8.5)
                        / 10.0
                        + te_specific.get("te_td_rate", 0.12) * 6.0
                    )

                    # Apply game script multipliers
                    script_multiplier = (
                        te_specific.get("te_winning_game_boost", 1.05)
                        if team_spread < -3
                        else te_specific.get("te_losing_game_penalty", 0.9)
                        if team_spread > 3
                        else 1.0
                    )
                    base_projection * script_multiplier

                    # TE-specific efficiency
                    te_target_rate = avg_targets / max(
                        15, 1
                    )  # Normalize against team average
                    red_zone_role = avg_rec_tds / max(
                        avg_targets * 0.25, 1
                    )  # TEs get more RZ looks
                    short_area_specialist = (
                        1.0 if (avg_rec_yds / max(avg_receptions, 1)) < 12 else 0.0
                    )

                    # Game script (TE-optimized)
                    over_under_line = features.get("total_line", 45)
                    team_spread = features.get("team_spread", 0)
                    team_itt = over_under_line / 2.0 - team_spread / 2.0

                    # TE benefits from both run-heavy (blocking) and pass-heavy (receiving) scripts
                    te_game_script = 1.0  # Base
                    if abs(team_spread) < 3:  # Close games = more TE involvement
                        te_game_script = 1.3
                    elif over_under_line > 50:  # High-scoring = more targets
                        te_game_script = 1.2
                    elif team_spread < -7:  # Big favorites = more goal line looks
                        te_game_script = 1.1

                    # Sanity checks
                    if team_itt < 5 or team_itt > 60:
                        team_itt = 22.5
                    if te_game_script <= 0 or te_game_script > 3:
                        te_game_script = 1.0

                    qb_features.update(
                        {
                            "te_target_rate": te_target_rate,
                            "red_zone_specialist": red_zone_role,
                            "short_area_role": short_area_specialist,
                            "receiving_efficiency": min(
                                avg_rec_yds / max(avg_targets, 1), 15.0
                            ),
                            "touchdown_dependency": min(
                                avg_rec_tds * 4, 1.5
                            ),  # TDs crucial for TE scoring
                            "dual_role_value": te_game_script,
                            # Game script features
                            "game_script_favorability": max(
                                0.4, min(te_game_script, 2.0)
                            ),
                            "implied_team_total": max(0.3, min(team_itt / 30.0, 2.0)),
                            "close_game_upside": 1.0 if abs(team_spread) < 3 else 0.0,
                            "goal_line_opportunities": 1.0 if team_spread < -7 else 0.0,
                            "ceiling_indicator": min(
                                avg_rec_yds + avg_rec_tds * 20, 150
                            )
                            / 150.0,
                        }
                    )

                # Enhanced WR-specific features for better prediction
                elif position == "WR":
                    # Get all WR-specific features
                    wr_features = get_wr_specific_features(
                        player_id,
                        player_name,
                        team_abbr,
                        opponent_abbr,
                        season,
                        week,
                        conn,
                    )
                    features.update(wr_features)

                    # Set default projections based on targets and game script
                    if "wr_avg_targets" in features and "wr_implied_total" in features:
                        # Base: 1 point per target + game script bonus + TD upside
                        base_projection = features["wr_avg_targets"] * 1.0
                        script_bonus = (
                            features["wr_implied_total"] - 20
                        ) * 0.2  # Higher totals = more points
                        td_projection = features.get("wr_avg_rec_tds", 0.3) * 6
                        features["baseline_projection"] = (
                            base_projection + script_bonus + td_projection
                        )

                    # Advanced WR efficiency metrics
                    avg_targets = features.get("avg_targets", 5)
                    avg_rec_yds = features.get("avg_receiving_yards", 60)
                    avg_rec_tds = features.get("avg_rec_tds", 0.3)
                    avg_receptions = features.get("avg_receptions", 3.5)

                    # WR-specific metrics
                    target_share = avg_targets / max(
                        25, 1
                    )  # Normalize against team target volume
                    air_yards_per_target = avg_rec_yds / max(avg_targets, 1)
                    catch_rate = avg_receptions / max(avg_targets, 1)
                    red_zone_target_rate = avg_rec_tds / max(avg_targets * 0.15, 1)

                    # Game script features (WR-optimized)
                    over_under_line = features.get("total_line", 45)
                    team_spread = features.get("team_spread", 0)
                    team_itt = over_under_line / 2.0 - team_spread / 2.0

                    # WR game script logic
                    is_shootout_game = over_under_line > 50
                    is_big_underdog = team_spread > 7
                    is_pass_heavy_script = is_shootout_game or is_big_underdog

                    wr_game_script = 1.0  # Default
                    if is_pass_heavy_script:
                        wr_game_script = 1.4  # Favorable for WRs
                    elif team_spread < -7:  # Big favorite = less passing
                        wr_game_script = 0.8

                    # Sanity checks
                    if team_itt < 5 or team_itt > 60:
                        team_itt = 22.5
                    if wr_game_script <= 0 or wr_game_script > 3:
                        wr_game_script = 1.0

                    qb_features.update(
                        {
                            "target_share_trend": min(target_share, 1.0),
                            "air_yards_per_target": min(air_yards_per_target, 25.0),
                            "catch_rate_trend": catch_rate,
                            "red_zone_involvement": red_zone_target_rate,
                            "route_efficiency": min(
                                avg_rec_yds / max(avg_receptions, 1), 25.0
                            ),
                            "big_play_upside": 1.0
                            if air_yards_per_target > 12
                            else 0.0,
                            # Game script features
                            "game_script_favorability": max(
                                0.3, min(wr_game_script, 2.5)
                            ),
                            "implied_team_total": max(0.3, min(team_itt / 30.0, 2.0)),
                            "shootout_potential": 1.0 if is_shootout_game else 0.0,
                            "pass_heavy_script": 1.0 if is_pass_heavy_script else 0.0,
                            "garbage_time_upside": 1.2
                            if (is_big_underdog and is_shootout_game)
                            else 1.0,
                            "ceiling_indicator": min(
                                avg_rec_yds + avg_rec_tds * 15, 200
                            )
                            / 200.0,
                        }
                    )

                # Enhanced DST-specific features for better prediction (Research-Based Improvements)
                elif position in ["DST", "DEF"]:
                    # Core historical defensive stats (fixed data without leakage)
                    avg_recent_fantasy = features.get("avg_recent_fantasy_points", 6)
                    avg_recent_pa = features.get("avg_recent_points_allowed", 20)
                    avg_recent_sacks = features.get("avg_recent_sacks", 2)
                    avg_recent_ints = features.get("avg_recent_interceptions", 0.8)
                    avg_recent_fumbles = features.get(
                        "avg_recent_fumbles_recovered", 0.5
                    )
                    avg_recent_tds = features.get("avg_recent_defensive_tds", 0.1)

                    # ADVANCED FEATURE 1: Pressure Rate Analysis (Key Research Finding)
                    # Pressure rate is the #1 predictor of DST success
                    pressure_attempts = (
                        avg_recent_sacks * 4
                    )  # Estimate total pressure attempts
                    pressure_rate = min(
                        avg_recent_sacks / max(pressure_attempts, 1), 0.5
                    )  # Cap at 50%
                    pressure_effectiveness = (
                        pressure_rate * 2.0
                    )  # Scale for feature importance

                    # ADVANCED FEATURE 2: Opponent Offensive Strength Analysis
                    over_under_line = features.get("total_line", 45)
                    team_spread = features.get("team_spread", 0)
                    opp_itt = (
                        over_under_line / 2.0 + team_spread / 2.0
                    )  # Opponent implied team total

                    # Opponent quality indicators
                    opp_passing_strength = max(
                        0.5, min(opp_itt / 25.0, 1.8)
                    )  # Higher ITT = stronger offense
                    opp_turnover_prone = (
                        1.0 if opp_itt < 20 else (1.2 if opp_itt < 18 else 0.8)
                    )
                    opp_offensive_line_quality = 1.0 - min(
                        pressure_rate, 0.4
                    )  # Inverse of pressure allowed

                    # ADVANCED FEATURE 3: Game Script Scenarios (Research-Based)
                    is_opp_pass_heavy = team_spread > 7 or over_under_line > 50
                    is_low_scoring = over_under_line < 42
                    is_blowout_potential = abs(team_spread) > 10
                    is_negative_game_script = (
                        team_spread < -7
                    )  # DST team heavily favored

                    # Multi-scenario game script analysis
                    game_script_multiplier = 1.0
                    if is_opp_pass_heavy and over_under_line > 52:
                        game_script_multiplier = 1.5  # High-volume passing game
                    elif is_low_scoring and abs(team_spread) < 3:
                        game_script_multiplier = 1.3  # Defensive slugfest
                    elif is_blowout_potential:
                        game_script_multiplier = (
                            1.4  # Desperate opponent or garbage time
                        )
                    elif is_negative_game_script:
                        game_script_multiplier = (
                            0.8  # DST team ahead, less opportunities
                        )

                    # ADVANCED FEATURE 4: Turnover Generation Composite
                    # Research shows turnovers correlate more with DST scoring than points allowed
                    total_turnovers = avg_recent_ints + avg_recent_fumbles
                    turnover_rate = min(
                        total_turnovers / 1.5, 2.0
                    )  # Normalize to expected range
                    turnover_upside = 1.0 + (
                        turnover_rate * 0.5
                    )  # Bonus for turnover-heavy defenses

                    # ADVANCED FEATURE 5: Points Allowed Buckets (Tiered Scoring)
                    # Research shows points allowed has non-linear fantasy impact
                    if avg_recent_pa <= 14:
                        pa_tier_bonus = 2.0  # Elite defense
                    elif avg_recent_pa <= 17:
                        pa_tier_bonus = 1.5  # Very good defense
                    elif avg_recent_pa <= 21:
                        pa_tier_bonus = 1.2  # Good defense
                    elif avg_recent_pa <= 24:
                        pa_tier_bonus = 1.0  # Average defense
                    else:
                        pa_tier_bonus = 0.7  # Poor defense

                    # ADVANCED FEATURE 6: Special Teams Impact
                    st_upside = 1.0 + (
                        avg_recent_tds * 3.0
                    )  # ST/Defensive TDs are game-changers
                    return_td_potential = min(
                        avg_recent_tds * 5.0, 1.5
                    )  # Cap special teams bonus

                    # ADVANCED FEATURE 7: Defensive Consistency vs Ceiling
                    def_floor = avg_recent_fantasy - (
                        avg_recent_fantasy * 0.4
                    )  # 40% floor variance
                    def_ceiling = (
                        avg_recent_fantasy
                        + (total_turnovers * 3)
                        + (avg_recent_tds * 6)
                    )
                    def_range = def_ceiling - def_floor
                    volatility_factor = min(
                        def_range / 8.0, 2.0
                    )  # Higher volatility = more upside

                    # Sanity checks
                    if opp_itt < 10 or opp_itt > 35:
                        opp_itt = 22.5
                    if game_script_multiplier <= 0.3 or game_script_multiplier > 2.0:
                        game_script_multiplier = 1.0

                    qb_features.update(
                        {
                            # Core defensive metrics (enhanced)
                            "pressure_rate_effectiveness": min(
                                pressure_effectiveness, 3.0
                            ),
                            "turnover_generation_composite": turnover_upside,
                            "points_allowed_tier_bonus": pa_tier_bonus,
                            "special_teams_upside": st_upside,
                            # Opponent analysis
                            "opponent_offensive_strength": opp_passing_strength,
                            "opponent_turnover_prone": opp_turnover_prone,
                            "opponent_line_protection": opp_offensive_line_quality,
                            "opponent_implied_total": max(
                                0.4, min(opp_itt / 25.0, 1.6)
                            ),
                            # Game script scenarios
                            "game_script_multiplier": game_script_multiplier,
                            "high_volume_passing_game": 1.0
                            if (is_opp_pass_heavy and over_under_line > 50)
                            else 0.0,
                            "defensive_game_environment": 1.0
                            if is_low_scoring
                            else 0.0,
                            "blowout_potential": 1.0 if is_blowout_potential else 0.0,
                            "negative_game_script": 1.0
                            if is_negative_game_script
                            else 0.0,
                            # Advanced metrics
                            "sack_rate_normalized": min(avg_recent_sacks / 2.5, 2.0),
                            "interception_rate_boosted": min(
                                avg_recent_ints * 3.0, 2.5
                            ),
                            "fumble_recovery_rate": min(avg_recent_fumbles * 4.0, 2.0),
                            "defensive_td_potential": return_td_potential,
                            "volatility_upside_factor": volatility_factor,
                            # Composite indicators
                            "defensive_floor_estimate": max(0.2, def_floor / 10.0),
                            "defensive_ceiling_estimate": min(def_ceiling / 15.0, 2.0),
                            "overall_dst_favorability": min(
                                (
                                    pressure_effectiveness
                                    + turnover_upside
                                    + pa_tier_bonus
                                    + game_script_multiplier
                                )
                                / 4.0,
                                2.5,
                            ),
                        }
                    )

                # Add all contextual features including new position-specific features
                all_features = {
                    "games_missed_last4": 0,
                    "practice_trend": 0,
                    "returning_from_injury": 0,
                    "team_injured_starters": 0,
                    "opp_injured_starters": 0,
                    "targets_ema": features.get("avg_targets", 0),
                    "routes_run_ema": 0,
                    "rush_att_ema": features.get("avg_rush_attempts", 0),
                    "snap_share_ema": 0.7,
                    "redzone_opps_ema": 0,
                    "air_yards_ema": 0,
                    "adot_ema": 0,
                    "yprr_ema": 0,
                    "yards_after_contact": 0,
                    "missed_tackles_forced": 0,
                    "pressure_rate": 0,
                    "opp_dvp_pos_allowed": 0,
                    "salary": 5000,
                    "home": 1 if is_home else 0,
                    "rest_days": 7,
                    "travel": 0,
                    "season_week": week,
                }
                all_features.update(qb_features)
                features.update(all_features)

            except Exception as e:
                logger.debug(
                    f"Error adding contextual features for player {player_id}: {e}"
                )
                # Add defaults if query fails
                features.update(
                    {
                        "injury_status_Out": 0,
                        "injury_status_Doubtful": 0,
                        "injury_status_Questionable": 0,
                        "injury_status_Probable": 1,
                        "games_missed_last4": 0,
                        "practice_trend": 0,
                        "returning_from_injury": 0,
                        "team_injured_starters": 0,
                        "opp_injured_starters": 0,
                        "targets_ema": 0,
                        "routes_run_ema": 0,
                        "rush_att_ema": 0,
                        "snap_share_ema": 0.7,
                        "redzone_opps_ema": 0,
                        "air_yards_ema": 0,
                        "adot_ema": 0,
                        "yprr_ema": 0,
                        "yards_after_contact": 0,
                        "missed_tackles_forced": 0,
                        "pressure_rate": 0,
                        "opp_dvp_pos_allowed": 0,
                        "salary": 5000,
                        "home": 1 if is_home else 0,
                        "rest_days": 7,
                        "travel": 0,
                        "season_week": week,
                    }
                )

            # Correlation features
            try:
                import importlib

                models_module = importlib.import_module("models")
                correlation_extractor = models_module.CorrelationFeatureExtractor(
                    db_path
                )
                correlation_features = (
                    correlation_extractor.extract_correlation_features(
                        player_id, game_id, position
                    )
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
            if fname == "total_line":
                total_line_idx = i
            elif fname == "team_itt":
                team_itt_idx = i
            elif fname == "game_tot_z":
                game_tot_z_idx = i
            elif fname == "team_itt_z":
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
            for (_season, _week), indices in week_groups.items():
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
            important_features = [
                "weather_",
                "spread",
                "total",
                "favorite",
                "underdog",
                "pace",
                "cold_weather",
                "hot_weather",
                "high_wind",
                "stadium",
            ]

            for i, fname in enumerate(feature_names):
                if any(keyword in fname for keyword in important_features):
                    non_constant_mask[i] = True  # Force keep important features

            if not np.all(non_constant_mask):
                constant_features = [
                    feature_names[i]
                    for i, keep in enumerate(non_constant_mask)
                    if not keep
                ]
                cleaning_progress.finish(
                    f"Removed {len(constant_features)} constant features"
                )

                X = X[:, non_constant_mask]
                feature_names = [
                    feature_names[i] for i, keep in enumerate(non_constant_mask) if keep
                ]
            else:
                cleaning_progress.update(3, 3)  # Final step

        # Apply feature validation if available
        try:
            import pandas as pd

            from utils_feature_validation import (
                load_expected_schema,
                validate_and_prepare_features,
            )

            # Convert to DataFrame for validation
            df = pd.DataFrame(X, columns=feature_names)

            # Load expected schema and validate
            try:
                expected_schema = load_expected_schema("feature_names.json")
                df = validate_and_prepare_features(
                    df, expected_schema, allow_extra=True
                )

                # Update arrays after validation
                X = df.values.astype(np.float32)
                feature_names = df.columns.tolist()
                pass  # Validation successful
            except Exception as ve:
                logger.warning(
                    f"Feature validation failed, continuing without validation: {ve}"
                )

        except ImportError:
            pass  # Validation not available

        if not cleaning_progress._finished:
            cleaning_progress.finish()

        # Filter for top performers (remove low-scoring games that add noise)
        X, y = _filter_for_top_performers(X, y, position)

        logger.info(
            f"Final dataset: {len(X)} training samples for {position} with {len(feature_names)} features"
        )
        return X, y, feature_names

    except Exception as e:
        logger.error(f"Error getting training data for {position}: {e}")
        return np.array([]), np.array([]), []
    finally:
        conn.close()


def compute_features_from_stats(
    player_stats: List[Tuple], target_game_date, lookback_weeks: int = 4
) -> Dict[str, float]:
    """Compute features from pre-loaded player stats."""
    features = {}

    if not player_stats:
        return features

    # Filter to recent games before target date
    cutoff_date = target_game_date - timedelta(weeks=lookback_weeks)
    recent_stats = [
        stat
        for stat in player_stats
        if parse_date_flexible(stat[14]).date() >= cutoff_date
        and parse_date_flexible(stat[14]).date() < target_game_date
    ]

    # If no recent stats, expand the window to get any historical data
    if not recent_stats:
        # Try with all available data before the target date
        recent_stats = [
            stat
            for stat in player_stats
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
    features["avg_fantasy_points"] = np.mean(recent_points) if recent_points else 0
    features["avg_passing_yards"] = (
        np.mean(recent_pass_yards) if recent_pass_yards else 0
    )
    features["avg_rushing_yards"] = (
        np.mean(recent_rush_yards) if recent_rush_yards else 0
    )
    features["avg_receiving_yards"] = (
        np.mean(recent_rec_yards) if recent_rec_yards else 0
    )
    features["avg_targets"] = np.mean(recent_targets) if recent_targets else 0
    features["avg_pass_tds"] = np.mean(recent_pass_tds) if recent_pass_tds else 0
    features["avg_rush_tds"] = np.mean(recent_rush_tds) if recent_rush_tds else 0
    features["avg_rec_tds"] = np.mean(recent_rec_tds) if recent_rec_tds else 0
    features["avg_interceptions"] = (
        np.mean(recent_interceptions) if recent_interceptions else 0
    )
    features["avg_fumbles"] = np.mean(recent_fumbles) if recent_fumbles else 0
    features["avg_rush_attempts"] = (
        np.mean(recent_rush_attempts) if recent_rush_attempts else 0
    )
    features["avg_receptions"] = np.mean(recent_receptions) if recent_receptions else 0

    # Advanced metrics
    if recent_rush_attempts and recent_rush_yards:
        features["yards_per_carry"] = np.mean(
            [
                y / max(a, 1)
                for y, a in zip(recent_rush_yards, recent_rush_attempts, strict=False)
            ]
        )
    else:
        features["yards_per_carry"] = 0

    if recent_receptions and recent_rec_yards:
        features["yards_per_reception"] = np.mean(
            [
                y / max(r, 1)
                for y, r in zip(recent_rec_yards, recent_receptions, strict=False)
            ]
        )
    else:
        features["yards_per_reception"] = 0

    if recent_targets and recent_receptions:
        features["catch_rate"] = np.mean(
            [
                r / max(t, 1)
                for r, t in zip(recent_receptions, recent_targets, strict=False)
            ]
        )
    else:
        features["catch_rate"] = 0

    # Games played and consistency metrics
    features["games_played"] = len(recent_stats)
    if recent_points:
        features["max_points"] = max(recent_points)
        features["min_points"] = min(recent_points)
        features["consistency"] = 1 - (
            np.std(recent_points) / (np.mean(recent_points) + 1e-6)
        )
    else:
        features["max_points"] = 0
        features["min_points"] = 0
        features["consistency"] = 0

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
    contest_id: str = None, db_path: str = "data/nfl_dfs.db"
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
            (contest_id,),
        ).fetchall()

        return [
            {
                "player_id": row[0],
                "name": row[1],
                "position": row[2],
                "salary": row[3],
                "roster_position": row[4],
                "team_abbr": row[5],
                "team_name": row[6],
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
        conn.execute(
            "DELETE FROM player_stats WHERE game_id NOT IN (SELECT id FROM games)"
        )

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
            issues["missing_fantasy_points"] = missing_points

        # Check for players without team
        orphan_players = conn.execute(
            "SELECT COUNT(*) FROM players WHERE team_id IS NULL"
        ).fetchone()[0]

        if orphan_players > 0:
            issues["orphan_players"] = orphan_players

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
                issues["stale_data_days"] = days_old

    except Exception as e:
        issues["validation_error"] = str(e)
    finally:
        conn.close()

    return issues


def import_spreadspoke_data(
    csv_path: str, db_path: str = "data/nfl_dfs.db", seasons: Optional[List[int]] = None
) -> None:
    """Import weather and betting data from spreadspoke CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    conn = get_db_connection(db_path)

    # Add weather columns to games table if they don't exist
    try:
        # Check what columns already exist
        columns = conn.execute("PRAGMA table_info(games)").fetchall()
        existing_cols = {col[1] for col in columns}

        weather_columns_to_add = [
            ("stadium", "TEXT"),
            ("stadium_neutral", "INTEGER DEFAULT 0"),
            ("weather_temperature", "INTEGER"),
            ("weather_wind_mph", "INTEGER"),
            ("weather_humidity", "INTEGER"),
            ("weather_detail", "TEXT"),
        ]

        for col_name, col_type in weather_columns_to_add:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE games ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column {col_name} to games table")

        # Create betting_odds table if it doesn't exist
        conn.execute("""
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
        """)

        conn.commit()

    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        raise

    # Create team abbreviation mapping for different naming conventions
    team_mapping = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LAR",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Oakland Raiders": "LV",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS",
        "Washington Redskins": "WAS",
        "Washington Football Team": "WAS",
    }

    # Get team IDs from database
    team_ids = {}
    for row in conn.execute("SELECT id, team_abbr FROM teams").fetchall():
        team_ids[row[1]] = row[0]

    games_inserted = 0
    odds_inserted = 0
    games_updated = 0

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Parse basic game info
                raw_game_date = row["schedule_date"]
                # Convert date from M/D/YYYY to YYYY-MM-DD format
                try:
                    parsed_date = parse_date_flexible(raw_game_date)
                    game_date = parsed_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.warning(f"Could not parse date {raw_game_date}: {e}")
                    continue
                season = int(row["schedule_season"])

                # Optional season filtering
                if seasons is not None and season not in seasons:
                    continue
                is_playoff = row["schedule_playoff"].upper() == "TRUE"

                # Handle playoff weeks (Wildcard, Division, Conference, Superbowl)
                week_str = row["schedule_week"]
                if is_playoff:
                    week_mapping = {
                        "Wildcard": 18,
                        "Division": 19,
                        "Conference": 20,
                        "Superbowl": 21,
                    }
                    week = week_mapping.get(week_str, 18)  # Default to 18 if unknown
                else:
                    week = int(week_str)

                # Map team names to abbreviations
                home_team = team_mapping.get(row["team_home"], row["team_home"])
                away_team = team_mapping.get(row["team_away"], row["team_away"])

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
                home_score = int(row["score_home"]) if row["score_home"] else None
                away_score = int(row["score_away"]) if row["score_away"] else None
                game_finished = (
                    1 if home_score is not None and away_score is not None else 0
                )

                # Parse weather data
                weather_temp = (
                    int(row["weather_temperature"])
                    if row["weather_temperature"]
                    else None
                )
                weather_wind = (
                    int(row["weather_wind_mph"]) if row["weather_wind_mph"] else None
                )
                weather_humidity = (
                    int(row["weather_humidity"]) if row["weather_humidity"] else None
                )
                weather_detail = (
                    row["weather_detail"] if row["weather_detail"] else None
                )

                # Parse betting data
                favorite_team = (
                    team_mapping.get(row["team_favorite_id"], row["team_favorite_id"])
                    if row["team_favorite_id"]
                    else None
                )
                spread_favorite = (
                    float(row["spread_favorite"]) if row["spread_favorite"] else None
                )
                over_under = (
                    float(row["over_under_line"]) if row["over_under_line"] else None
                )

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

                stadium = row["stadium"] if row["stadium"] else None
                stadium_neutral = 1 if row["stadium_neutral"].upper() == "TRUE" else 0

                try:
                    # Check if game already exists
                    existing_game = conn.execute(
                        "SELECT id FROM games WHERE id = ?", (game_id,)
                    ).fetchone()

                    if existing_game:
                        # Update existing game with weather/stadium data only
                        conn.execute(
                            """
                            UPDATE games SET
                                stadium = ?, stadium_neutral = ?,
                                weather_temperature = ?, weather_wind_mph = ?,
                                weather_humidity = ?, weather_detail = ?
                            WHERE id = ?
                        """,
                            (
                                stadium,
                                stadium_neutral,
                                weather_temp,
                                weather_wind,
                                weather_humidity,
                                weather_detail,
                                game_id,
                            ),
                        )
                        games_updated += 1
                    else:
                        # Insert new game (if it doesn't exist in your data)
                        conn.execute(
                            """
                            INSERT INTO games (
                                id, game_date, season, week, home_team_id, away_team_id,
                                home_score, away_score, game_finished, stadium, stadium_neutral,
                                weather_temperature, weather_wind_mph, weather_humidity, weather_detail
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                game_id,
                                game_date,
                                season,
                                week,
                                home_team_id,
                                away_team_id,
                                home_score,
                                away_score,
                                game_finished,
                                stadium,
                                stadium_neutral,
                                weather_temp,
                                weather_wind,
                                weather_humidity,
                                weather_detail,
                            ),
                        )
                        games_inserted += 1

                    # Insert betting odds if available
                    if spread_favorite or over_under:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO betting_odds (
                                game_id, favorite_team, spread_favorite, over_under_line,
                                home_team_spread, away_team_spread, source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                game_id,
                                favorite_team,
                                spread_favorite,
                                over_under,
                                home_spread,
                                away_spread,
                                "spreadspoke",
                            ),
                        )
                        odds_inserted += 1

                except Exception as e:
                    logger.error(f"Error inserting game {game_id}: {e}")
                    continue

        conn.commit()
        logger.info(
            f"Spreadspoke import complete: {games_inserted} games inserted, {games_updated} games updated, {odds_inserted} betting records inserted"
        )

    except Exception as e:
        logger.error(f"Error importing spreadspoke data: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def collect_odds_data(
    target_date: str = None, db_path: str = "data/nfl_dfs.db"
) -> None:
    """Collect NFL betting odds from The Odds API for upcoming games.

    Args:
        target_date: Date in YYYY-MM-DD format. If None, collects all upcoming games.
        db_path: Path to the SQLite database
    """
    odds_api_key = os.getenv("ODDS_API_KEY")
    if not odds_api_key or odds_api_key == "your_key_here":
        raise ValueError(
            "ODDS_API_KEY environment variable not set. Please set it in your .env file."
        )

    conn = get_db_connection(db_path)

    try:
        # API endpoint for NFL odds
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": odds_api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        logger.info("Fetching NFL odds from The Odds API...")
        response = requests.get(url, params=params)
        response.raise_for_status()

        odds_data = response.json()
        logger.info(f"Retrieved {len(odds_data)} games with odds data")

        odds_inserted = 0

        for game in odds_data:
            game["id"]
            commence_time = game["commence_time"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Parse the commence time to check if it matches target date
            game_date = datetime.fromisoformat(
                commence_time.replace("Z", "+00:00")
            ).date()
            if target_date:
                target_date_obj = parse_date_flexible(target_date).date()
                if game_date != target_date_obj:
                    continue

            # Find matching game in our database (check both games table and draftkings_salaries)
            db_game = conn.execute(
                """
                SELECT id FROM games
                WHERE (home_team_id = (SELECT id FROM teams WHERE team_abbr = ?)
                       OR home_team_id = (SELECT id FROM teams WHERE team_name = ?))
                  AND (away_team_id = (SELECT id FROM teams WHERE team_abbr = ?)
                       OR away_team_id = (SELECT id FROM teams WHERE team_name = ?))
                  AND date(game_date) = date(?)
            """,
                (home_team, home_team, away_team, away_team, commence_time[:10]),
            ).fetchone()

            # If not found in games table, check draftkings_salaries table for upcoming games
            if not db_game:
                # Convert team names to abbreviations for DraftKings format matching
                away_abbr = conn.execute(
                    "SELECT team_abbr FROM teams WHERE team_name = ? OR team_abbr = ?",
                    (away_team, away_team),
                ).fetchone()
                home_abbr = conn.execute(
                    "SELECT team_abbr FROM teams WHERE team_name = ? OR team_abbr = ?",
                    (home_team, home_team),
                ).fetchone()

                if away_abbr and home_abbr:
                    away_abbr = away_abbr[0]
                    home_abbr = home_abbr[0]

                    # Format date to match DraftKings format (MM/DD/YYYY)
                    dk_date = game_date.strftime("%m/%d/%Y")

                    # Check if we have DraftKings data for this matchup - format: "AWAY@HOME MM/DD/YYYY"
                    dk_game = conn.execute(
                        """
                        SELECT DISTINCT game_info FROM draftkings_salaries
                        WHERE game_info LIKE ?
                    """,
                        (f"{away_abbr}@{home_abbr} {dk_date}%",),
                    ).fetchone()

                    # Use a meaningful game_id for upcoming games regardless of DK slate availability
                    db_game_id = f"{game_date}_{away_abbr}@{home_abbr}"

                    # Create minimal game record for foreign key constraint
                    away_team_id = conn.execute(
                        "SELECT id FROM teams WHERE team_abbr = ?", (away_abbr,)
                    ).fetchone()
                    home_team_id = conn.execute(
                        "SELECT id FROM teams WHERE team_abbr = ?", (home_abbr,)
                    ).fetchone()

                    if away_team_id and home_team_id:
                        away_team_id = away_team_id[0]
                        home_team_id = home_team_id[0]

                        # Insert minimal game record if it doesn't exist
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO games (
                                id, game_date, season, week, home_team_id, away_team_id,
                                home_score, away_score, game_finished
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                db_game_id,
                                game_date.strftime("%Y-%m-%d"),
                                game_date.year,
                                1,
                                home_team_id,
                                away_team_id,
                                0,
                                0,
                                0,
                            ),
                        )

                        if dk_game:
                            logger.info(
                                f"Found upcoming game in DraftKings data: {away_abbr}@{home_abbr} on {dk_date}"
                            )
                        else:
                            logger.info(
                                f"No DraftKings slate found; created upcoming game: {away_abbr}@{home_abbr} on {dk_date}"
                            )
                    else:
                        logger.warning(
                            f"Could not find team IDs for {away_abbr}/{home_abbr}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Could not find team abbreviations for {away_team} / {home_team}"
                    )
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
            if game.get("bookmakers"):
                bookmaker = game["bookmakers"][0]  # Use first bookmaker

                for market in bookmaker.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_team:
                                home_spread = float(outcome["point"])
                            elif outcome["name"] == away_team:
                                away_spread = float(outcome["point"])

                        # Determine favorite
                        if home_spread is not None and away_spread is not None:
                            if home_spread < away_spread:
                                spread_favorite = abs(home_spread)
                                spread_fav_team = home_team
                            else:
                                spread_favorite = abs(away_spread)
                                spread_fav_team = away_team

                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                over_under = float(outcome["point"])
                                break

            # Insert or update betting odds
            conn.execute(
                """
                INSERT OR REPLACE INTO betting_odds (
                    game_id, favorite_team, spread_favorite, over_under_line,
                    home_team_spread, away_team_spread, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    db_game_id,
                    spread_fav_team,
                    spread_favorite,
                    over_under,
                    home_spread,
                    away_spread,
                    "odds_api",
                ),
            )

            odds_inserted += 1
            logger.info(
                f"Processed odds for {away_team} @ {home_team}: spread={spread_favorite}, o/u={over_under}"
            )

        conn.commit()
        logger.info(f"Odds collection complete: {odds_inserted} records processed")

        # Check remaining API quota
        remaining_requests = response.headers.get("x-requests-remaining")
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


def collect_injury_data(
    seasons: List[int] = None, db_path: str = "data/nfl_dfs.db"
) -> None:
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

        if "injury_status" not in columns:
            logger.info("Adding injury_status column to players table...")
            cursor.execute(
                "ALTER TABLE players ADD COLUMN injury_status TEXT DEFAULT NULL"
            )
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
                        nfl_status = injury_row.get("report_status")
                        if not nfl_status or pd.isna(nfl_status):
                            continue

                        nfl_status = str(nfl_status).upper()

                        # Map NFL injury statuses to our standardized codes
                        status_mapping = {
                            "OUT": "OUT",
                            "DOUBTFUL": "D",
                            "QUESTIONABLE": "Q",
                            "PROBABLE": "P",
                            "NOTE": None,  # Skip these
                            "INJURED_RESERVE": "IR",
                            "PHYSICALLY_UNABLE_TO_PERFORM": "PUP",
                            "NON_FOOTBALL_INJURY": "NFI",
                            "SUSPENSION": "SUSP",
                            "RESERVE_COVID_19": "COV",
                            "PRACTICE_SQUAD_INJURED": "PS-INJ",
                        }

                        injury_status = status_mapping.get(nfl_status)
                        if injury_status is None:
                            if nfl_status == "NOTE":
                                continue  # Skip notes
                            # If status not in mapping, use original if it's short enough
                            injury_status = (
                                nfl_status[:10] if len(nfl_status) <= 10 else None
                            )

                        if not injury_status:
                            continue

                        # Get player identifiers
                        gsis_id = injury_row.get("gsis_id")
                        player_name = injury_row.get("full_name", "")
                        team_abbr = injury_row.get("team")

                        if not gsis_id:
                            continue

                        # Try to find player by GSIS ID first (most reliable)
                        player_record = conn.execute(
                            "SELECT id, player_name FROM players WHERE gsis_id = ?",
                            (gsis_id,),
                        ).fetchone()

                        # If not found by GSIS ID, try name and team
                        if not player_record and player_name and team_abbr:
                            # Get team ID
                            team_record = conn.execute(
                                "SELECT id FROM teams WHERE team_abbr = ?", (team_abbr,)
                            ).fetchone()

                            if team_record:
                                team_id = team_record[0]
                                # Try both player_name and display_name columns
                                player_record = conn.execute(
                                    """SELECT id, player_name FROM players
                                       WHERE (player_name LIKE ? OR display_name LIKE ?) AND team_id = ?""",
                                    (f"%{player_name}%", f"%{player_name}%", team_id),
                                ).fetchone()

                        if player_record:
                            player_id = player_record[0]

                            # Update injury status
                            conn.execute(
                                "UPDATE players SET injury_status = ? WHERE id = ?",
                                (injury_status, player_id),
                            )
                            season_updates += 1

                    except Exception as e:
                        logger.warning(f"Error processing injury record: {e}")
                        continue

                injury_progress.finish(
                    f"Completed season {season}: {season_updates} updates"
                )
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
    position: str, test_season: int = 2023, db_path: str = "data/nfl_dfs.db"
) -> Dict[str, float]:
    """Backtest model using exact production pipeline for accurate performance metrics."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score

    from models import ModelConfig, create_model

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
        logger.info(
            f"Backtesting {len(test_data)} {position} performances from {test_season}"
        )

        if not test_data:
            return {"error": "No test data found"}

        # Load trained model
        model_path = f"models/{position.lower()}_model.pth"
        if not Path(model_path).exists():
            return {"error": f"No trained model found: {model_path}"}

        # Get expected feature count from training data
        X_train, _, feature_names = get_training_data(position, [2022, 2023], db_path)
        expected_features = len(feature_names)

        # Use ensemble for QB, WR, TE (matches training and prediction configuration, RB neural-only, DST CatBoost-only)
        config = ModelConfig(position=position, features=feature_names)
        use_ensemble = position in ["QB", "WR", "TE"]
        model = create_model(position, config, use_ensemble=use_ensemble)

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
                    logger.debug(
                        f"Prediction failed for {player_name} in {game_id}: {e}"
                    )

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
        std_error = np.std(predictions - actuals)  # Consistency

        logger.info(
            f"Production backtesting complete: {len(predictions)} valid predictions"
        )

        return {
            "mae": mae,
            "r2": r2,
            "mean_error": mean_error,
            "std_error": std_error,
            "valid_predictions": len(predictions),
            "failed_predictions": failed_predictions,
            "success_rate": len(predictions) / len(test_data),
        }

    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def populate_dfs_scores_for_season(
    season: int, db_path: str = "data/nfl_dfs.db"
) -> None:
    """
    Populate DFS scores table with data from player_stats and dst_stats for a given season.

    Args:
        season: Season year to populate data for
        db_path: Path to SQLite database
    """
    conn = get_db_connection(db_path)

    try:
        # Query all player stats and game data for the season
        player_query = """
            SELECT
                ps.player_id,
                ps.game_id,
                ps.fantasy_points as dfs_points,
                p.position,
                p.team_id,
                g.season,
                g.week,
                CASE
                    WHEN g.home_team_id = p.team_id THEN g.away_team_id
                    ELSE g.home_team_id
                END as opponent_id
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            JOIN games g ON ps.game_id = g.id
            WHERE g.season = ? AND ps.fantasy_points > 0
        """

        player_data = pd.read_sql_query(player_query, conn, params=[season])

        if not player_data.empty:
            # Insert player DFS scores
            insert_dfs_scores(player_data, db_path)
            logger.info(f"Populated {len(player_data)} player DFS scores for {season}")

        # Query all DST stats for the season
        dst_query = """
            SELECT
                COALESCE(p.id, -1) as player_id,
                ds.game_id,
                ds.fantasy_points as dfs_points,
                'DST' as position,
                t.id as team_id,
                g.season,
                g.week,
                CASE
                    WHEN g.home_team_id = t.id THEN g.away_team_id
                    ELSE g.home_team_id
                END as opponent_id
            FROM dst_stats ds
            JOIN teams t ON ds.team_abbr = t.team_abbr
            JOIN games g ON ds.game_id = g.id
            LEFT JOIN players p ON p.team_id = t.id AND p.position = 'DST'
            WHERE g.season = ? AND ds.fantasy_points > 0
        """

        dst_data = pd.read_sql_query(dst_query, conn, params=[season])

        if not dst_data.empty:
            # For DST, create defense players if they don't exist
            for idx, row in dst_data.iterrows():
                if row["player_id"] == -1:  # No defense player found
                    team_abbr_query = "SELECT team_abbr FROM teams WHERE id = ?"
                    team_abbr = conn.execute(
                        team_abbr_query, [row["team_id"]]
                    ).fetchone()
                    if team_abbr:
                        defense_player_id = get_or_create_defense_player(
                            team_abbr[0], conn
                        )
                        dst_data.loc[idx, "player_id"] = defense_player_id

            # Remove any rows that still have -1 player_id
            dst_data = dst_data[dst_data["player_id"] != -1]

            if not dst_data.empty:
                insert_dfs_scores(dst_data, db_path)
                logger.info(f"Populated {len(dst_data)} DST DFS scores for {season}")

        # DFS scores populated successfully

    except Exception as e:
        logger.error(f"Failed to populate DFS scores for season {season}: {e}")
        raise
    finally:
        conn.close()


def insert_dfs_scores(df: pd.DataFrame, db_path: str = "data/nfl_dfs.db") -> None:
    """
    Insert or replace DFS scores into the dfs_scores table.

    Args:
        df: DataFrame containing DFS scores with columns:
            - player_id: INTEGER (foreign key to players table)
            - season: INTEGER
            - week: INTEGER
            - team_id: INTEGER (foreign key to teams table)
            - position: TEXT
            - opponent_id: INTEGER (foreign key to teams table)
            - game_id: TEXT (foreign key to games table, optional)
            - dfs_points: REAL
        db_path: Path to SQLite database
    """
    conn = get_db_connection(db_path)

    try:
        # Prepare data for insertion - handle NaN values
        insert_data = []
        for _, row in df.iterrows():
            try:
                # Skip rows with invalid data
                if (
                    pd.isna(row["player_id"])
                    or pd.isna(row["team_id"])
                    or pd.isna(row["opponent_id"])
                ):
                    continue
                if (
                    pd.isna(row["season"])
                    or pd.isna(row["week"])
                    or pd.isna(row["dfs_points"])
                ):
                    continue

                data_tuple = (
                    int(row["player_id"]),
                    int(row["season"]),
                    int(row["week"]),
                    int(row["team_id"]),
                    str(row["position"]),
                    int(row["opponent_id"]),
                    row.get("game_id"),  # Can be None
                    float(row["dfs_points"]),
                )
                insert_data.append(data_tuple)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue

        # Use INSERT OR REPLACE to handle uniqueness constraint
        insert_sql = """
            INSERT OR REPLACE INTO dfs_scores
            (player_id, season, week, team_id, position, opponent_id, game_id, dfs_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Insert in batches for efficiency
        batch_size = 1000
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i : i + batch_size]
            conn.executemany(insert_sql, batch)

        conn.commit()
        logger.info(f"Inserted/updated {len(insert_data)} DFS scores")

    except Exception as e:
        logger.error(f"Failed to insert DFS scores: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def calculate_vs_team_avg(
    player_id: int,
    opponent_id: int,
    current_season: int,
    current_week: int,
    conn: sqlite3.Connection,
) -> float:
    """
    Calculate on-the-fly vs team average for a specific player against a specific opponent.

    Args:
        player_id: Player ID
        opponent_id: Opponent team ID
        current_season: Current season (to exclude future games)
        current_week: Current week (to exclude future games)
        conn: Database connection

    Returns:
        Average DFS points vs this opponent, or 0.0 if no historical data
    """
    query = """
        SELECT dfs_points
        FROM dfs_scores
        WHERE player_id = ? AND opponent_id = ?
        AND ((season < ?) OR (season = ? AND week < ?))
    """

    historical_scores = conn.execute(
        query, [player_id, opponent_id, current_season, current_season, current_week]
    ).fetchall()

    if historical_scores:
        return float(np.mean([row[0] for row in historical_scores]))
    else:
        return 0.0
