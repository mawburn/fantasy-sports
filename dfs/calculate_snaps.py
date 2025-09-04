#!/usr/bin/env python3
"""Calculate snap counts and usage metrics from NFL data.

Uses nfl_data_py's import_snap_counts() to get accurate snap data
and calculates additional usage metrics from play-by-play data.
"""

import logging
import sqlite3
from typing import Dict, List, Tuple

import nfl_data_py as nfl
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_snap_counts_from_pbp(season: int, db_path: str = "data/nfl_dfs.db") -> None:
    """Calculate snap counts using nfl_data_py's snap count data.

    This uses the dedicated snap count function for accurate data.
    """
    logger.info(f"Collecting accurate snap counts for {season} season using nfl.import_snap_counts()")

    try:
        # Load actual snap count data from nfl_data_py
        logger.info(f"Loading snap count data for {season}")
        snap_data = nfl.import_snap_counts([season])

        if snap_data.empty:
            logger.warning(f"No snap count data available for {season}")
            return

        logger.info(f"Processing {len(snap_data)} player-week snap records")

        # Load play-by-play data for additional usage metrics
        pbp = nfl.import_pbp_data([season])

        if not pbp.empty:
            # Calculate additional usage metrics from play-by-play
            usage_metrics = calculate_usage_metrics(pbp)
        else:
            usage_metrics = pd.DataFrame()
            logger.warning("No play-by-play data for usage metrics")

        # Update database with snap counts and usage metrics
        update_database_with_snaps(snap_data, usage_metrics, season, db_path)

    except Exception as e:
        logger.error(f"Error calculating snap counts: {e}")




def calculate_usage_metrics(plays: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional usage metrics like target share, rush share, etc."""
    metrics = []

    for game_id, game_plays in plays.groupby('game_id'):
        for team in game_plays['posteam'].unique():
            if pd.isna(team):
                continue

            team_plays = game_plays[game_plays['posteam'] == team]

            # Calculate target shares
            pass_plays = team_plays[team_plays['play_type'] == 'pass']
            total_targets = len(pass_plays[pass_plays['receiver_player_id'].notna()])

            if total_targets > 0:
                target_counts = pass_plays['receiver_player_id'].value_counts()
                for player_id, targets in target_counts.items():
                    if not pd.isna(player_id):
                        metrics.append({
                            'game_id': game_id,
                            'player_id': player_id,
                            'team': team,
                            'targets': targets,
                            'target_share': targets / total_targets,
                            'metric_type': 'target_share'
                        })

            # Calculate rush attempt shares
            run_plays = team_plays[team_plays['play_type'] == 'run']
            total_rushes = len(run_plays[run_plays['rusher_player_id'].notna()])

            if total_rushes > 0:
                rush_counts = run_plays['rusher_player_id'].value_counts()
                for player_id, rushes in rush_counts.items():
                    if not pd.isna(player_id):
                        metrics.append({
                            'game_id': game_id,
                            'player_id': player_id,
                            'team': team,
                            'rush_attempts': rushes,
                            'rush_attempt_share': rushes / total_rushes,
                            'metric_type': 'rush_share'
                        })

            # Calculate red zone usage
            rz_plays = team_plays[team_plays['yardline_100'] <= 20]

            # Red zone targets
            rz_pass = rz_plays[rz_plays['play_type'] == 'pass']
            rz_target_counts = rz_pass['receiver_player_id'].value_counts()

            for player_id, rz_targets in rz_target_counts.items():
                if not pd.isna(player_id):
                    metrics.append({
                        'game_id': game_id,
                        'player_id': player_id,
                        'team': team,
                        'red_zone_targets': rz_targets,
                        'metric_type': 'rz_targets'
                    })

            # Red zone rushes
            rz_run = rz_plays[rz_plays['play_type'] == 'run']
            rz_rush_counts = rz_run['rusher_player_id'].value_counts()

            for player_id, rz_rushes in rz_rush_counts.items():
                if not pd.isna(player_id):
                    metrics.append({
                        'game_id': game_id,
                        'player_id': player_id,
                        'team': team,
                        'red_zone_rushes': rz_rushes,
                        'metric_type': 'rz_rushes'
                    })

    return pd.DataFrame(metrics)


def update_database_with_snaps(
    snap_data: pd.DataFrame,
    usage_metrics: pd.DataFrame,
    season: int,
    db_path: str
) -> None:
    """Update the database with snap counts and usage metrics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout - fail fast

    try:
        # Update snap counts from nfl_data_py snap data
        if not snap_data.empty:
            logger.info(f"Updating snap counts for {len(snap_data)} player-weeks")

            update_count = 0
            batch_count = 0
            for _, row in snap_data.iterrows():
                # Get player_id and game_id from our database
                player_name = row.get('player', row.get('player_name', ''))
                team = row.get('team', '')
                week = row.get('week', 0)
                position = row.get('position', '')

                # Get offense snap count and percentage
                offense_snaps = row.get('offense_snaps', 0)
                offense_pct = row.get('offense_pct', 0)

                # Also check for defense and special teams snaps
                defense_snaps = row.get('defense_snaps', 0)
                defense_pct = row.get('defense_pct', 0)
                st_snaps = row.get('st_snaps', 0)
                st_pct = row.get('st_pct', 0)

                # Use offense snaps for offensive players, defense for DST
                # Note: nfl_data_py returns percentages as decimals (0.0 to 1.0)
                # We need to convert to percentage (0 to 100) for storage
                if position in ['QB', 'RB', 'WR', 'TE', 'FB']:
                    snap_count = offense_snaps if pd.notna(offense_snaps) else 0
                    snap_pct = (offense_pct * 100) if pd.notna(offense_pct) else 0
                elif position in ['DST', 'DEF']:
                    snap_count = defense_snaps if pd.notna(defense_snaps) else 0
                    snap_pct = (defense_pct * 100) if pd.notna(defense_pct) else 0
                else:
                    # For other positions, use max of offense/defense
                    snap_count = max(
                        offense_snaps if pd.notna(offense_snaps) else 0,
                        defense_snaps if pd.notna(defense_snaps) else 0
                    )
                    snap_pct = max(
                        (offense_pct * 100) if pd.notna(offense_pct) else 0,
                        (defense_pct * 100) if pd.notna(defense_pct) else 0
                    )

                # Find the player and game in our database
                cursor.execute("""
                    SELECT p.id, g.id
                    FROM players p
                    JOIN teams t ON t.team_abbr = ?
                    JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id)
                    WHERE p.player_name = ?
                    AND g.season = ?
                    AND g.week = ?
                """, (team, player_name, season, week))

                result = cursor.fetchone()
                if result:
                    player_id, game_id = result
                    cursor.execute("""
                        UPDATE player_stats
                        SET snap_count = ?, snap_percentage = ?
                        WHERE player_id = ?
                        AND game_id = ?
                    """, (int(snap_count), float(snap_pct), player_id, game_id))

        # Update usage metrics
        if not usage_metrics.empty:
            # Update target shares
            target_shares = usage_metrics[usage_metrics['metric_type'] == 'target_share']
            for _, row in target_shares.iterrows():
                # First find the player in our database by matching gsis_id
                cursor.execute("""
                    SELECT id FROM players WHERE gsis_id = ?
                """, (row['player_id'],))
                player_result = cursor.fetchone()

                if player_result:
                    player_id = player_result[0]
                    cursor.execute("""
                        UPDATE player_stats
                        SET target_share = ?
                        WHERE player_id = ?
                        AND game_id = ?
                    """, (row['target_share'], player_id, row['game_id']))

            # Update rush shares
            rush_shares = usage_metrics[usage_metrics['metric_type'] == 'rush_share']
            for _, row in rush_shares.iterrows():
                cursor.execute("""
                    SELECT id FROM players WHERE gsis_id = ?
                """, (row['player_id'],))
                player_result = cursor.fetchone()

                if player_result:
                    player_id = player_result[0]
                    cursor.execute("""
                        UPDATE player_stats
                        SET rush_attempt_share = ?
                        WHERE player_id = ?
                        AND game_id = ?
                    """, (row['rush_attempt_share'], player_id, row['game_id']))

            # Update red zone targets
            rz_targets = usage_metrics[usage_metrics['metric_type'] == 'rz_targets']
            for _, row in rz_targets.iterrows():
                cursor.execute("""
                    SELECT id FROM players WHERE gsis_id = ?
                """, (row['player_id'],))
                player_result = cursor.fetchone()

                if player_result:
                    player_id = player_result[0]
                    cursor.execute("""
                        UPDATE player_stats
                        SET red_zone_targets = ?
                        WHERE player_id = ?
                        AND game_id = ?
                    """, (row['red_zone_targets'], player_id, row['game_id']))

            # Update red zone touches (rushes)
            rz_rushes = usage_metrics[usage_metrics['metric_type'] == 'rz_rushes']
            for _, row in rz_rushes.iterrows():
                cursor.execute("""
                    SELECT id FROM players WHERE gsis_id = ?
                """, (row['player_id'],))
                player_result = cursor.fetchone()

                if player_result:
                    player_id = player_result[0]
                    cursor.execute("""
                        UPDATE player_stats
                        SET red_zone_touches = COALESCE(red_zone_touches, 0) + ?
                        WHERE player_id = ?
                        AND game_id = ?
                    """, (row['red_zone_rushes'], player_id, row['game_id']))

        conn.commit()
        logger.info(f"Updated snap counts and usage metrics for {season}")

    except Exception as e:
        logger.error(f"Error updating database: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023

    logger.info(f"Calculating snap counts for {season} season")
    calculate_snap_counts_from_pbp(season)
    logger.info("Snap count calculation complete")
