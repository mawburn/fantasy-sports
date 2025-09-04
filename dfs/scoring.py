"""Fantasy sports scoring calculations.

This module contains functions for calculating DraftKings fantasy points
for different positions according to official DraftKings NFL Classic rules.
"""

from typing import Dict
import pandas as pd


def calculate_dk_fantasy_points(player_data: pd.Series) -> float:
    """Calculate DraftKings fantasy points according to NFL Classic rules.

    Scoring based on official DraftKings rules:
    - Passing: 0.04 pts/yard, 4 pts/TD, -1 pt/INT, +3 bonus at 300+ yards
    - Rushing: 0.1 pts/yard, 6 pts/TD, +3 bonus at 100+ yards
    - Receiving: 0.1 pts/yard, 6 pts/TD, 1 pt/reception, +3 bonus at 100+ yards
    - 2-pt conversions: 2 pts (pass, run, or catch)
    - Fumbles lost: -1 pt
    - Special teams TD: 6 pts
    """
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
    """Calculate DraftKings DST fantasy points according to NFL Classic rules.

    DraftKings DST Scoring:
    - Sack: 1 point
    - Interception: 2 points
    - Fumble Recovery: 2 points
    - Any Defensive/Special Teams TD: 6 points
    - Safety: 2 points
    - Blocked Kick: 2 points
    - Points Allowed tiers:
        - 0: 10 points
        - 1-6: 7 points
        - 7-13: 4 points
        - 14-20: 1 point
        - 21-27: 0 points
        - 28-34: -1 point
        - 35+: -4 points
    """
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
        points -= 1
    else:
        points -= 4

    # Defensive stats
    points += stats.get("sacks", 0) * 1
    points += stats.get("interceptions", 0) * 2
    points += stats.get("fumbles_recovered", 0) * 2
    points += stats.get("safeties", 0) * 2
    points += stats.get("blocked_kicks", 0) * 2

    # Defensive TDs (all types count as 6)
    points += stats.get("defensive_tds", 0) * 6
    points += stats.get("special_teams_tds", 0) * 6
    points += stats.get("return_tds", 0) * 6

    return round(points, 2)
