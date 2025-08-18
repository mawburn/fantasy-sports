"""Defensive statistics aggregation from play-by-play data.

This module calculates team defensive performance metrics from NFL play-by-play data,
providing crucial opponent context for fantasy predictions.

Key Calculations:
- Position-specific fantasy points allowed (QB, RB, WR, TE)
- Yards and touchdowns allowed by type (pass vs rush)
- Situational defense (red zone, third down)
- Rolling averages for recent performance
- Defensive rankings across all metrics
"""

import logging

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Team, TeamDefensiveStats

logger = logging.getLogger(__name__)


class DefensiveStatsCollector:
    """Collects and aggregates team defensive statistics from play-by-play data."""

    def __init__(self, db_session: Session = None):
        """Initialize the defensive stats collector."""
        self.db = db_session or next(get_db())

        # DraftKings scoring system for calculating fantasy points allowed
        self.scoring = {
            "passing_td": 4,
            "passing_yard": 0.04,
            "interception": -1,
            "rushing_td": 6,
            "rushing_yard": 0.1,
            "receiving_td": 6,
            "receiving_yard": 0.1,
            "reception": 1,  # PPR scoring
            "fumble_lost": -1,
            "2pt_conversion": 2,
        }

    def collect_defensive_stats(self, seasons: list[int]) -> int:
        """Collect and aggregate defensive statistics for specified seasons.

        Args:
            seasons: List of seasons to process

        Returns:
            Number of defensive stat records created/updated
        """
        total_records = 0

        for season in seasons:
            logger.info(f"Processing defensive stats for {season} season")

            try:
                # Load play-by-play data
                pbp_df = nfl.import_pbp_data([season])

                # Process each week
                for week in range(1, 19):  # Regular season weeks 1-18
                    week_records = self._process_week(pbp_df, season, week)
                    total_records += week_records

            except Exception as e:
                logger.exception(f"Error processing {season} defensive stats: {e}")
                continue

        logger.info(f"Created/updated {total_records} defensive stat records")
        return total_records

    def _process_week(self, pbp_df: pd.DataFrame, season: int, week: int) -> int:
        """Process defensive stats for a specific week.

        Args:
            pbp_df: Play-by-play dataframe
            season: NFL season
            week: Week number

        Returns:
            Number of records created/updated
        """
        # Filter to specific week
        week_pbp = pbp_df[pbp_df["week"] == week].copy()

        if week_pbp.empty:
            return 0

        # Get unique teams that played defense this week
        teams = week_pbp["defteam"].unique()
        teams = [t for t in teams if pd.notna(t)]

        records_created = 0

        for team_abbr in teams:
            # Calculate defensive stats for this team
            stats = self._calculate_team_defensive_stats(week_pbp, team_abbr)

            # Add identifiers
            stats["season"] = season
            stats["week"] = week

            # Get team ID from database
            team = self.db.query(Team).filter(Team.team_abbr == team_abbr).first()
            if not team:
                logger.warning(f"Team {team_abbr} not found in database")
                continue

            stats["team_id"] = team.id

            # Calculate rolling averages if we have prior weeks
            if week > 4:
                stats.update(self._calculate_rolling_averages(team.id, season, week))

            # Save to database
            if self._save_defensive_stats(stats):
                records_created += 1

        # Calculate rankings for this week
        self._calculate_weekly_rankings(season, week)

        return records_created

    def _calculate_team_defensive_stats(self, pbp_df: pd.DataFrame, team: str) -> dict:
        """Calculate defensive statistics for a team from play-by-play data.

        Args:
            pbp_df: Play-by-play data for the week
            team: Team abbreviation

        Returns:
            Dictionary of defensive statistics
        """
        # Filter to plays where this team was on defense
        def_plays = pbp_df[pbp_df["defteam"] == team].copy()

        stats = {}

        # Overall defensive metrics
        stats["plays_faced"] = len(def_plays)
        stats["total_yards_allowed"] = def_plays["yards_gained"].sum()
        stats["yards_per_play_allowed"] = (
            stats["total_yards_allowed"] / stats["plays_faced"] if stats["plays_faced"] > 0 else 0
        )

        # Passing defense
        pass_plays = def_plays[def_plays["pass_attempt"] == 1]
        stats["pass_attempts_faced"] = len(pass_plays)
        stats["completions_allowed"] = pass_plays["complete_pass"].sum()
        stats["pass_yards_allowed"] = pass_plays["passing_yards"].sum()
        stats["pass_tds_allowed"] = pass_plays["pass_touchdown"].sum()
        stats["completion_pct_allowed"] = (
            stats["completions_allowed"] / stats["pass_attempts_faced"] * 100
            if stats["pass_attempts_faced"] > 0
            else 0
        )
        stats["yards_per_pass_allowed"] = (
            stats["pass_yards_allowed"] / stats["pass_attempts_faced"]
            if stats["pass_attempts_faced"] > 0
            else 0
        )
        stats["sacks"] = def_plays["sack"].sum()
        stats["interceptions"] = def_plays["interception"].sum()

        # Rushing defense
        rush_plays = def_plays[def_plays["rush_attempt"] == 1]
        stats["rush_attempts_faced"] = len(rush_plays)
        stats["rush_yards_allowed"] = rush_plays["rushing_yards"].sum()
        stats["rush_tds_allowed"] = rush_plays["rush_touchdown"].sum()
        stats["yards_per_rush_allowed"] = (
            stats["rush_yards_allowed"] / stats["rush_attempts_faced"]
            if stats["rush_attempts_faced"] > 0
            else 0
        )
        stats["stuffed_runs"] = len(rush_plays[rush_plays["yards_gained"] <= 0])
        stats["explosive_runs_allowed"] = len(rush_plays[rush_plays["yards_gained"] >= 15])

        # Turnovers and scoring
        stats["fumbles_recovered"] = def_plays["fumble_lost"].sum()
        stats["defensive_tds"] = def_plays["td_team"].apply(lambda x: 1 if x == team else 0).sum()
        stats["safeties"] = def_plays["safety"].sum()

        # Calculate total points allowed (approximate from TDs)
        stats["total_points_allowed"] = (
            stats["pass_tds_allowed"] + stats["rush_tds_allowed"]
        ) * 7 + stats[
            "safeties"
        ] * 2  # Assume PATs

        # Situational defense
        third_downs = def_plays[def_plays["down"] == 3]
        stats["third_down_attempts_faced"] = len(third_downs)
        stats["third_down_conversions_allowed"] = third_downs["first_down"].sum()
        stats["third_down_pct_allowed"] = (
            stats["third_down_conversions_allowed"] / stats["third_down_attempts_faced"] * 100
            if stats["third_down_attempts_faced"] > 0
            else 0
        )

        # Red zone defense (inside 20 yard line)
        red_zone = def_plays[def_plays["yardline_100"] <= 20]
        stats["red_zone_attempts_faced"] = len(red_zone)
        stats["red_zone_tds_allowed"] = red_zone["touchdown"].sum()
        stats["red_zone_pct_allowed"] = (
            stats["red_zone_tds_allowed"] / stats["red_zone_attempts_faced"] * 100
            if stats["red_zone_attempts_faced"] > 0
            else 0
        )

        # Calculate position-specific fantasy points allowed
        stats.update(self._calculate_fantasy_points_allowed(def_plays))

        return stats

    def _calculate_fantasy_points_allowed(self, def_plays: pd.DataFrame) -> dict:
        """Calculate fantasy points allowed by position.

        Args:
            def_plays: Plays where team was on defense

        Returns:
            Dictionary with fantasy points allowed by position
        """
        fantasy_allowed = {
            "qb_fantasy_points_allowed": 0,
            "rb_fantasy_points_allowed": 0,
            "wr_fantasy_points_allowed": 0,
            "te_fantasy_points_allowed": 0,
        }

        # Group by passer (for QB stats)
        if "passer_player_id" in def_plays.columns:
            qb_stats = (
                def_plays.groupby("passer_player_id")
                .agg(
                    {
                        "passing_yards": "sum",
                        "pass_touchdown": "sum",
                        "interception": "sum",
                        "rushing_yards": "sum",
                        "rush_touchdown": "sum",
                    }
                )
                .fillna(0)
            )

            # Calculate QB fantasy points
            for _, row in qb_stats.iterrows():
                fp = (
                    row["passing_yards"] * self.scoring["passing_yard"]
                    + row["pass_touchdown"] * self.scoring["passing_td"]
                    + row["interception"] * self.scoring["interception"]
                    + row["rushing_yards"] * self.scoring["rushing_yard"]
                    + row["rush_touchdown"] * self.scoring["rushing_td"]
                )
                fantasy_allowed["qb_fantasy_points_allowed"] += fp

        # Group by rusher (for RB stats)
        if "rusher_player_id" in def_plays.columns:
            rb_stats = (
                def_plays.groupby("rusher_player_id")
                .agg(
                    {
                        "rushing_yards": "sum",
                        "rush_touchdown": "sum",
                        "receiving_yards": "sum",
                        "pass_touchdown": "sum",
                        "complete_pass": "sum",  # Receptions
                    }
                )
                .fillna(0)
            )

            for _, row in rb_stats.iterrows():
                fp = (
                    row["rushing_yards"] * self.scoring["rushing_yard"]
                    + row["rush_touchdown"] * self.scoring["rushing_td"]
                    + row["receiving_yards"] * self.scoring["receiving_yard"]
                    + row["pass_touchdown"] * self.scoring["receiving_td"]
                    + row["complete_pass"] * self.scoring["reception"]
                )
                fantasy_allowed["rb_fantasy_points_allowed"] += fp

        # Group by receiver (for WR/TE stats)
        if "receiver_player_id" in def_plays.columns:
            rec_stats = (
                def_plays.groupby("receiver_player_id")
                .agg(
                    {
                        "receiving_yards": "sum",
                        "pass_touchdown": "sum",
                        "complete_pass": "sum",
                    }
                )
                .fillna(0)
            )

            # Note: This is simplified - would need player position data to split WR/TE
            total_rec_fp = 0
            for _, row in rec_stats.iterrows():
                fp = (
                    row["receiving_yards"] * self.scoring["receiving_yard"]
                    + row["pass_touchdown"] * self.scoring["receiving_td"]
                    + row["complete_pass"] * self.scoring["reception"]
                )
                total_rec_fp += fp

            # Split between WR and TE (approximate 70/30 split)
            fantasy_allowed["wr_fantasy_points_allowed"] = total_rec_fp * 0.7
            fantasy_allowed["te_fantasy_points_allowed"] = total_rec_fp * 0.3

        return fantasy_allowed

    def _calculate_rolling_averages(self, team_id: int, season: int, current_week: int) -> dict:
        """Calculate 4-week rolling averages for defensive metrics.

        Args:
            team_id: Team database ID
            season: Current season
            current_week: Current week number

        Returns:
            Dictionary with rolling average statistics
        """
        # Get last 4 weeks of data
        recent_stats = (
            self.db.query(TeamDefensiveStats)
            .filter(
                TeamDefensiveStats.team_id == team_id,
                TeamDefensiveStats.season == season,
                TeamDefensiveStats.week >= current_week - 4,
                TeamDefensiveStats.week < current_week,
            )
            .all()
        )

        if not recent_stats:
            return {}

        # Calculate averages
        rolling = {
            "rolling_yards_allowed": np.mean([s.total_yards_allowed for s in recent_stats]),
            "rolling_points_allowed": np.mean([s.total_points_allowed for s in recent_stats]),
            "rolling_qb_fantasy_allowed": np.mean(
                [s.qb_fantasy_points_allowed for s in recent_stats]
            ),
            "rolling_rb_fantasy_allowed": np.mean(
                [s.rb_fantasy_points_allowed for s in recent_stats]
            ),
            "rolling_wr_fantasy_allowed": np.mean(
                [s.wr_fantasy_points_allowed for s in recent_stats]
            ),
            "rolling_te_fantasy_allowed": np.mean(
                [s.te_fantasy_points_allowed for s in recent_stats]
            ),
        }

        return rolling

    def _calculate_weekly_rankings(self, season: int, week: int):
        """Calculate defensive rankings for all teams in a given week.

        Args:
            season: NFL season
            week: Week number
        """
        # Get all defensive stats for this week
        week_stats = (
            self.db.query(TeamDefensiveStats)
            .filter(TeamDefensiveStats.season == season, TeamDefensiveStats.week == week)
            .all()
        )

        if not week_stats:
            return

        # Create dataframe for easier ranking
        df = pd.DataFrame(
            [
                {
                    "id": s.id,
                    "total_yards_allowed": s.total_yards_allowed,
                    "total_points_allowed": s.total_points_allowed,
                    "pass_yards_allowed": s.pass_yards_allowed,
                    "rush_yards_allowed": s.rush_yards_allowed,
                    "qb_fantasy_allowed": s.qb_fantasy_points_allowed,
                    "rb_fantasy_allowed": s.rb_fantasy_points_allowed,
                    "wr_fantasy_allowed": s.wr_fantasy_points_allowed,
                    "te_fantasy_allowed": s.te_fantasy_points_allowed,
                }
                for s in week_stats
            ]
        )

        # Calculate rankings (1 = best defense, 32 = worst)
        df["overall_def_rank"] = df["total_yards_allowed"].rank()
        df["pass_def_rank"] = df["pass_yards_allowed"].rank()
        df["rush_def_rank"] = df["rush_yards_allowed"].rank()
        df["qb_def_rank"] = df["qb_fantasy_allowed"].rank()
        df["rb_def_rank"] = df["rb_fantasy_allowed"].rank()
        df["wr_def_rank"] = df["wr_fantasy_allowed"].rank()
        df["te_def_rank"] = df["te_fantasy_allowed"].rank()

        # Update database records with rankings
        for _, row in df.iterrows():
            stat = next(s for s in week_stats if s.id == row["id"])
            stat.overall_def_rank = int(row["overall_def_rank"])
            stat.pass_def_rank = int(row["pass_def_rank"])
            stat.rush_def_rank = int(row["rush_def_rank"])
            stat.qb_def_rank = int(row["qb_def_rank"])
            stat.rb_def_rank = int(row["rb_def_rank"])
            stat.wr_def_rank = int(row["wr_def_rank"])
            stat.te_def_rank = int(row["te_def_rank"])

        self.db.commit()

    def _save_defensive_stats(self, stats: dict) -> bool:
        """Save or update defensive statistics in the database.

        Args:
            stats: Dictionary of defensive statistics

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if record already exists
            existing = (
                self.db.query(TeamDefensiveStats)
                .filter(
                    TeamDefensiveStats.team_id == stats["team_id"],
                    TeamDefensiveStats.season == stats["season"],
                    TeamDefensiveStats.week == stats["week"],
                )
                .first()
            )

            if existing:
                # Update existing record
                for key, value in stats.items():
                    setattr(existing, key, value)
            else:
                # Create new record
                new_stat = TeamDefensiveStats(**stats)
                self.db.add(new_stat)

            self.db.commit()
            return True

        except Exception as e:
            logger.exception(f"Error saving defensive stats: {e}")
            self.db.rollback()
            return False
