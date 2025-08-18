"""Collect and calculate fantasy points for team defenses.

This module aggregates defensive statistics at the team level and calculates
fantasy points based on standard DFS scoring rules for team defenses.

DFS Defense Scoring (DraftKings Standard):
- Sack: 1 point
- Interception: 2 points
- Fumble Recovery: 2 points
- Defensive/Special Teams TD: 6 points
- Safety: 2 points
- Blocked Kick: 2 points
- Kickoff/Punt Return TD: 6 points
- Points Allowed Scoring:
  - 0 points: 10 fantasy points
  - 1-6 points: 7 fantasy points
  - 7-13 points: 4 fantasy points
  - 14-20 points: 1 fantasy point
  - 21-27 points: 0 fantasy points
  - 28-34 points: -1 fantasy point
  - 35+ points: -4 fantasy points
"""

import logging

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import get_db

logger = logging.getLogger(__name__)


class DefenseFantasyCollector:
    """Collect and calculate fantasy points for team defenses."""

    def __init__(self, db_session: Session | None = None):
        """Initialize the defense fantasy collector.

        Args:
            db_session: Optional database session (creates new one if None)
        """
        self.db = db_session or next(get_db())

    def collect_defense_fantasy_data(self, season: int, week: int | None = None) -> pd.DataFrame:
        """Collect defense/special teams fantasy data for a season.

        This aggregates team-level defensive statistics and calculates
        fantasy points based on DFS scoring rules.

        Args:
            season: NFL season year
            week: Optional specific week (None for all weeks)

        Returns:
            DataFrame with defense fantasy data
        """
        logger.info(f"Collecting defense fantasy data for {season} week {week or 'all'}")

        try:
            # Import play-by-play data for defensive stats
            pbp_df = nfl.import_pbp_data([season])

            if week:
                pbp_df = pbp_df[pbp_df["week"] == week]

            # Filter to regular season and playoffs
            pbp_df = pbp_df[pbp_df["season_type"].isin(["REG", "POST"])]

            # Initialize results list
            defense_stats = []

            # Get unique games
            games = pbp_df[["game_id", "week", "defteam", "posteam"]].drop_duplicates()

            for game_id in pbp_df["game_id"].unique():
                game_plays = pbp_df[pbp_df["game_id"] == game_id]

                # Get teams involved
                teams_in_game = game_plays["defteam"].dropna().unique()

                for team in teams_in_game:
                    # Calculate defensive stats when this team is on defense
                    def_plays = game_plays[game_plays["defteam"] == team]
                    opp_plays = game_plays[game_plays["posteam"] == team]

                    # Basic defensive stats
                    sacks = def_plays["sack"].sum()
                    interceptions = def_plays["interception"].sum()
                    fumbles_recovered = def_plays[
                        (def_plays["fumble_lost"] == 1)
                        & (def_plays["fumble_recovery_1_team"] == team)
                    ].shape[0]

                    # Defensive TDs (pick-6, fumble return, etc.)
                    def_tds = def_plays[
                        (def_plays["touchdown"] == 1)
                        & (def_plays["td_team"] == team)
                        & (def_plays["defteam"] == team)
                    ].shape[0]

                    # Special teams TDs (kickoff/punt returns)
                    special_tds = game_plays[
                        (game_plays["touchdown"] == 1)
                        & (game_plays["td_team"] == team)
                        & (game_plays["play_type"].isin(["kickoff", "punt"]))
                    ].shape[0]

                    # Safeties
                    safeties = def_plays["safety"].sum()

                    # Blocked kicks
                    blocked_kicks = def_plays[
                        def_plays["play_type"].isin(["field_goal", "extra_point", "punt"])
                        & (def_plays["blocked_player_id"].notna())
                    ].shape[0]

                    # Points allowed (when team is on defense)
                    points_allowed = (
                        opp_plays[
                            (opp_plays["touchdown"] == 1) & (opp_plays["td_team"] != team)
                        ].shape[0]
                        * 7
                    )  # Simplified - assumes all TDs are 7 points

                    # Add field goals allowed
                    points_allowed += (
                        opp_plays[(opp_plays["field_goal_result"] == "made")].shape[0] * 3
                    )

                    # Calculate fantasy points
                    fantasy_points = self._calculate_defense_fantasy_points(
                        sacks=sacks,
                        interceptions=interceptions,
                        fumbles_recovered=fumbles_recovered,
                        def_tds=def_tds,
                        special_tds=special_tds,
                        safeties=safeties,
                        blocked_kicks=blocked_kicks,
                        points_allowed=points_allowed,
                    )

                    # Get game info
                    game_info = game_plays[["week", "game_date"]].iloc[0]

                    defense_stats.append(
                        {
                            "game_id": game_id,
                            "team": team,
                            "season": season,
                            "week": game_info["week"],
                            "game_date": game_info["game_date"],
                            "sacks": sacks,
                            "interceptions": interceptions,
                            "fumbles_recovered": fumbles_recovered,
                            "def_tds": def_tds,
                            "special_tds": special_tds,
                            "safeties": safeties,
                            "blocked_kicks": blocked_kicks,
                            "points_allowed": points_allowed,
                            "fantasy_points": fantasy_points,
                            "fantasy_points_ppr": fantasy_points,  # Same for DEF
                        }
                    )

            df = pd.DataFrame(defense_stats)

            if not df.empty:
                logger.info(f"Collected {len(df)} defense fantasy records")

            return df

        except Exception as e:
            logger.exception(f"Error collecting defense fantasy data: {e}")
            return pd.DataFrame()

    def _calculate_defense_fantasy_points(
        self,
        sacks: int,
        interceptions: int,
        fumbles_recovered: int,
        def_tds: int,
        special_tds: int,
        safeties: int,
        blocked_kicks: int,
        points_allowed: int,
    ) -> float:
        """Calculate fantasy points for a defense.

        Uses DraftKings standard scoring.

        Args:
            sacks: Number of sacks
            interceptions: Number of interceptions
            fumbles_recovered: Number of fumble recoveries
            def_tds: Defensive touchdowns
            special_tds: Special teams touchdowns
            safeties: Number of safeties
            blocked_kicks: Number of blocked kicks
            points_allowed: Total points allowed

        Returns:
            Total fantasy points
        """
        points = 0.0

        # Basic defensive stats
        points += sacks * 1.0
        points += interceptions * 2.0
        points += fumbles_recovered * 2.0
        points += def_tds * 6.0
        points += special_tds * 6.0
        points += safeties * 2.0
        points += blocked_kicks * 2.0

        # Points allowed bonus/penalty
        if points_allowed == 0:
            points += 10.0
        elif points_allowed <= 6:
            points += 7.0
        elif points_allowed <= 13:
            points += 4.0
        elif points_allowed <= 20:
            points += 1.0
        elif points_allowed <= 27:
            points += 0.0
        elif points_allowed <= 34:
            points -= 1.0
        else:
            points -= 4.0

        return points

    def save_to_database(self, df: pd.DataFrame) -> int:
        """Save defense fantasy data to database.

        Creates team defense "players" with position='DEF' for each team.

        Args:
            df: DataFrame with defense fantasy data

        Returns:
            Number of records saved
        """
        if df.empty:
            return 0

        saved_count = 0

        try:
            # First, ensure we have DEF "players" for each team
            teams = df["team"].unique()

            for team_abbr in teams:
                # Check if DEF player exists for this team
                result = self.db.execute(
                    text(
                        """
                        SELECT id FROM players
                        WHERE team_id = (SELECT id FROM teams WHERE team_abbr = :team_abbr)
                        AND position = 'DEF'
                    """
                    ),
                    {"team_abbr": team_abbr},
                ).fetchone()

                if not result:
                    # Create DEF player for this team
                    team_result = self.db.execute(
                        text("SELECT id, team_name FROM teams WHERE team_abbr = :team_abbr"),
                        {"team_abbr": team_abbr},
                    ).fetchone()

                    if team_result:
                        team_id, team_name = team_result
                        # Generate unique player_id for DEF
                        player_id_str = f"DEF-{team_abbr}"

                        self.db.execute(
                            text(
                                """
                                INSERT INTO players (
                                    player_id, display_name, position, team_id,
                                    jersey_number, status, created_at, updated_at
                                ) VALUES (
                                    :player_id, :display_name, 'DEF', :team_id,
                                    0, 'ACT', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                                )
                            """
                            ),
                            {
                                "player_id": player_id_str,
                                "display_name": f"{team_name} Defense",
                                "team_id": team_id,
                            },
                        )
                        logger.info(f"Created DEF player for {team_name}")

            self.db.commit()

            # Now save the fantasy stats
            for _, row in df.iterrows():
                try:
                    # Get player_id for this team's DEF
                    player_result = self.db.execute(
                        text(
                            """
                            SELECT p.id FROM players p
                            JOIN teams t ON p.team_id = t.id
                            WHERE t.team_abbr = :team_abbr
                            AND p.position = 'DEF'
                        """
                        ),
                        {"team_abbr": row["team"]},
                    ).fetchone()

                    if not player_result:
                        logger.warning(f"No DEF player found for team {row['team']}")
                        continue

                    player_id = player_result[0]

                    # Get game_id
                    game_result = self.db.execute(
                        text(
                            """
                            SELECT id FROM games
                            WHERE season = :season
                            AND week = :week
                            AND (
                                home_team_id = (SELECT id FROM teams WHERE team_abbr = :team_abbr)
                                OR away_team_id = (SELECT id FROM teams WHERE team_abbr = :team_abbr)
                            )
                        """
                        ),
                        {"season": row["season"], "week": row["week"], "team_abbr": row["team"]},
                    ).fetchone()

                    if not game_result:
                        logger.warning(f"No game found for {row['team']} in week {row['week']}")
                        continue

                    game_id = game_result[0]

                    # Check if stats already exist
                    existing = self.db.execute(
                        text(
                            """
                            SELECT id FROM player_stats
                            WHERE player_id = :player_id AND game_id = :game_id
                        """
                        ),
                        {"player_id": player_id, "game_id": game_id},
                    ).fetchone()

                    if existing:
                        # Update existing record
                        self.db.execute(
                            text(
                                """
                                UPDATE player_stats
                                SET fantasy_points = :fantasy_points,
                                    fantasy_points_ppr = :fantasy_points_ppr,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE player_id = :player_id AND game_id = :game_id
                            """
                            ),
                            {
                                "player_id": player_id,
                                "game_id": game_id,
                                "fantasy_points": row["fantasy_points"],
                                "fantasy_points_ppr": row["fantasy_points_ppr"],
                            },
                        )
                    else:
                        # Insert new record
                        self.db.execute(
                            text(
                                """
                                INSERT INTO player_stats (
                                    player_id, game_id, fantasy_points, fantasy_points_ppr,
                                    created_at, updated_at
                                ) VALUES (
                                    :player_id, :game_id, :fantasy_points, :fantasy_points_ppr,
                                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                                )
                            """
                            ),
                            {
                                "player_id": player_id,
                                "game_id": game_id,
                                "fantasy_points": row["fantasy_points"],
                                "fantasy_points_ppr": row["fantasy_points_ppr"],
                            },
                        )

                    saved_count += 1

                except Exception as e:
                    logger.exception(f"Error saving defense stats for {row['team']}: {e}")
                    continue

            self.db.commit()
            logger.info(f"Saved {saved_count} defense fantasy records to database")

        except Exception as e:
            logger.exception(f"Error saving defense data to database: {e}")
            self.db.rollback()

        return saved_count


def main():
    """Run defense fantasy collection for multiple seasons."""
    collector = DefenseFantasyCollector()

    seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    for season in seasons:
        logger.info(f"Processing {season} season defense data")

        # Collect defense fantasy data
        df = collector.collect_defense_fantasy_data(season)

        if not df.empty:
            # Save to database
            saved = collector.save_to_database(df)
            logger.info(f"Saved {saved} defense records for {season}")
        else:
            logger.warning(f"No defense data found for {season}")

    logger.info("Defense fantasy collection complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
