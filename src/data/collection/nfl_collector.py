"""NFL data collection using nfl_data_py."""

import logging
from datetime import datetime

import nfl_data_py as nfl
import pandas as pd

from ...config.settings import settings
from ...database.connection import SessionLocal
from ...database.models import Game, PlayByPlay, Player, PlayerStats, Team

logger = logging.getLogger(__name__)


class NFLDataCollector:
    """Collects and stores NFL data using nfl_data_py."""

    def __init__(self):
        self.current_season = datetime.now().year
        if datetime.now().month < 9:  # Before September, use previous year
            self.current_season -= 1

    def collect_teams(self, seasons: list[int] | None = None) -> int:
        """Collect NFL team data and store in database."""
        # Team data doesn't vary by season, so seasons parameter is acknowledged but not used
        _ = seasons  # Acknowledge parameter to avoid unused warning
        logger.info("Collecting NFL team data...")

        try:
            # Get team data from nfl_data_py
            teams_df = nfl.import_team_desc()

            teams_added = 0
            session = SessionLocal()
            try:
                for _, row in teams_df.iterrows():
                    # Check if team already exists
                    existing_team = (
                        session.query(Team).filter_by(team_abbr=row["team_abbr"]).first()
                    )

                    if not existing_team:
                        team = Team(
                            team_abbr=row["team_abbr"],
                            team_name=row["team_name"],
                            conference=row["team_conf"],
                            division=row["team_division"],
                        )
                        session.add(team)
                        teams_added += 1
                    else:
                        # Update existing team info
                        existing_team.team_name = row["team_name"]
                        existing_team.conference = row["team_conf"]
                        existing_team.division = row["team_division"]
                        existing_team.updated_at = datetime.now()

                session.commit()
            finally:
                session.close()

            logger.info(f"Teams processed: {len(teams_df)}, new teams added: {teams_added}")
            return teams_added

        except Exception:
            logger.exception("Error collecting teams")
            raise

    def collect_players(self, seasons: list[int] | None = None) -> int:
        """Collect NFL player data and store in database."""
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting player data for seasons: {seasons}")

        try:
            players_added = 0

            for season in seasons:
                # Get roster data
                rosters_df = nfl.import_seasonal_rosters([season])

                session = SessionLocal()
            try:
                # Get team mapping for foreign keys
                teams = {team.team_abbr: team.id for team in session.query(Team).all()}

                for _, row in rosters_df.iterrows():
                    # Check if player already exists
                    existing_player = (
                        session.query(Player).filter_by(player_id=row["player_id"]).first()
                    )

                    team_id = teams.get(row.get("team"))

                    if not existing_player:
                        player = Player(
                            player_id=row["player_id"],
                            display_name=row.get("player_name", ""),
                            first_name=row.get("first_name", ""),
                            last_name=row.get("last_name", ""),
                            position=row.get("position", ""),
                            jersey_number=row.get("jersey_number"),
                            team_id=team_id,
                            height=row.get("height"),
                            weight=row.get("weight"),
                            age=row.get("age"),
                            years_exp=row.get("years_exp"),
                            college=row.get("college"),
                            rookie_year=row.get("rookie_year"),
                            status=row.get("status", "Active"),
                        )
                        session.add(player)
                        players_added += 1
                    else:
                        # Update player info
                        existing_player.display_name = row.get(
                            "player_name", existing_player.display_name
                        )
                        existing_player.team_id = team_id
                        existing_player.position = row.get("position", existing_player.position)
                        existing_player.jersey_number = row.get(
                            "jersey_number", existing_player.jersey_number
                        )
                        existing_player.status = row.get("status", existing_player.status)
                        existing_player.updated_at = datetime.now()

                session.commit()
            finally:
                session.close()

            logger.info(f"Players processed, new players added: {players_added}")
            return players_added

        except Exception:
            logger.exception("Error collecting players")
            raise

    def collect_schedules(self, seasons: list[int] | None = None) -> int:
        """Collect NFL schedule data and store in database."""
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting schedule data for seasons: {seasons}")

        try:
            games_added = 0

            for season in seasons:
                # Get schedule data
                schedule_df = nfl.import_schedules([season])

                session = SessionLocal()
            try:
                # Get team mapping
                teams = {team.team_abbr: team.id for team in session.query(Team).all()}

                for _, row in schedule_df.iterrows():
                    # Check if game already exists
                    existing_game = session.query(Game).filter_by(game_id=row["game_id"]).first()

                    home_team_id = teams.get(row["home_team"])
                    away_team_id = teams.get(row["away_team"])

                    if not existing_game and home_team_id and away_team_id:
                        game = Game(
                            game_id=row["game_id"],
                            season=season,
                            week=row["week"],
                            game_date=pd.to_datetime(row["gameday"]),
                            home_team_id=home_team_id,
                            away_team_id=away_team_id,
                            game_type=row["game_type"],
                            home_score=row.get("home_score"),
                            away_score=row.get("away_score"),
                            stadium=row.get("stadium"),
                            roof_type=row.get("roof"),
                            game_finished=not pd.isna(row.get("home_score")),
                        )
                        session.add(game)
                        games_added += 1
                    elif existing_game:
                        # Update game info (scores, etc.)
                        existing_game.home_score = row.get("home_score", existing_game.home_score)
                        existing_game.away_score = row.get("away_score", existing_game.away_score)
                        existing_game.game_finished = not pd.isna(row.get("home_score"))
                        existing_game.updated_at = datetime.now()

                session.commit()
            finally:
                session.close()

            logger.info(f"Games processed, new games added: {games_added}")
            return games_added

        except Exception:
            logger.exception("Error collecting schedules")
            raise

    def collect_player_stats(
        self, seasons: list[int] | None = None, weeks: list[int] | None = None
    ) -> int:
        """Collect NFL player statistics and store in database."""
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting player stats for seasons: {seasons}")

        try:
            stats_added = 0

            for season in seasons:
                # Get weekly stats
                weekly_df = nfl.import_weekly_data([season])

                # Filter by weeks if specified
                if weeks is not None:
                    weekly_df = weekly_df[weekly_df["week"].isin(weeks)]

                session = SessionLocal()
            try:
                # Get player and game mappings
                players = {p.player_id: p.id for p in session.query(Player).all()}
                games = {
                    g.game_id: g.id for g in session.query(Game).filter_by(season=season).all()
                }

                for _, row in weekly_df.iterrows():
                    player_id = players.get(row["player_id"])
                    game_id = games.get(row["game_id"])

                    if not player_id or not game_id:
                        continue

                    # Check if stats already exist
                    existing_stats = (
                        session.query(PlayerStats)
                        .filter_by(player_id=player_id, game_id=game_id)
                        .first()
                    )

                    if not existing_stats:
                        stats = PlayerStats(
                            player_id=player_id,
                            game_id=game_id,
                            passing_yards=row.get("passing_yards", 0) or 0,
                            passing_tds=row.get("passing_tds", 0) or 0,
                            passing_interceptions=row.get("interceptions", 0) or 0,
                            passing_attempts=row.get("attempts", 0) or 0,
                            passing_completions=row.get("completions", 0) or 0,
                            rushing_yards=row.get("rushing_yards", 0) or 0,
                            rushing_tds=row.get("rushing_tds", 0) or 0,
                            rushing_attempts=row.get("carries", 0) or 0,
                            receiving_yards=row.get("receiving_yards", 0) or 0,
                            receiving_tds=row.get("receiving_tds", 0) or 0,
                            receptions=row.get("receptions", 0) or 0,
                            targets=row.get("targets", 0) or 0,
                            fumbles_lost=row.get("fumbles_lost", 0) or 0,
                            two_point_conversions=row.get("two_point_conversions", 0) or 0,
                            fantasy_points=row.get("fantasy_points", 0) or 0,
                            fantasy_points_ppr=row.get("fantasy_points_ppr", 0) or 0,
                        )
                        session.add(stats)
                        stats_added += 1
                    else:
                        # Update existing stats
                        for field in [
                            "passing_yards",
                            "passing_tds",
                            "passing_interceptions",
                            "passing_attempts",
                            "passing_completions",
                            "rushing_yards",
                            "rushing_tds",
                            "rushing_attempts",
                            "receiving_yards",
                            "receiving_tds",
                            "receptions",
                            "targets",
                            "fumbles_lost",
                            "two_point_conversions",
                            "fantasy_points",
                            "fantasy_points_ppr",
                        ]:
                            if field in ["attempts", "completions", "carries"]:
                                # Map field names
                                db_field = (
                                    f"passing_{field}"
                                    if field in ["attempts", "completions"]
                                    else f"rushing_{field}"
                                )
                                setattr(existing_stats, db_field, row.get(field, 0) or 0)
                            else:
                                setattr(existing_stats, field, row.get(field, 0) or 0)
                        existing_stats.updated_at = datetime.now()

                session.commit()
            finally:
                session.close()

            logger.info(f"Player stats processed, new stats added: {stats_added}")
            return stats_added

        except Exception:
            logger.exception("Error collecting player stats")
            raise

    def collect_play_by_play(
        self, seasons: list[int] | None = None, weeks: list[int] | None = None
    ) -> int:
        """Collect NFL play-by-play data and store in database."""
        if seasons is None:
            seasons = [self.current_season]

        logger.info(f"Collecting play-by-play data for seasons: {seasons}")

        try:
            plays_added = 0

            for season in seasons:
                # Get play-by-play data
                pbp_df = nfl.import_pbp_data([season])

                # Filter by weeks if specified
                if weeks is not None:
                    pbp_df = pbp_df[pbp_df["week"].isin(weeks)]

                session = SessionLocal()
                try:
                    # Get game mapping
                    games = {
                        g.game_id: g.id for g in session.query(Game).filter_by(season=season).all()
                    }

                    # Process in batches for memory efficiency
                    batch_size = 1000
                    for i in range(0, len(pbp_df), batch_size):
                        batch = pbp_df.iloc[i : i + batch_size]

                        for _, row in batch.iterrows():
                            game_id = games.get(row["game_id"])

                            if not game_id:
                                continue

                            # Check if play already exists
                            existing_play = (
                                session.query(PlayByPlay).filter_by(play_id=row["play_id"]).first()
                            )

                            if not existing_play:
                                play = PlayByPlay(
                                    play_id=row["play_id"],
                                    game_id=game_id,
                                    season=season,
                                    week=row.get("week"),
                                    posteam=row.get("posteam"),
                                    defteam=row.get("defteam"),
                                    quarter=row.get("qtr"),
                                    time=row.get("time"),
                                    down=row.get("down"),
                                    ydstogo=row.get("ydstogo"),
                                    yardline_100=row.get("yardline_100"),
                                    play_type=row.get("play_type"),
                                    desc=row.get("desc", "")[:500],  # Truncate to fit column
                                    yards_gained=row.get("yards_gained"),
                                    first_down=bool(row.get("first_down", 0)),
                                    touchdown=bool(row.get("touchdown", 0)),
                                    interception=bool(row.get("interception", 0)),
                                    fumble=bool(row.get("fumble", 0)),
                                    sack=bool(row.get("sack", 0)),
                                    penalty=bool(row.get("penalty", 0)),
                                    epa=row.get("epa"),
                                    wp=row.get("wp"),
                                    wpa=row.get("wpa"),
                                    home_score=row.get("total_home_score"),
                                    away_score=row.get("total_away_score"),
                                )
                                session.add(play)
                                plays_added += 1

                        # Commit in batches to avoid memory issues
                        session.commit()
                        logger.debug(
                            f"Processed batch {i // batch_size + 1}, plays added: {plays_added}"
                        )

                finally:
                    session.close()

            logger.info(f"Play-by-play data processed, new plays added: {plays_added}")
            return plays_added

        except Exception:
            logger.exception("Error collecting play-by-play data")
            raise

    def collect_all_data(self, seasons: list[int] | None = None) -> dict[str, int]:
        """Collect all NFL data (teams, players, schedules, stats)."""
        if seasons is None:
            seasons = list(
                range(
                    self.current_season - settings.nfl_seasons_to_load + 1, self.current_season + 1
                )
            )

        logger.info(f"Starting full data collection for seasons: {seasons}")

        results = {}

        try:
            # Collect in order of dependencies
            results["teams"] = self.collect_teams()
            results["players"] = self.collect_players(seasons)
            results["games"] = self.collect_schedules(seasons)
            results["stats"] = self.collect_player_stats(seasons)
            results["play_by_play"] = self.collect_play_by_play(seasons)

            logger.info(f"Data collection complete: {results}")
            return results

        except Exception:
            logger.exception("Error in full data collection")
            raise
