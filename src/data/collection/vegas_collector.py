"""Vegas betting lines collection for game context.

This module collects betting lines and game totals that provide crucial context
for fantasy predictions. Vegas lines are excellent predictors of game flow and
scoring opportunities.

Data Sources:
- nfl_data_py: import_sc_lines() provides betting data
- Multiple sportsbooks for consensus lines
"""

import logging

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Game, Team, VegasLines

logger = logging.getLogger(__name__)


class VegasLinesCollector:
    """Collects Vegas betting lines and game totals."""

    def __init__(self, db_session: Session = None):
        """Initialize the Vegas lines collector."""
        self.db = db_session or next(get_db())

    def collect_vegas_lines(self, seasons: list[int]) -> int:
        """Collect Vegas betting lines for specified seasons.

        Args:
            seasons: List of seasons to collect

        Returns:
            Number of lines collected
        """
        total_lines = 0

        for season in seasons:
            logger.info(f"Collecting Vegas lines for {season} season")

            try:
                # Import schedules which contain Vegas lines
                schedules_df = nfl.import_schedules([season])

                if schedules_df.empty:
                    logger.warning(f"No schedules found for {season}")
                    continue

                # Filter to games with Vegas data and rename columns to match expected format
                lines_df = schedules_df[schedules_df["spread_line"].notna()].copy()

                # Ensure we have the expected column names
                if "home_team" not in lines_df.columns and "home" in lines_df.columns:
                    lines_df["home_team"] = lines_df["home"]
                if "away_team" not in lines_df.columns and "away" in lines_df.columns:
                    lines_df["away_team"] = lines_df["away"]

                # Process each game's lines
                lines_count = self._process_lines(lines_df, season)
                total_lines += lines_count

            except Exception as e:
                logger.exception(f"Error collecting Vegas lines for {season}: {e}")
                continue

        logger.info(f"Collected {total_lines} Vegas line records")
        return total_lines

    def _process_lines(self, lines_df: pd.DataFrame, season: int) -> int:
        """Process Vegas lines dataframe and save to database.

        Args:
            lines_df: DataFrame with betting lines
            season: NFL season

        Returns:
            Number of lines saved
        """
        lines_saved = 0

        # Process each row (game)
        for _, row in lines_df.iterrows():
            try:
                # Find matching game in database
                game = self._find_game(row, season)
                if not game:
                    continue

                # Extract line data
                line_data = self._extract_line_data(row, game.id)

                # Save to database
                if self._save_vegas_line(line_data):
                    lines_saved += 1

            except Exception as e:
                logger.exception(f"Error processing line: {e}")
                continue

        return lines_saved

    def _find_game(self, row: pd.Series, season: int) -> Game | None:
        """Find the game record matching the Vegas line.

        Args:
            row: Vegas line data row
            season: NFL season

        Returns:
            Game object if found, None otherwise
        """
        # Extract team abbreviations from the data
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        week = row.get("week")

        if not home_team or not away_team or pd.isna(week):
            return None

        # Get team IDs
        home = self.db.query(Team).filter(Team.team_abbr == home_team).first()
        away = self.db.query(Team).filter(Team.team_abbr == away_team).first()

        if not home or not away:
            logger.warning(f"Teams not found: {home_team} vs {away_team}")
            return None

        # Find the game
        game = (
            self.db.query(Game)
            .filter(
                Game.season == season,
                Game.week == int(week),
                Game.home_team_id == home.id,
                Game.away_team_id == away.id,
            )
            .first()
        )

        if not game:
            logger.warning(f"Game not found: Week {week} {away_team} @ {home_team}")

        return game

    def _extract_line_data(self, row: pd.Series, game_id: int) -> dict:
        """Extract Vegas line data from dataframe row.

        Args:
            row: Data row with Vegas lines
            game_id: Database game ID

        Returns:
            Dictionary with line data
        """
        data = {
            "game_id": game_id,
            "source": "consensus",  # Default to consensus lines
        }

        # Extract spread (negative = home favored)
        if "spread_line" in row:
            data["spread"] = float(row["spread_line"]) if not pd.isna(row["spread_line"]) else None

        # Extract total (over/under)
        if "total_line" in row:
            data["total"] = float(row["total_line"]) if not pd.isna(row["total_line"]) else None

        # Calculate implied team totals
        if data.get("spread") is not None and data.get("total") is not None:
            # Home team total = (Total - Spread) / 2
            # Away team total = (Total + Spread) / 2
            data["home_team_total"] = (data["total"] - data["spread"]) / 2
            data["away_team_total"] = (data["total"] + data["spread"]) / 2

        # Extract moneyline odds if available
        if "home_moneyline" in row:
            data["home_moneyline"] = (
                int(row["home_moneyline"]) if not pd.isna(row["home_moneyline"]) else None
            )
        if "away_moneyline" in row:
            data["away_moneyline"] = (
                int(row["away_moneyline"]) if not pd.isna(row["away_moneyline"]) else None
            )

        # Track opening lines if available
        if "spread_open" in row:
            data["opening_spread"] = (
                float(row["spread_open"]) if not pd.isna(row["spread_open"]) else None
            )
            if data.get("spread") is not None and data.get("opening_spread") is not None:
                data["spread_movement"] = data["spread"] - data["opening_spread"]

        if "total_open" in row:
            data["opening_total"] = (
                float(row["total_open"]) if not pd.isna(row["total_open"]) else None
            )
            if data.get("total") is not None and data.get("opening_total") is not None:
                data["total_movement"] = data["total"] - data["opening_total"]

        return data

    def _save_vegas_line(self, line_data: dict) -> bool:
        """Save Vegas line to database.

        Args:
            line_data: Dictionary with line information

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if line already exists
            existing = (
                self.db.query(VegasLines)
                .filter(
                    VegasLines.game_id == line_data["game_id"],
                    VegasLines.source == line_data["source"],
                )
                .first()
            )

            if existing:
                # Update existing line
                for key, value in line_data.items():
                    setattr(existing, key, value)
            else:
                # Create new line
                new_line = VegasLines(**line_data)
                self.db.add(new_line)

            self.db.commit()
            return True

        except Exception as e:
            logger.exception(f"Error saving Vegas line: {e}")
            self.db.rollback()
            return False

    def get_game_context(self, game_id: int) -> dict | None:
        """Get Vegas context for a specific game.

        Args:
            game_id: Database game ID

        Returns:
            Dictionary with Vegas context or None
        """
        line = self.db.query(VegasLines).filter(VegasLines.game_id == game_id).first()

        if not line:
            return None

        return {
            "spread": line.spread,
            "total": line.total,
            "home_team_total": line.home_team_total,
            "away_team_total": line.away_team_total,
            "is_high_total": line.total > 48 if line.total else False,  # High-scoring game
            "is_low_total": line.total < 42 if line.total else False,  # Low-scoring game
            "is_blowout": abs(line.spread) > 7 if line.spread else False,  # Large spread
        }
