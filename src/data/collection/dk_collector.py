"""DraftKings data collection and CSV processing."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz

from ...database.connection import SessionLocal
from ...database.models import DraftKingsContest, DraftKingsSalary, Player

logger = logging.getLogger(__name__)


class DKPlayerMatcher:
    """Match DraftKings players to database players using fuzzy matching."""

    def __init__(self, session):
        self.session = session
        self.name_cache = {}
        self._load_player_cache()

    def _load_player_cache(self):
        """Load player data into cache for faster matching."""
        players = self.session.query(Player).all()
        for player in players:
            key = f"{player.display_name}_{player.team.team_abbr if player.team else ''}"
            self.name_cache[key] = player.id

    def match_player(self, dk_name: str, dk_team: str, position: str) -> int | None:
        """
        Match DraftKings player to database player ID.

        Args:
            dk_name: Player name from DraftKings
            dk_team: Team abbreviation from DraftKings
            position: Player position

        Returns:
            Player ID if match found, None otherwise
        """
        # Try exact match first
        exact_key = f"{dk_name}_{dk_team}"
        if exact_key in self.name_cache:
            return self.name_cache[exact_key]

        # Try fuzzy matching
        best_match_id = None
        best_score = 0

        for cached_key, player_id in self.name_cache.items():
            cached_name, cached_team = cached_key.split("_", 1)

            # Skip if team doesn't match (unless empty)
            if dk_team and cached_team and dk_team != cached_team:
                continue

            # Calculate name similarity
            name_score = fuzz.ratio(dk_name.lower(), cached_name.lower())

            # Boost score for exact team match
            if dk_team == cached_team:
                name_score += 10

            if name_score > best_score and name_score >= 85:  # Threshold for match
                best_score = name_score
                best_match_id = player_id

        if best_match_id:
            logger.debug(
                f"Fuzzy matched '{dk_name}' to player ID {best_match_id} (score: {best_score})"
            )
        else:
            logger.warning(f"Could not match player: {dk_name} ({dk_team}, {position})")

        return best_match_id


class DKCSVParser:
    """Parse DraftKings salary CSV files."""

    EXPECTED_COLUMNS = [
        "Name",
        "Position",
        "Team",
        "Opponent",
        "Game Info",
        "Salary",
        "AvgPointsPerGame",
        "TeamAbbrev",
        "ID",
    ]

    POSITION_MAPPING = {
        "QB": "QB",
        "RB": "RB",
        "WR": "WR",
        "TE": "TE",
        "DST": "DST",
        "DEF": "DST",
        "FLEX": "FLEX",
    }

    def parse_salary_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse DraftKings salary CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Cleaned and validated DataFrame

        Raises:
            ValueError: If file format is invalid
        """
        try:
            # Read CSV with proper encoding
            df = pd.read_csv(file_path, encoding="utf-8-sig")

            # Validate columns
            self._validate_columns(df)

            # Clean and transform data
            df = self._clean_salary_data(df)

            return df

        except Exception as e:
            logger.error(f"Failed to parse DK CSV {file_path}: {e}")
            raise ValueError(f"CSV parsing failed: {e}")

    def _validate_columns(self, df: pd.DataFrame):
        """Validate that required columns are present."""
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize salary data."""
        # Remove special characters from names
        df["Name"] = df["Name"].str.replace(r"[^A-Za-z\s\.\-\']", "", regex=True)

        # Ensure salary is integer
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)

        # Map positions
        df["Position"] = df["Position"].map(self.POSITION_MAPPING).fillna(df["Position"])

        # Parse game info to extract home/away status
        df = self._parse_game_info(df)

        # Remove rows with invalid data
        df = df.dropna(subset=["Name", "Position", "Team"])

        return df

    def _parse_game_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse game info field to extract additional context."""
        # Game info typically looks like "LAR@SF" or "BUF vs MIA"
        df["is_home"] = df["Game Info"].str.contains(r"vs", case=False, na=False)

        return df


class DKSalaryValidator:
    """Validate DraftKings salary data."""

    SALARY_RANGES = {
        "QB": (4000, 9500),
        "RB": (3000, 10500),
        "WR": (3000, 10500),
        "TE": (2500, 8500),
        "DST": (2000, 5500),
    }

    def validate_salaries(self, salary_data: pd.DataFrame) -> dict[str, list[str]]:
        """
        Validate salary data integrity.

        Returns:
            Dictionary with 'errors' and 'warnings' keys
        """
        errors = []
        warnings = []

        # Check salary ranges
        for position, (min_sal, max_sal) in self.SALARY_RANGES.items():
            pos_data = salary_data[salary_data["Position"] == position]

            if pos_data.empty:
                continue

            out_of_range = pos_data[(pos_data["Salary"] < min_sal) | (pos_data["Salary"] > max_sal)]

            if not out_of_range.empty:
                warnings.append(f"Unusual salaries for {position}: {out_of_range['Name'].tolist()}")

        # Check for duplicates
        duplicates = salary_data[salary_data.duplicated(subset=["Name", "Team"])]
        if not duplicates.empty:
            errors.append(f"Duplicate players: {duplicates['Name'].tolist()}")

        # Verify salary cap feasibility (basic check)
        total_min_salary = salary_data.groupby("Position")["Salary"].min().sum()
        if total_min_salary > 50000:  # DK salary cap
            warnings.append("Minimum salaries exceed salary cap - check data")

        return {"errors": errors, "warnings": warnings}


class DraftKingsCollector:
    """Collect and process DraftKings data."""

    def __init__(self):
        self.parser = DKCSVParser()
        self.validator = DKSalaryValidator()

    def process_salary_file(self, file_path: Path, contest_name: str = None) -> dict[str, int]:
        """
        Process a DraftKings salary CSV file.

        Args:
            file_path: Path to the salary CSV file
            contest_name: Optional contest name (derived from filename if not provided)

        Returns:
            Dictionary with processing results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing DraftKings salary file: {file_path}")

        # Parse CSV
        salary_df = self.parser.parse_salary_file(file_path)

        # Validate data
        validation_results = self.validator.validate_salaries(salary_df)

        if validation_results["errors"]:
            logger.error(f"Validation errors: {validation_results['errors']}")
            raise ValueError(f"Data validation failed: {validation_results['errors']}")

        if validation_results["warnings"]:
            logger.warning(f"Validation warnings: {validation_results['warnings']}")

        # Store in database
        results = self._store_salary_data(salary_df, contest_name or file_path.stem)

        logger.info(f"DraftKings data processing complete: {results}")
        return results

    def _store_salary_data(self, salary_df: pd.DataFrame, contest_name: str) -> dict[str, int]:
        """Store salary data in database."""
        session = SessionLocal()
        results = {"contests": 0, "salaries": 0, "unmatched_players": 0}

        try:
            # Create or get contest
            contest = self._get_or_create_contest(session, contest_name, salary_df)
            if contest:
                results["contests"] = 1

            # Initialize player matcher
            matcher = DKPlayerMatcher(session)

            # Process each player salary
            for _, row in salary_df.iterrows():
                player_id = matcher.match_player(row["Name"], row.get("Team", ""), row["Position"])

                if not player_id:
                    results["unmatched_players"] += 1
                    continue

                # Check if salary already exists
                existing_salary = (
                    session.query(DraftKingsSalary)
                    .filter_by(player_id=player_id, contest_id=contest.id)
                    .first()
                )

                if not existing_salary:
                    salary = DraftKingsSalary(
                        player_id=player_id,
                        contest_id=contest.id,
                        salary=row["Salary"],
                        position=row["Position"],
                        dk_player_name=row["Name"],
                        dk_team_abbr=row.get("Team", ""),
                        game_info=row.get("Game Info", ""),
                    )
                    session.add(salary)
                    results["salaries"] += 1
                else:
                    # Update existing salary
                    existing_salary.salary = row["Salary"]
                    existing_salary.dk_player_name = row["Name"]
                    existing_salary.updated_at = datetime.now()

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing salary data: {e}")
            raise
        finally:
            session.close()

        return results

    def _get_or_create_contest(
        self, session, contest_name: str, salary_df: pd.DataFrame
    ) -> DraftKingsContest:
        """Get existing contest or create new one."""
        # Try to find existing contest
        existing_contest = (
            session.query(DraftKingsContest).filter_by(contest_name=contest_name).first()
        )

        if existing_contest:
            return existing_contest

        # Create new contest
        contest = DraftKingsContest(
            contest_id=f"dk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            contest_name=contest_name,
            contest_type="Classic",  # Default type
            entry_fee=0.0,  # Default values - can be updated later
            total_prizes=0.0,
            max_entries=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            slate_id=contest_name,
            is_live=True,
        )
        session.add(contest)
        session.flush()  # Get the ID

        return contest

    def bulk_process_files(self, directory: Path) -> dict[str, any]:
        """
        Process all CSV files in a directory.

        Args:
            directory: Directory containing DK salary CSV files

        Returns:
            Summary of processing results
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {directory}")
            return {"files_processed": 0, "total_contests": 0, "total_salaries": 0}

        logger.info(f"Found {len(csv_files)} CSV files to process")

        total_results = {
            "files_processed": 0,
            "total_contests": 0,
            "total_salaries": 0,
            "errors": [],
        }

        for csv_file in csv_files:
            try:
                results = self.process_salary_file(csv_file)
                total_results["files_processed"] += 1
                total_results["total_contests"] += results.get("contests", 0)
                total_results["total_salaries"] += results.get("salaries", 0)

                logger.info(f"Processed {csv_file.name}: {results}")

            except Exception as e:
                error_msg = f"Failed to process {csv_file.name}: {e}"
                logger.error(error_msg)
                total_results["errors"].append(error_msg)

        return total_results
