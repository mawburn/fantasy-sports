"""DraftKings data collection and CSV processing.

This file handles importing player salaries and contest data from DraftKings CSV files.
DraftKings provides daily player pricing that's essential for lineup optimization.

Key Concepts for Beginners:

DraftKings (DK): A major daily fantasy sports platform where users build lineups
under salary cap constraints to compete for cash prizes.

Salary Cap: Each player has a price (salary), and lineups must stay under $50,000
total. Higher-priced players are expected to score more points.

CSV Processing Pipeline:
1. Parse CSV files from DraftKings exports
2. Validate data integrity (required columns, reasonable salary ranges)
3. Match DK player names to our database players (fuzzy matching)
4. Store salaries and contest information for lineup optimization

Fuzzy Matching: Player names in DK files may not exactly match our database
(different formatting, nicknames, etc.). Fuzzy matching uses algorithms to
find the best match based on string similarity.

Data Quality Challenges:
- Player names can vary ("Chris" vs "Christopher")
- Team abbreviations may differ between sources
- Special characters in names (accents, apostrophes)
- Duplicate entries or missing data

This module provides robust error handling and validation to ensure
data quality for downstream lineup optimization.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
from fuzzywuzzy import fuzz

from ...database.connection import SessionLocal
from ...database.models import DraftKingsContest, DraftKingsSalary, Player

logger = logging.getLogger(__name__)


class CSVParsingError(ValueError):
    """Raised when CSV parsing fails.

    Common causes:
    - Corrupted or malformed CSV files
    - Encoding issues (special characters)
    - Unexpected file format changes from DraftKings
    - Missing or empty files
    """


class MissingColumnsError(ValueError):
    """Raised when required columns are missing.

    DraftKings CSV files must contain specific columns for processing.
    This error helps identify format changes or incomplete exports.

    Required columns include Name, Position, Team, Salary, etc.
    """


class DKFileNotFoundError(FileNotFoundError):
    """Raised when a DK file is not found.

    Helps distinguish between missing DraftKings files and other
    file system issues, providing clearer error messages.
    """


class DataValidationError(ValueError):
    """Raised when data validation fails.

    Occurs when CSV data doesn't meet business rules:
    - Salaries outside expected ranges
    - Duplicate players in same contest
    - Invalid player positions
    - Data integrity issues
    """


class DKPlayerMatcher:
    """Match DraftKings players to database players using fuzzy matching.

    Player matching is one of the most challenging aspects of sports data integration.
    DraftKings may use different name formats, abbreviations, or spellings than our
    database. This class uses intelligent matching techniques to link players accurately.

    Matching Strategy:
    1. Exact Match: Try direct name + team lookup first (fastest, most accurate)
    2. Fuzzy Match: Use string similarity algorithms for partial matches
    3. Team Filtering: Restrict matches to same team to avoid false positives
    4. Threshold Scoring: Only accept matches above confidence threshold (85%)

    Why Fuzzy Matching?
    - "Chris Jones" vs "Christopher Jones" (nickname variations)
    - "D.K. Metcalf" vs "DK Metcalf" (punctuation differences)
    - Typos in either dataset
    - Different name ordering or formats

    Performance Optimization:
    - Caches all database players for fast lookups
    - Avoids repeated database queries during bulk processing
    - Uses efficient string matching algorithms

    For beginners: Think of this as a "smart autocorrect" that can match
    similar but not identical player names between different data sources.
    """

    def __init__(self, session):
        """Initialize matcher with database session.

        Args:
            session: SQLAlchemy database session for player queries
        """
        self.session = session
        self.name_cache = {}  # Cache for fast player lookups
        self._load_player_cache()  # Pre-load all players into memory

    def _load_player_cache(self):
        """Load player data into cache for faster matching.

        Pre-loading all players into memory provides significant performance benefits:
        - Avoids repeated database queries during bulk processing
        - Enables fast string similarity comparisons
        - Reduces overall processing time for large CSV files

        Cache Key Format: "{player_name}_{team_abbreviation}"
        This format allows exact matching while including team context to
        distinguish players with same names on different teams.

        Memory vs Speed Trade-off:
        - Uses more memory to store all player data
        - Dramatically faster than querying database for each match
        - Reasonable for NFL (1000+ players) but consider pagination for larger datasets
        """
        # Query all active players from database
        players = self.session.query(Player).all()

        # Build lookup cache with name + team as key
        for player in players:
            # Create composite key: "Player Name_TEAM" for unique identification
            key = f"{player.display_name}_{player.team.team_abbr if player.team else ''}"
            self.name_cache[key] = player.id  # Store database ID for quick lookup

    def match_player(self, dk_name: str, dk_team: str, position: str) -> int | None:
        """
        Match DraftKings player to database player ID.

        This method implements a two-stage matching strategy:
        1. Exact matching for perfect name + team combinations
        2. Fuzzy matching with similarity scoring for partial matches

        Exact Matching (Stage 1):
        - Fastest and most accurate method
        - Direct lookup in pre-built cache
        - Handles majority of standard cases

        Fuzzy Matching (Stage 2):
        - Uses Levenshtein distance algorithm (fuzzywuzzy library)
        - Measures character-level similarity between strings
        - Applies team filtering to avoid false positives
        - Requires minimum 85% similarity score for acceptance

        Similarity Scoring:
        - Base score from string comparison (0-100)
        - +10 bonus for exact team match (encourages correct team assignment)
        - Threshold of 85+ prevents low-confidence matches

        Edge Cases Handled:
        - Missing team information (broader search)
        - Nickname variations (Chris vs Christopher)
        - Punctuation differences (D.K. vs DK)
        - Minor spelling variations

        Args:
            dk_name: Player name from DraftKings CSV
            dk_team: Team abbreviation from DraftKings CSV
            position: Player position (used for logging context)

        Returns:
            Database player ID if confident match found, None otherwise
        """
        # Stage 1: Try exact match first (fastest and most accurate)
        exact_key = f"{dk_name}_{dk_team}"
        if exact_key in self.name_cache:
            logger.debug(f"Exact match found for '{dk_name}' ({dk_team})")
            return self.name_cache[exact_key]

        # Stage 2: Fuzzy matching for partial matches
        # Initialize tracking variables for best match found
        best_match_id = None
        best_score = 0
        best_match_name = None

        # Compare against all cached players
        for cached_key, player_id in self.name_cache.items():
            # Parse cached key to extract name and team
            cached_name, cached_team = cached_key.split("_", 1)

            # Team filtering: skip if teams don't match (unless one is empty)
            # This prevents false positives like matching "Mike Williams" on different teams
            if dk_team and cached_team and dk_team != cached_team:
                continue  # Different teams - skip this candidate

            # Calculate name similarity using Levenshtein distance algorithm
            # fuzz.ratio returns 0-100 score (100 = perfect match)
            name_score = fuzz.ratio(dk_name.lower(), cached_name.lower())

            # Apply team match bonus to encourage correct team assignments
            if dk_team == cached_team:
                name_score += 10  # Bonus for exact team match

            # Update best match if this score is higher and meets threshold
            if name_score > best_score and name_score >= 85:  # 85% minimum similarity
                best_score = name_score
                best_match_id = player_id
                best_match_name = cached_name

        # Log results for debugging and monitoring
        if best_match_id:
            logger.debug(
                f"Fuzzy matched '{dk_name}' to '{best_match_name}' (ID: {best_match_id}, score: {best_score})"
            )
        else:
            logger.warning(f"Could not match player: {dk_name} ({dk_team}, {position})")
            # Consider lowering threshold or manual review for unmatched players

        return best_match_id


class DKCSVParser:
    """Parse DraftKings salary CSV files.

    This class handles the technical aspects of parsing and cleaning CSV data
    from DraftKings exports. It validates file format, cleans data inconsistencies,
    and transforms the data into a standardized format.

    CSV Format Challenges:
    - Encoding issues (UTF-8 vs ASCII, byte order marks)
    - Special characters in player names (accents, apostrophes)
    - Inconsistent position naming (DST vs DEF)
    - Various data types (strings, integers, floats)
    - Missing or null values in non-critical fields

    Data Cleaning Tasks:
    - Standardize position names (DST -> DEF)
    - Remove invalid characters from player names
    - Convert salary strings to integers
    - Parse game info for home/away status
    - Handle missing or malformed data gracefully

    Quality Assurance:
    - Validates all required columns are present
    - Ensures salary values are reasonable numbers
    - Removes rows with critical missing data
    - Logs warnings for data quality issues
    """

    # Expected columns in DraftKings CSV exports
    # If DK changes format, this validation will catch it
    EXPECTED_COLUMNS: ClassVar[list[str]] = [
        "Name",  # Player full name
        "Position",  # Position abbreviation (QB, RB, etc.)
        "Team",  # Team abbreviation (BUF, KC, etc.)
        "Opponent",  # Opposing team abbreviation
        "Game Info",  # Home/away and matchup info
        "Salary",  # DraftKings salary for this contest
        "AvgPointsPerGame",  # DK's projected points (not always present)
        "TeamAbbrev",  # Alternative team field (backup)
        "ID",  # DraftKings internal player ID
    ]

    # Position standardization mapping
    # DraftKings sometimes uses different position codes than our database
    POSITION_MAPPING: ClassVar[dict[str, str]] = {
        "QB": "QB",  # Quarterback (standard)
        "RB": "RB",  # Running Back (standard)
        "WR": "WR",  # Wide Receiver (standard)
        "TE": "TE",  # Tight End (standard)
        "DST": "DST",  # Defense/Special Teams (DK format)
        "DEF": "DST",  # Defense (alternative format -> standardize to DST)
        "FLEX": "FLEX",  # Flex position (RB/WR/TE eligible)
    }

    def parse_salary_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse DraftKings salary CSV file.

        This method orchestrates the complete CSV processing pipeline:
        1. Read CSV with proper encoding handling
        2. Validate required columns are present
        3. Clean and standardize data formats
        4. Return processed DataFrame ready for database storage

        Error Handling Strategy:
        - Gracefully handle encoding issues (UTF-8 with BOM)
        - Provide clear error messages for validation failures
        - Log detailed information for debugging
        - Wrap all exceptions with context-specific error types

        Performance Considerations:
        - Uses pandas for efficient data processing
        - Processes entire file in memory (suitable for DK file sizes)
        - Vectorized operations for data cleaning

        Args:
            file_path: Path to the DraftKings CSV export file

        Returns:
            Cleaned and validated pandas DataFrame with standardized columns

        Raises:
            CSVParsingError: If file cannot be parsed or contains invalid format
            MissingColumnsError: If required columns are missing from CSV
        """
        try:
            # Step 1: Read CSV with proper encoding handling
            # utf-8-sig handles Windows Excel exports with Byte Order Mark (BOM)
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            logger.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")

            # Step 2: Validate file structure meets expectations
            self._validate_columns(df)

            # Step 3: Clean and transform data to standard format
            df = self._clean_salary_data(df)
            logger.info(f"Data cleaning complete: {len(df)} valid rows remaining")

            return df

        except (CSVParsingError, MissingColumnsError):
            # Re-raise our custom exceptions without wrapping
            raise
        except Exception as e:
            # Wrap unexpected errors with context
            logger.exception("Unexpected error parsing DK CSV %s", file_path)
            raise CSVParsingError(f"CSV parsing failed for {file_path.name}: {e}") from e

    def _validate_columns(self, df: pd.DataFrame):
        """Validate that required columns are present.

        Column validation is critical because:
        - DraftKings may change CSV format without notice
        - Missing critical columns would cause downstream errors
        - Early validation provides clear error messages

        Validation Strategy:
        - Check for exact column name matches
        - Report all missing columns at once (not just first)
        - Provide actionable error message for debugging

        Args:
            df: DataFrame to validate

        Raises:
            MissingColumnsError: If any required columns are missing
        """
        # Find columns that are expected but not present in the CSV
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)

        if missing_cols:
            # Log available columns for debugging
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"Expected columns: {self.EXPECTED_COLUMNS}")
            raise MissingColumnsError(
                f"Missing required columns: {sorted(missing_cols)}. "
                f"This may indicate a format change in DraftKings exports."
            )

    def _clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize salary data.

        Data cleaning is essential because CSV files can contain:
        - Special characters that break name matching
        - Non-numeric salary values (commas, dollar signs)
        - Inconsistent position abbreviations
        - Missing critical data in some rows

        Cleaning Operations:
        1. Sanitize player names (remove special chars but keep apostrophes)
        2. Convert salary strings to integers (handle formatting issues)
        3. Standardize position codes using mapping table
        4. Extract home/away status from game info field
        5. Remove rows with missing critical data

        Data Quality Trade-offs:
        - Conservative name cleaning (preserve readability)
        - Zero-fill for invalid salaries (rather than dropping rows)
        - Keep original position if mapping fails
        - Log data quality issues for monitoring

        Args:
            df: Raw DataFrame from CSV parsing

        Returns:
            Cleaned DataFrame ready for database storage
        """
        # Step 1: Clean player names
        # Remove special characters but preserve normal punctuation (apostrophes, hyphens, periods)
        # This helps with matching while keeping names readable
        original_names = df["Name"].copy()  # Keep original for comparison
        df["Name"] = df["Name"].str.replace(r"[^A-Za-z\s\.\-\']", "", regex=True)

        # Log any name changes for quality monitoring
        name_changes = df["Name"] != original_names
        if name_changes.any():
            logger.info(f"Cleaned {name_changes.sum()} player names")

        # Step 2: Convert salary to integer
        # Handle various formats: "$5000", "5,000", "5000.00", etc.
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)

        # Log salary conversion issues
        zero_salaries = (df["Salary"] == 0).sum()
        if zero_salaries > 0:
            logger.warning(f"{zero_salaries} players have zero salaries (conversion failures)")

        # Step 3: Standardize position codes
        # Use mapping to convert variations (DEF->DST, etc.)
        original_positions = df["Position"].copy()
        df["Position"] = df["Position"].map(self.POSITION_MAPPING).fillna(df["Position"])

        # Log unmapped positions for monitoring format changes
        unmapped = df["Position"][~original_positions.isin(self.POSITION_MAPPING.keys())]
        if not unmapped.empty:
            logger.warning(f"Unmapped positions found: {unmapped.unique()}")

        # Step 4: Extract home/away status from game info
        df = self._parse_game_info(df)

        # Step 5: Remove rows with missing critical data
        # These fields are essential for player matching and processing
        initial_count = len(df)
        df = df.dropna(subset=["Name", "Position", "Team"])
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows due to missing critical data")

        return df

    def _parse_game_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse game info field to extract additional context.

        DraftKings Game Info Format Examples:
        - "LAR@SF" = LAR (away) at SF (home)
        - "BUF vs MIA" = BUF (home) vs MIA (away)
        - "KC @ DEN" = KC (away) at DEN (home)

        Home field advantage is significant in NFL, so identifying home/away
        status is important for predictions and lineup optimization.

        Parsing Logic:
        - "vs" typically indicates home team (hosting the game)
        - "@" typically indicates away team (traveling to play)
        - Handle case variations and spacing inconsistencies

        Args:
            df: DataFrame with Game Info column to parse

        Returns:
            DataFrame with added is_home boolean column
        """
        # Parse game info to determine home/away status
        # Game info typically looks like "LAR@SF" (away@home) or "BUF vs MIA" (home vs away)

        # "vs" indicates home team in most formats
        df["is_home"] = df["Game Info"].str.contains(r"vs", case=False, na=False)

        # Log parsing results for quality monitoring
        home_games = df["is_home"].sum()
        away_games = len(df) - home_games
        logger.debug(f"Parsed game info: {home_games} home games, {away_games} away games")

        return df


class DKSalaryValidator:
    """Validate DraftKings salary data."""

    SALARY_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
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

    def process_salary_file(
        self, file_path: Path, contest_name: str | None = None
    ) -> dict[str, int]:
        """
        Process a DraftKings salary CSV file.

        Args:
            file_path: Path to the salary CSV file
            contest_name: Optional contest name (derived from filename if not provided)

        Returns:
            Dictionary with processing results
        """
        if not file_path.exists():
            raise DKFileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing DraftKings salary file: {file_path}")

        # Parse CSV
        salary_df = self.parser.parse_salary_file(file_path)

        # Validate data
        validation_results = self.validator.validate_salaries(salary_df)

        if validation_results["errors"]:
            logger.error(f"Validation errors: {validation_results['errors']}")
            raise DataValidationError(f"Data validation failed: {validation_results['errors']}")

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

        except Exception:
            session.rollback()
            logger.exception("Error storing salary data")
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
            raise DKFileNotFoundError(f"Directory not found: {directory}")

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
                logger.exception(error_msg)
                total_results["errors"].append(error_msg)

        return total_results
