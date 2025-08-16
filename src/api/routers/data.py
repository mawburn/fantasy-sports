"""
Data access API endpoints for NFL DFS System.

This module defines all the REST API endpoints for accessing NFL data,
including players, teams, games, statistics, and DraftKings contest information.

Key FastAPI Concepts Used:
- APIRouter: Groups related endpoints together
- Depends(): Dependency injection for database sessions
- Query parameters: URL parameters for filtering/pagination
- Response models: Pydantic models that define API response structure
- HTTPException: Standard way to return HTTP error responses

Database Patterns:
- SQLAlchemy ORM for database queries
- Query building with method chaining (query.filter().limit())
- JOIN operations for cross-table queries
- Pagination with offset/limit
"""

# FastAPI imports for routing and request handling
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

# SQLAlchemy for database operations
from sqlalchemy.orm import Session

# Import Pydantic response schemas that define API response structure
from src.api.schemas import ContestResponse  # DraftKings contest information
from src.api.schemas import CSVUploadResponse  # File upload response
from src.api.schemas import GameResponse  # NFL game details
from src.api.schemas import PlayerResponse  # Player information
from src.api.schemas import PlayerStatsResponse  # Player performance statistics
from src.api.schemas import SalaryResponse  # DraftKings salary information
from src.api.schemas import TeamResponse  # NFL team information

# Database connection and ORM models
from src.database.connection import get_db
from src.database.models import DraftKingsContest, DraftKingsSalary, Game, Player, PlayerStats, Team

# Create router instance - this groups all endpoints under /api/data
router = APIRouter()


# ========== PLAYER ENDPOINTS ==========


@router.get(
    "/players", response_model=list[PlayerResponse]
)  # Return list of PlayerResponse objects
async def get_players(
    # Query parameters are extracted from URL query string (?position=QB&team=KC)
    position: str | None = Query(None, description="Filter by position (QB, RB, WR, TE, K, DEF)"),
    team: str | None = Query(None, description="Filter by team abbreviation"),
    status: str | None = Query(None, description="Filter by player status"),
    # Pagination parameters with validation (ge=greater/equal, le=less/equal)
    limit: int = Query(100, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    # Dependency injection - FastAPI automatically provides database session
    db: Session = Depends(get_db),
):
    """
    Get all players with optional filtering and pagination.

    This endpoint demonstrates several important patterns:
    - Query parameter validation and documentation
    - Optional filtering with SQLAlchemy
    - Database JOINs for related data
    - Pagination for large datasets

    Example URLs:
    - /api/data/players - Get all players (first 100)
    - /api/data/players?position=QB - Get all quarterbacks
    - /api/data/players?team=KC&limit=50 - Get 50 Kansas City players

    Args:
        position: NFL position code (QB, RB, WR, TE, K, DEF)
        team: Team abbreviation (KC, NE, DAL, etc.)
        status: Player status (Active, Inactive, etc.)
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        db: Database session (injected by FastAPI)

    Returns:
        List of PlayerResponse objects containing player information
    """
    # Start with base query - this is SQLAlchemy's query builder pattern
    query = db.query(Player)

    # Add filters conditionally - only if parameter was provided
    # This is more efficient than always applying filters
    if position:
        # Convert to uppercase for consistent matching
        query = query.filter(Player.position == position.upper())
    if team:
        # JOIN with Team table to filter by team abbreviation
        # This demonstrates relationship querying in SQLAlchemy
        query = query.join(Team).filter(Team.team_abbr == team.upper())
    if status:
        query = query.filter(Player.status == status)

    # Apply pagination and execute query
    # offset() skips records, limit() restricts count, all() fetches results
    players = query.offset(offset).limit(limit).all()

    # FastAPI automatically converts ORM objects to Pydantic response models
    return players


@router.get("/players/{player_id}", response_model=PlayerResponse)  # Path parameter in URL
async def get_player(
    # Path parameter - extracted from URL path like /players/123
    player_id: int,  # FastAPI automatically converts string to int and validates
    db: Session = Depends(get_db),
):
    """
    Get a specific player by their unique ID.

    This endpoint demonstrates:
    - Path parameters (variables in URL path)
    - Single record retrieval with error handling
    - HTTP 404 responses for missing data

    Args:
        player_id: The unique database ID of the player
        db: Database session (injected by FastAPI)

    Returns:
        PlayerResponse: Single player object

    Raises:
        HTTPException: 404 if player not found
    """
    # first() returns single result or None (vs all() which returns list)
    player = db.query(Player).filter(Player.id == player_id).first()

    # Always check for None when using first() - data might not exist
    if not player:
        # HTTPException is FastAPI's standard way to return HTTP error responses
        raise HTTPException(status_code=404, detail="Player not found")

    return player


@router.get("/players/search", response_model=list[PlayerResponse])
async def search_players(
    # ... (ellipsis) means parameter is required - will return 422 if missing
    # min_length validation ensures meaningful search terms
    name: str = Query(..., min_length=2, description="Search by player name"),
    db: Session = Depends(get_db),
):
    """
    Search for players by name using partial matching.

    This endpoint demonstrates:
    - Required query parameters with validation
    - Case-insensitive partial text searching with LIKE/ILIKE
    - Limited results to prevent performance issues

    Example: /api/data/players/search?name=maho -> finds "Patrick Mahomes"

    Args:
        name: Player name or partial name to search for (minimum 2 characters)
        db: Database session (injected by FastAPI)

    Returns:
        List of PlayerResponse objects matching the search term
    """
    # ilike() is case-insensitive LIKE - finds partial matches
    # %name% pattern matches text containing the search term anywhere
    players = (
        db.query(Player).filter(Player.display_name.ilike(f"%{name}%")).limit(50).all()
    )  # Limit results to prevent slow queries

    return players


# ========== TEAM ENDPOINTS ==========


@router.get("/teams", response_model=list[TeamResponse])
async def get_teams(
    conference: str | None = Query(None, description="Filter by conference (AFC, NFC)"),
    division: str | None = Query(None, description="Filter by division"),
    db: Session = Depends(get_db),
):
    """
    Get all NFL teams with optional filtering by conference or division.

    The NFL has 32 teams organized into:
    - 2 Conferences: AFC (American Football Conference) and NFC (National Football Conference)
    - 8 Divisions: AFC/NFC North, South, East, West

    Args:
        conference: Filter by AFC or NFC
        division: Filter by division name (North, South, East, West)
        db: Database session (injected by FastAPI)

    Returns:
        List of TeamResponse objects containing team information
    """
    query = db.query(Team)

    if conference:
        # Exact match for conference (AFC or NFC)
        query = query.filter(Team.conference == conference.upper())
    if division:
        # Partial match for division using ILIKE (case-insensitive LIKE)
        query = query.filter(Team.division.ilike(f"%{division}%"))

    # No pagination needed - only 32 NFL teams total
    teams = query.all()
    return teams


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a specific NFL team by its unique ID.

    Args:
        team_id: The unique database ID of the team
        db: Database session (injected by FastAPI)

    Returns:
        TeamResponse: Single team object

    Raises:
        HTTPException: 404 if team not found
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


# ========== GAME ENDPOINTS ==========


@router.get("/games", response_model=list[GameResponse])
async def get_games(
    season: int | None = Query(None, description="Filter by season year"),
    week: int | None = Query(None, description="Filter by week number"),
    team: str | None = Query(None, description="Filter by team (home or away)"),
    game_type: str | None = Query(None, description="Filter by game type (REG, POST, PRE)"),
    finished: bool | None = Query(None, description="Filter by game completion status"),
    limit: int = Query(100, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Get NFL games with comprehensive filtering options.

    NFL games are categorized by:
    - Season: Year (e.g., 2023, 2024)
    - Week: 1-18 for regular season, 19+ for playoffs
    - Game Type: REG (regular season), POST (playoffs), PRE (preseason)

    This demonstrates complex filtering with OR conditions for team matching.

    Args:
        season: NFL season year
        week: Week number within the season
        team: Team abbreviation - matches if team is home OR away
        game_type: REG, POST, or PRE
        finished: True for completed games, False for upcoming games
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session (injected by FastAPI)

    Returns:
        List of GameResponse objects ordered by date (newest first)
    """
    query = db.query(Game)

    # Apply filters conditionally
    if season:
        query = query.filter(Game.season == season)
    if week:
        query = query.filter(Game.week == week)
    if team:
        team_upper = team.upper()
        # Complex JOIN with OR condition - find games where team is home OR away
        # This demonstrates advanced SQLAlchemy query patterns
        query = query.join(Team, (Team.id == Game.home_team_id) | (Team.id == Game.away_team_id))
        query = query.filter(Team.team_abbr == team_upper)
    if game_type:
        query = query.filter(Game.game_type == game_type.upper())
    if finished is not None:
        # Boolean filtering - can specifically filter for completed/upcoming games
        query = query.filter(Game.game_finished == finished)

    # Order by date descending (newest first) and apply pagination
    games = query.order_by(Game.game_date.desc()).offset(offset).limit(limit).all()
    return games


@router.get("/games/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a specific NFL game by its unique ID.

    Args:
        game_id: The unique database ID of the game
        db: Database session (injected by FastAPI)

    Returns:
        GameResponse: Single game object with team and timing information

    Raises:
        HTTPException: 404 if game not found
    """
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


# ========== PLAYER STATISTICS ENDPOINTS ==========


@router.get("/stats", response_model=list[PlayerStatsResponse])
async def get_player_stats(
    player_id: int | None = Query(None, description="Filter by player ID"),
    game_id: int | None = Query(None, description="Filter by game ID"),
    season: int | None = Query(None, description="Filter by season"),
    week: int | None = Query(None, description="Filter by week"),
    position: str | None = Query(None, description="Filter by position"),
    min_fantasy_points: float | None = Query(None, description="Minimum fantasy points"),
    limit: int = Query(100, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Get player performance statistics with comprehensive filtering.

    Player statistics include all the raw NFL stats (yards, TDs, etc.) plus
    calculated fantasy points based on standard scoring systems.

    This endpoint demonstrates:
    - Multiple JOIN operations for related data
    - Numeric filtering (min_fantasy_points)
    - Complex ordering with NULL handling

    Args:
        player_id: Specific player to get stats for
        game_id: Specific game to get stats from
        season: NFL season year
        week: Week number within season
        position: Player position (QB, RB, WR, TE, etc.)
        min_fantasy_points: Only return performances above this threshold
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session (injected by FastAPI)

    Returns:
        List of PlayerStatsResponse objects ordered by fantasy points (highest first)
    """
    query = db.query(PlayerStats)

    # Apply filters - multiple JOINs are handled automatically by SQLAlchemy
    if player_id:
        query = query.filter(PlayerStats.player_id == player_id)
    if game_id:
        query = query.filter(PlayerStats.game_id == game_id)
    if season:
        # JOIN with Game table to filter by season
        query = query.join(Game).filter(Game.season == season)
    if week:
        # Multiple JOINs are automatically optimized by SQLAlchemy
        query = query.join(Game).filter(Game.week == week)
    if position:
        # JOIN with Player table to filter by position
        query = query.join(Player).filter(Player.position == position.upper())
    if min_fantasy_points:
        # Numeric comparison for fantasy points threshold
        query = query.filter(PlayerStats.fantasy_points >= min_fantasy_points)

    # Order by fantasy points descending, with NULL values at end
    # nullslast() ensures NULL fantasy_points appear at the end
    stats = (
        query.order_by(PlayerStats.fantasy_points.desc().nullslast())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return stats


@router.get("/stats/player/{player_id}", response_model=list[PlayerStatsResponse])
async def get_player_game_stats(
    player_id: int,
    season: int | None = Query(None, description="Filter by season"),
    limit: int = Query(20, ge=1, le=100, description="Limit number of results"),
    db: Session = Depends(get_db),
):
    """
    Get all game-by-game statistics for a specific player.

    This endpoint is useful for analyzing player performance trends,
    consistency, and recent form. Results are ordered by most recent games first.

    Args:
        player_id: The unique database ID of the player
        season: Optional season filter (defaults to all seasons)
        limit: Maximum number of games to return (default 20)
        db: Database session (injected by FastAPI)

    Returns:
        List of PlayerStatsResponse objects ordered by game date (newest first)

    Raises:
        HTTPException: 404 if no stats found for this player
    """
    query = db.query(PlayerStats).filter(PlayerStats.player_id == player_id)

    if season:
        query = query.join(Game).filter(Game.season == season)

    # Order by game date descending to show most recent games first
    stats = query.join(Game).order_by(Game.game_date.desc()).limit(limit).all()

    # Check if player has any stats - return 404 if not
    if not stats:
        raise HTTPException(status_code=404, detail="No stats found for this player")

    return stats


# ========== DRAFTKINGS CONTEST ENDPOINTS ==========


@router.get("/contests", response_model=list[ContestResponse])
async def get_contests(
    contest_type: str | None = Query(None, description="Filter by contest type"),
    is_live: bool | None = Query(None, description="Filter by live status"),
    min_prize_pool: float | None = Query(None, description="Minimum total prizes"),
    limit: int = Query(50, ge=1, le=200, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Get DraftKings fantasy contests with filtering options.

    DraftKings offers various contest types:
    - GPP (Guaranteed Prize Pool): Tournaments with large prize pools
    - Cash: 50/50s and Double-ups with ~50% win rate
    - Showdown: Single-game contests
    - Multiplier: Win 2x, 3x, etc. your entry fee

    Args:
        contest_type: Partial match on contest type (GPP, Cash, Showdown, etc.)
        is_live: True for currently running contests, False for upcoming
        min_prize_pool: Only show contests with at least this prize pool
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session (injected by FastAPI)

    Returns:
        List of ContestResponse objects ordered by start time (newest first)
    """
    query = db.query(DraftKingsContest)

    if contest_type:
        # ilike allows partial matching: 'gpp' matches 'GPP Tournament'
        query = query.filter(DraftKingsContest.contest_type.ilike(f"%{contest_type}%"))
    if is_live is not None:
        # Boolean filtering for live status
        query = query.filter(DraftKingsContest.is_live == is_live)
    if min_prize_pool:
        # Numeric filtering for minimum prize pool
        query = query.filter(DraftKingsContest.total_prizes >= min_prize_pool)

    # Order by start time descending (newest/upcoming first)
    contests = query.order_by(DraftKingsContest.start_time.desc()).offset(offset).limit(limit).all()
    return contests


@router.get("/contests/{contest_id}", response_model=ContestResponse)
async def get_contest(
    contest_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a specific DraftKings contest by its unique ID.

    Args:
        contest_id: The unique database ID of the contest
        db: Database session (injected by FastAPI)

    Returns:
        ContestResponse: Single contest object with full details

    Raises:
        HTTPException: 404 if contest not found
    """
    contest = db.query(DraftKingsContest).filter(DraftKingsContest.id == contest_id).first()
    if not contest:
        raise HTTPException(status_code=404, detail="Contest not found")
    return contest


# ========== DRAFTKINGS SALARY ENDPOINTS ==========


@router.get("/salaries", response_model=list[SalaryResponse])
async def get_salaries(
    contest_id: int | None = Query(None, description="Filter by contest ID"),
    player_id: int | None = Query(None, description="Filter by player ID"),
    position: str | None = Query(None, description="Filter by position"),
    min_salary: int | None = Query(None, description="Minimum salary"),
    max_salary: int | None = Query(None, description="Maximum salary"),
    limit: int = Query(100, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Get DraftKings salary information with comprehensive filtering.

    Player salaries change based on:
    - Recent performance (good games = higher salary)
    - Injury status (injured/questionable players get lower salaries)
    - Matchup difficulty (easier matchups = higher salaries)
    - Contest type (Showdown vs Classic have different salary structures)

    This is crucial data for lineup optimization since you must stay
    under the salary cap while maximizing projected points.

    Args:
        contest_id: Specific contest to get salaries from
        player_id: Specific player salary information
        position: Filter by position (QB, RB, WR, TE, DEF)
        min_salary: Minimum salary threshold (useful for finding expensive players)
        max_salary: Maximum salary threshold (useful for finding value plays)
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session (injected by FastAPI)

    Returns:
        List of SalaryResponse objects ordered by salary (highest first)
    """
    query = db.query(DraftKingsSalary)

    # Apply all filters conditionally
    if contest_id:
        query = query.filter(DraftKingsSalary.contest_id == contest_id)
    if player_id:
        query = query.filter(DraftKingsSalary.player_id == player_id)
    if position:
        query = query.filter(DraftKingsSalary.position == position.upper())
    if min_salary:
        # >= for minimum salary (find expensive players)
        query = query.filter(DraftKingsSalary.salary >= min_salary)
    if max_salary:
        # <= for maximum salary (find value plays)
        query = query.filter(DraftKingsSalary.salary <= max_salary)

    # Order by salary descending (most expensive first)
    salaries = query.order_by(DraftKingsSalary.salary.desc()).offset(offset).limit(limit).all()
    return salaries


@router.get("/salaries/contest/{contest_id}", response_model=list[SalaryResponse])
async def get_contest_salaries(
    contest_id: int,
    position: str | None = Query(None, description="Filter by position"),
    db: Session = Depends(get_db),
):
    """
    Get all player salaries for a specific DraftKings contest.

    This endpoint is essential for lineup building as it provides
    the complete salary structure for a contest. Each contest has
    different salaries based on the player pool and contest type.

    Args:
        contest_id: The unique database ID of the contest
        position: Optional position filter to see salaries for specific positions
        db: Database session (injected by FastAPI)

    Returns:
        List of SalaryResponse objects ordered by salary (highest first)

    Raises:
        HTTPException: 404 if no salaries found for this contest
    """
    query = db.query(DraftKingsSalary).filter(DraftKingsSalary.contest_id == contest_id)

    if position:
        query = query.filter(DraftKingsSalary.position == position.upper())

    # Order by salary descending - no pagination since we want all contest salaries
    salaries = query.order_by(DraftKingsSalary.salary.desc()).all()

    # Validate that contest exists and has salaries
    if not salaries:
        raise HTTPException(status_code=404, detail="No salaries found for this contest")

    return salaries


# ========== FILE UPLOAD ENDPOINTS ==========


@router.post("/upload/draftkings", response_model=CSVUploadResponse)
async def upload_draftkings_csv(
    file: UploadFile = File(..., description="DraftKings salary CSV file"),
    contest_name: str | None = Query(
        None, description="Custom contest name (defaults to filename)"
    ),
):
    """
    Upload and process a DraftKings salary CSV file.

    This endpoint provides a web interface for uploading DraftKings CSV files,
    complementing the existing CLI functionality. It handles:
    - File validation and format checking
    - CSV parsing and data extraction
    - Player matching and database storage
    - Comprehensive error reporting and feedback

    The endpoint maintains compatibility with the existing CLI workflow while
    providing a user-friendly web interface for data uploads.

    CSV Format Requirements:
    - Must contain required columns: Name, Position, Team, Salary, etc.
    - Should follow standard DraftKings export format
    - Encoding should be UTF-8 or UTF-8 with BOM

    Processing Pipeline:
    1. Validate file format and extension
    2. Read and parse CSV content using existing DKCSVParser
    3. Validate data integrity using DKSalaryValidator
    4. Match players to database records using fuzzy matching
    5. Store contest and salary data in database
    6. Return detailed processing results

    Args:
        file: Uploaded CSV file from DraftKings export
        contest_name: Optional custom contest name (derives from filename if not provided)

    Returns:
        CSVUploadResponse with processing results, counts, and any warnings/errors

    Raises:
        HTTPException: 400 for file validation errors, 500 for processing errors
    """
    import tempfile
    from pathlib import Path

    # Import the DraftKings collector and exception types
    from src.data.collection.dk_collector import (
        CSVParsingError,
        DataValidationError,
        DraftKingsCollector,
        MissingColumnsError,
    )

    # Step 1: Validate file before processing
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    # Check file size (reasonable limit for CSV files)
    # Read file content to check size and validate format
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Reset file pointer for subsequent reading
        await file.seek(0)

        # Basic size check (10MB limit for CSV files)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {e!s}") from e

    # Step 2: Process the CSV file using existing DraftKings collector
    try:
        # Create temporary file to save uploaded content
        # Using tempfile ensures proper cleanup and secure file handling
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as temp_file:
            # Write uploaded content to temporary file
            await file.seek(0)  # Ensure we're at the beginning
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Process the file using existing DraftKings collector
        collector = DraftKingsCollector()

        # Derive contest name from filename if not provided
        final_contest_name = contest_name or Path(file.filename).stem

        # Use the existing process_salary_file method
        results = collector.process_salary_file(temp_path, final_contest_name)

        # Clean up temporary file
        temp_path.unlink()

        # Step 3: Return success response with detailed results
        return CSVUploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            contests_created=results.get("contests", 0),
            salaries_processed=results.get("salaries", 0),
            unmatched_players=results.get("unmatched_players", 0),
            filename=file.filename,
            contest_name=final_contest_name,
            warnings=None,  # Validation warnings would be captured here
            errors=None,
        )

    except (CSVParsingError, MissingColumnsError) as e:
        # Handle file format and parsing errors
        # Clean up temp file if it exists
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()

        raise HTTPException(status_code=400, detail=f"CSV parsing error: {e!s}") from e

    except DataValidationError as e:
        # Handle data validation errors
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()

        raise HTTPException(status_code=400, detail=f"Data validation error: {e!s}") from e

    except Exception as e:
        # Handle unexpected errors
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()

        # Log the full error for debugging while returning safe message to user
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error processing {file.filename}")

        raise HTTPException(
            status_code=500, detail=f"Internal server error processing file: {e!s}"
        ) from e


# ========== STATISTICAL ANALYSIS ENDPOINTS ==========


@router.get("/stats/summary/{player_id}")
async def get_player_stats_summary(
    player_id: int,
    season: int | None = Query(None, description="Filter by season"),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive statistical summary for a player.

    This endpoint demonstrates advanced SQL aggregate functions using SQLAlchemy.
    It calculates various statistical measures that are useful for:
    - Player evaluation and comparison
    - Fantasy value assessment
    - Consistency analysis
    - Historical performance review

    Args:
        player_id: The unique database ID of the player
        season: Optional season filter (all seasons if not provided)
        db: Database session (injected by FastAPI)

    Returns:
        Dict with comprehensive player statistics including averages, totals, and ranges

    Raises:
        HTTPException: 404 if player not found or has no statistics
    """
    # Import func here to access SQL aggregate functions
    from sqlalchemy import func

    # Build complex aggregate query using SQLAlchemy's func module
    # This demonstrates how to use SQL functions like COUNT, AVG, MAX, MIN, SUM
    query = db.query(
        # Counting functions
        func.count(PlayerStats.id).label("games_played"),
        # Statistical measures for fantasy points
        func.avg(PlayerStats.fantasy_points).label("avg_fantasy_points"),
        func.max(PlayerStats.fantasy_points).label("max_fantasy_points"),
        func.min(PlayerStats.fantasy_points).label("min_fantasy_points"),
        # Cumulative statistics by category
        func.sum(PlayerStats.passing_yards).label("total_passing_yards"),
        func.sum(PlayerStats.rushing_yards).label("total_rushing_yards"),
        func.sum(PlayerStats.receiving_yards).label("total_receiving_yards"),
        func.sum(PlayerStats.passing_tds).label("total_passing_tds"),
        func.sum(PlayerStats.rushing_tds).label("total_rushing_tds"),
        func.sum(PlayerStats.receiving_tds).label("total_receiving_tds"),
    ).filter(PlayerStats.player_id == player_id)

    # Optional season filter using JOIN
    if season:
        query = query.join(Game).filter(Game.season == season)

    # Execute the aggregate query - returns single row with all calculated values
    result = query.first()

    # Validate that player has statistics
    if not result or result.games_played == 0:
        raise HTTPException(status_code=404, detail="No stats found for this player")

    # Get player basic information for context
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    # Return comprehensive summary combining player info with calculated statistics
    # Note the defensive programming with "or 0" to handle NULL values
    return {
        "player_id": player_id,
        "player_name": player.display_name,
        "position": player.position,
        # Game participation
        "games_played": result.games_played,
        # Fantasy point statistics (rounded for readability)
        "avg_fantasy_points": round(float(result.avg_fantasy_points or 0), 2),
        "max_fantasy_points": float(result.max_fantasy_points or 0),
        "min_fantasy_points": float(result.min_fantasy_points or 0),
        # Total yardage by category
        "total_passing_yards": result.total_passing_yards or 0,
        "total_rushing_yards": result.total_rushing_yards or 0,
        "total_receiving_yards": result.total_receiving_yards or 0,
        # Total touchdowns by category
        "total_passing_tds": result.total_passing_tds or 0,
        "total_rushing_tds": result.total_rushing_tds or 0,
        "total_receiving_tds": result.total_receiving_tds or 0,
    }
