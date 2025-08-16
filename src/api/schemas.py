"""
Pydantic schemas for API request/response models.

This module defines all the data structures used for API communication using
Pydantic, a powerful data validation library. Pydantic provides:

- Automatic data validation and type conversion
- JSON serialization/deserialization
- OpenAPI/Swagger documentation generation
- IDE support with type hints
- Runtime type checking

Key Pydantic Concepts:
- BaseModel: Base class for all data models
- ConfigDict: Configuration options for model behavior
- Type hints: Define expected data types with validation
- Optional fields: Use | None or Optional[] for nullable fields
- from_attributes: Allows creation from SQLAlchemy ORM objects

Schema Organization:
- Response schemas: Define API output structure
- Request schemas: Define API input validation
- Model configuration: Controls serialization behavior
"""

from datetime import date, datetime  # For date/time field types

# Pydantic imports for data validation and API schemas
from pydantic import BaseModel, ConfigDict

# ========== NFL DATA SCHEMAS ==========


class TeamResponse(BaseModel):
    """
    Schema for NFL team data in API responses.

    This schema defines the structure of team data returned by API endpoints.
    All fields are required unless marked with | None (optional).

    The from_attributes=True configuration allows Pydantic to create instances
    directly from SQLAlchemy ORM objects, automatically mapping attributes.
    """

    # Configure model behavior
    model_config = ConfigDict(from_attributes=True)  # Enable creation from SQLAlchemy ORM objects

    # Core identification
    id: int  # Database primary key
    team_abbr: str  # Short code (e.g., "KC", "NE", "DAL")
    team_name: str  # Full name (e.g., "Kansas City Chiefs")

    # NFL organization structure
    conference: str  # "AFC" or "NFC"
    division: str  # "North", "South", "East", "West"

    # Audit timestamps - automatically managed by database
    created_at: datetime  # When record was first created
    updated_at: datetime  # When record was last modified


class PlayerResponse(BaseModel):
    """
    Schema for NFL player data in API responses.

    Contains comprehensive player information including personal details,
    physical attributes, career information, and current status.

    Many fields are optional (| None) because player data from various
    sources may be incomplete or not applicable (e.g., some positions
    don't typically have certain stats).
    """

    model_config = ConfigDict(from_attributes=True)

    # Core identification
    id: int  # Database primary key
    player_id: str  # External NFL player ID
    display_name: str  # Full display name (e.g., "Patrick Mahomes")
    first_name: str | None = None  # First name only
    last_name: str | None = None  # Last name only

    # Playing information
    position: str  # Position code (QB, RB, WR, TE, K, DEF)
    jersey_number: int | None = None  # Jersey number (can change between seasons)
    team_id: int | None = None  # Foreign key to Team table

    # Physical attributes
    height: int | None = None  # Height in inches
    weight: int | None = None  # Weight in pounds
    age: int | None = None  # Current age
    birthdate: date | None = None  # Birth date for age calculations

    # Career information
    years_exp: int | None = None  # Years of NFL experience
    college: str | None = None  # College/university attended
    rookie_year: int | None = None  # First NFL season

    # Current status
    status: str  # "Active", "Inactive", "Injured Reserve", etc.

    # Audit timestamps
    created_at: datetime
    updated_at: datetime


class GameResponse(BaseModel):
    """
    Schema for NFL game data in API responses.

    Contains comprehensive game information including teams, timing,
    weather conditions, and final scores. This data is crucial for
    fantasy sports as weather and game conditions affect player performance.
    """

    model_config = ConfigDict(from_attributes=True)

    # Core identification
    id: int  # Database primary key
    game_id: str  # External NFL game identifier

    # Schedule information
    season: int  # NFL season year (e.g., 2024)
    week: int  # Week number (1-18 regular season, 19+ playoffs)
    game_date: datetime  # Date and time of game

    # Team matchup
    home_team_id: int  # Foreign key to home Team
    away_team_id: int  # Foreign key to away Team

    # Game classification
    game_type: str  # "REG" (regular season), "POST" (playoffs), "PRE" (preseason)

    # Final results (None if game not completed)
    home_score: int | None = None  # Home team final score
    away_score: int | None = None  # Away team final score

    # Weather conditions (important for fantasy performance)
    weather_temperature: int | None = None  # Temperature in Fahrenheit
    weather_wind_speed: int | None = None  # Wind speed in mph
    weather_description: str | None = None  # Text description ("Clear", "Rain", etc.)

    # Venue information
    stadium: str | None = None  # Stadium name
    roof_type: str | None = None  # "Dome", "Open", "Retractable"

    # Game status
    game_finished: bool  # True if game is completed, False if upcoming/in-progress

    # Audit timestamps
    created_at: datetime
    updated_at: datetime


class PlayerStatsResponse(BaseModel):
    """
    Schema for player performance statistics in API responses.

    Contains comprehensive game-by-game statistics for players across all
    major statistical categories. These raw stats are used to calculate
    fantasy points and evaluate player performance.

    Statistics are organized by category:
    - Passing: QB-focused stats
    - Rushing: Running stats for all positions
    - Receiving: Pass-catching stats
    - Miscellaneous: Fumbles, conversions
    - Fantasy: Calculated point totals
    """

    model_config = ConfigDict(from_attributes=True)

    # Core identification
    id: int  # Database primary key
    player_id: int  # Foreign key to Player table
    game_id: int  # Foreign key to Game table

    # Passing statistics (primarily for QBs)
    passing_yards: int  # Total passing yards
    passing_tds: int  # Passing touchdowns
    passing_interceptions: int  # Interceptions thrown
    passing_attempts: int  # Pass attempts
    passing_completions: int  # Completed passes

    # Rushing statistics (RBs, QBs, WRs, etc.)
    rushing_yards: int  # Total rushing yards
    rushing_tds: int  # Rushing touchdowns
    rushing_attempts: int  # Rushing attempts (carries)

    # Receiving statistics (WRs, TEs, RBs)
    receiving_yards: int  # Total receiving yards
    receiving_tds: int  # Receiving touchdowns
    receptions: int  # Catches made
    targets: int  # Times targeted by QB

    # Miscellaneous statistics
    fumbles_lost: int  # Fumbles lost (negative points)
    two_point_conversions: int  # 2-point conversion scores

    # Calculated fantasy points (may be None if not calculated)
    fantasy_points: float | None = None  # Standard fantasy scoring
    fantasy_points_ppr: float | None = None  # PPR (Point Per Reception) scoring

    # Audit timestamps
    created_at: datetime
    updated_at: datetime


# ========== DRAFTKINGS DATA SCHEMAS ==========


class ContestResponse(BaseModel):
    """
    Schema for DraftKings contest information in API responses.

    Represents daily fantasy sports contests with entry requirements,
    prize structures, and timing information. This data is essential for:
    - Contest selection and strategy
    - Entry fee budgeting
    - Prize pool analysis
    - Contest type optimization
    """

    model_config = ConfigDict(from_attributes=True)

    # Core identification
    id: int  # Database primary key
    contest_id: str  # DraftKings contest identifier
    contest_name: str  # Display name (e.g., "NFL $100K Touchdown")

    # Contest classification and structure
    contest_type: str  # "GPP", "50/50", "Double-Up", "Multiplier", etc.
    entry_fee: float  # Cost to enter contest ($)
    total_prizes: float  # Total prize pool ($)

    # Entry management
    max_entries: int  # Maximum number of entries allowed
    current_entries: int  # Current number of entries (for tracking fill rate)

    # Timing
    start_time: datetime  # When contest starts (lineup lock)
    end_time: datetime  # When contest ends (all games finished)

    # Organization
    slate_id: str | None = None  # Associated game slate identifier
    sport: str  # "NFL", "NBA", etc.

    # Contest characteristics
    is_live: bool  # True if contest is currently running
    is_guaranteed: bool  # True if prize pool is guaranteed regardless of entries

    # Audit timestamps
    created_at: datetime
    updated_at: datetime


class SalaryResponse(BaseModel):
    """
    Schema for DraftKings player salary information in API responses.

    Player salaries are central to daily fantasy sports as they create
    the constraint optimization problem: maximize projected points while
    staying under the salary cap.

    Salaries change based on:
    - Recent player performance
    - Injury status and availability
    - Matchup difficulty
    - Public perception and ownership
    """

    model_config = ConfigDict(from_attributes=True)

    # Core identification
    id: int  # Database primary key
    player_id: int  # Foreign key to Player table
    contest_id: int  # Foreign key to Contest table

    # Pricing information
    salary: int  # Player cost in salary cap dollars

    # Position information
    position: str  # Basic position (QB, RB, WR, TE, DEF)
    roster_position: str | None = None  # Specific roster slot (FLEX, UTIL, etc.)

    # DraftKings-specific data (may differ from our normalized data)
    dk_player_name: str  # Player name as displayed on DraftKings
    dk_team_abbr: str | None = None  # Team abbreviation from DraftKings
    game_info: str | None = None  # Game context (e.g., "@KC 1:00PM")

    # Audit timestamps
    created_at: datetime
    updated_at: datetime


class PlayerStatsSummaryResponse(BaseModel):
    """
    Schema for aggregated player statistics summary in API responses.

    Provides high-level statistical analysis across multiple games,
    useful for player evaluation, comparison, and fantasy value assessment.

    Contains both cumulative totals and calculated averages.
    """

    # Player identification
    player_id: int  # Database player ID
    player_name: str  # Player display name
    position: str  # Player position

    # Performance summary
    games_played: int  # Number of games with statistics
    avg_fantasy_points: float  # Average fantasy points per game
    max_fantasy_points: float  # Best single-game performance
    min_fantasy_points: float  # Worst single-game performance

    # Cumulative totals by category
    total_passing_yards: int  # Season/career passing yards
    total_rushing_yards: int  # Season/career rushing yards
    total_receiving_yards: int  # Season/career receiving yards
    total_passing_tds: int  # Season/career passing touchdowns
    total_rushing_tds: int  # Season/career rushing touchdowns
    total_receiving_tds: int  # Season/career receiving touchdowns


# ========== MACHINE LEARNING PREDICTION SCHEMAS ==========


class PlayerPredictionRequest(BaseModel):
    """
    Request schema for single player prediction API calls.

    Used with POST /api/predictions/player to request predictions
    with structured request body instead of query parameters.
    """

    player_id: int  # Database ID of player to predict
    game_date: datetime  # Date/time of game for prediction context


class PredictionResponse(BaseModel):
    """
    Response schema for machine learning fantasy point predictions.

    Contains the complete prediction output including:
    - Point estimate (primary prediction)
    - Confidence score (model certainty)
    - Floor/ceiling bounds (prediction range)
    - Model metadata for tracking

    This schema is used for both single player and batch predictions.
    """

    # Player identification
    player_id: int  # Database player ID
    player_name: str  # Player display name
    position: str  # Player position

    # Core prediction
    predicted_points: float  # Expected fantasy points (primary value)
    confidence_score: float  # Model confidence (0.0 to 1.0)

    # Prediction bounds (may be None for simpler models)
    floor: float | None = None  # Lower bound (conservative estimate)
    ceiling: float | None = None  # Upper bound (optimistic estimate)

    # Metadata for tracking and debugging
    model_version: str | None = None  # Which model version made this prediction
    prediction_date: datetime  # When prediction was generated


class SlatePredictionResponse(BaseModel):
    """
    Response schema for batch slate predictions.

    Contains predictions for all players in games on a specific date,
    along with summary statistics and error reporting.

    Used by the /api/predictions/slate endpoint for comprehensive
    daily fantasy analysis.
    """

    # Request context
    game_date: datetime  # Date that was predicted for
    total_predictions: int  # Number of successful predictions generated

    # Prediction results
    predictions: list[PredictionResponse]  # Individual player predictions

    # Error reporting (None if all predictions succeeded)
    errors: list[str] | None = None  # Any errors encountered during processing


# ========== FILE UPLOAD SCHEMAS ==========


class CSVUploadResponse(BaseModel):
    """
    Response schema for DraftKings CSV file upload operations.

    Contains comprehensive processing results including:
    - Success/failure status
    - Count of records processed
    - Data validation warnings
    - Error details for debugging

    Used by the /api/data/upload/draftkings endpoint to provide
    detailed feedback on CSV file processing operations.
    """

    # Processing status
    success: bool  # True if file was processed successfully
    message: str  # Human-readable status message

    # Processing results (None if processing failed)
    contests_created: int | None = None  # Number of new contests created
    salaries_processed: int | None = None  # Number of salary records processed
    unmatched_players: int | None = None  # Number of players that couldn't be matched

    # Data quality feedback
    warnings: list[str] | None = None  # Non-fatal validation warnings
    errors: list[str] | None = None  # Processing errors that occurred

    # File metadata
    filename: str | None = None  # Original filename for reference
    contest_name: str | None = None  # Derived or provided contest name
