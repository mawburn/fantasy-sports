"""SQLAlchemy database models."""

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Team(Base):
    """NFL team model."""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    team_abbr = Column(String(5), unique=True, nullable=False, index=True)
    team_name = Column(String(50), nullable=False)
    conference = Column(String(3), nullable=False)  # AFC/NFC
    division = Column(String(10), nullable=False)  # North/South/East/West

    # Relationships
    home_games = relationship(
        "Game", foreign_keys="[Game.home_team_id]", back_populates="home_team"
    )
    away_games = relationship(
        "Game", foreign_keys="[Game.away_team_id]", back_populates="away_team"
    )
    players = relationship("Player", back_populates="team")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Player(Base):
    """NFL player model."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String(20), unique=True, nullable=False, index=True)  # nfl_data_py player_id
    display_name = Column(String(100), nullable=False, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    position = Column(String(5), nullable=False, index=True)  # QB, RB, WR, TE, K, DEF
    jersey_number = Column(Integer)

    # Team information
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True, index=True)
    team = relationship("Team", back_populates="players")

    # Physical attributes
    height = Column(Integer)  # inches
    weight = Column(Integer)  # pounds
    age = Column(Integer)
    birthdate = Column(Date)

    # Career info
    years_exp = Column(Integer)
    college = Column(String(100))
    rookie_year = Column(Integer)

    # Status
    status = Column(String(20), default="Active")  # Active, Inactive, IR, etc.

    # Relationships
    stats = relationship("PlayerStats", back_populates="player")
    salaries = relationship("DraftKingsSalary", back_populates="player")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index("idx_player_position_team", "position", "team_id"),
        Index("idx_player_name", "display_name"),
    )


class Game(Base):
    """NFL game model."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String(20), unique=True, nullable=False, index=True)  # nfl_data_py game_id
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    game_date = Column(DateTime, nullable=False, index=True)

    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")

    # Game info
    game_type = Column(String(10), nullable=False)  # REG, POST, PRE
    home_score = Column(Integer)
    away_score = Column(Integer)

    # Weather (if available)
    weather_temperature = Column(Integer)
    weather_wind_speed = Column(Integer)
    weather_description = Column(String(100))

    # Stadium info
    stadium = Column(String(100))
    roof_type = Column(String(20))  # dome, outdoors, retractable

    # Game status
    game_finished = Column(Boolean, default=False)

    # Relationships
    player_stats = relationship("PlayerStats", back_populates="game")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("season", "week", "home_team_id", "away_team_id"),
        Index("idx_game_season_week", "season", "week"),
        Index("idx_game_date", "game_date"),
    )


class PlayerStats(Base):
    """Player game statistics."""

    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)

    # Relationships
    player = relationship("Player", back_populates="stats")
    game = relationship("Game", back_populates="player_stats")

    # Passing stats
    passing_yards = Column(Integer, default=0)
    passing_tds = Column(Integer, default=0)
    passing_interceptions = Column(Integer, default=0)
    passing_attempts = Column(Integer, default=0)
    passing_completions = Column(Integer, default=0)

    # Rushing stats
    rushing_yards = Column(Integer, default=0)
    rushing_tds = Column(Integer, default=0)
    rushing_attempts = Column(Integer, default=0)

    # Receiving stats
    receiving_yards = Column(Integer, default=0)
    receiving_tds = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    targets = Column(Integer, default=0)

    # Other stats
    fumbles_lost = Column(Integer, default=0)
    two_point_conversions = Column(Integer, default=0)

    # Fantasy points
    fantasy_points = Column(Float)
    fantasy_points_ppr = Column(Float)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("player_id", "game_id"),
        Index("idx_stats_player_game", "player_id", "game_id"),
    )


class DraftKingsContest(Base):
    """DraftKings contest information."""

    __tablename__ = "draftkings_contests"

    id = Column(Integer, primary_key=True, index=True)
    contest_id = Column(String(50), unique=True, nullable=False, index=True)
    contest_name = Column(String(200), nullable=False)
    contest_type = Column(String(50), nullable=False)  # Classic, Showdown, etc.

    # Contest details
    entry_fee = Column(Float, nullable=False)
    total_prizes = Column(Float, nullable=False)
    max_entries = Column(Integer, nullable=False)
    current_entries = Column(Integer, default=0)

    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    # Game info
    slate_id = Column(String(50), index=True)
    sport = Column(String(10), default="NFL")

    # Status
    is_live = Column(Boolean, default=True)
    is_guaranteed = Column(Boolean, default=False)

    # Relationships
    salaries = relationship("DraftKingsSalary", back_populates="contest")

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PlayByPlay(Base):
    """Play-by-play data model."""

    __tablename__ = "play_by_play"

    id = Column(Integer, primary_key=True, index=True)
    play_id = Column(String(50), unique=True, nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)

    # Relationships
    game = relationship("Game")

    # Play details
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    posteam = Column(String(5), index=True)  # Team with possession
    defteam = Column(String(5), index=True)  # Defending team

    # Time and situation
    quarter = Column(Integer)
    time = Column(String(10))  # Game clock
    down = Column(Integer)
    ydstogo = Column(Integer)  # Yards to go for first down
    yardline_100 = Column(Integer)  # Yards from goal line (0-100)

    # Play description
    play_type = Column(String(20), index=True)
    desc = Column(String(500))  # Play description

    # Outcome
    yards_gained = Column(Integer)
    first_down = Column(Boolean, default=False)
    touchdown = Column(Boolean, default=False)
    interception = Column(Boolean, default=False)
    fumble = Column(Boolean, default=False)
    sack = Column(Boolean, default=False)
    penalty = Column(Boolean, default=False)

    # Advanced metrics
    epa = Column(Float)  # Expected Points Added
    wp = Column(Float)  # Win Probability
    wpa = Column(Float)  # Win Probability Added

    # Scoring
    home_score = Column(Integer)
    away_score = Column(Integer)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints and indexes
    __table_args__ = (
        Index("idx_pbp_game_play", "game_id", "play_id"),
        Index("idx_pbp_season_week", "season", "week"),
        Index("idx_pbp_team", "posteam"),
    )


class DraftKingsSalary(Base):
    """DraftKings player salary information."""

    __tablename__ = "draftkings_salaries"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    contest_id = Column(Integer, ForeignKey("draftkings_contests.id"), nullable=False, index=True)

    # Relationships
    player = relationship("Player", back_populates="salaries")
    contest = relationship("DraftKingsContest", back_populates="salaries")

    # Salary info
    salary = Column(Integer, nullable=False)
    position = Column(String(10), nullable=False)
    roster_position = Column(String(10))  # FLEX, UTIL, etc.

    # Player info from DK
    dk_player_name = Column(String(100), nullable=False)
    dk_team_abbr = Column(String(5))

    # Game info
    game_info = Column(String(50))  # "LAR@SF", opponent info

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("player_id", "contest_id"),
        Index("idx_salary_contest_position", "contest_id", "position"),
        Index("idx_salary_player_contest", "player_id", "contest_id"),
    )


class InjuryReport(Base):
    """NFL injury report model."""

    __tablename__ = "injury_reports"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)

    # Relationships
    player = relationship("Player")
    game = relationship("Game")

    # Injury details
    injury_status = Column(
        String(20), nullable=False, index=True
    )  # Out, Doubtful, Questionable, Probable
    injury_designation = Column(String(50))  # Body part (Knee, Shoulder, etc.)
    injury_description = Column(String(200))  # Additional details

    # Report metadata
    report_date = Column(Date, nullable=False, index=True)
    practice_status = Column(String(20))  # DNP, Limited, Full

    # Historical tracking
    days_missed = Column(Integer, default=0)
    games_missed = Column(Integer, default=0)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("player_id", "game_id", "report_date"),
        Index("idx_injury_player_status", "player_id", "injury_status"),
        Index("idx_injury_game_date", "game_id", "report_date"),
    )


class ScoringRules(Base):
    """DraftKings scoring rules model."""

    __tablename__ = "scoring_rules"

    id = Column(Integer, primary_key=True, index=True)
    contest_type = Column(String(50), nullable=False, index=True)  # Classic, Showdown, etc.
    position = Column(String(10), nullable=False, index=True)  # QB, RB, WR, TE, K, DEF

    # Passing scoring
    passing_yards_per_point = Column(Float, default=25.0)  # Yards needed for 1 point
    passing_td_points = Column(Float, default=4.0)
    passing_int_points = Column(Float, default=-2.0)
    passing_2pt_points = Column(Float, default=2.0)

    # Rushing scoring
    rushing_yards_per_point = Column(Float, default=10.0)
    rushing_td_points = Column(Float, default=6.0)
    rushing_2pt_points = Column(Float, default=2.0)

    # Receiving scoring
    receiving_yards_per_point = Column(Float, default=10.0)
    receiving_td_points = Column(Float, default=6.0)
    reception_points = Column(Float, default=0.0)  # PPR bonus
    receiving_2pt_points = Column(Float, default=2.0)

    # Other scoring
    fumble_lost_points = Column(Float, default=-2.0)
    fumble_recovery_td_points = Column(Float, default=6.0)

    # Kicker scoring
    fg_0_39_points = Column(Float, default=3.0)
    fg_40_49_points = Column(Float, default=4.0)
    fg_50_plus_points = Column(Float, default=5.0)
    fg_miss_points = Column(Float, default=-1.0)
    extra_point_points = Column(Float, default=1.0)
    extra_point_miss_points = Column(Float, default=-1.0)

    # Defense scoring
    def_td_points = Column(Float, default=6.0)
    def_int_points = Column(Float, default=2.0)
    def_fumble_recovery_points = Column(Float, default=2.0)
    def_safety_points = Column(Float, default=2.0)
    def_sack_points = Column(Float, default=1.0)
    def_points_allowed_0_points = Column(Float, default=10.0)
    def_points_allowed_1_6_points = Column(Float, default=7.0)
    def_points_allowed_7_13_points = Column(Float, default=4.0)
    def_points_allowed_14_20_points = Column(Float, default=1.0)
    def_points_allowed_21_27_points = Column(Float, default=0.0)
    def_points_allowed_28_34_points = Column(Float, default=-1.0)
    def_points_allowed_35_plus_points = Column(Float, default=-4.0)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("contest_type", "position"),
        Index("idx_scoring_contest_pos", "contest_type", "position"),
    )


class DraftKingsLineup(Base):
    """DraftKings lineup model."""

    __tablename__ = "draftkings_lineups"

    id = Column(Integer, primary_key=True, index=True)
    lineup_id = Column(String(50), unique=True, nullable=False, index=True)
    contest_id = Column(Integer, ForeignKey("draftkings_contests.id"), nullable=False, index=True)

    # Relationships
    contest = relationship("DraftKingsContest")
    entries = relationship("DraftKingsEntry", back_populates="lineup")

    # Lineup composition
    qb_player_id = Column(Integer, ForeignKey("players.id"))
    rb1_player_id = Column(Integer, ForeignKey("players.id"))
    rb2_player_id = Column(Integer, ForeignKey("players.id"))
    wr1_player_id = Column(Integer, ForeignKey("players.id"))
    wr2_player_id = Column(Integer, ForeignKey("players.id"))
    wr3_player_id = Column(Integer, ForeignKey("players.id"))
    te_player_id = Column(Integer, ForeignKey("players.id"))
    flex_player_id = Column(Integer, ForeignKey("players.id"))  # RB/WR/TE
    def_player_id = Column(Integer, ForeignKey("players.id"))

    # Player relationships
    qb_player = relationship("Player", foreign_keys=[qb_player_id])
    rb1_player = relationship("Player", foreign_keys=[rb1_player_id])
    rb2_player = relationship("Player", foreign_keys=[rb2_player_id])
    wr1_player = relationship("Player", foreign_keys=[wr1_player_id])
    wr2_player = relationship("Player", foreign_keys=[wr2_player_id])
    wr3_player = relationship("Player", foreign_keys=[wr3_player_id])
    te_player = relationship("Player", foreign_keys=[te_player_id])
    flex_player = relationship("Player", foreign_keys=[flex_player_id])
    def_player = relationship("Player", foreign_keys=[def_player_id])

    # Lineup metrics
    total_salary = Column(Integer, nullable=False)
    salary_remaining = Column(Integer, default=0)
    projected_points = Column(Float)
    actual_points = Column(Float)

    # Optimization metadata
    optimization_strategy = Column(String(50))  # Cash, GPP, Balanced, etc.
    generated_by = Column(String(50))  # Algorithm/Manual
    confidence_score = Column(Float)  # Model confidence 0-1

    # Performance tracking
    rank = Column(Integer)  # Final contest rank
    percentile = Column(Float)  # Performance percentile
    roi = Column(Float)  # Return on investment

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        Index("idx_lineup_contest", "contest_id"),
        Index("idx_lineup_performance", "actual_points", "rank"),
    )


class DraftKingsEntry(Base):
    """DraftKings contest entry model."""

    __tablename__ = "draftkings_entries"

    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(String(50), unique=True, nullable=False, index=True)
    contest_id = Column(Integer, ForeignKey("draftkings_contests.id"), nullable=False, index=True)
    lineup_id = Column(Integer, ForeignKey("draftkings_lineups.id"), nullable=False, index=True)

    # Relationships
    contest = relationship("DraftKingsContest")
    lineup = relationship("DraftKingsLineup", back_populates="entries")

    # Entry details
    entry_fee = Column(Float, nullable=False)
    username = Column(String(100))  # DK username (if available)

    # Results
    final_score = Column(Float)
    final_rank = Column(Integer)
    prize_won = Column(Float, default=0.0)

    # Entry metadata
    entry_time = Column(DateTime, nullable=False)
    is_late_swap = Column(Boolean, default=False)
    swap_count = Column(Integer, default=0)

    # Performance metrics
    roi = Column(Float)  # (prize_won - entry_fee) / entry_fee
    profit = Column(Float)  # prize_won - entry_fee

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        Index("idx_entry_contest_lineup", "contest_id", "lineup_id"),
        Index("idx_entry_performance", "final_score", "final_rank"),
        Index("idx_entry_roi", "roi"),
    )


# ML-Specific Models


class FeatureStore(Base):
    """Feature store for ML model features."""

    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)

    # Relationships
    player = relationship("Player")
    game = relationship("Game")

    # Feature metadata
    feature_version = Column(String(20), nullable=False, index=True)  # v1.0, v1.1, etc.
    position = Column(String(10), nullable=False, index=True)

    # Feature storage (JSON-like storage for flexibility)
    features_json = Column(String, nullable=False)  # Serialized feature dict
    feature_names = Column(String)  # Comma-separated feature names
    feature_count = Column(Integer, nullable=False)

    # Feature quality metrics
    completeness_score = Column(Float)  # 0-1, percentage of non-null features
    quality_score = Column(Float)  # 0-1, overall feature quality

    # Timestamps
    computed_at = Column(DateTime, nullable=False, index=True)
    game_date = Column(DateTime, nullable=False, index=True)  # Denormalized for performance

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("player_id", "game_id", "feature_version"),
        Index("idx_features_player_game", "player_id", "game_id"),
        Index("idx_features_version_pos", "feature_version", "position"),
        Index("idx_features_date", "game_date"),
    )


class ModelMetadata(Base):
    """ML model metadata and versioning."""

    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(50), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)

    # Model details
    position = Column(String(10), nullable=False, index=True)  # QB, RB, WR, TE, DEF
    model_type = Column(String(50), nullable=False)  # XGBoost, LightGBM, Neural Network
    version = Column(String(20), nullable=False, index=True)

    # Training details
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    training_data_size = Column(Integer, nullable=False)
    validation_data_size = Column(Integer, nullable=False)

    # Hyperparameters (JSON storage)
    hyperparameters = Column(String)  # Serialized hyperparameter dict
    feature_names = Column(String)  # Comma-separated feature names
    feature_count = Column(Integer, nullable=False)

    # Performance metrics
    mae_validation = Column(Float)  # Mean Absolute Error on validation
    rmse_validation = Column(Float)  # Root Mean Square Error
    r2_validation = Column(Float)  # R-squared score
    mape_validation = Column(Float)  # Mean Absolute Percentage Error

    # Model status
    status = Column(
        String(20), nullable=False, default="training"
    )  # training, validated, deployed, retired
    is_active = Column(Boolean, default=False)  # Currently deployed model
    deployment_date = Column(DateTime)
    retirement_date = Column(DateTime)

    # File storage
    model_path = Column(String(500))  # Path to serialized model file
    preprocessor_path = Column(String(500))  # Path to feature preprocessor

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        Index("idx_model_position_status", "position", "status"),
        Index("idx_model_active", "is_active"),
        Index("idx_model_version", "version"),
    )


class PredictionResult(Base):
    """ML model prediction results."""

    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(50), ForeignKey("model_metadata.model_id"), nullable=False, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)

    # Relationships
    model = relationship("ModelMetadata")
    player = relationship("Player")
    game = relationship("Game")

    # Predictions
    predicted_points = Column(Float, nullable=False)
    predicted_floor = Column(Float)  # 25th percentile prediction
    predicted_ceiling = Column(Float)  # 75th percentile prediction
    confidence_score = Column(Float)  # Model confidence 0-1

    # Actual results (filled in after game)
    actual_points = Column(Float)
    prediction_error = Column(Float)  # actual - predicted
    absolute_error = Column(Float)  # |actual - predicted|

    # Feature importance for this prediction
    top_features = Column(String)  # JSON of top contributing features
    feature_weights = Column(String)  # JSON of feature importance scores

    # Prediction metadata
    prediction_date = Column(DateTime, nullable=False, index=True)
    game_date = Column(DateTime, nullable=False, index=True)  # Denormalized
    position = Column(String(10), nullable=False, index=True)  # Denormalized

    # Model performance tracking
    is_outlier = Column(Boolean, default=False)  # Flagged as prediction outlier
    outlier_reason = Column(String(200))  # Why flagged as outlier

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint("model_id", "player_id", "game_id"),
        Index("idx_pred_model_game", "model_id", "game_id"),
        Index("idx_pred_player_date", "player_id", "game_date"),
        Index("idx_pred_error", "absolute_error"),
        Index("idx_pred_performance", "position", "game_date"),
    )


class TrainingDataset(Base):
    """Training dataset versioning and metadata."""

    __tablename__ = "training_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(50), unique=True, nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False)

    # Dataset details
    position = Column(String(10), nullable=False, index=True)
    version = Column(String(20), nullable=False, index=True)

    # Date range
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False, index=True)

    # Dataset size
    total_samples = Column(Integer, nullable=False)
    training_samples = Column(Integer, nullable=False)
    validation_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)

    # Feature information
    feature_version = Column(String(20), nullable=False)
    feature_count = Column(Integer, nullable=False)
    feature_names = Column(String)  # Comma-separated

    # Data quality metrics
    missing_data_percentage = Column(Float)
    outlier_percentage = Column(Float)
    class_balance_score = Column(Float)  # For classification tasks

    # File storage
    dataset_path = Column(String(500))  # Path to dataset files
    metadata_path = Column(String(500))  # Path to dataset metadata

    # Usage tracking
    models_trained = Column(Integer, default=0)  # Number of models trained on this dataset
    last_used = Column(DateTime)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        Index("idx_dataset_position_version", "position", "version"),
        Index("idx_dataset_date_range", "start_date", "end_date"),
        Index("idx_dataset_usage", "last_used"),
    )


class BacktestResult(Base):
    """Model backtesting results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(String(50), unique=True, nullable=False, index=True)
    model_id = Column(String(50), ForeignKey("model_metadata.model_id"), nullable=False, index=True)

    # Relationships
    model = relationship("ModelMetadata")

    # Backtest configuration
    backtest_name = Column(String(100), nullable=False)
    position = Column(String(10), nullable=False, index=True)

    # Date range
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False, index=True)

    # Performance metrics
    total_predictions = Column(Integer, nullable=False)
    mae = Column(Float, nullable=False)  # Mean Absolute Error
    rmse = Column(Float, nullable=False)  # Root Mean Square Error
    r2_score = Column(Float, nullable=False)  # R-squared
    mape = Column(Float, nullable=False)  # Mean Absolute Percentage Error

    # Consistency metrics
    consistency_score = Column(Float)  # How consistent are predictions
    calibration_score = Column(Float)  # How well calibrated are confidence intervals

    # Financial metrics (for optimization validation)
    roi_simulated = Column(Float)  # Simulated ROI if used for lineups
    profit_simulated = Column(Float)  # Simulated profit
    sharpe_ratio = Column(Float)  # Risk-adjusted returns

    # Detailed results storage
    detailed_results_path = Column(String(500))  # Path to detailed prediction results

    # Execution metadata
    execution_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        Index("idx_backtest_model_date", "model_id", "start_date"),
        Index("idx_backtest_position_perf", "position", "mae"),
        Index("idx_backtest_roi", "roi_simulated"),
    )
