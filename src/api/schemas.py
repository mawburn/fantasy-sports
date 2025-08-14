"""Pydantic schemas for API responses."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class TeamResponse(BaseModel):
    """Team response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    team_abbr: str
    team_name: str
    conference: str
    division: str
    created_at: datetime
    updated_at: datetime


class PlayerResponse(BaseModel):
    """Player response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str
    display_name: str
    first_name: str | None = None
    last_name: str | None = None
    position: str
    jersey_number: int | None = None
    team_id: int | None = None
    height: int | None = None
    weight: int | None = None
    age: int | None = None
    birthdate: date | None = None
    years_exp: int | None = None
    college: str | None = None
    rookie_year: int | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class GameResponse(BaseModel):
    """Game response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    game_id: str
    season: int
    week: int
    game_date: datetime
    home_team_id: int
    away_team_id: int
    game_type: str
    home_score: int | None = None
    away_score: int | None = None
    weather_temperature: int | None = None
    weather_wind_speed: int | None = None
    weather_description: str | None = None
    stadium: str | None = None
    roof_type: str | None = None
    game_finished: bool
    created_at: datetime
    updated_at: datetime


class PlayerStatsResponse(BaseModel):
    """Player stats response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    game_id: int
    passing_yards: int
    passing_tds: int
    passing_interceptions: int
    passing_attempts: int
    passing_completions: int
    rushing_yards: int
    rushing_tds: int
    rushing_attempts: int
    receiving_yards: int
    receiving_tds: int
    receptions: int
    targets: int
    fumbles_lost: int
    two_point_conversions: int
    fantasy_points: float | None = None
    fantasy_points_ppr: float | None = None
    created_at: datetime
    updated_at: datetime


class ContestResponse(BaseModel):
    """DraftKings contest response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    contest_id: str
    contest_name: str
    contest_type: str
    entry_fee: float
    total_prizes: float
    max_entries: int
    current_entries: int
    start_time: datetime
    end_time: datetime
    slate_id: str | None = None
    sport: str
    is_live: bool
    is_guaranteed: bool
    created_at: datetime
    updated_at: datetime


class SalaryResponse(BaseModel):
    """DraftKings salary response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    contest_id: int
    salary: int
    position: str
    roster_position: str | None = None
    dk_player_name: str
    dk_team_abbr: str | None = None
    game_info: str | None = None
    created_at: datetime
    updated_at: datetime


class PlayerStatsSummaryResponse(BaseModel):
    """Player stats summary response schema."""

    player_id: int
    player_name: str
    position: str
    games_played: int
    avg_fantasy_points: float
    max_fantasy_points: float
    min_fantasy_points: float
    total_passing_yards: int
    total_rushing_yards: int
    total_receiving_yards: int
    total_passing_tds: int
    total_rushing_tds: int
    total_receiving_tds: int
