"""Data access API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import (
    ContestResponse,
    GameResponse,
    PlayerResponse,
    PlayerStatsResponse,
    SalaryResponse,
    TeamResponse,
)
from src.database.connection import get_db
from src.database.models import DraftKingsContest, DraftKingsSalary, Game, Player, PlayerStats, Team

router = APIRouter()


@router.get("/players", response_model=list[PlayerResponse])
async def get_players(
    position: str | None = Query(None, description="Filter by position (QB, RB, WR, TE, K, DEF)"),
    team: str | None = Query(None, description="Filter by team abbreviation"),
    status: str | None = Query(None, description="Filter by player status"),
    limit: int = Query(100, ge=1, le=1000, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """Get players with optional filtering."""
    query = db.query(Player)

    if position:
        query = query.filter(Player.position == position.upper())
    if team:
        query = query.join(Team).filter(Team.team_abbr == team.upper())
    if status:
        query = query.filter(Player.status == status)

    players = query.offset(offset).limit(limit).all()
    return players


@router.get("/players/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific player by ID."""
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


@router.get("/players/search", response_model=list[PlayerResponse])
async def search_players(
    name: str = Query(..., min_length=2, description="Search by player name"),
    db: Session = Depends(get_db),
):
    """Search players by name."""
    players = db.query(Player).filter(Player.display_name.ilike(f"%{name}%")).limit(50).all()
    return players


@router.get("/teams", response_model=list[TeamResponse])
async def get_teams(
    conference: str | None = Query(None, description="Filter by conference (AFC, NFC)"),
    division: str | None = Query(None, description="Filter by division"),
    db: Session = Depends(get_db),
):
    """Get all teams with optional filtering."""
    query = db.query(Team)

    if conference:
        query = query.filter(Team.conference == conference.upper())
    if division:
        query = query.filter(Team.division.ilike(f"%{division}%"))

    teams = query.all()
    return teams


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific team by ID."""
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


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
    """Get games with optional filtering."""
    query = db.query(Game)

    if season:
        query = query.filter(Game.season == season)
    if week:
        query = query.filter(Game.week == week)
    if team:
        team_upper = team.upper()
        query = query.join(Team, (Team.id == Game.home_team_id) | (Team.id == Game.away_team_id))
        query = query.filter(Team.team_abbr == team_upper)
    if game_type:
        query = query.filter(Game.game_type == game_type.upper())
    if finished is not None:
        query = query.filter(Game.game_finished == finished)

    games = query.order_by(Game.game_date.desc()).offset(offset).limit(limit).all()
    return games


@router.get("/games/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific game by ID."""
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


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
    """Get player statistics with optional filtering."""
    query = db.query(PlayerStats)

    if player_id:
        query = query.filter(PlayerStats.player_id == player_id)
    if game_id:
        query = query.filter(PlayerStats.game_id == game_id)
    if season:
        query = query.join(Game).filter(Game.season == season)
    if week:
        query = query.join(Game).filter(Game.week == week)
    if position:
        query = query.join(Player).filter(Player.position == position.upper())
    if min_fantasy_points:
        query = query.filter(PlayerStats.fantasy_points >= min_fantasy_points)

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
    """Get all game stats for a specific player."""
    query = db.query(PlayerStats).filter(PlayerStats.player_id == player_id)

    if season:
        query = query.join(Game).filter(Game.season == season)

    stats = query.join(Game).order_by(Game.game_date.desc()).limit(limit).all()

    if not stats:
        raise HTTPException(status_code=404, detail="No stats found for this player")

    return stats


@router.get("/contests", response_model=list[ContestResponse])
async def get_contests(
    contest_type: str | None = Query(None, description="Filter by contest type"),
    is_live: bool | None = Query(None, description="Filter by live status"),
    min_prize_pool: float | None = Query(None, description="Minimum total prizes"),
    limit: int = Query(50, ge=1, le=200, description="Limit number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """Get DraftKings contests with optional filtering."""
    query = db.query(DraftKingsContest)

    if contest_type:
        query = query.filter(DraftKingsContest.contest_type.ilike(f"%{contest_type}%"))
    if is_live is not None:
        query = query.filter(DraftKingsContest.is_live == is_live)
    if min_prize_pool:
        query = query.filter(DraftKingsContest.total_prizes >= min_prize_pool)

    contests = query.order_by(DraftKingsContest.start_time.desc()).offset(offset).limit(limit).all()
    return contests


@router.get("/contests/{contest_id}", response_model=ContestResponse)
async def get_contest(
    contest_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific contest by ID."""
    contest = db.query(DraftKingsContest).filter(DraftKingsContest.id == contest_id).first()
    if not contest:
        raise HTTPException(status_code=404, detail="Contest not found")
    return contest


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
    """Get DraftKings salary information with optional filtering."""
    query = db.query(DraftKingsSalary)

    if contest_id:
        query = query.filter(DraftKingsSalary.contest_id == contest_id)
    if player_id:
        query = query.filter(DraftKingsSalary.player_id == player_id)
    if position:
        query = query.filter(DraftKingsSalary.position == position.upper())
    if min_salary:
        query = query.filter(DraftKingsSalary.salary >= min_salary)
    if max_salary:
        query = query.filter(DraftKingsSalary.salary <= max_salary)

    salaries = query.order_by(DraftKingsSalary.salary.desc()).offset(offset).limit(limit).all()
    return salaries


@router.get("/salaries/contest/{contest_id}", response_model=list[SalaryResponse])
async def get_contest_salaries(
    contest_id: int,
    position: str | None = Query(None, description="Filter by position"),
    db: Session = Depends(get_db),
):
    """Get all salaries for a specific contest."""
    query = db.query(DraftKingsSalary).filter(DraftKingsSalary.contest_id == contest_id)

    if position:
        query = query.filter(DraftKingsSalary.position == position.upper())

    salaries = query.order_by(DraftKingsSalary.salary.desc()).all()

    if not salaries:
        raise HTTPException(status_code=404, detail="No salaries found for this contest")

    return salaries


@router.get("/stats/summary/{player_id}")
async def get_player_stats_summary(
    player_id: int,
    season: int | None = Query(None, description="Filter by season"),
    db: Session = Depends(get_db),
):
    """Get statistical summary for a player (averages, totals, etc.)."""
    from sqlalchemy import func

    query = db.query(
        func.count(PlayerStats.id).label("games_played"),
        func.avg(PlayerStats.fantasy_points).label("avg_fantasy_points"),
        func.max(PlayerStats.fantasy_points).label("max_fantasy_points"),
        func.min(PlayerStats.fantasy_points).label("min_fantasy_points"),
        func.sum(PlayerStats.passing_yards).label("total_passing_yards"),
        func.sum(PlayerStats.rushing_yards).label("total_rushing_yards"),
        func.sum(PlayerStats.receiving_yards).label("total_receiving_yards"),
        func.sum(PlayerStats.passing_tds).label("total_passing_tds"),
        func.sum(PlayerStats.rushing_tds).label("total_rushing_tds"),
        func.sum(PlayerStats.receiving_tds).label("total_receiving_tds"),
    ).filter(PlayerStats.player_id == player_id)

    if season:
        query = query.join(Game).filter(Game.season == season)

    result = query.first()

    if not result or result.games_played == 0:
        raise HTTPException(status_code=404, detail="No stats found for this player")

    # Get player info
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    return {
        "player_id": player_id,
        "player_name": player.display_name,
        "position": player.position,
        "games_played": result.games_played,
        "avg_fantasy_points": round(float(result.avg_fantasy_points or 0), 2),
        "max_fantasy_points": float(result.max_fantasy_points or 0),
        "min_fantasy_points": float(result.min_fantasy_points or 0),
        "total_passing_yards": result.total_passing_yards or 0,
        "total_rushing_yards": result.total_rushing_yards or 0,
        "total_receiving_yards": result.total_receiving_yards or 0,
        "total_passing_tds": result.total_passing_tds or 0,
        "total_rushing_tds": result.total_rushing_tds or 0,
        "total_receiving_tds": result.total_receiving_tds or 0,
    }
