"""
Optimization API endpoints for DFS lineup generation.

This module provides REST API endpoints for building optimized lineups using
the mathematical optimization capabilities in src.optimization.lineup_builder.

Endpoints:
- /lineup: Generate single optimal lineup
- /alternate: Generate multiple lineup options
- /validate: Validate lineup against constraints
- /adjust: Make manual adjustments to existing lineups

The endpoints use the LineupBuilder class which implements:
- Linear programming optimization (guaranteed optimal)
- Genetic algorithm optimization (evolutionary approach)
- Stacking optimization (correlated players)
- Tournament optimization (high-ceiling focused)
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.optimization.lineup_builder import LineupBuilder, LineupConstraints

logger = logging.getLogger(__name__)


# Request/Response Models
class OptimizeLineupRequest(BaseModel):
    """Request model for lineup optimization."""

    contest_id: int = Field(..., description="DraftKings contest ID")
    algorithm: str = Field(
        default="linear_programming",
        description="Optimization algorithm: linear_programming, greedy, genetic, stacking",
    )
    salary_cap: int = Field(default=50000, description="Salary cap constraint")
    min_salary: int = Field(default=45000, description="Minimum salary to use")
    positions: dict[str, int] | None = Field(default=None, description="Position requirements")

    # Stacking options
    enable_stacking: bool = Field(default=False, description="Enable QB-WR/TE stacking")
    stack_teams: list[str] | None = Field(default=None, description="Teams to stack")
    min_stack_count: int = Field(default=1, description="Minimum players in stack")

    # Tournament options
    tournament_mode: bool = Field(
        default=False, description="Optimize for tournaments (use ceiling)"
    )
    ceiling_weight: float = Field(default=0.3, description="Weight for ceiling vs projection")

    # Ownership options
    ownership_mode: str = Field(
        default="none", description="Ownership strategy: none, contrarian, conservative"
    )
    ownership_weight: float = Field(default=0.2, description="Weight for ownership adjustments")


class AlternateLineupsRequest(BaseModel):
    """Request model for generating multiple lineup options."""

    contest_id: int = Field(..., description="DraftKings contest ID")
    num_lineups: int = Field(default=5, ge=1, le=20, description="Number of lineups to generate")
    diversity_factor: float = Field(default=0.3, description="Diversity factor between lineups")
    algorithm: str = Field(default="linear_programming", description="Base optimization algorithm")


class ValidateLineupRequest(BaseModel):
    """Request model for lineup validation."""

    player_ids: list[int] = Field(..., description="List of player IDs in lineup")
    contest_id: int = Field(..., description="DraftKings contest ID")
    salary_cap: int = Field(default=50000, description="Salary cap to validate against")


class AdjustLineupRequest(BaseModel):
    """Request model for lineup adjustments."""

    current_lineup: list[int] = Field(..., description="Current player IDs")
    remove_player_id: int | None = Field(default=None, description="Player ID to remove")
    add_player_id: int | None = Field(default=None, description="Player ID to add")
    contest_id: int = Field(..., description="DraftKings contest ID")


class PlayerProjectionResponse(BaseModel):
    """Response model for player projection."""

    player_id: int
    name: str
    position: str
    roster_slot: str | None = None  # The actual roster slot (e.g., "FLEX" for flex position)
    team: str
    salary: int
    projected_points: float
    floor: float
    ceiling: float
    value: float


class LineupResponse(BaseModel):
    """Response model for optimized lineup."""

    lineup: list[PlayerProjectionResponse]
    total_salary: int
    projected_points: float
    lineup_value: float
    optimization_status: str
    constraint_violations: list[str]
    stacking_analysis: dict[str, Any] | None = None


class ValidationResponse(BaseModel):
    """Response model for lineup validation."""

    is_valid: bool
    total_salary: int
    projected_points: float
    constraint_violations: list[str]
    position_counts: dict[str, int]


# Create router
router = APIRouter(prefix="/optimize", tags=["optimization"])


@router.post("/lineup", response_model=LineupResponse)
async def optimize_lineup(
    request: OptimizeLineupRequest, db: Session = Depends(get_db)
) -> LineupResponse:
    """
    Generate optimal single-entry lineup for DraftKings contest.

    Uses mathematical optimization to find the lineup that maximizes projected
    points while satisfying all DraftKings constraints (salary cap, positions).

    Supports multiple optimization algorithms:
    - linear_programming: Guaranteed optimal solution (recommended)
    - greedy: Fast heuristic approach
    - genetic: Evolutionary optimization for complex constraints
    - stacking: Correlated player optimization
    """
    try:
        builder = LineupBuilder(db)

        # Get player pool for contest
        player_pool = builder.get_player_pool(
            contest_id=request.contest_id, min_salary=3000, max_salary=15000
        )

        if not player_pool:
            raise HTTPException(
                status_code=404, detail=f"No players found for contest {request.contest_id}"
            )

        # Add ML predictions to player pool
        from src.api.routers.predictions import PredictionService
        
        prediction_service = PredictionService(db)
        
        # Generate predictions for each player
        # Use position-based averages as fallback when models aren't available
        position_averages = {
            "QB": 18.0,
            "RB": 12.0,
            "WR": 11.0,
            "TE": 8.0,
            "DST": 7.0,
        }
        
        for player in player_pool:
            try:
                # Try to get prediction from trained models
                position = "DEF" if player.position == "DST" else player.position
                # Get game date from contest (assuming next Sunday for now)
                from datetime import datetime, timedelta
                today = datetime.now()
                days_ahead = 6 - today.weekday()  # Sunday is 6
                if days_ahead <= 0:
                    days_ahead += 7
                game_date = today + timedelta(days=days_ahead)
                
                predictions = prediction_service.predict_player(
                    player.player_id, game_date, position
                )
                
                if predictions:
                    player.projected_points = predictions.predicted_points
                    player.floor = predictions.floor
                    player.ceiling = predictions.ceiling
                else:
                    # Fallback for players without predictions
                    player.projected_points = position_averages.get(player.position, 10.0)
                    player.floor = player.projected_points * 0.7
                    player.ceiling = player.projected_points * 1.3
                    
            except Exception as e:
                # Use position-based fallback if model prediction fails
                logger.debug(f"Using fallback for player {player.player_id}: {e}")
                player.projected_points = position_averages.get(player.position, 10.0)
                player.floor = player.projected_points * 0.7
                player.ceiling = player.projected_points * 1.3

        # Set up constraints
        constraints = LineupConstraints(
            salary_cap=request.salary_cap,
            min_salary=request.min_salary,
            positions=request.positions,
            allow_qb_stack=request.enable_stacking,
            allow_rb_def_stack=request.enable_stacking,
        )

        # Choose optimization algorithm
        if request.algorithm == "linear_programming":
            result = builder.build_linear_programming_lineup(player_pool, constraints)
        elif request.algorithm == "greedy":
            result = builder.build_greedy_lineup(player_pool, constraints)
        elif request.algorithm == "genetic":
            result = builder.build_genetic_algorithm_lineup(player_pool, constraints)
        elif request.algorithm == "stacking" and request.enable_stacking:
            result = builder.build_lineup_with_stacking(
                player_pool,
                constraints,
                qb_stack_teams=request.stack_teams,
                min_qb_stack_count=request.min_stack_count,
            )
        elif request.tournament_mode:
            result = builder.build_tournament_optimized_lineup(
                player_pool, constraints, ceiling_weight=request.ceiling_weight
            )
        elif request.ownership_mode != "none":
            result = builder.optimize_with_ownership_projections(
                player_pool,
                constraints,
                ownership_weight=request.ownership_weight,
                contrarian_mode=(request.ownership_mode == "contrarian"),
            )
        else:
            result = builder.build_linear_programming_lineup(player_pool, constraints)

        # Convert to response format with proper roster slot assignment
        # DraftKings lineup order: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
        lineup_players = []
        position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0}
        position_limits = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
        
        # First pass: assign base positions
        base_assigned = []
        flex_candidate = None
        
        for p in result.lineup:
            if position_counts.get(p.position, 0) < position_limits.get(p.position, 0):
                # Base position slot
                lineup_players.append(
                    PlayerProjectionResponse(
                        player_id=p.player_id,
                        name=p.name,
                        position=p.position,
                        roster_slot=p.position,  # Base position
                        team=p.team_abbr,
                        salary=p.salary,
                        projected_points=p.projected_points,
                        floor=p.floor,
                        ceiling=p.ceiling,
                        value=p.value,
                    )
                )
                position_counts[p.position] = position_counts.get(p.position, 0) + 1
                base_assigned.append(p.player_id)
            elif p.position in ["RB", "WR", "TE"] and not flex_candidate:
                # This is the FLEX player
                flex_candidate = p
                
        # Add FLEX player if found
        if flex_candidate:
            lineup_players.append(
                PlayerProjectionResponse(
                    player_id=flex_candidate.player_id,
                    name=flex_candidate.name,
                    position=flex_candidate.position,
                    roster_slot="FLEX",  # Marked as FLEX
                    team=flex_candidate.team_abbr,
                    salary=flex_candidate.salary,
                    projected_points=flex_candidate.projected_points,
                    floor=flex_candidate.floor,
                    ceiling=flex_candidate.ceiling,
                    value=flex_candidate.value,
                )
            )

        # Add stacking analysis if stacking was used
        stacking_analysis = None
        if request.enable_stacking and result.lineup:
            stacking_analysis = builder._analyze_stacking(result.lineup)

        return LineupResponse(
            lineup=lineup_players,
            total_salary=result.total_salary,
            projected_points=result.projected_points,
            lineup_value=result.lineup_value,
            optimization_status=result.optimization_status,
            constraint_violations=result.constraint_violations,
            stacking_analysis=stacking_analysis,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e!s}") from e


@router.post("/alternate", response_model=list[LineupResponse])
async def generate_alternate_lineups(
    request: AlternateLineupsRequest, db: Session = Depends(get_db)
) -> list[LineupResponse]:
    """
    Generate multiple diverse lineup options for analysis and comparison.

    This endpoint creates several different lineups using the same player pool,
    with diversity controls to ensure different strategic approaches.

    Useful for:
    - Comparing different lineup construction strategies
    - Research and analysis of optimal approaches
    - Single-entry classic contests where you want options
    """
    try:
        builder = LineupBuilder(db)

        # Get player pool
        player_pool = builder.get_player_pool(
            contest_id=request.contest_id, min_salary=3000, max_salary=15000
        )

        if not player_pool:
            raise HTTPException(
                status_code=404, detail=f"No players found for contest {request.contest_id}"
            )

        # Add placeholder projections
        for player in player_pool:
            player.projected_points = 10.0
            player.floor = 7.0
            player.ceiling = 15.0

        # Generate multiple lineups
        constraints = LineupConstraints()
        results = builder.generate_multiple_lineups(
            player_pool,
            constraints,
            num_lineups=request.num_lineups,
            diversity_factor=request.diversity_factor,
        )

        # Convert to response format
        lineup_responses = []
        for result in results:
            lineup_players = [
                PlayerProjectionResponse(
                    player_id=p.player_id,
                    name=p.name,
                    position=p.position,
                    team=p.team_abbr,
                    salary=p.salary,
                    projected_points=p.projected_points,
                    floor=p.floor,
                    ceiling=p.ceiling,
                    value=p.value,
                )
                for p in result.lineup
            ]

            lineup_responses.append(
                LineupResponse(
                    lineup=lineup_players,
                    total_salary=result.total_salary,
                    projected_points=result.projected_points,
                    lineup_value=result.lineup_value,
                    optimization_status=result.optimization_status,
                    constraint_violations=result.constraint_violations,
                )
            )

        return lineup_responses

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Alternate lineup generation failed: {e!s}"
        ) from e


@router.post("/validate", response_model=ValidationResponse)
async def validate_lineup(
    request: ValidateLineupRequest, db: Session = Depends(get_db)
) -> ValidationResponse:
    """
    Validate a lineup against DraftKings constraints.

    Checks if a proposed lineup satisfies all requirements:
    - Salary cap compliance
    - Position requirements
    - Player availability in contest
    - No duplicate players
    """
    try:
        builder = LineupBuilder(db)

        # Get player data for the provided player IDs
        player_pool = builder.get_player_pool(
            contest_id=request.contest_id, min_salary=0, max_salary=20000
        )

        # Filter to only requested players
        player_dict = {p.player_id: p for p in player_pool}
        lineup_players = []
        missing_players = []

        for player_id in request.player_ids:
            if player_id in player_dict:
                lineup_players.append(player_dict[player_id])
            else:
                missing_players.append(player_id)

        if missing_players:
            raise HTTPException(
                status_code=400, detail=f"Players not found in contest: {missing_players}"
            )

        # Validate lineup
        constraints = LineupConstraints(salary_cap=request.salary_cap)
        violations = builder.validate_lineup(lineup_players, constraints)

        # Calculate position counts
        position_counts = {}
        for player in lineup_players:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Calculate metrics
        total_salary = sum(p.salary for p in lineup_players)
        projected_points = sum(p.projected_points for p in lineup_players)

        return ValidationResponse(
            is_valid=len(violations) == 0,
            total_salary=total_salary,
            projected_points=projected_points,
            constraint_violations=violations,
            position_counts=position_counts,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e!s}") from e


@router.post("/adjust", response_model=LineupResponse)
async def adjust_lineup(
    request: AdjustLineupRequest, db: Session = Depends(get_db)
) -> LineupResponse:
    """
    Make manual adjustments to an existing lineup.

    Allows swapping players in/out of a lineup while maintaining constraints.
    If the adjustment violates constraints, suggests alternative players.
    """
    try:
        builder = LineupBuilder(db)

        # Get current lineup players
        player_pool = builder.get_player_pool(
            contest_id=request.contest_id, min_salary=0, max_salary=20000
        )

        player_dict = {p.player_id: p for p in player_pool}

        # Build current lineup
        current_lineup = []
        for player_id in request.current_lineup:
            if player_id in player_dict:
                current_lineup.append(player_dict[player_id])

        # Apply adjustments
        adjusted_lineup = current_lineup.copy()

        # Remove player if specified
        if request.remove_player_id:
            adjusted_lineup = [
                p for p in adjusted_lineup if p.player_id != request.remove_player_id
            ]

        # Add player if specified
        if request.add_player_id and request.add_player_id in player_dict:
            new_player = player_dict[request.add_player_id]
            # Check if player is already in lineup
            if not any(p.player_id == request.add_player_id for p in adjusted_lineup):
                adjusted_lineup.append(new_player)

        # Validate adjusted lineup
        constraints = LineupConstraints()
        violations = builder.validate_lineup(adjusted_lineup, constraints)

        # If lineup is invalid, try to auto-correct
        if violations and len(adjusted_lineup) < 9:
            # Try to fill missing positions
            position_counts = {}
            for player in adjusted_lineup:
                pos = player.position
                position_counts[pos] = position_counts.get(pos, 0) + 1

            # Find missing positions
            used_player_ids = {p.player_id for p in adjusted_lineup}
            remaining_salary = constraints.salary_cap - sum(p.salary for p in adjusted_lineup)

            for pos, required in constraints.positions.items():
                current_count = position_counts.get(pos, 0)
                if pos == "FLEX":
                    # FLEX calculation is more complex
                    continue

                needed = required - current_count
                if needed > 0:
                    # Find available players for this position within salary
                    available = [
                        p
                        for p in player_pool
                        if (
                            p.position == pos
                            and p.player_id not in used_player_ids
                            and p.salary <= remaining_salary
                        )
                    ]

                    if available:
                        # Add best value player
                        best_player = max(available, key=lambda x: x.value)
                        adjusted_lineup.append(best_player)
                        used_player_ids.add(best_player.player_id)
                        remaining_salary -= best_player.salary
                        break

        # Convert to optimization result format
        total_salary = sum(p.salary for p in adjusted_lineup)
        projected_points = sum(p.projected_points for p in adjusted_lineup)
        lineup_value = projected_points / (total_salary / 1000) if total_salary > 0 else 0

        # Re-validate final lineup
        final_violations = builder.validate_lineup(adjusted_lineup, constraints)

        lineup_players = [
            PlayerProjectionResponse(
                player_id=p.player_id,
                name=p.name,
                position=p.position,
                team=p.team_abbr,
                salary=p.salary,
                projected_points=p.projected_points,
                floor=p.floor,
                ceiling=p.ceiling,
                value=p.value,
            )
            for p in adjusted_lineup
        ]

        return LineupResponse(
            lineup=lineup_players,
            total_salary=total_salary,
            projected_points=projected_points,
            lineup_value=lineup_value,
            optimization_status="adjusted",
            constraint_violations=final_violations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lineup adjustment failed: {e!s}") from e
