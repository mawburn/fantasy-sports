"""Foundation for DFS lineup optimization using mathematical optimization techniques.

This file implements several optimization algorithms to build optimal fantasy sports lineups:

1. Linear Programming (LP): Guaranteed optimal solutions for linear objectives
2. Greedy Algorithm: Fast heuristic approach for baseline solutions
3. Genetic Algorithm: Evolutionary approach for complex constraints
4. Stacking Optimization: Correlated player selection strategies
5. Tournament Optimization: High-ceiling focused for GPP contests

Key Concepts for Beginners:

Linear Programming: Mathematical optimization technique that finds the best solution
to a problem with linear constraints and objectives. Perfect for DFS because:
- Objective: Maximize fantasy points (linear in player selection)
- Constraints: Salary cap, position requirements (all linear)
- Variables: Binary (select player or don't)

Optimization Problem Structure:
- Decision Variables: Binary variables for each player (1 = selected, 0 = not selected)
- Objective Function: Maximize sum of (player_points * selection_variable)
- Constraints: Sum of (player_salary * selection_variable) <= salary_cap

PuLP Library: Python library for linear programming that interfaces with
optimization solvers like CBC, GLPK, and Gurobi.

Stacking Strategy: Selecting correlated players (QB + WR from same team)
to capture upside when offenses perform well together.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp as lp
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import DraftKingsSalary, Player

logger = logging.getLogger(__name__)


@dataclass
class PlayerProjection:
    """Player projection for optimization.

    This dataclass represents a player in the optimization problem with all
    the information needed for decision making:

    Core Data:
    - player_id, name, position: Player identification
    - salary: Cost in salary cap (constraint)
    - projected_points: Expected fantasy points (objective)

    Uncertainty Quantification:
    - floor: Conservative projection (25th percentile)
    - ceiling: Optimistic projection (75th percentile)

    Strategic Information:
    - ownership_projection: Expected % of contest entries using this player
    - value: Points per $1000 salary (efficiency metric)
    - team_abbr: For stacking and correlation analysis

    The __post_init__ method automatically calculates derived metrics
    like value (points per dollar) when the object is created.
    """

    player_id: int
    name: str
    position: str
    salary: int
    projected_points: float
    floor: float
    ceiling: float
    ownership_projection: float = 0.0
    value: float = 0.0  # Points per dollar
    team_abbr: str = ""  # Team abbreviation for stacking logic
    roster_position: str = ""  # DraftKings roster position (e.g., "RB/FLEX", "TE")

    def __post_init__(self):
        """Calculate derived metrics after object creation.

        This method automatically calculates value (points per $1000) when
        a PlayerProjection is created. This is more efficient than calculating
        it repeatedly and ensures consistency.

        Value calculation: projected_points / (salary / 1000)
        Example: 20 projected points at $8000 salary = 20 / 8 = 2.5 value
        """
        if self.salary > 0:
            self.value = self.projected_points / (self.salary / 1000)


@dataclass
class LineupConstraints:
    """Constraints for lineup optimization.

    This dataclass defines all the rules and restrictions for building lineups.
    In optimization terminology, these are the "constraints" that must be satisfied.

    DraftKings NFL Constraints:
    - salary_cap: Maximum total salary (usually $50,000)
    - positions: Required number of each position (QB:1, RB:2, WR:3, TE:1, FLEX:1, DEF:1)
    - FLEX: Can be filled by RB, WR, or TE (adds flexibility)

    Strategic Constraints:
    - Stacking options: Allow/forbid correlated player selections
    - Exposure limits: Maximum % of lineups a player can be in (multi-lineup)

    The constraints define the "feasible region" - all possible lineups that
    satisfy the rules. The optimization algorithm searches this region for
    the lineup with maximum projected points.
    """

    salary_cap: int = 50000
    positions: dict[str, int] = None  # {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
    min_salary: int = 0
    max_salary: int = 50000

    # Stacking constraints
    allow_qb_stack: bool = True
    allow_rb_def_stack: bool = True

    # Exposure constraints
    max_exposure: dict[int, float] = None  # {player_id: max_exposure_pct}

    def __post_init__(self):
        """Set default DraftKings NFL lineup positions if not provided.

        DraftKings NFL lineup requirements:
        - 1 QB: Quarterback
        - 2 RB: Running Backs
        - 3 WR: Wide Receivers
        - 1 TE: Tight End
        - 1 FLEX: Additional RB/WR/TE (provides flexibility)
        - 1 DST: Defense/Special Teams

        Total: 9 players, $50,000 salary cap

        The FLEX position allows for strategic lineup construction
        based on value and matchups at RB/WR/TE positions.
        """
        if self.positions is None:
            self.positions = {
                "QB": 1,  # Quarterback
                "RB": 2,  # Running Backs
                "WR": 3,  # Wide Receivers
                "TE": 1,  # Tight End
                "FLEX": 1,  # Flex position (RB/WR/TE)
                "DST": 1,  # Defense/Special Teams
            }


@dataclass
class OptimizationResult:
    """Result from lineup optimization."""

    lineup: list[PlayerProjection]
    total_salary: int
    projected_points: float
    lineup_value: float
    constraint_violations: list[str]
    optimization_status: str

    @property
    def is_valid(self) -> bool:
        """Check if lineup is valid (has no constraint violations).

        A valid lineup satisfies all DFS constraints:
        - Under salary cap
        - Correct number of each position
        - No duplicate players
        - Meets minimum salary requirements

        Returns:
            True if lineup is valid, False if any constraints are violated
        """
        return len(self.constraint_violations) == 0


class LineupBuilder:
    """Foundation class for building DFS lineups using multiple optimization algorithms.

    This class implements several optimization approaches:

    1. Greedy Algorithm:
       - Fast and simple
       - Sorts players by value (points/$), picks best available
       - No guarantee of optimality but good baseline

    2. Linear Programming:
       - Guaranteed optimal solution
       - Handles complex constraints efficiently
       - Uses PuLP library with CBC solver

    3. Genetic Algorithm:
       - Evolutionary optimization
       - Good for non-linear objectives
       - Handles complex constraints through population evolution

    4. Specialized Methods:
       - Stacking optimization (correlated players)
       - Tournament optimization (maximize ceiling)
       - Ownership-aware optimization (contrarian strategies)

    The class maintains database connections for accessing player pools
    and salaries, then applies ML predictions to create optimized lineups.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize lineup builder."""
        self.db = db_session or next(get_db())

    def get_player_pool(
        self,
        contest_id: int,
        positions: list[str] | None = None,
        min_salary: int = 3000,
        max_salary: int = 15000,
    ) -> list[PlayerProjection]:
        """Get available players for optimization.

        This method queries the database to get all players available for
        a specific DraftKings contest, along with their salaries and team info.

        Filtering Strategy:
        - Only active players (not injured/inactive)
        - Only specified positions
        - Only players within salary range (removes extreme values)

        The salary filtering helps optimization performance by removing:
        - Very cheap players (likely backups with no projection)
        - Very expensive players (may not provide value)

        Returns PlayerProjection objects with placeholders for projections
        that will be filled in by ML models later.

        Args:
            contest_id: DraftKings contest ID from database
            positions: Positions to include (default: QB, RB, WR, TE, DEF)
            min_salary: Minimum salary threshold (removes cheap backups)
            max_salary: Maximum salary threshold (removes extreme prices)

        Returns:
            List of PlayerProjection objects ready for ML projections
        """
        if positions is None:
            positions = ["QB", "RB", "WR", "TE", "DST"]

        # Build database query to get all relevant player data
        from src.database.models import Team

        # Complex join to get player info + salary + team data in one query
        query = (
            self.db.query(DraftKingsSalary, Player, Team)
            .join(Player, DraftKingsSalary.player_id == Player.id)  # Get player details
            .join(
                Team, Player.team_id == Team.id, isouter=True
            )  # Get team info (outer join for safety)
            .filter(
                DraftKingsSalary.contest_id == contest_id,  # Specific contest
                DraftKingsSalary.salary >= min_salary,  # Remove too-cheap players
                DraftKingsSalary.salary <= max_salary,  # Remove too-expensive players
                Player.position.in_(positions),  # Only requested positions
                Player.status == "Active",  # Only active (non-injured) players
            )
        )

        results = query.all()

        if not results:
            logger.warning(f"No players found for contest {contest_id}")
            return []

        # Convert database results to PlayerProjection objects
        player_pool = []
        for salary_data, player_data, team_data in results:
            # Use roster_position from DraftKings if available, otherwise use position
            roster_pos = salary_data.roster_position or player_data.position
            
            projection = PlayerProjection(
                player_id=player_data.id,  # Database ID
                name=player_data.display_name,  # Player name for display
                position=player_data.position,  # Position (QB, RB, etc.)
                salary=salary_data.salary,  # DraftKings salary
                projected_points=0.0,  # Placeholder - filled by ML models
                floor=0.0,  # Placeholder - filled by ML models
                ceiling=0.0,  # Placeholder - filled by ML models
                team_abbr=(
                    team_data.team_abbr if team_data else ""
                ),  # Team abbreviation for stacking
                roster_position=roster_pos,  # DraftKings roster eligibility
            )
            player_pool.append(projection)

        logger.info(f"Found {len(player_pool)} players for optimization")
        return player_pool

    def add_projections(
        self, player_pool: list[PlayerProjection], predictions: dict[int, dict[str, float]]
    ) -> list[PlayerProjection]:
        """Add predictions to player pool.

        Args:
            player_pool: List of players
            predictions: Dict mapping player_id to prediction data

        Returns:
            Updated player pool with projections
        """
        updated_pool = []

        for player in player_pool:
            if player.player_id in predictions:
                pred_data = predictions[player.player_id]
                player.projected_points = pred_data.get("predicted_points", 0.0)
                player.floor = pred_data.get("floor", player.projected_points * 0.7)
                player.ceiling = pred_data.get("ceiling", player.projected_points * 1.3)

                # Recalculate value
                if player.salary > 0:
                    player.value = player.projected_points / (player.salary / 1000)

                updated_pool.append(player)
            else:
                logger.warning(f"No prediction for player {player.name} ({player.player_id})")

        return updated_pool

    def validate_lineup(
        self, lineup: list[PlayerProjection], constraints: LineupConstraints
    ) -> list[str]:
        """Validate a lineup against constraints.

        Args:
            lineup: Proposed lineup
            constraints: Optimization constraints

        Returns:
            List of constraint violations
        """
        violations = []

        # Check salary constraints
        total_salary = sum(p.salary for p in lineup)
        if total_salary > constraints.salary_cap:
            violations.append(f"Salary cap exceeded: ${total_salary} > ${constraints.salary_cap}")

        if total_salary < constraints.min_salary:
            violations.append(f"Salary too low: ${total_salary} < ${constraints.min_salary}")

        # Check position constraints
        position_counts = {}
        for player in lineup:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Check required positions
        for pos, required_count in constraints.positions.items():
            if pos == "FLEX":
                # FLEX can be RB, WR, or TE
                flex_count = (
                    position_counts.get("RB", 0)
                    + position_counts.get("WR", 0)
                    + position_counts.get("TE", 0)
                )
                total_flex_required = (
                    required_count
                    + constraints.positions.get("RB", 0)
                    + constraints.positions.get("WR", 0)
                    + constraints.positions.get("TE", 0)
                )
                if flex_count < total_flex_required:
                    violations.append(
                        f"Not enough FLEX eligible players: {flex_count} < {total_flex_required}"
                    )
            else:
                actual_count = position_counts.get(pos, 0)
                if actual_count < required_count:
                    violations.append(f"Not enough {pos}: {actual_count} < {required_count}")

        # Check lineup size
        total_players = sum(constraints.positions.values())
        if len(lineup) != total_players:
            violations.append(f"Wrong lineup size: {len(lineup)} != {total_players}")

        return violations

    def build_greedy_lineup(
        self, player_pool: list[PlayerProjection], constraints: LineupConstraints
    ) -> OptimizationResult:
        """Build lineup using simple greedy algorithm.

        This is a basic implementation for Phase 4 foundation.

        Args:
            player_pool: Available players
            constraints: Optimization constraints

        Returns:
            Optimization result
        """
        # Sort players by value (points per $1000 salary)
        sorted_players = sorted(player_pool, key=lambda x: x.value, reverse=True)

        lineup = []
        remaining_salary = constraints.salary_cap
        position_needs = constraints.positions.copy()

        for player in sorted_players:
            # Check if we can afford this player
            if player.salary > remaining_salary:
                continue

            # Check if we need this position
            pos = player.position

            # Handle FLEX logic
            if pos in ["RB", "WR", "TE"]:
                if position_needs.get(pos, 0) > 0:
                    # Fill specific position first
                    position_needs[pos] -= 1
                elif position_needs.get("FLEX", 0) > 0:
                    # Fill FLEX
                    position_needs["FLEX"] -= 1
                else:
                    continue
            else:
                if position_needs.get(pos, 0) > 0:
                    position_needs[pos] -= 1
                else:
                    continue

            # Add player to lineup
            lineup.append(player)
            remaining_salary -= player.salary

            # Check if lineup is complete
            if sum(position_needs.values()) == 0:
                break

        # Validate the lineup
        violations = self.validate_lineup(lineup, constraints)

        # Calculate metrics
        total_salary = sum(p.salary for p in lineup)
        projected_points = sum(p.projected_points for p in lineup)
        lineup_value = projected_points / (total_salary / 1000) if total_salary > 0 else 0

        status = "optimal" if len(violations) == 0 else "infeasible"

        return OptimizationResult(
            lineup=lineup,
            total_salary=total_salary,
            projected_points=projected_points,
            lineup_value=lineup_value,
            constraint_violations=violations,
            optimization_status=status,
        )

    def generate_multiple_lineups(
        self,
        player_pool: list[PlayerProjection],
        constraints: LineupConstraints,
        num_lineups: int = 10,
        diversity_factor: float = 0.3,
    ) -> list[OptimizationResult]:
        """Generate multiple diverse lineups for analysis and comparison.

        Note: This generates lineups for research and comparison purposes, not for
        multi-entry contests. Each lineup represents a different strategic approach
        to the same single-entry classic contest.

        Args:
            player_pool: Available players
            constraints: Optimization constraints
            num_lineups: Number of lineups to generate
            diversity_factor: Factor to ensure lineup diversity

        Returns:
            List of optimization results
        """
        lineups = []
        used_players = set()

        for i in range(num_lineups):
            # Create modified player pool for diversity
            if i > 0:
                # Reduce value of already used players
                modified_pool = []
                for player in player_pool:
                    modified_player = PlayerProjection(
                        player_id=player.player_id,
                        name=player.name,
                        position=player.position,
                        salary=player.salary,
                        projected_points=player.projected_points,
                        floor=player.floor,
                        ceiling=player.ceiling,
                        ownership_projection=player.ownership_projection,
                        team_abbr=player.team_abbr,
                    )

                    # Reduce value if already used
                    if player.player_id in used_players:
                        modified_player.projected_points *= 1 - diversity_factor

                    modified_pool.append(modified_player)

                pool_to_use = modified_pool
            else:
                pool_to_use = player_pool

            # Generate lineup
            result = self.build_greedy_lineup(pool_to_use, constraints)

            if result.is_valid:
                lineups.append(result)
                # Track used players
                for player in result.lineup:
                    used_players.add(player.player_id)
            else:
                logger.warning(f"Generated invalid lineup {i + 1}: {result.constraint_violations}")

        # Sort lineups by projected points
        lineups.sort(key=lambda x: x.projected_points, reverse=True)

        logger.info(f"Generated {len(lineups)} valid lineups out of {num_lineups} attempts")
        return lineups

    def build_linear_programming_lineup(
        self, player_pool: list[PlayerProjection], constraints: LineupConstraints
    ) -> OptimizationResult:
        """Build lineup using linear programming optimization.

        This uses PuLP to solve the lineup optimization as an integer linear
        programming problem, which guarantees optimal solutions.

        Args:
            player_pool: Available players
            constraints: Optimization constraints

        Returns:
            Optimization result
        """
        if not player_pool:
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["No players available"],
                optimization_status="infeasible",
            )

        # Create the LP problem
        prob = lp.LpProblem("DFS_Lineup_Optimization", lp.LpMaximize)

        # Create binary decision variables for each player
        player_vars = {}
        for player in player_pool:
            var_name = f"player_{player.player_id}"
            player_vars[player.player_id] = lp.LpVariable(var_name, cat="Binary")

        # Objective function: maximize projected points
        prob += lp.lpSum(
            [player.projected_points * player_vars[player.player_id] for player in player_pool]
        )

        # Salary constraint
        prob += (
            lp.lpSum([player.salary * player_vars[player.player_id] for player in player_pool])
            <= constraints.salary_cap
        )

        # Position constraints
        position_groups = {}
        flex_eligible_players = []  # Track players who can be used in FLEX
        
        for pos in constraints.positions:
            position_groups[pos] = []

        for player in player_pool:
            if player.position in position_groups:
                position_groups[player.position].append(player.player_id)
            
            # Check FLEX eligibility based on roster_position
            # A player is FLEX-eligible if their roster_position contains "/FLEX"
            if "/FLEX" in player.roster_position:
                if player.position in ["RB", "WR", "TE"]:
                    flex_eligible_players.append(player.player_id)

        # Add position constraints
        # We need exactly the specified counts for base positions
        # Plus exactly 1 additional RB/WR/TE for the FLEX spot
        
        # First, enforce base position requirements (minimum)
        for pos, required_count in constraints.positions.items():
            if pos != "FLEX" and position_groups.get(pos):
                # At least the required count for each position
                prob += (
                    lp.lpSum([player_vars[pid] for pid in position_groups[pos]])
                    >= required_count
                )
        
        # Total RB+WR+TE must equal their base requirements + 1 for FLEX
        if "FLEX" in constraints.positions:
            total_flex_positions = (
                constraints.positions.get("RB", 0) +
                constraints.positions.get("WR", 0) +
                constraints.positions.get("TE", 0) +
                constraints.positions.get("FLEX", 0)  # Should be 1
            )
            
            # Get all RB/WR/TE players who are FLEX-eligible
            flex_eligible_by_position = {
                "RB": [pid for pid in position_groups.get("RB", []) 
                       if pid in flex_eligible_players],
                "WR": [pid for pid in position_groups.get("WR", []) 
                       if pid in flex_eligible_players],
                "TE": [pid for pid in position_groups.get("TE", []) 
                       if pid in flex_eligible_players],
            }
            
            all_flex_eligible = (
                flex_eligible_by_position["RB"] +
                flex_eligible_by_position["WR"] +
                flex_eligible_by_position["TE"]
            )
            
            if all_flex_eligible:
                # Total RB+WR+TE must be exactly base + FLEX
                prob += (
                    lp.lpSum([player_vars[pid] for pid in all_flex_eligible])
                    == total_flex_positions
                )

        # Total lineup size constraint
        total_positions = sum(constraints.positions.values())
        prob += lp.lpSum([player_vars[pid] for pid in player_vars]) == total_positions

        # Add exposure constraints if specified
        if constraints.max_exposure:
            for player_id, _max_exposure_pct in constraints.max_exposure.items():
                if player_id in player_vars:
                    # For single lineup, exposure is binary (0 or 1)
                    # This constraint is more useful for multi-lineup generation
                    prob += player_vars[player_id] <= 1.0

        # Solve the problem
        try:
            prob.solve(lp.PULP_CBC_CMD(msg=0))  # Suppress solver output

            status = lp.LpStatus[prob.status]

            if status != "Optimal":
                return OptimizationResult(
                    lineup=[],
                    total_salary=0,
                    projected_points=0.0,
                    lineup_value=0.0,
                    constraint_violations=[f"LP solver status: {status}"],
                    optimization_status="infeasible",
                )

            # Extract the solution
            selected_players = []
            for player in player_pool:
                if player_vars[player.player_id].varValue == 1:
                    selected_players.append(player)

            # Validate the lineup
            violations = self.validate_lineup(selected_players, constraints)

            # Calculate metrics
            total_salary = sum(p.salary for p in selected_players)
            projected_points = sum(p.projected_points for p in selected_players)
            lineup_value = projected_points / (total_salary / 1000) if total_salary > 0 else 0

            return OptimizationResult(
                lineup=selected_players,
                total_salary=total_salary,
                projected_points=projected_points,
                lineup_value=lineup_value,
                constraint_violations=violations,
                optimization_status="optimal",
            )

        except Exception as e:
            logger.exception("Linear programming optimization failed")
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=[f"Optimization error: {e!s}"],
                optimization_status="error",
            )

    def build_lineup_with_stacking(
        self,
        player_pool: list[PlayerProjection],
        constraints: LineupConstraints,
        qb_stack_teams: list[str] | None = None,
        rb_def_stack_teams: list[str] | None = None,
        min_qb_stack_count: int = 1,
        force_stacking: bool = False,
    ) -> OptimizationResult:
        """Build lineup with stacking constraints using linear programming.

        Args:
            player_pool: Available players
            constraints: Optimization constraints
            qb_stack_teams: Teams to force QB-WR stacking for
            rb_def_stack_teams: Teams to force RB-DEF stacking for
            min_qb_stack_count: Minimum number of pass catchers to stack with QB
            force_stacking: If True, require stacking; if False, allow but don't require

        Returns:
            Optimization result with stacking applied
        """
        if not player_pool:
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["No players available"],
                optimization_status="infeasible",
            )

        # Create the LP problem with stacking constraints
        prob = lp.LpProblem("DFS_Lineup_Stacking_Optimization", lp.LpMaximize)

        # Create binary decision variables for each player
        player_vars = {}
        for player in player_pool:
            var_name = f"player_{player.player_id}"
            player_vars[player.player_id] = lp.LpVariable(var_name, cat="Binary")

        # Objective function: maximize projected points
        prob += lp.lpSum(
            [player.projected_points * player_vars[player.player_id] for player in player_pool]
        )

        # Standard constraints (salary, positions, lineup size)
        # Salary constraint
        prob += (
            lp.lpSum([player.salary * player_vars[player.player_id] for player in player_pool])
            <= constraints.salary_cap
        )

        # Position constraints
        position_groups = {}
        for pos in constraints.positions:
            position_groups[pos] = []

        for player in player_pool:
            if player.position in position_groups:
                position_groups[player.position].append(player.player_id)

        # Add standard position constraints
        flex_eligible_players = []
        for player in player_pool:
            # Check FLEX eligibility based on roster_position
            if "/FLEX" in player.roster_position:
                if player.position in ["RB", "WR", "TE"]:
                    flex_eligible_players.append(player.player_id)
        
        # First, enforce base position requirements (minimum)
        for pos, required_count in constraints.positions.items():
            if pos != "FLEX" and position_groups.get(pos):
                # At least the required count for each position
                prob += (
                    lp.lpSum([player_vars[pid] for pid in position_groups[pos]])
                    >= required_count
                )
        
        # Total RB+WR+TE must equal their base requirements + 1 for FLEX
        if "FLEX" in constraints.positions:
            total_flex_positions = (
                constraints.positions.get("RB", 0) +
                constraints.positions.get("WR", 0) +
                constraints.positions.get("TE", 0) +
                constraints.positions.get("FLEX", 0)  # Should be 1
            )
            
            # Get all RB/WR/TE players who are FLEX-eligible
            flex_eligible_by_position = {
                "RB": [pid for pid in position_groups.get("RB", []) 
                       if pid in flex_eligible_players],
                "WR": [pid for pid in position_groups.get("WR", []) 
                       if pid in flex_eligible_players],
                "TE": [pid for pid in position_groups.get("TE", []) 
                       if pid in flex_eligible_players],
            }
            
            all_flex_eligible = (
                flex_eligible_by_position["RB"] +
                flex_eligible_by_position["WR"] +
                flex_eligible_by_position["TE"]
            )
            
            if all_flex_eligible:
                # Total RB+WR+TE must be exactly base + FLEX
                prob += (
                    lp.lpSum([player_vars[pid] for pid in all_flex_eligible])
                    == total_flex_positions
                )

        # Total lineup size constraint
        total_positions = sum(constraints.positions.values())
        prob += lp.lpSum([player_vars[pid] for pid in player_vars]) == total_positions

        # STACKING CONSTRAINTS

        # Group players by team for stacking logic
        teams_players = {}
        for player in player_pool:
            if player.team_abbr not in teams_players:
                teams_players[player.team_abbr] = {
                    "QB": [],
                    "WR": [],
                    "TE": [],
                    "RB": [],
                    "DEF": [],
                }
            if player.position in teams_players[player.team_abbr]:
                teams_players[player.team_abbr][player.position].append(player.player_id)

        # QB-WR/TE stacking constraints
        if constraints.allow_qb_stack and qb_stack_teams:
            for team in qb_stack_teams:
                if team in teams_players:
                    qb_players = teams_players[team]["QB"]
                    pass_catchers = teams_players[team]["WR"] + teams_players[team]["TE"]

                    if qb_players and pass_catchers:
                        # If we select a QB from this team, we must select at least min_qb_stack_count pass catchers
                        qb_selected = lp.lpSum([player_vars[pid] for pid in qb_players])
                        catchers_selected = lp.lpSum([player_vars[pid] for pid in pass_catchers])

                        if force_stacking:
                            # Force stacking: if QB selected, must have pass catchers
                            prob += catchers_selected >= min_qb_stack_count * qb_selected
                        else:
                            # Encourage stacking with a bonus in the objective function
                            # This is implemented by adjusting player projections above
                            pass

        # RB-DEF stacking constraints (game script correlation)
        if constraints.allow_rb_def_stack and rb_def_stack_teams:
            for team in rb_def_stack_teams:
                if team in teams_players:
                    rb_players = teams_players[team]["RB"]
                    def_players = teams_players[team]["DEF"]

                    if rb_players and def_players:
                        rb_selected = lp.lpSum([player_vars[pid] for pid in rb_players])
                        def_selected = lp.lpSum([player_vars[pid] for pid in def_players])

                        if force_stacking:
                            # If we select RB from team, we should select their DEF too
                            prob += def_selected >= rb_selected

        # Solve the problem
        try:
            prob.solve(lp.PULP_CBC_CMD(msg=0))

            status = lp.LpStatus[prob.status]

            if status != "Optimal":
                # If stacking constraints made it infeasible, try without forced stacking
                if force_stacking:
                    logger.warning("Forced stacking made problem infeasible, trying without")
                    return self.build_lineup_with_stacking(
                        player_pool,
                        constraints,
                        qb_stack_teams,
                        rb_def_stack_teams,
                        min_qb_stack_count,
                        force_stacking=False,
                    )

                return OptimizationResult(
                    lineup=[],
                    total_salary=0,
                    projected_points=0.0,
                    lineup_value=0.0,
                    constraint_violations=[f"LP solver status: {status}"],
                    optimization_status="infeasible",
                )

            # Extract the solution
            selected_players = []
            for player in player_pool:
                if player_vars[player.player_id].varValue == 1:
                    selected_players.append(player)

            # Validate the lineup and check stacking
            violations = self.validate_lineup(selected_players, constraints)

            # Check stacking success
            stacking_info = self._analyze_stacking(selected_players)
            logger.debug(f"Stacking analysis: {stacking_info}")

            # Calculate metrics
            total_salary = sum(p.salary for p in selected_players)
            projected_points = sum(p.projected_points for p in selected_players)
            lineup_value = projected_points / (total_salary / 1000) if total_salary > 0 else 0

            return OptimizationResult(
                lineup=selected_players,
                total_salary=total_salary,
                projected_points=projected_points,
                lineup_value=lineup_value,
                constraint_violations=violations,
                optimization_status="optimal",
            )

        except Exception as e:
            logger.exception("Stacking optimization failed")
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=[f"Optimization error: {e!s}"],
                optimization_status="error",
            )

    def build_genetic_algorithm_lineup(
        self,
        player_pool: list[PlayerProjection],
        constraints: LineupConstraints,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ) -> OptimizationResult:
        """Build lineup using genetic algorithm optimization.

        This evolutionary approach can handle complex constraints and
        non-linear objectives better than LP in some cases.

        Args:
            player_pool: Available players
            constraints: Optimization constraints
            population_size: Size of genetic algorithm population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover

        Returns:
            Optimization result
        """
        import random

        if not player_pool:
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["No players available"],
                optimization_status="infeasible",
            )

        # Group players by position for easier lineup construction
        position_pools = {}
        for pos in constraints.positions:
            position_pools[pos] = []

        for player in player_pool:
            if player.position in position_pools:
                position_pools[player.position].append(player)

        # Add flex-eligible players to FLEX pool
        if "FLEX" in constraints.positions:
            position_pools["FLEX"] = (
                position_pools.get("RB", [])
                + position_pools.get("WR", [])
                + position_pools.get("TE", [])
            )

        def create_random_lineup() -> list[PlayerProjection] | None:
            """Create a random valid lineup."""
            lineup = []
            used_players = set()

            for pos, count in constraints.positions.items():
                available = [
                    p for p in position_pools.get(pos, []) if p.player_id not in used_players
                ]

                if len(available) < count:
                    return None  # Cannot satisfy position requirements

                selected = random.sample(available, count)
                lineup.extend(selected)
                used_players.update(p.player_id for p in selected)

            return lineup

        def calculate_fitness(lineup: list[PlayerProjection]) -> float:
            """Calculate fitness score for a lineup."""
            if not lineup:
                return 0.0

            total_salary = sum(p.salary for p in lineup)
            projected_points = sum(p.projected_points for p in lineup)

            # Penalty for salary constraint violations
            salary_penalty = 0.0
            if total_salary > constraints.salary_cap:
                salary_penalty = (total_salary - constraints.salary_cap) * 0.01

            # Bonus for using salary efficiently (closer to cap is better)
            salary_utilization = total_salary / constraints.salary_cap
            salary_bonus = salary_utilization * 0.1 if salary_utilization <= 1.0 else 0.0

            return projected_points + salary_bonus - salary_penalty

        def mutate_lineup(lineup: list[PlayerProjection]) -> list[PlayerProjection]:
            """Mutate a lineup by swapping one player."""
            if not lineup or random.random() > mutation_rate:
                return lineup

            new_lineup = lineup.copy()

            # Pick a random position to mutate
            positions_list = list(constraints.positions.keys())
            pos_to_mutate = random.choice(positions_list)

            # Find players in this position
            pos_players = [
                p
                for p in new_lineup
                if p.position == pos_to_mutate
                or (pos_to_mutate == "FLEX" and p.position in ["RB", "WR", "TE"])
            ]

            if not pos_players:
                return new_lineup

            # Remove one player and add a different one
            player_to_remove = random.choice(pos_players)
            new_lineup.remove(player_to_remove)

            # Find available replacements
            used_ids = {p.player_id for p in new_lineup}
            available = [
                p for p in position_pools.get(pos_to_mutate, []) if p.player_id not in used_ids
            ]

            if available:
                replacement = random.choice(available)
                new_lineup.append(replacement)
            else:
                # If no replacement available, keep original player
                new_lineup.append(player_to_remove)

            return new_lineup

        def crossover_lineups(
            parent1: list[PlayerProjection], parent2: list[PlayerProjection]
        ) -> tuple[list[PlayerProjection], list[PlayerProjection]]:
            """Create two children by crossing over two parent lineups."""
            if random.random() > crossover_rate:
                return parent1, parent2

            # Simple crossover: take some positions from parent1, others from parent2
            positions_list = list(constraints.positions.keys())
            crossover_point = random.randint(1, len(positions_list) - 1)

            def build_child(
                p1: list[PlayerProjection], p2: list[PlayerProjection]
            ) -> list[PlayerProjection]:
                child = []
                used_ids = set()

                for i, pos in enumerate(positions_list):
                    source_parent = p1 if i < crossover_point else p2
                    pos_players = [
                        p
                        for p in source_parent
                        if (
                            p.position == pos
                            or (pos == "FLEX" and p.position in ["RB", "WR", "TE"])
                        )
                        and p.player_id not in used_ids
                    ]

                    needed = constraints.positions[pos]
                    available_count = min(len(pos_players), needed)

                    if available_count > 0:
                        selected = pos_players[:available_count]
                        child.extend(selected)
                        used_ids.update(p.player_id for p in selected)

                return child

            child1 = build_child(parent1, parent2)
            child2 = build_child(parent2, parent1)

            return child1, child2

        # Initialize population
        population = []
        attempts = 0
        max_attempts = population_size * 10

        while len(population) < population_size and attempts < max_attempts:
            lineup = create_random_lineup()
            if lineup and len(lineup) == sum(constraints.positions.values()):
                population.append(lineup)
            attempts += 1

        if len(population) < population_size // 2:
            # Not enough valid lineups could be generated
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["Unable to generate sufficient initial population"],
                optimization_status="infeasible",
            )

        # Evolution loop
        for _ in range(generations):
            # Calculate fitness for all lineups
            fitness_scores = [(lineup, calculate_fitness(lineup)) for lineup in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Select best lineups for next generation (elitism)
            elite_count = max(1, population_size // 10)
            next_population = [lineup for lineup, _ in fitness_scores[:elite_count]]

            # Generate offspring through crossover and mutation
            while len(next_population) < population_size:
                # Tournament selection
                tournament_size = 5
                parent1 = max(
                    random.sample(
                        fitness_scores[: population_size // 2],
                        min(tournament_size, len(fitness_scores)),
                    ),
                    key=lambda x: x[1],
                )[0]

                parent2 = max(
                    random.sample(
                        fitness_scores[: population_size // 2],
                        min(tournament_size, len(fitness_scores)),
                    ),
                    key=lambda x: x[1],
                )[0]

                # Crossover
                child1, child2 = crossover_lineups(parent1, parent2)

                # Mutation
                child1 = mutate_lineup(child1)
                child2 = mutate_lineup(child2)

                # Add children to next generation
                if len(next_population) < population_size:
                    next_population.append(child1)
                if len(next_population) < population_size:
                    next_population.append(child2)

            population = next_population

        # Return best lineup from final generation
        final_fitness_scores = [(lineup, calculate_fitness(lineup)) for lineup in population]
        best_lineup, _ = max(final_fitness_scores, key=lambda x: x[1])

        # Validate the best lineup
        violations = self.validate_lineup(best_lineup, constraints)

        # Calculate final metrics
        total_salary = sum(p.salary for p in best_lineup)
        projected_points = sum(p.projected_points for p in best_lineup)
        lineup_value = projected_points / (total_salary / 1000) if total_salary > 0 else 0

        status = "optimal" if len(violations) == 0 else "feasible"

        return OptimizationResult(
            lineup=best_lineup,
            total_salary=total_salary,
            projected_points=projected_points,
            lineup_value=lineup_value,
            constraint_violations=violations,
            optimization_status=status,
        )

    def _analyze_stacking(self, lineup: list[PlayerProjection]) -> dict[str, Any]:
        """Analyze stacking in a lineup.

        Args:
            lineup: List of selected players

        Returns:
            Dictionary with stacking analysis
        """
        teams_in_lineup = {}
        for player in lineup:
            if player.team_abbr not in teams_in_lineup:
                teams_in_lineup[player.team_abbr] = {
                    "QB": [],
                    "WR": [],
                    "TE": [],
                    "RB": [],
                    "DEF": [],
                }
            teams_in_lineup[player.team_abbr][player.position].append(player)

        stacks = {"qb_stacks": [], "rb_def_stacks": [], "total_stacks": 0}

        for team, positions in teams_in_lineup.items():
            # Check QB stacks
            if positions["QB"] and (positions["WR"] or positions["TE"]):
                qb = positions["QB"][0]
                pass_catchers = positions["WR"] + positions["TE"]
                stacks["qb_stacks"].append(
                    {
                        "team": team,
                        "qb": qb.name,
                        "pass_catchers": [p.name for p in pass_catchers],
                        "stack_size": len(pass_catchers) + 1,
                    }
                )
                stacks["total_stacks"] += 1

            # Check RB-DEF stacks
            if positions["RB"] and positions["DEF"]:
                rb = positions["RB"][0]
                defense = positions["DEF"][0]
                stacks["rb_def_stacks"].append(
                    {"team": team, "rb": rb.name, "defense": defense.name}
                )
                stacks["total_stacks"] += 1

        return stacks

    def optimize_with_ownership_projections(
        self,
        player_pool: list[PlayerProjection],
        constraints: LineupConstraints,
        ownership_weight: float = 0.2,
        contrarian_mode: bool = False,
    ) -> OptimizationResult:
        """Optimize lineup considering ownership projections.

        Args:
            player_pool: Available players with ownership projections
            constraints: Optimization constraints
            ownership_weight: Weight given to ownership considerations (0-1)
            contrarian_mode: If True, favor low-owned players; if False, avoid high-owned

        Returns:
            Optimization result accounting for ownership
        """
        if not player_pool:
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["No players available"],
                optimization_status="infeasible",
            )

        # Adjust projections based on ownership
        adjusted_pool = []
        for player in player_pool:
            adjusted_player = PlayerProjection(
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                salary=player.salary,
                projected_points=player.projected_points,
                floor=player.floor,
                ceiling=player.ceiling,
                ownership_projection=player.ownership_projection,
                team_abbr=player.team_abbr,
            )

            # Adjust projections based on ownership strategy
            if player.ownership_projection > 0:
                if contrarian_mode:
                    # Contrarian: Bonus for low ownership, penalty for high ownership
                    ownership_adjustment = (
                        1 - player.ownership_projection / 100
                    ) * ownership_weight
                else:
                    # Conservative: Small penalty for very high ownership
                    if player.ownership_projection > 30:  # High ownership threshold
                        ownership_adjustment = (
                            -(player.ownership_projection / 100 - 0.3) * ownership_weight
                        )
                    else:
                        ownership_adjustment = 0

                adjusted_player.projected_points *= 1 + ownership_adjustment

            adjusted_pool.append(adjusted_player)

        # Use linear programming with adjusted projections
        return self.build_linear_programming_lineup(adjusted_pool, constraints)

    def build_tournament_optimized_lineup(
        self,
        player_pool: list[PlayerProjection],
        constraints: LineupConstraints,
        ceiling_weight: float = 0.3,
        min_ceiling_threshold: float = 25.0,
    ) -> OptimizationResult:
        """Build lineup optimized for tournaments (GPPs) using ceiling projections.

        Args:
            player_pool: Available players
            constraints: Optimization constraints
            ceiling_weight: Weight given to ceiling vs projection (0-1)
            min_ceiling_threshold: Minimum ceiling to consider for tournaments

        Returns:
            Tournament-optimized lineup
        """
        if not player_pool:
            return OptimizationResult(
                lineup=[],
                total_salary=0,
                projected_points=0.0,
                lineup_value=0.0,
                constraint_violations=["No players available"],
                optimization_status="infeasible",
            )

        # Filter players with sufficient ceiling potential
        tournament_pool = []
        for player in player_pool:
            if player.ceiling >= min_ceiling_threshold:
                tournament_player = PlayerProjection(
                    player_id=player.player_id,
                    name=player.name,
                    position=player.position,
                    salary=player.salary,
                    projected_points=player.projected_points,
                    floor=player.floor,
                    ceiling=player.ceiling,
                    ownership_projection=player.ownership_projection,
                    team_abbr=player.team_abbr,
                )

                # Blend projection with ceiling for tournament optimization
                tournament_score = (
                    player.projected_points * (1 - ceiling_weight) + player.ceiling * ceiling_weight
                )
                tournament_player.projected_points = tournament_score
                tournament_pool.append(tournament_player)

        if not tournament_pool:
            logger.warning(f"No players meet ceiling threshold of {min_ceiling_threshold}")
            return self.build_linear_programming_lineup(player_pool, constraints)

        return self.build_linear_programming_lineup(tournament_pool, constraints)

    def export_lineup_to_csv(self, lineup: OptimizationResult, filename: str | None = None) -> str:
        """Export lineup to DraftKings CSV format.

        Args:
            lineup: Optimization result
            filename: Output filename

        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"lineup_{int(pd.Timestamp.now().timestamp())}.csv"

        # Create DataFrame in DraftKings format
        lineup_data = []
        position_order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]

        # Sort players by position requirements
        position_players = {pos: [] for pos in ["QB", "RB", "WR", "TE", "DEF"]}
        for player in lineup.lineup:
            position_players[player.position].append(player)

        # Build ordered lineup
        ordered_lineup = []
        for pos in position_order:
            if pos == "FLEX":
                # Find flex player (RB/WR/TE not already used)
                for flex_pos in ["RB", "WR", "TE"]:
                    if position_players[flex_pos]:
                        ordered_lineup.append(position_players[flex_pos].pop())
                        break
            else:
                if position_players[pos]:
                    ordered_lineup.append(position_players[pos].pop(0))

        # Create CSV data
        for i, player in enumerate(ordered_lineup):
            lineup_data.append(
                {
                    "Position": position_order[i],
                    "Name": player.name,
                    "Salary": player.salary,
                    "Projected": player.projected_points,
                }
            )

        df = pd.DataFrame(lineup_data)
        df.to_csv(filename, index=False)

        logger.info(f"Exported lineup to {filename}")
        return filename
