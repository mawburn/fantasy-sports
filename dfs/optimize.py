"""Simplified lineup optimization for DFS contests.

This module consolidates all optimization algorithms into a single file:
1. Linear programming with PuLP (guaranteed optimal solutions)
2. Greedy algorithm (fast baseline approach)
3. Stacking logic (QB-WR correlations, RB-DEF game script)
4. Multiple optimization strategies (cash games vs tournaments)
5. Constraint handling (salary cap, positions, exposure)

No complex classes or abstractions - just functions that build optimal lineups.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import pulp as lp
except ImportError:
    print("Warning: PuLP not available. Install with: pip install pulp")
    lp = None

logger = logging.getLogger(__name__)

# DraftKings NFL lineup requirements
DEFAULT_POSITIONS = {
    "QB": 1,    # Quarterback
    "RB": 2,    # Running Backs
    "WR": 3,    # Wide Receivers
    "TE": 1,    # Tight End
    "FLEX": 1,  # Flex position (RB/WR/TE)
    "DST": 1,   # Defense/Special Teams
}

DEFAULT_SALARY_CAP = 50000


@dataclass
class Player:
    """Simple player data structure for optimization."""
    player_id: int
    name: str
    position: str
    salary: int
    projected_points: float
    floor: float = 0.0
    ceiling: float = 0.0
    ownership_projection: float = 0.0
    team_abbr: str = ""
    roster_position: str = ""  # DraftKings roster eligibility (e.g., "RB/FLEX")
    injury_status: Optional[str] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        self.value = self.projected_points / (self.salary / 1000) if self.salary > 0 else 0


@dataclass
class LineupConstraints:
    """Constraints for lineup optimization."""
    salary_cap: int = DEFAULT_SALARY_CAP
    positions: Dict[str, int] = None
    min_salary: int = 0
    max_salary: int = DEFAULT_SALARY_CAP

    # Stacking constraints
    allow_qb_stack: bool = True
    allow_rb_def_stack: bool = True
    min_qb_stack_count: int = 1  # Minimum pass catchers to stack with QB

    # Exposure constraints (for multi-lineup generation)
    max_exposure: Dict[int, float] = None

    def __post_init__(self):
        """Set default positions if not provided."""
        if self.positions is None:
            self.positions = DEFAULT_POSITIONS.copy()


@dataclass
class OptimizationResult:
    """Result from lineup optimization."""
    lineup: List[Player]
    total_salary: int
    projected_points: float
    lineup_value: float
    constraint_violations: List[str]
    optimization_status: str
    stacking_info: Dict[str, Any] = None

    @property
    def is_valid(self) -> bool:
        """Check if lineup is valid (no constraint violations)."""
        return len(self.constraint_violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export."""
        return {
            'players': [
                {
                    'name': p.name,
                    'position': p.position,
                    'salary': p.salary,
                    'projected_points': p.projected_points,
                    'team': p.team_abbr
                }
                for p in self.lineup
            ],
            'total_salary': self.total_salary,
            'projected_points': self.projected_points,
            'lineup_value': self.lineup_value,
            'is_valid': self.is_valid,
            'stacking_info': self.stacking_info
        }


def validate_lineup(lineup: List[Player], constraints: LineupConstraints) -> List[str]:
    """Validate a lineup against constraints."""
    violations = []

    if not lineup:
        violations.append("Empty lineup")
        return violations

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
                position_counts.get("RB", 0) +
                position_counts.get("WR", 0) +
                position_counts.get("TE", 0)
            )
            total_flex_required = (
                required_count +
                constraints.positions.get("RB", 0) +
                constraints.positions.get("WR", 0) +
                constraints.positions.get("TE", 0)
            )
            if flex_count < total_flex_required:
                violations.append(f"Not enough FLEX eligible players: {flex_count} < {total_flex_required}")
        else:
            actual_count = position_counts.get(pos, 0)
            if actual_count < required_count:
                violations.append(f"Not enough {pos}: {actual_count} < {required_count}")

    # Check for duplicate players
    player_ids = [p.player_id for p in lineup]
    if len(player_ids) != len(set(player_ids)):
        violations.append("Duplicate players in lineup")

    # Check lineup size
    total_players = sum(constraints.positions.values())
    if len(lineup) != total_players:
        violations.append(f"Wrong lineup size: {len(lineup)} != {total_players}")

    return violations


def build_greedy_lineup(
    player_pool: List[Player],
    constraints: LineupConstraints
) -> OptimizationResult:
    """Build lineup using simple greedy algorithm (sorts by value)."""
    if not player_pool:
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=["No players available"], optimization_status="infeasible"
        )

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
    violations = validate_lineup(lineup, constraints)

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


def build_linear_programming_lineup(
    player_pool: List[Player],
    constraints: LineupConstraints
) -> OptimizationResult:
    """Build lineup using linear programming optimization (guaranteed optimal)."""
    if lp is None:
        logger.error("PuLP not available - falling back to greedy")
        return build_greedy_lineup(player_pool, constraints)

    if not player_pool:
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=["No players available"], optimization_status="infeasible"
        )

    # Create the LP problem
    prob = lp.LpProblem("DFS_Lineup_Optimization", lp.LpMaximize)

    # Create binary decision variables for each player
    player_vars = {}
    for player in player_pool:
        var_name = f"player_{player.player_id}"
        player_vars[player.player_id] = lp.LpVariable(var_name, cat="Binary")

    # Objective function: maximize projected points
    prob += lp.lpSum([
        player.projected_points * player_vars[player.player_id]
        for player in player_pool
    ])

    # Salary constraint
    prob += lp.lpSum([
        player.salary * player_vars[player.player_id]
        for player in player_pool
    ]) <= constraints.salary_cap

    # Position constraints
    position_groups = {}
    flex_eligible_players = []

    for pos in constraints.positions:
        position_groups[pos] = []

    for player in player_pool:
        if player.position in position_groups:
            position_groups[player.position].append(player.player_id)

        # Check FLEX eligibility - player is FLEX-eligible if roster_position contains "/FLEX"
        if "/FLEX" in player.roster_position or player.position in ["RB", "WR", "TE"]:
            flex_eligible_players.append(player.player_id)

    # Add position constraints
    # First, enforce base position requirements (minimum)
    for pos, required_count in constraints.positions.items():
        if pos != "FLEX" and position_groups.get(pos):
            prob += lp.lpSum([
                player_vars[pid] for pid in position_groups[pos]
            ]) >= required_count

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
            "RB": [pid for pid in position_groups.get("RB", []) if pid in flex_eligible_players],
            "WR": [pid for pid in position_groups.get("WR", []) if pid in flex_eligible_players],
            "TE": [pid for pid in position_groups.get("TE", []) if pid in flex_eligible_players],
        }

        all_flex_eligible = (
            flex_eligible_by_position["RB"] +
            flex_eligible_by_position["WR"] +
            flex_eligible_by_position["TE"]
        )

        if all_flex_eligible:
            prob += lp.lpSum([
                player_vars[pid] for pid in all_flex_eligible
            ]) == total_flex_positions

    # Total lineup size constraint
    total_positions = sum(constraints.positions.values())
    prob += lp.lpSum([player_vars[pid] for pid in player_vars]) == total_positions

    # Solve the problem
    try:
        prob.solve(lp.PULP_CBC_CMD(msg=0))  # Suppress solver output

        status = lp.LpStatus[prob.status]

        if status != "Optimal":
            return OptimizationResult(
                lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
                constraint_violations=[f"LP solver status: {status}"],
                optimization_status="infeasible"
            )

        # Extract the solution
        selected_players = []
        for player in player_pool:
            if player_vars[player.player_id].varValue == 1:
                selected_players.append(player)

        # Validate the lineup
        violations = validate_lineup(selected_players, constraints)

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
            optimization_status="optimal"
        )

    except Exception as e:
        logger.exception("Linear programming optimization failed")
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=[f"Optimization error: {str(e)}"],
            optimization_status="error"
        )


def analyze_stacking(lineup: List[Player]) -> Dict[str, Any]:
    """Analyze stacking in a lineup."""
    teams_in_lineup = {}
    for player in lineup:
        if player.team_abbr not in teams_in_lineup:
            teams_in_lineup[player.team_abbr] = {
                "QB": [], "WR": [], "TE": [], "RB": [], "DST": []
            }
        teams_in_lineup[player.team_abbr][player.position].append(player)

    stacks = {"qb_stacks": [], "rb_def_stacks": [], "total_stacks": 0}

    for team, positions in teams_in_lineup.items():
        # Check QB stacks
        if positions["QB"] and (positions["WR"] or positions["TE"]):
            qb = positions["QB"][0]
            pass_catchers = positions["WR"] + positions["TE"]
            stacks["qb_stacks"].append({
                "team": team,
                "qb": qb.name,
                "pass_catchers": [p.name for p in pass_catchers],
                "stack_size": len(pass_catchers) + 1,
            })
            stacks["total_stacks"] += 1

        # Check RB-DEF stacks
        if positions["RB"] and positions["DST"]:
            rb = positions["RB"][0]
            defense = positions["DST"][0]
            stacks["rb_def_stacks"].append({
                "team": team,
                "rb": rb.name,
                "defense": defense.name
            })
            stacks["total_stacks"] += 1

    return stacks


def build_stacking_lineup(
    player_pool: List[Player],
    constraints: LineupConstraints,
    qb_stack_teams: List[str] = None,
    rb_def_stack_teams: List[str] = None,
    force_stacking: bool = False
) -> OptimizationResult:
    """Build lineup with stacking constraints using linear programming."""
    if lp is None:
        logger.error("PuLP not available for stacking optimization")
        return build_greedy_lineup(player_pool, constraints)

    if not player_pool:
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=["No players available"], optimization_status="infeasible"
        )

    # Create the LP problem with stacking constraints
    prob = lp.LpProblem("DFS_Lineup_Stacking_Optimization", lp.LpMaximize)

    # Create binary decision variables for each player
    player_vars = {}
    for player in player_pool:
        var_name = f"player_{player.player_id}"
        player_vars[player.player_id] = lp.LpVariable(var_name, cat="Binary")

    # Objective function: maximize projected points
    prob += lp.lpSum([
        player.projected_points * player_vars[player.player_id]
        for player in player_pool
    ])

    # Standard constraints (salary, positions, lineup size)
    prob += lp.lpSum([
        player.salary * player_vars[player.player_id]
        for player in player_pool
    ]) <= constraints.salary_cap

    # Position constraints (same as linear programming)
    position_groups = {}
    for pos in constraints.positions:
        position_groups[pos] = []

    for player in player_pool:
        if player.position in position_groups:
            position_groups[player.position].append(player.player_id)

    flex_eligible_players = []
    for player in player_pool:
        if "/FLEX" in player.roster_position or player.position in ["RB", "WR", "TE"]:
            flex_eligible_players.append(player.player_id)

    # Add standard position constraints
    for pos, required_count in constraints.positions.items():
        if pos != "FLEX" and position_groups.get(pos):
            prob += lp.lpSum([
                player_vars[pid] for pid in position_groups[pos]
            ]) >= required_count

    # FLEX constraint
    if "FLEX" in constraints.positions:
        total_flex_positions = (
            constraints.positions.get("RB", 0) +
            constraints.positions.get("WR", 0) +
            constraints.positions.get("TE", 0) +
            constraints.positions.get("FLEX", 0)
        )

        flex_eligible_by_position = {
            "RB": [pid for pid in position_groups.get("RB", []) if pid in flex_eligible_players],
            "WR": [pid for pid in position_groups.get("WR", []) if pid in flex_eligible_players],
            "TE": [pid for pid in position_groups.get("TE", []) if pid in flex_eligible_players],
        }

        all_flex_eligible = (
            flex_eligible_by_position["RB"] +
            flex_eligible_by_position["WR"] +
            flex_eligible_by_position["TE"]
        )

        if all_flex_eligible:
            prob += lp.lpSum([
                player_vars[pid] for pid in all_flex_eligible
            ]) == total_flex_positions

    # Total lineup size constraint
    total_positions = sum(constraints.positions.values())
    prob += lp.lpSum([player_vars[pid] for pid in player_vars]) == total_positions

    # STACKING CONSTRAINTS
    # Group players by team for stacking logic
    teams_players = {}
    for player in player_pool:
        if player.team_abbr not in teams_players:
            teams_players[player.team_abbr] = {
                "QB": [], "WR": [], "TE": [], "RB": [], "DST": []
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
                    qb_selected = lp.lpSum([player_vars[pid] for pid in qb_players])
                    catchers_selected = lp.lpSum([player_vars[pid] for pid in pass_catchers])

                    if force_stacking:
                        # Force stacking: if QB selected, must have pass catchers
                        prob += catchers_selected >= constraints.min_qb_stack_count * qb_selected

    # RB-DEF stacking constraints (game script correlation)
    if constraints.allow_rb_def_stack and rb_def_stack_teams:
        for team in rb_def_stack_teams:
            if team in teams_players:
                rb_players = teams_players[team]["RB"]
                def_players = teams_players[team]["DST"]

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
                return build_stacking_lineup(
                    player_pool, constraints, qb_stack_teams, rb_def_stack_teams, force_stacking=False
                )

            return OptimizationResult(
                lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
                constraint_violations=[f"LP solver status: {status}"],
                optimization_status="infeasible"
            )

        # Extract the solution
        selected_players = []
        for player in player_pool:
            if player_vars[player.player_id].varValue == 1:
                selected_players.append(player)

        # Validate the lineup and check stacking
        violations = validate_lineup(selected_players, constraints)
        stacking_info = analyze_stacking(selected_players)

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
            stacking_info=stacking_info
        )

    except Exception as e:
        logger.exception("Stacking optimization failed")
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=[f"Optimization error: {str(e)}"],
            optimization_status="error"
        )


def build_tournament_lineup(
    player_pool: List[Player],
    constraints: LineupConstraints,
    ceiling_weight: float = 0.3,
    min_ceiling_threshold: float = 25.0
) -> OptimizationResult:
    """Build lineup optimized for tournaments (GPPs) using ceiling projections."""
    if not player_pool:
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=["No players available"], optimization_status="infeasible"
        )

    # Filter players with sufficient ceiling potential
    tournament_pool = []
    for player in player_pool:
        if player.ceiling >= min_ceiling_threshold:
            # Create new player with adjusted projections
            tournament_player = Player(
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                salary=player.salary,
                projected_points=player.projected_points,
                floor=player.floor,
                ceiling=player.ceiling,
                ownership_projection=player.ownership_projection,
                team_abbr=player.team_abbr,
                roster_position=player.roster_position,
                injury_status=player.injury_status
            )

            # Blend projection with ceiling for tournament optimization
            tournament_score = (
                player.projected_points * (1 - ceiling_weight) +
                player.ceiling * ceiling_weight
            )
            tournament_player.projected_points = tournament_score
            tournament_pool.append(tournament_player)

    if not tournament_pool:
        logger.warning(f"No players meet ceiling threshold of {min_ceiling_threshold}")
        return build_linear_programming_lineup(player_pool, constraints)

    return build_linear_programming_lineup(tournament_pool, constraints)


def build_contrarian_lineup(
    player_pool: List[Player],
    constraints: LineupConstraints,
    ownership_weight: float = 0.3
) -> OptimizationResult:
    """Build lineup that avoids high-owned players for contrarian strategy."""
    if not player_pool:
        return OptimizationResult(
            lineup=[], total_salary=0, projected_points=0.0, lineup_value=0.0,
            constraint_violations=["No players available"], optimization_status="infeasible"
        )

    # Adjust projections based on ownership (penalize high ownership)
    contrarian_pool = []
    for player in player_pool:
        contrarian_player = Player(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            salary=player.salary,
            projected_points=player.projected_points,
            floor=player.floor,
            ceiling=player.ceiling,
            ownership_projection=player.ownership_projection,
            team_abbr=player.team_abbr,
            roster_position=player.roster_position,
            injury_status=player.injury_status
        )

        # Penalty for high ownership
        if player.ownership_projection > 0:
            ownership_penalty = (player.ownership_projection / 100) * ownership_weight
            contrarian_player.projected_points *= (1 - ownership_penalty)

        contrarian_pool.append(contrarian_player)

    return build_linear_programming_lineup(contrarian_pool, constraints)


def generate_multiple_lineups(
    player_pool: List[Player],
    constraints: LineupConstraints,
    num_lineups: int = 5,
    diversity_factor: float = 0.3,
    strategy: str = "balanced"  # "balanced", "tournament", "contrarian"
) -> List[OptimizationResult]:
    """Generate multiple diverse lineups."""
    lineups = []
    used_players = set()

    for i in range(num_lineups):
        # Create modified player pool for diversity
        if i > 0:
            modified_pool = []
            for player in player_pool:
                modified_player = Player(
                    player_id=player.player_id,
                    name=player.name,
                    position=player.position,
                    salary=player.salary,
                    projected_points=player.projected_points,
                    floor=player.floor,
                    ceiling=player.ceiling,
                    ownership_projection=player.ownership_projection,
                    team_abbr=player.team_abbr,
                    roster_position=player.roster_position,
                    injury_status=player.injury_status
                )

                # Reduce value if already used
                if player.player_id in used_players:
                    modified_player.projected_points *= (1 - diversity_factor)

                modified_pool.append(modified_player)

            pool_to_use = modified_pool
        else:
            pool_to_use = player_pool

        # Generate lineup based on strategy
        if strategy == "tournament":
            result = build_tournament_lineup(pool_to_use, constraints)
        elif strategy == "contrarian":
            result = build_contrarian_lineup(pool_to_use, constraints)
        else:  # balanced
            result = build_linear_programming_lineup(pool_to_use, constraints)

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


def format_lineup_display(result: OptimizationResult) -> str:
    """Format lineup with proper FLEX identification."""
    if not result.is_valid or not result.lineup:
        return "Invalid lineup"

    # Group players by position
    position_players = {}
    for player in result.lineup:
        pos = player.position
        if pos not in position_players:
            position_players[pos] = []
        position_players[pos].append(player)

    # DraftKings roster requirements
    roster_requirements = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}

    # Track which players are assigned to base positions vs FLEX
    assigned_players = []
    flex_player = None

    # Assign base positions first (highest salary players get base positions)
    for pos, required in roster_requirements.items():
        pos_players = position_players.get(pos, [])
        pos_players.sort(key=lambda x: x.salary, reverse=True)

        for i in range(min(required, len(pos_players))):
            assigned_players.append((pos, pos_players[i]))

    # Find the FLEX player (remaining RB/WR/TE)
    assigned_ids = {player.player_id for _, player in assigned_players}

    for player in result.lineup:
        if player.player_id not in assigned_ids and player.position in ['RB', 'WR', 'TE']:
            flex_player = player
            break

    # Build display
    display_lines = []
    display_lines.append(f"=== OPTIMAL LINEUP ===")
    display_lines.append(f"Total Salary: ${result.total_salary:,}")
    display_lines.append(f"Projected Points: {result.projected_points:.1f}")
    display_lines.append("")

    # Group assigned players by position for display
    assigned_by_pos = {}
    for pos, player in assigned_players:
        if pos not in assigned_by_pos:
            assigned_by_pos[pos] = []
        assigned_by_pos[pos].append(player)

    # Show lineup in DraftKings order
    lineup_order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    position_counters = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0}

    for slot in lineup_order:
        if slot == "FLEX":
            if flex_player:
                injury_indicator = f" ({flex_player.injury_status})" if flex_player.injury_status else ""
                display_lines.append(f"FLEX ({flex_player.position}):  {flex_player.name}{injury_indicator} - ${flex_player.salary} - {flex_player.projected_points:.1f} pts")
            else:
                display_lines.append("FLEX: (not assigned)")
        else:
            pos_players = assigned_by_pos.get(slot, [])
            if position_counters[slot] < len(pos_players):
                player = pos_players[position_counters[slot]]
                injury_indicator = f" ({player.injury_status})" if player.injury_status else ""
                display_lines.append(f"{slot}:   {player.name}{injury_indicator} - ${player.salary} - {player.projected_points:.1f} pts")
                position_counters[slot] += 1
            else:
                display_lines.append(f"{slot}: (not assigned)")

    return "\n".join(display_lines)

def export_lineup_to_csv(lineup: OptimizationResult, filename: str = None) -> str:
    """Export lineup to DraftKings CSV format."""
    if filename is None:
        filename = f"lineup_{int(pd.Timestamp.now().timestamp())}.csv"

    # Create DataFrame in DraftKings format
    lineup_data = []
    position_order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]

    # Use same logic as display format to properly identify FLEX
    position_players = {}
    for player in lineup.lineup:
        pos = player.position
        if pos not in position_players:
            position_players[pos] = []
        position_players[pos].append(player)

    # Assign base positions first
    roster_requirements = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    assigned_players = []
    flex_player = None

    for pos, required in roster_requirements.items():
        pos_players = position_players.get(pos, [])
        pos_players.sort(key=lambda x: x.salary, reverse=True)

        for i in range(min(required, len(pos_players))):
            assigned_players.append((pos, pos_players[i]))

    # Find FLEX player
    assigned_ids = {player.player_id for _, player in assigned_players}
    for player in lineup.lineup:
        if player.player_id not in assigned_ids and player.position in ['RB', 'WR', 'TE']:
            flex_player = player
            break

    # Build ordered lineup for CSV
    assigned_by_pos = {}
    for pos, player in assigned_players:
        if pos not in assigned_by_pos:
            assigned_by_pos[pos] = []
        assigned_by_pos[pos].append(player)

    position_counters = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0}
    ordered_lineup = []

    for slot in position_order:
        if slot == "FLEX":
            if flex_player:
                ordered_lineup.append(flex_player)
        else:
            pos_players = assigned_by_pos.get(slot, [])
            if position_counters[slot] < len(pos_players):
                ordered_lineup.append(pos_players[position_counters[slot]])
                position_counters[slot] += 1

    # Create CSV data
    for i, player in enumerate(ordered_lineup):
        if player:  # Make sure player exists
            display_position = position_order[i]
            if display_position == "FLEX":
                display_position = f"FLEX ({player.position})"

            # Add injury indicator to name if player is injured
            display_name = player.name
            if player.injury_status:
                display_name += f" ({player.injury_status})"

            lineup_data.append({
                "Position": display_position,
                "Name": display_name,
                "Salary": player.salary,
                "Projected": player.projected_points,
                "Team": player.team_abbr,
                "Injury": player.injury_status or ""
            })

    df = pd.DataFrame(lineup_data)
    df.to_csv(filename, index=False)

    logger.info(f"Exported lineup to {filename}")
    return filename


# Convenience functions for common use cases
def optimize_cash_game_lineup(
    player_pool: List[Player],
    salary_cap: int = DEFAULT_SALARY_CAP
) -> OptimizationResult:
    """Optimize lineup for cash games (high floor, consistent scoring)."""
    constraints = LineupConstraints(salary_cap=salary_cap)

    # For cash games, use floor instead of projection for safer plays
    cash_pool = []
    for player in player_pool:
        cash_player = Player(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            salary=player.salary,
            projected_points=max(player.floor, player.projected_points * 0.8),  # Conservative
            floor=player.floor,
            ceiling=player.ceiling,
            team_abbr=player.team_abbr,
            roster_position=player.roster_position,
            injury_status=player.injury_status
        )
        cash_pool.append(cash_player)

    return build_linear_programming_lineup(cash_pool, constraints)


def optimize_tournament_lineup(
    player_pool: List[Player],
    salary_cap: int = DEFAULT_SALARY_CAP,
    enable_stacking: bool = True
) -> OptimizationResult:
    """Optimize lineup for tournaments (high ceiling, upside focused)."""
    constraints = LineupConstraints(salary_cap=salary_cap)

    if enable_stacking:
        # Find teams with good stacking potential
        team_qbs = {}
        for player in player_pool:
            if player.position == "QB":
                team_qbs[player.team_abbr] = player

        # Suggest stacking for top QB teams
        qb_stack_teams = sorted(
            team_qbs.keys(),
            key=lambda team: team_qbs[team].ceiling,
            reverse=True
        )[:3]  # Top 3 QB teams for stacking

        return build_stacking_lineup(
            player_pool, constraints, qb_stack_teams=qb_stack_teams, force_stacking=False
        )
    else:
        return build_tournament_lineup(player_pool, constraints)


# Example usage and testing functions
def test_optimization():
    """Test optimization functions with sample data."""
    # Create sample players
    sample_players = [
        Player(1, "Josh Allen", "QB", 8500, 22.5, 18.0, 28.0, team_abbr="BUF", roster_position="QB"),
        Player(2, "Christian McCaffrey", "RB", 9000, 20.8, 15.5, 26.0, team_abbr="SF", roster_position="RB/FLEX", injury_status="Q"),
        Player(3, "Stefon Diggs", "WR", 7800, 16.2, 12.0, 22.0, team_abbr="BUF", roster_position="WR/FLEX"),
        Player(4, "Travis Kelce", "TE", 7200, 14.5, 10.0, 20.0, team_abbr="KC", roster_position="TE/FLEX", injury_status="D"),
        Player(5, "Buffalo Bills", "DST", 2800, 8.5, 5.0, 15.0, team_abbr="BUF", roster_position="DST"),
        # Add more players to make valid lineups...
    ]

    constraints = LineupConstraints()

    # Test greedy algorithm
    result = build_greedy_lineup(sample_players, constraints)
    print(f"Greedy result: {result.optimization_status}, Points: {result.projected_points}")

    # Test linear programming
    if lp is not None:
        result = build_linear_programming_lineup(sample_players, constraints)
        print(f"LP result: {result.optimization_status}, Points: {result.projected_points}")

    return result


if __name__ == "__main__":
    # Run basic tests
    test_optimization()
