"""CLI commands for game selection and contest recommendation."""

import json
import logging

import typer
from rich.console import Console
from rich.table import Table

from src.database.connection import get_db
from src.optimization.game_selection import (
    GameSelectionEngine,
    GameSelectionSettings,
    RiskTolerance,
)

# Create CLI app and console for rich output
app = typer.Typer(help="Game selection and contest recommendation commands")
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command("recommend")
def recommend_contests(
    bankroll: float = typer.Argument(..., help="Total bankroll in dollars"),
    risk_tolerance: str = typer.Option(
        "moderate", help="Risk tolerance (conservative, moderate, aggressive)"
    ),
    max_entry_fee: float | None = typer.Option(
        None, help="Maximum entry fee per contest (default: 10% of bankroll)"
    ),
    max_total_risk: float | None = typer.Option(
        None, help="Maximum total risk (default: 10% of bankroll)"
    ),
    min_fun_score: float = typer.Option(60.0, help="Minimum fun score (0-100)"),
    min_expected_value: float = typer.Option(
        -0.10, help="Minimum expected value (negative allows some -EV for fun)"
    ),
    max_contests: int = typer.Option(10, help="Maximum number of contest recommendations"),
    skill_edge: float = typer.Option(
        0.0, help="Your estimated skill edge over field (-0.5 to 0.5)"
    ),
    output_format: str = typer.Option("table", help="Output format (table, json)"),
) -> None:
    """Get personalized contest recommendations based on your preferences.

    This command analyzes available DraftKings contests and recommends
    the best options based on your bankroll, risk tolerance, and preferences.

    Examples:
        # Conservative recommendations for $1000 bankroll
        uv run python -m src.cli.game_selection recommend 1000 --risk-tolerance conservative

        # Aggressive player with skill edge
        uv run python -m src.cli.game_selection recommend 500 --risk-tolerance aggressive --skill-edge 0.1

        # High-roller with specific constraints
        uv run python -m src.cli.game_selection recommend 10000 --max-entry-fee 200 --min-fun-score 80
    """
    try:
        # Validate and convert risk tolerance
        risk_tolerance_map = {
            "conservative": RiskTolerance.CONSERVATIVE,
            "moderate": RiskTolerance.MODERATE,
            "aggressive": RiskTolerance.AGGRESSIVE,
        }

        if risk_tolerance.lower() not in risk_tolerance_map:
            console.print(f"❌ Invalid risk tolerance: {risk_tolerance}", style="red")
            console.print("Valid options: conservative, moderate, aggressive")
            raise typer.Exit(1)

        risk_level = risk_tolerance_map[risk_tolerance.lower()]

        # Set defaults based on bankroll
        if max_entry_fee is None:
            max_entry_fee = bankroll * 0.10  # 10% of bankroll

        if max_total_risk is None:
            max_total_risk = bankroll * 0.10  # 10% of bankroll

        # Create settings
        settings = GameSelectionSettings(
            bankroll=bankroll,
            max_entry_fee=max_entry_fee,
            max_total_risk=max_total_risk,
            risk_tolerance=risk_level,
            min_fun_score=min_fun_score,
            min_expected_value=min_expected_value,
            max_contests=max_contests,
        )

        # Initialize engine and get recommendations
        console.print(
            f"🎯 Analyzing contests for {risk_tolerance} player with ${bankroll:,.0f} bankroll..."
        )

        with next(get_db()) as db:
            engine = GameSelectionEngine(db)
            recommendations = engine.recommend_contests(settings, skill_edge)

        if not recommendations:
            console.print("😞 No contests found matching your criteria.", style="yellow")
            console.print("\n💡 Try adjusting your filters:")
            console.print("   • Lower minimum fun score")
            console.print("   • Allow more negative EV")
            console.print("   • Increase maximum entry fee")
            console.print("   • Change risk tolerance")
            return

        # Display results
        if output_format.lower() == "json":
            _display_json_output(recommendations)
        else:
            _display_table_output(recommendations, settings)

        # Show summary
        total_cost = sum(c.entry_fee for c in recommendations)
        total_ev = sum(c.expected_value for c in recommendations)
        avg_fun = sum(c.fun_score for c in recommendations) / len(recommendations)

        console.print("\n📊 Portfolio Summary:")
        console.print(f"   💰 Total Entry Fees: ${total_cost:.2f}")
        console.print(f"   📈 Total Expected Value: ${total_ev:+.2f}")
        console.print(f"   🎉 Average Fun Score: {avg_fun:.1f}/100")
        console.print(f"   🎲 Bankroll Risk: {(total_cost / bankroll) * 100:.1f}%")

        # Risk warnings
        if total_cost / bankroll > 0.15:
            console.print(
                "\n⚠️  High bankroll risk detected! Consider reducing total exposure.",
                style="yellow",
            )

        if total_ev < -5.0:
            console.print(
                "\n⚠️  Negative expected value portfolio. Playing for fun only!", style="yellow"
            )

    except Exception as e:
        console.print(f"❌ Error generating recommendations: {e}", style="red")
        logger.exception("Failed to generate contest recommendations")
        raise typer.Exit(1) from e


@app.command("analyze")
def analyze_contest(
    contest_id: str = typer.Argument(..., help="DraftKings contest ID to analyze"),
    skill_edge: float = typer.Option(0.0, help="Your estimated skill edge (-0.5 to 0.5)"),
    output_format: str = typer.Option("table", help="Output format (table, json)"),
) -> None:
    """Analyze a specific DraftKings contest in detail.

    This command provides comprehensive analysis of a single contest
    including fun score, expected value, risk metrics, and recommendations.

    Examples:
        # Analyze specific contest
        uv run python -m src.cli.game_selection analyze CONTEST123

        # Analyze with skill edge
        uv run python -m src.cli.game_selection analyze CONTEST123 --skill-edge 0.15
    """
    try:
        console.print(f"🔍 Analyzing contest {contest_id}...")

        with next(get_db()) as db:
            engine = GameSelectionEngine(db)

            # Get contest from database
            from src.database.models import DraftKingsContest

            contest = (
                db.query(DraftKingsContest)
                .filter(DraftKingsContest.contest_id == contest_id)
                .first()
            )

            if not contest:
                console.print(f"❌ Contest not found: {contest_id}", style="red")
                raise typer.Exit(1)

            # Analyze contest
            metrics = engine.contest_analyzer.analyze_contest(contest)
            metrics.expected_value = engine.ev_calculator.calculate_ev(metrics, skill_edge)

        # Display results
        if output_format.lower() == "json":
            _display_contest_json(metrics)
        else:
            _display_contest_table(metrics, skill_edge)

    except Exception as e:
        console.print(f"❌ Error analyzing contest: {e}", style="red")
        logger.exception(f"Failed to analyze contest {contest_id}")
        raise typer.Exit(1) from e


@app.command("summary")
def show_summary(
    bankroll: float = typer.Argument(..., help="Total bankroll for analysis"),
    risk_tolerance: str = typer.Option("moderate", help="Risk tolerance level"),
) -> None:
    """Show summary of available contests and opportunities.

    This command provides an overview of the contest landscape
    to help understand available opportunities.
    """
    try:
        # Convert risk tolerance
        risk_tolerance_map = {
            "conservative": RiskTolerance.CONSERVATIVE,
            "moderate": RiskTolerance.MODERATE,
            "aggressive": RiskTolerance.AGGRESSIVE,
        }

        risk_level = risk_tolerance_map.get(risk_tolerance.lower())
        if not risk_level:
            console.print(f"❌ Invalid risk tolerance: {risk_tolerance}", style="red")
            raise typer.Exit(1)

        settings = GameSelectionSettings(
            bankroll=bankroll,
            max_entry_fee=bankroll * 0.1,
            max_total_risk=bankroll * 0.1,
            risk_tolerance=risk_level,
        )

        console.print(f"📈 Contest Landscape Summary for ${bankroll:,.0f} bankroll...")

        with next(get_db()) as db:
            engine = GameSelectionEngine(db)
            recommendations = engine.recommend_contests(settings)

            # Calculate summary statistics
            if recommendations:
                total_contests = len(recommendations) * 5  # Estimate
                total_entry_fees = sum(c.entry_fee for c in recommendations)
                total_ev = sum(c.expected_value for c in recommendations)
                avg_fun = sum(c.fun_score for c in recommendations) / len(recommendations)

                console.print("\n📊 Available Opportunities:")
                console.print(f"   🎯 Contests Analyzed: ~{total_contests}")
                console.print(f"   ✅ Contests Recommended: {len(recommendations)}")
                console.print(f"   💰 Total Entry Fees: ${total_entry_fees:.2f}")
                console.print(f"   📈 Portfolio Expected Value: ${total_ev:+.2f}")
                console.print(f"   🎉 Average Fun Score: {avg_fun:.1f}/100")
                console.print(
                    f"   🎲 Portfolio Risk: {(total_entry_fees / bankroll) * 100:.1f}% of bankroll"
                )
            else:
                console.print("😞 No suitable contests found for your criteria.", style="yellow")

    except Exception as e:
        console.print(f"❌ Error generating summary: {e}", style="red")
        logger.exception("Failed to generate contest summary")
        raise typer.Exit(1) from e


def _display_table_output(recommendations: list, settings: GameSelectionSettings) -> None:
    """Display recommendations in rich table format."""
    table = Table(title=f"🎯 Contest Recommendations ({len(recommendations)} contests)")

    table.add_column("Contest", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Entry Fee", justify="right", style="green")
    table.add_column("Field Size", justify="right")
    table.add_column("Fun Score", justify="center", style="yellow")
    table.add_column("Expected Value", justify="right", style="bright_green")
    table.add_column("Win %", justify="center")
    table.add_column("Risk", justify="center", style="red")

    for contest in recommendations:
        # Truncate long contest names
        name = (
            contest.contest_name[:25] + "..."
            if len(contest.contest_name) > 28
            else contest.contest_name
        )

        # Format expected value with color
        ev_color = "bright_green" if contest.expected_value >= 0 else "red"
        ev_text = f"${contest.expected_value:+.2f}"

        # Risk indicator
        if contest.downside_risk < 0.5:
            risk_indicator = "🟢 Low"
        elif contest.downside_risk < 0.8:
            risk_indicator = "🟡 Med"
        else:
            risk_indicator = "🔴 High"

        table.add_row(
            name,
            contest.contest_type.value.upper(),
            f"${contest.entry_fee:.0f}",
            f"{contest.field_size:,}",
            f"{contest.fun_score:.0f}/100",
            f"[{ev_color}]{ev_text}[/{ev_color}]",
            f"{contest.win_probability * 100:.0f}%",
            risk_indicator,
        )

    console.print(table)


def _display_json_output(recommendations: list) -> None:
    """Display recommendations in JSON format."""
    output = []
    for contest in recommendations:
        output.append(
            {
                "contest_id": contest.contest_id,
                "contest_name": contest.contest_name,
                "contest_type": contest.contest_type.value,
                "entry_fee": contest.entry_fee,
                "field_size": contest.field_size,
                "fun_score": contest.fun_score,
                "expected_value": contest.expected_value,
                "win_probability": contest.win_probability,
                "downside_risk": contest.downside_risk,
                "recommended_entries": contest.recommended_entries,
            }
        )

    console.print(json.dumps(output, indent=2))


def _display_contest_table(metrics, skill_edge: float) -> None:
    """Display single contest analysis in table format."""
    table = Table(title=f"🔍 Contest Analysis: {metrics.contest_name}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Assessment", style="yellow")

    # Basic info
    table.add_row("Contest Type", metrics.contest_type.value.upper(), "")
    table.add_row("Entry Fee", f"${metrics.entry_fee:.2f}", "")
    table.add_row("Field Size", f"{metrics.field_size:,}", "")
    table.add_row("Total Prizes", f"${metrics.total_prizes:,.2f}", "")

    table.add_row("", "", "")  # Separator

    # Key metrics
    fun_assessment = (
        "Excellent" if metrics.fun_score >= 80 else "Good" if metrics.fun_score >= 60 else "Fair"
    )
    table.add_row("Fun Score", f"{metrics.fun_score:.1f}/100", fun_assessment)

    ev_assessment = (
        "Positive Edge"
        if metrics.expected_value > 0
        else "Break Even"
        if metrics.expected_value > -0.1
        else "Negative"
    )
    ev_color = "bright_green" if metrics.expected_value >= 0 else "red"
    table.add_row(
        "Expected Value", f"[{ev_color}]${metrics.expected_value:+.2f}[/{ev_color}]", ev_assessment
    )

    table.add_row("Win Probability", f"{metrics.win_probability * 100:.1f}%", "")
    table.add_row("Top 10% Probability", f"{metrics.top_finish_probability * 100:.1f}%", "")

    risk_assessment = (
        "Low"
        if metrics.downside_risk < 0.5
        else "Medium"
        if metrics.downside_risk < 0.8
        else "High"
    )
    table.add_row("Downside Risk", f"{metrics.downside_risk * 100:.0f}%", risk_assessment)

    if skill_edge != 0:
        table.add_row("", "", "")  # Separator
        table.add_row(
            "Skill Edge Applied",
            f"{skill_edge:+.1%}",
            f"{'Advantage' if skill_edge > 0 else 'Disadvantage'}",
        )

    console.print(table)


def _display_contest_json(metrics) -> None:
    """Display single contest analysis in JSON format."""
    output = {
        "contest_id": metrics.contest_id,
        "contest_name": metrics.contest_name,
        "contest_type": metrics.contest_type.value,
        "entry_fee": metrics.entry_fee,
        "field_size": metrics.field_size,
        "total_prizes": metrics.total_prizes,
        "fun_score": metrics.fun_score,
        "expected_value": metrics.expected_value,
        "win_probability": metrics.win_probability,
        "top_finish_probability": metrics.top_finish_probability,
        "variance": metrics.variance,
        "downside_risk": metrics.downside_risk,
        "kelly_fraction": metrics.kelly_fraction,
        "field_competitiveness": metrics.field_competitiveness,
        "overlay_percentage": metrics.overlay_percentage,
    }

    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
