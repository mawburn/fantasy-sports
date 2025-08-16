"""API endpoints for game selection and contest recommendation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.optimization.game_selection import (
    ContestMetrics,
    GameSelectionEngine,
    GameSelectionSettings,
    RiskTolerance,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/game-selection", tags=["Game Selection"])


class ContestRecommendationRequest(BaseModel):
    """Request model for contest recommendations."""

    # Financial constraints
    bankroll: float = Field(..., gt=0, description="Total available bankroll in dollars")
    max_entry_fee: float = Field(..., gt=0, description="Maximum single contest entry fee")
    max_total_risk: float = Field(..., gt=0, description="Maximum total dollars at risk")

    # User preferences
    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.MODERATE, description="Risk tolerance level"
    )
    min_fun_score: float = Field(
        default=60.0, ge=0, le=100, description="Minimum fun score threshold"
    )
    min_expected_value: float = Field(
        default=-0.10, description="Minimum expected value (negative allows some -EV for fun)"
    )
    max_field_size: int = Field(default=100000, gt=0, description="Maximum contest field size")
    max_contests: int = Field(
        default=10, gt=0, le=50, description="Maximum number of contest recommendations"
    )

    # Skill assessment
    skill_edge: float = Field(
        default=0.0, ge=-0.5, le=0.5, description="Estimated skill edge over field (-0.5 to 0.5)"
    )


class ContestRecommendationResponse(BaseModel):
    """Response model for contest recommendations."""

    # Contest identification
    contest_id: str
    contest_name: str
    contest_type: str

    # Basic info
    entry_fee: float
    field_size: int
    total_prizes: float

    # Calculated metrics
    fun_score: float
    expected_value: float
    win_probability: float
    top_finish_probability: float

    # Risk metrics
    variance: float
    downside_risk: float
    kelly_fraction: float

    # Recommendations
    recommended_entries: int
    bankroll_percentage: float

    # Reasoning
    recommendation_reason: str


class GameSelectionSummary(BaseModel):
    """Summary of game selection results."""

    total_contests_analyzed: int
    contests_recommended: int
    total_entry_fees: float
    total_bankroll_risk: float
    expected_portfolio_ev: float
    portfolio_fun_score: float


@router.post("/recommend", response_model=list[ContestRecommendationResponse])
async def recommend_contests(
    request: ContestRecommendationRequest, db: Session = Depends(get_db)
) -> list[ContestRecommendationResponse]:
    """Get personalized contest recommendations.

    This endpoint analyzes available DraftKings contests and recommends
    the best options based on user preferences, risk tolerance, and bankroll.

    The recommendation engine considers:
    - Fun factor (entertainment value)
    - Expected value (mathematical edge)
    - Risk assessment (variance and downside)
    - Field competitiveness (skill level)
    - Portfolio optimization (diversification)
    """
    try:
        # Convert request to settings
        settings = GameSelectionSettings(
            bankroll=request.bankroll,
            max_entry_fee=request.max_entry_fee,
            max_total_risk=request.max_total_risk,
            risk_tolerance=request.risk_tolerance,
            min_fun_score=request.min_fun_score,
            min_expected_value=request.min_expected_value,
            max_field_size=request.max_field_size,
            max_contests=request.max_contests,
        )

        # Initialize game selection engine
        engine = GameSelectionEngine(db)

        # Get recommendations
        recommendations = engine.recommend_contests(settings, request.skill_edge)

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No contests found matching your criteria. Try adjusting your filters.",
            )

        # Convert to response format
        response_contests = []
        for contest in recommendations:
            # Generate recommendation reason
            reason = _generate_recommendation_reason(contest, request.risk_tolerance)

            response_contests.append(
                ContestRecommendationResponse(
                    contest_id=contest.contest_id,
                    contest_name=contest.contest_name,
                    contest_type=contest.contest_type.value,
                    entry_fee=contest.entry_fee,
                    field_size=contest.field_size,
                    total_prizes=contest.total_prizes,
                    fun_score=contest.fun_score,
                    expected_value=contest.expected_value,
                    win_probability=contest.win_probability,
                    top_finish_probability=contest.top_finish_probability,
                    variance=contest.variance,
                    downside_risk=contest.downside_risk,
                    kelly_fraction=contest.kelly_fraction,
                    recommended_entries=contest.recommended_entries,
                    bankroll_percentage=contest.bankroll_percentage,
                    recommendation_reason=reason,
                )
            )

        return response_contests

    except Exception as e:
        logger.exception("Error generating contest recommendations")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recommendations: {e!s}"
        ) from e


@router.post("/analyze", response_model=ContestRecommendationResponse)
async def analyze_contest(
    contest_id: str,
    skill_edge: float = Query(default=0.0, ge=-0.5, le=0.5),
    db: Session = Depends(get_db),
) -> ContestRecommendationResponse:
    """Analyze a specific contest and get detailed metrics.

    This endpoint provides comprehensive analysis of a single contest
    including fun score, expected value, risk metrics, and recommendations.
    """
    try:
        from src.database.models import DraftKingsContest

        # Get contest from database
        contest = (
            db.query(DraftKingsContest).filter(DraftKingsContest.contest_id == contest_id).first()
        )

        if not contest:
            raise HTTPException(status_code=404, detail=f"Contest not found: {contest_id}")

        # Analyze contest
        engine = GameSelectionEngine(db)
        metrics = engine.contest_analyzer.analyze_contest(contest)

        # Adjust for skill edge
        metrics.expected_value = engine.ev_calculator.calculate_ev(metrics, skill_edge)

        # Generate recommendation reason
        reason = _generate_recommendation_reason(metrics, RiskTolerance.MODERATE)

        return ContestRecommendationResponse(
            contest_id=metrics.contest_id,
            contest_name=metrics.contest_name,
            contest_type=metrics.contest_type.value,
            entry_fee=metrics.entry_fee,
            field_size=metrics.field_size,
            total_prizes=metrics.total_prizes,
            fun_score=metrics.fun_score,
            expected_value=metrics.expected_value,
            win_probability=metrics.win_probability,
            top_finish_probability=metrics.top_finish_probability,
            variance=metrics.variance,
            downside_risk=metrics.downside_risk,
            kelly_fraction=metrics.kelly_fraction,
            recommended_entries=1,
            bankroll_percentage=0.0,
            recommendation_reason=reason,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error analyzing contest {contest_id}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze contest: {e!s}") from e


@router.get("/summary")
async def get_selection_summary(
    bankroll: float = Query(..., gt=0),
    risk_tolerance: RiskTolerance = Query(default=RiskTolerance.MODERATE),
    db: Session = Depends(get_db),
) -> GameSelectionSummary:
    """Get summary statistics for available contests.

    This endpoint provides an overview of the contest landscape
    to help users understand available opportunities.
    """
    try:
        # Create basic settings for analysis
        settings = GameSelectionSettings(
            bankroll=bankroll,
            max_entry_fee=bankroll * 0.1,  # 10% of bankroll max
            max_total_risk=bankroll * 0.1,
            risk_tolerance=risk_tolerance,
        )

        # Get recommendations
        engine = GameSelectionEngine(db)
        recommendations = engine.recommend_contests(settings)

        # Calculate summary statistics
        total_entry_fees = sum(c.entry_fee for c in recommendations)
        total_ev = sum(c.expected_value for c in recommendations)
        avg_fun_score = (
            sum(c.fun_score for c in recommendations) / len(recommendations)
            if recommendations
            else 0
        )

        # Get total contests analyzed (this would need database query in real implementation)
        total_analyzed = len(recommendations) * 5  # Estimate

        return GameSelectionSummary(
            total_contests_analyzed=total_analyzed,
            contests_recommended=len(recommendations),
            total_entry_fees=total_entry_fees,
            total_bankroll_risk=total_entry_fees / bankroll if bankroll > 0 else 0,
            expected_portfolio_ev=total_ev,
            portfolio_fun_score=avg_fun_score,
        )

    except Exception as e:
        logger.exception("Error generating selection summary")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e!s}") from e


def _generate_recommendation_reason(contest: ContestMetrics, risk_tolerance: RiskTolerance) -> str:
    """Generate human-readable recommendation reason."""
    reasons = []

    # Fun score reasoning
    if contest.fun_score >= 80:
        reasons.append("High entertainment value")
    elif contest.fun_score >= 60:
        reasons.append("Good entertainment value")

    # Expected value reasoning
    if contest.expected_value > 0:
        reasons.append("Positive expected value")
    elif contest.expected_value > -0.05:
        reasons.append("Break-even expected value")

    # Contest type reasoning
    if contest.contest_type.value == "cash":
        reasons.append("Lower variance cash game")
    elif contest.contest_type.value == "gpp":
        reasons.append("High upside tournament")

    # Risk tolerance alignment
    if risk_tolerance == RiskTolerance.CONSERVATIVE and contest.downside_risk < 0.6:
        reasons.append("Low risk for conservative approach")
    elif risk_tolerance == RiskTolerance.AGGRESSIVE and contest.variance > contest.entry_fee * 5:
        reasons.append("High variance for aggressive strategy")

    # Field size reasoning
    if contest.field_size < 1000:
        reasons.append("Smaller field size")

    # Default reason
    if not reasons:
        reasons.append("Meets your criteria")

    return "; ".join(reasons)
