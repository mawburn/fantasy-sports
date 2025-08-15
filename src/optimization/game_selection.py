"""Game Selection Engine for DraftKings contest recommendation.

This module helps users choose which DraftKings contests to enter based on:
1. Fun Factor: Engagement and entertainment value  
2. Expected Value: Mathematical expectation of profit
3. Risk Assessment: Variance and downside protection
4. Field Size: Competition level and skill requirements
5. Payout Structure: Top-heavy vs flat distributions

The goal is to optimize for entertainment value rather than pure expected value,
since this system is designed for recreational users who want engaging contests.

Key Concepts:

Fun Score: Proprietary metric combining:
- Contest excitement (payout structure)
- Achievable goals (realistic win probability) 
- Field competitiveness (skill level)
- Entry variance (multiple lineups vs single)

Expected Value (EV): Mathematical expectation calculated as:
EV = (Win_Probability * Payout) - Entry_Fee

Risk Metrics:
- Variance: Spread of possible outcomes
- Downside Risk: Probability of significant loss
- Kelly Criterion: Optimal bet sizing for bankroll management
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import DraftKingsContest

logger = logging.getLogger(__name__)


class ContestType(Enum):
    """DraftKings contest types with different optimization strategies."""
    
    GPP = "gpp"  # Guaranteed Prize Pool (tournaments)
    CASH = "cash"  # 50/50s and double-ups
    H2H = "head_to_head"  # Head-to-head matches
    SATELLITE = "satellite"  # Qualifier contests
    SHOWDOWN = "showdown"  # Single-game contests
    MULTIPLIER = "multiplier"  # 3x, 5x, 10x contests


class RiskTolerance(Enum):
    """User risk tolerance levels for contest recommendation."""
    
    CONSERVATIVE = "conservative"  # Focus on cash games and low variance
    MODERATE = "moderate"  # Balanced mix of contest types
    AGGRESSIVE = "aggressive"  # High variance GPPs and satellites


@dataclass
class ContestMetrics:
    """Comprehensive metrics for a DraftKings contest.
    
    This dataclass contains all the calculated metrics needed for contest
    evaluation and recommendation.
    """
    
    # Contest identification
    contest_id: str
    contest_name: str
    contest_type: ContestType
    
    # Basic contest info
    entry_fee: float
    max_entries: int
    total_prizes: float
    field_size: int
    
    # Calculated metrics
    fun_score: float  # 0-100 proprietary engagement metric
    expected_value: float  # Dollars expected return
    win_probability: float  # Probability of any cash finish
    top_finish_probability: float  # Probability of top 10% finish
    
    # Risk metrics
    variance: float  # Outcome variance in dollars
    downside_risk: float  # Probability of losing >50% of bankroll
    kelly_fraction: float  # Optimal bet size per Kelly Criterion
    
    # Competition metrics
    field_competitiveness: float  # 0-1 estimated field skill level
    overlay_percentage: float  # Value above expected (negative = rake)
    
    # User-specific
    recommended_entries: int  # Suggested number of lineups
    bankroll_percentage: float  # Suggested % of bankroll to risk


@dataclass 
class GameSelectionSettings:
    """User preferences for contest selection."""
    
    # Financial constraints
    bankroll: float  # Total available bankroll
    max_entry_fee: float  # Maximum single contest entry
    max_total_risk: float  # Maximum total dollars at risk
    
    # Risk preferences
    risk_tolerance: RiskTolerance
    min_fun_score: float = 60.0  # Minimum entertainment threshold
    min_expected_value: float = -0.10  # Minimum EV (negative allows some -EV for fun)
    max_field_size: int = 100000  # Maximum contest size
    
    # Contest preferences  
    preferred_types: List[ContestType] = None
    avoid_types: List[ContestType] = None
    max_contests: int = 10  # Maximum recommendations
    
    def __post_init__(self):
        """Set default preferences based on risk tolerance."""
        if self.preferred_types is None:
            if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
                self.preferred_types = [ContestType.CASH, ContestType.H2H]
            elif self.risk_tolerance == RiskTolerance.MODERATE:
                self.preferred_types = [ContestType.CASH, ContestType.GPP, ContestType.MULTIPLIER]
            else:  # AGGRESSIVE
                self.preferred_types = [ContestType.GPP, ContestType.SATELLITE, ContestType.MULTIPLIER]
        
        if self.avoid_types is None:
            if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
                self.avoid_types = [ContestType.SATELLITE]
            else:
                self.avoid_types = []


class ContestAnalyzer:
    """Analyzes DraftKings contests to extract key metrics."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize contest analyzer."""
        self.db = db_session or next(get_db())
    
    def analyze_contest(self, contest: DraftKingsContest) -> ContestMetrics:
        """Analyze a single contest and calculate all metrics.
        
        Args:
            contest: DraftKings contest from database
            
        Returns:
            Complete contest metrics for decision making
        """
        # Determine contest type
        contest_type = self._classify_contest_type(contest)
        
        # Calculate payout structure metrics
        payout_metrics = self._analyze_payout_structure(contest)
        
        # Calculate competition metrics
        competition_metrics = self._analyze_field_competitiveness(contest)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(contest, payout_metrics)
        
        # Calculate fun score
        fun_score = self._calculate_fun_score(contest, payout_metrics, competition_metrics)
        
        return ContestMetrics(
            contest_id=contest.contest_id,
            contest_name=contest.name,
            contest_type=contest_type,
            entry_fee=contest.entry_fee,
            max_entries=contest.max_entries,
            total_prizes=contest.total_prizes,
            field_size=contest.entries,
            fun_score=fun_score,
            expected_value=payout_metrics["expected_value"],
            win_probability=payout_metrics["win_probability"],
            top_finish_probability=payout_metrics["top_finish_probability"], 
            variance=risk_metrics["variance"],
            downside_risk=risk_metrics["downside_risk"],
            kelly_fraction=risk_metrics["kelly_fraction"],
            field_competitiveness=competition_metrics["skill_level"],
            overlay_percentage=competition_metrics["overlay"],
            recommended_entries=1,  # Will be calculated by recommendation engine
            bankroll_percentage=0.0  # Will be calculated by recommendation engine
        )
    
    def _classify_contest_type(self, contest: DraftKingsContest) -> ContestType:
        """Classify contest type based on name and structure."""
        name = contest.name.lower()
        
        # Simple classification based on contest names
        if any(term in name for term in ["50/50", "double up", "head to head"]):
            return ContestType.CASH
        elif any(term in name for term in ["gpp", "milly maker", "tournament"]):
            return ContestType.GPP
        elif "showdown" in name or "single game" in name:
            return ContestType.SHOWDOWN
        elif any(term in name for term in ["satellite", "qualifier"]):
            return ContestType.SATELLITE
        elif any(term in name for term in ["3x", "5x", "10x", "multiplier"]):
            return ContestType.MULTIPLIER
        else:
            # Default classification based on payout structure
            payout_percentage = contest.total_prizes / (contest.entry_fee * contest.entries) if contest.entries > 0 else 0
            
            if payout_percentage > 0.8:  # High payout percentage = cash game
                return ContestType.CASH
            else:  # Lower payout percentage = GPP
                return ContestType.GPP
    
    def _analyze_payout_structure(self, contest: DraftKingsContest) -> Dict[str, float]:
        """Analyze contest payout structure for EV calculation."""
        if contest.entries == 0:
            return {
                "expected_value": -contest.entry_fee,
                "win_probability": 0.0,
                "top_finish_probability": 0.0
            }
        
        # Calculate basic metrics
        total_entry_fees = contest.entry_fee * contest.entries
        rake_percentage = 1 - (contest.total_prizes / total_entry_fees) if total_entry_fees > 0 else 0
        
        # Estimate payout distribution based on contest type
        contest_type = self._classify_contest_type(contest)
        
        if contest_type == ContestType.CASH:
            # Cash games pay ~45% of field
            win_probability = 0.45
            avg_payout = contest.total_prizes / (contest.entries * win_probability) if contest.entries > 0 else 0
            expected_value = (win_probability * avg_payout) - contest.entry_fee
            top_finish_probability = 0.1  # Top 10% in cash games
            
        elif contest_type == ContestType.GPP:
            # GPPs pay ~20% of field, top heavy
            win_probability = 0.20
            
            # Estimate top-heavy distribution
            # Top 1% gets 30% of prize pool, next 4% gets 25%, next 15% gets 45%
            if contest.entries > 0:
                top_1_pct_payout = 0.30 * contest.total_prizes / (0.01 * contest.entries)
                top_5_pct_payout = 0.25 * contest.total_prizes / (0.04 * contest.entries)
                remaining_payout = 0.45 * contest.total_prizes / (0.15 * contest.entries)
                
                # Expected value calculation
                prob_top_1 = 0.01
                prob_top_5 = 0.04
                prob_remaining = 0.15
                
                expected_value = (
                    prob_top_1 * top_1_pct_payout +
                    prob_top_5 * top_5_pct_payout +
                    prob_remaining * remaining_payout
                ) - contest.entry_fee
                
                top_finish_probability = 0.05  # Top 5% in GPPs
            else:
                expected_value = -contest.entry_fee
                top_finish_probability = 0.0
                
        else:
            # Default calculation for other contest types
            win_probability = 0.30
            avg_payout = contest.total_prizes / (contest.entries * win_probability) if contest.entries > 0 else 0
            expected_value = (win_probability * avg_payout) - contest.entry_fee
            top_finish_probability = 0.05
        
        return {
            "expected_value": expected_value,
            "win_probability": win_probability,
            "top_finish_probability": top_finish_probability,
            "rake_percentage": rake_percentage
        }
    
    def _analyze_field_competitiveness(self, contest: DraftKingsContest) -> Dict[str, float]:
        """Analyze field skill level and overlay opportunities."""
        # Estimate field skill based on contest characteristics
        skill_factors = []
        
        # Entry fee factor (higher fee = more skilled players)
        if contest.entry_fee >= 100:
            skill_factors.append(0.8)
        elif contest.entry_fee >= 25:
            skill_factors.append(0.6)
        elif contest.entry_fee >= 5:
            skill_factors.append(0.4)
        else:
            skill_factors.append(0.2)
        
        # Field size factor (larger field = more recreational players)
        if contest.entries >= 10000:
            skill_factors.append(0.3)
        elif contest.entries >= 1000:
            skill_factors.append(0.5)
        else:
            skill_factors.append(0.7)
        
        # Contest type factor
        contest_type = self._classify_contest_type(contest)
        if contest_type == ContestType.CASH:
            skill_factors.append(0.6)  # Cash games attract skilled players
        elif contest_type == ContestType.GPP:
            skill_factors.append(0.4)  # GPPs have more recreational players
        else:
            skill_factors.append(0.5)
        
        # Average skill level
        skill_level = np.mean(skill_factors)
        
        # Calculate overlay (value above expected)
        total_fees = contest.entry_fee * contest.entries if contest.entries > 0 else contest.entry_fee
        overlay = (contest.total_prizes - total_fees) / total_fees if total_fees > 0 else 0
        
        return {
            "skill_level": skill_level,
            "overlay": overlay
        }
    
    def _calculate_risk_metrics(self, contest: DraftKingsContest, payout_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk and variance metrics."""
        expected_value = payout_metrics["expected_value"]
        win_probability = payout_metrics["win_probability"]
        
        # Estimate variance based on contest type
        contest_type = self._classify_contest_type(contest)
        
        if contest_type == ContestType.CASH:
            # Cash games have lower variance
            avg_win = contest.total_prizes / (contest.entries * win_probability) if contest.entries > 0 and win_probability > 0 else 0
            variance = win_probability * (avg_win - contest.entry_fee) ** 2 + (1 - win_probability) * (contest.entry_fee) ** 2
        else:
            # GPPs have higher variance due to top-heavy payouts
            # Simplified variance calculation
            variance = contest.entry_fee * contest.entry_fee * 10  # High variance multiplier
        
        # Downside risk (probability of losing more than 50% of entry)
        if contest_type == ContestType.CASH:
            downside_risk = 1 - win_probability  # Lose entire entry in cash games
        else:
            downside_risk = 0.8  # High probability of losing most/all in GPPs
        
        # Kelly Criterion for optimal bet sizing
        if win_probability > 0 and expected_value > 0:
            avg_payout = contest.total_prizes / (contest.entries * win_probability) if contest.entries > 0 else 0
            if avg_payout > contest.entry_fee:
                kelly_fraction = (win_probability * avg_payout - (1 - win_probability) * contest.entry_fee) / avg_payout
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
            else:
                kelly_fraction = 0
        else:
            kelly_fraction = 0
        
        return {
            "variance": variance,
            "downside_risk": downside_risk,
            "kelly_fraction": kelly_fraction
        }
    
    def _calculate_fun_score(self, contest: DraftKingsContest, payout_metrics: Dict[str, float], competition_metrics: Dict[str, float]) -> float:
        """Calculate proprietary fun score (0-100)."""
        score_components = []
        
        # Excitement factor (based on potential payout)
        max_potential_payout = contest.total_prizes * 0.3  # Assume top prize is ~30% of pool
        excitement_ratio = max_potential_payout / contest.entry_fee if contest.entry_fee > 0 else 0
        excitement_score = min(100, excitement_ratio * 2)  # Scale to 0-100
        score_components.append(("excitement", excitement_score, 0.3))
        
        # Achievability factor (realistic win probability)
        win_prob = payout_metrics["win_probability"]
        if win_prob > 0.4:  # High win probability
            achievability_score = 90
        elif win_prob > 0.2:  # Moderate win probability
            achievability_score = 70
        elif win_prob > 0.1:  # Low but reasonable
            achievability_score = 50
        else:  # Very low win probability
            achievability_score = 20
        score_components.append(("achievability", achievability_score, 0.25))
        
        # Field competitiveness (more fun when beatable)
        skill_level = competition_metrics["skill_level"]
        competitiveness_score = (1 - skill_level) * 100  # Invert skill level
        score_components.append(("competitiveness", competitiveness_score, 0.2))
        
        # Value factor (overlay and EV)
        expected_value = payout_metrics["expected_value"]
        if expected_value > 0:
            value_score = 100
        elif expected_value > -0.1 * contest.entry_fee:  # Small negative EV ok for fun
            value_score = 80
        elif expected_value > -0.2 * contest.entry_fee:
            value_score = 60
        else:
            value_score = 20
        score_components.append(("value", value_score, 0.15))
        
        # Variance factor (some variance is fun, too much is stressful)
        contest_type = self._classify_contest_type(contest)
        if contest_type == ContestType.CASH:
            variance_score = 60  # Lower variance
        elif contest_type == ContestType.GPP:
            variance_score = 85  # Higher variance is exciting
        else:
            variance_score = 70
        score_components.append(("variance", variance_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        # Apply field size penalty for very large fields
        if contest.entries > 50000:
            total_score *= 0.9  # 10% penalty for massive fields
        elif contest.entries > 100000:
            total_score *= 0.8  # 20% penalty for extremely large fields
        
        return max(0, min(100, total_score))


class ExpectedValueCalculator:
    """Calculates expected value for contests with different strategies."""
    
    def calculate_ev(self, contest_metrics: ContestMetrics, skill_edge: float = 0.0) -> float:
        """Calculate expected value given user skill edge.
        
        Args:
            contest_metrics: Contest metrics from analyzer
            skill_edge: User's estimated edge over field (0.0 = average, 0.1 = 10% edge)
            
        Returns:
            Expected value in dollars
        """
        base_ev = contest_metrics.expected_value
        
        # Adjust for skill edge
        if skill_edge > 0:
            # Positive skill edge improves EV
            skill_adjusted_ev = base_ev + (skill_edge * contest_metrics.entry_fee)
        else:
            # No adjustment for average or below-average skill
            skill_adjusted_ev = base_ev
        
        return skill_adjusted_ev
    
    def calculate_multi_entry_ev(self, contest_metrics: ContestMetrics, num_entries: int, skill_edge: float = 0.0) -> float:
        """Calculate EV for multiple entries in same contest."""
        single_entry_ev = self.calculate_ev(contest_metrics, skill_edge)
        
        # Multi-entry reduces variance but doesn't change expected value linearly
        # There are diminishing returns due to increased correlation
        if num_entries == 1:
            return single_entry_ev
        elif num_entries <= 3:
            # Small multi-entry has minimal correlation penalty
            return single_entry_ev * num_entries * 0.95
        else:
            # Larger multi-entry has more correlation penalty
            return single_entry_ev * num_entries * 0.90


class RiskAssessment:
    """Assesses risk metrics for contest selection."""
    
    def assess_portfolio_risk(self, selected_contests: List[ContestMetrics], bankroll: float) -> Dict[str, float]:
        """Assess risk of a portfolio of contests.
        
        Args:
            selected_contests: List of contests to analyze
            bankroll: User's total bankroll
            
        Returns:
            Dictionary of risk metrics
        """
        if not selected_contests:
            return {
                "total_risk": 0.0,
                "bankroll_percentage": 0.0,
                "variance": 0.0,
                "max_loss_probability": 0.0,
                "kelly_fraction": 0.0
            }
        
        # Calculate total exposure
        total_entry_fees = sum(contest.entry_fee for contest in selected_contests)
        bankroll_percentage = total_entry_fees / bankroll if bankroll > 0 else 1.0
        
        # Calculate portfolio variance (assuming some correlation between contests)
        individual_variances = [contest.variance for contest in selected_contests]
        correlation_factor = 0.3  # Assume 30% correlation between contests
        
        # Portfolio variance with correlation
        portfolio_variance = sum(individual_variances)
        if len(selected_contests) > 1:
            # Add correlation terms
            correlation_terms = 0
            for i in range(len(individual_variances)):
                for j in range(i + 1, len(individual_variances)):
                    correlation_terms += 2 * correlation_factor * math.sqrt(individual_variances[i] * individual_variances[j])
            portfolio_variance += correlation_terms
        
        # Maximum loss probability (probability of losing more than 50% of investment)
        max_loss_prob = np.mean([contest.downside_risk for contest in selected_contests])
        
        # Portfolio Kelly fraction
        total_ev = sum(contest.expected_value for contest in selected_contests)
        if total_ev > 0:
            portfolio_kelly = min(0.25, total_ev / (bankroll * 0.1))  # Conservative Kelly
        else:
            portfolio_kelly = 0.0
        
        return {
            "total_risk": total_entry_fees,
            "bankroll_percentage": bankroll_percentage,
            "variance": portfolio_variance,
            "max_loss_probability": max_loss_prob,
            "kelly_fraction": portfolio_kelly
        }
    
    def is_within_risk_tolerance(self, portfolio_risk: Dict[str, float], settings: GameSelectionSettings) -> bool:
        """Check if portfolio risk is within user tolerance."""
        # Check bankroll percentage
        if portfolio_risk["bankroll_percentage"] > 0.10:  # Never risk more than 10% of bankroll
            return False
        
        # Check total risk amount
        if portfolio_risk["total_risk"] > settings.max_total_risk:
            return False
        
        # Check risk tolerance specific limits
        if settings.risk_tolerance == RiskTolerance.CONSERVATIVE:
            return portfolio_risk["bankroll_percentage"] <= 0.05  # 5% max for conservative
        elif settings.risk_tolerance == RiskTolerance.MODERATE:
            return portfolio_risk["bankroll_percentage"] <= 0.08  # 8% max for moderate
        else:  # AGGRESSIVE
            return portfolio_risk["bankroll_percentage"] <= 0.10  # 10% max for aggressive
        
        return True


class GameSelectionEngine:
    """Main engine for contest selection and recommendation."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize game selection engine."""
        self.db = db_session or next(get_db())
        self.contest_analyzer = ContestAnalyzer(self.db)
        self.ev_calculator = ExpectedValueCalculator()
        self.risk_assessor = RiskAssessment()
    
    def recommend_contests(self, settings: GameSelectionSettings, skill_edge: float = 0.0) -> List[ContestMetrics]:
        """Generate contest recommendations based on user preferences.
        
        Args:
            settings: User preferences and constraints
            skill_edge: User's estimated skill edge (0.0 = average)
            
        Returns:
            List of recommended contests sorted by overall score
        """
        logger.info(f"Generating contest recommendations for {settings.risk_tolerance.value} user")
        
        # Get available contests from database
        available_contests = self._get_available_contests(settings)
        
        if not available_contests:
            logger.warning("No contests found matching user criteria")
            return []
        
        # Analyze all contests
        analyzed_contests = []
        for contest in available_contests:
            try:
                metrics = self.contest_analyzer.analyze_contest(contest)
                
                # Adjust EV for user skill
                metrics.expected_value = self.ev_calculator.calculate_ev(metrics, skill_edge)
                
                # Apply user filters
                if self._passes_user_filters(metrics, settings):
                    analyzed_contests.append(metrics)
                    
            except Exception as e:
                logger.error(f"Error analyzing contest {contest.contest_id}: {e}")
                continue
        
        if not analyzed_contests:
            logger.warning("No contests passed user filters")
            return []
        
        # Score and rank contests
        scored_contests = self._score_contests(analyzed_contests, settings)
        
        # Select optimal portfolio
        recommended_contests = self._select_optimal_portfolio(scored_contests, settings)
        
        logger.info(f"Recommended {len(recommended_contests)} contests")
        return recommended_contests
    
    def _get_available_contests(self, settings: GameSelectionSettings) -> List[DraftKingsContest]:
        """Get available contests from database based on user settings."""
        query = self.db.query(DraftKingsContest)
        
        # Filter by entry fee
        query = query.filter(DraftKingsContest.entry_fee <= settings.max_entry_fee)
        
        # Filter by field size
        query = query.filter(DraftKingsContest.entries <= settings.max_field_size)
        
        # Get active contests (this would need to be based on start time in real implementation)
        # For now, just get all contests
        contests = query.limit(100).all()  # Limit to prevent overwhelming analysis
        
        return contests
    
    def _passes_user_filters(self, metrics: ContestMetrics, settings: GameSelectionSettings) -> bool:
        """Check if contest passes user-defined filters."""
        # Check minimum fun score
        if metrics.fun_score < settings.min_fun_score:
            return False
        
        # Check minimum expected value
        if metrics.expected_value < settings.min_expected_value:
            return False
        
        # Check preferred contest types
        if settings.preferred_types and metrics.contest_type not in settings.preferred_types:
            return False
        
        # Check avoided contest types
        if settings.avoid_types and metrics.contest_type in settings.avoid_types:
            return False
        
        return True
    
    def _score_contests(self, contests: List[ContestMetrics], settings: GameSelectionSettings) -> List[ContestMetrics]:
        """Score contests based on user preferences."""
        for contest in contests:
            # Calculate composite score based on user risk tolerance
            if settings.risk_tolerance == RiskTolerance.CONSERVATIVE:
                # Conservative users prioritize: EV > Fun > Low Risk
                score = (
                    0.4 * self._normalize_ev(contest.expected_value) +
                    0.3 * contest.fun_score +
                    0.3 * (100 - contest.downside_risk * 100)
                )
            elif settings.risk_tolerance == RiskTolerance.MODERATE:
                # Moderate users prioritize: Fun > EV > Moderate Risk
                score = (
                    0.5 * contest.fun_score +
                    0.3 * self._normalize_ev(contest.expected_value) +
                    0.2 * (100 - contest.downside_risk * 100)
                )
            else:  # AGGRESSIVE
                # Aggressive users prioritize: Fun > High Variance > EV
                score = (
                    0.6 * contest.fun_score +
                    0.2 * self._normalize_variance(contest.variance, contest.entry_fee) +
                    0.2 * self._normalize_ev(contest.expected_value)
                )
            
            # Store score for sorting
            contest.composite_score = score
        
        # Sort by composite score
        return sorted(contests, key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
    
    def _normalize_ev(self, expected_value: float, max_ev: float = 10.0) -> float:
        """Normalize expected value to 0-100 scale."""
        # Clamp EV to reasonable range and normalize
        clamped_ev = max(-5.0, min(max_ev, expected_value))
        return ((clamped_ev + 5.0) / (max_ev + 5.0)) * 100
    
    def _normalize_variance(self, variance: float, entry_fee: float) -> float:
        """Normalize variance to 0-100 scale (higher variance = higher score for aggressive users)."""
        if entry_fee == 0:
            return 0
        
        # Variance relative to entry fee
        relative_variance = variance / (entry_fee * entry_fee)
        return min(100, relative_variance * 10)  # Scale appropriately
    
    def _select_optimal_portfolio(self, scored_contests: List[ContestMetrics], settings: GameSelectionSettings) -> List[ContestMetrics]:
        """Select optimal portfolio of contests within risk constraints."""
        selected_contests = []
        total_risk = 0.0
        
        for contest in scored_contests:
            # Check if adding this contest exceeds risk limits
            potential_risk = total_risk + contest.entry_fee
            potential_portfolio = selected_contests + [contest]
            
            # Calculate portfolio risk
            portfolio_risk = self.risk_assessor.assess_portfolio_risk(potential_portfolio, settings.bankroll)
            
            # Check risk tolerance
            if self.risk_assessor.is_within_risk_tolerance(portfolio_risk, settings):
                selected_contests.append(contest)
                total_risk = potential_risk
                
                # Update recommended entries and bankroll percentage
                contest.recommended_entries = 1  # Single entry for now
                contest.bankroll_percentage = contest.entry_fee / settings.bankroll
                
                # Stop if we've reached max contests
                if len(selected_contests) >= settings.max_contests:
                    break
            else:
                # Skip this contest if it would exceed risk tolerance
                continue
        
        return selected_contests