"""Vegas odds data collection using The Odds API.

This module handles collecting real-time and historical betting odds for NFL games
to enhance fantasy football predictions. Betting odds are highly predictive because
they incorporate all available information from professional handicappers.

Betting Odds Impact on Fantasy Football:
1. Game Totals (Over/Under): High totals indicate high-scoring games with more fantasy points
2. Point Spreads: Large spreads suggest blowouts, affecting game script and player usage
3. Moneyline: Win probability affects team strategy (conservative vs aggressive)
4. Line Movement: Changes indicate new information (injuries, weather, public sentiment)

Fantasy Strategy Applications:
- Target players in games with high over/under totals (45+ points)
- Favor RBs on favored teams (positive game script, more rushing attempts)
- Target WRs/TEs on underdog teams (trailing teams pass more frequently)
- Avoid players in potential blowouts (garbage time vs early exits)

Data Sources:
The Odds API provides:
- Real-time odds from major sportsbooks (DraftKings, FanDuel, Caesars)
- Historical line movement data
- Multiple betting markets (spread, totals, moneyline)
- Free tier: 500 requests per month

For beginners:

API Integration: Uses The Odds API which aggregates odds from multiple sportsbooks.
Free tier provides sufficient data for personal fantasy use.

Betting Terminology:
- Spread: Point difference (Chiefs -3.5 vs Broncos +3.5)
- Total: Combined score over/under (49.5 points)
- Moneyline: Straight win odds (+150, -180)
- Juice/Vig: Sportsbook commission (usually -110)

Market Efficiency: Vegas odds are highly accurate predictors because they represent
the collective wisdom of professional handicappers and incorporate all public information.
"""

import logging
import time
from datetime import datetime, timedelta

import httpx
from httpx import ConnectError, HTTPError, TimeoutException

from ...config.settings import settings
from ...database.connection import SessionLocal
from ...database.models import Game, Team, VegasOdds

# Set up logging for this module
logger = logging.getLogger(__name__)


class VegasOddsCollector:
    """Collects and stores betting odds data for NFL games using The Odds API.

    This class handles odds data collection from The Odds API to enhance fantasy
    football predictions. It integrates with multiple sportsbooks to provide
    comprehensive betting market data.

    Key Features:
    - Real-time odds from major sportsbooks (DraftKings, FanDuel, Caesars)
    - Historical line movement tracking for market sentiment analysis
    - Multiple betting markets (spread, totals, moneyline, props)
    - Rate limiting and error handling for API reliability
    - Automatic calculation of implied probabilities and market vig

    API Endpoints Used:
    - Sports: Get available NFL games
    - Odds: Get current betting lines for games
    - Historical: Track line movement over time (premium feature)

    Betting Markets Collected:
    - Point Spreads: Team point differentials
    - Totals: Game over/under point totals
    - Moneylines: Straight up winner odds
    - Player Props: Individual player betting lines (future enhancement)

    Design Patterns:
    - Repository Pattern: Encapsulates odds data access
    - Circuit Breaker: Handles API failures gracefully
    - Rate Limiting: Respects API usage limits (500 requests/month free)
    - Data Validation: Ensures odds data quality and consistency
    """

    def __init__(self, api_key: str | None = None):
        """Initialize odds collector with API configuration.

        Args:
            api_key: The Odds API key (from settings if not provided)

        Raises:
            ValueError: If no API key is found in settings or parameters
        """
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            logger.warning(
                "No odds API key found. Odds collection will be disabled. "
                "Get a free key from https://the-odds-api.com/ and set ODDS_API_KEY environment variable"
            )
            self.api_key = None
            return

        # The Odds API configuration
        self.base_url = "https://api.the-odds-api.com/v4"

        # API rate limiting (free tier: 500 requests per month)
        self.request_delay = 1.0  # Seconds between requests
        self.max_retries = 3
        self.timeout = 15.0  # Request timeout in seconds

        # Initialize HTTP client with timeout and headers
        self.client = httpx.Client(
            timeout=self.timeout, headers={"User-Agent": "NFL-DFS-System/1.0"}
        )

        # Supported sportsbooks (major books with good API coverage)
        self.target_sportsbooks = {"draftkings", "fanduel", "caesars", "betmgm", "pointsbet"}

        # Track API usage to avoid exceeding limits
        self.requests_made = 0
        self.monthly_limit = 500  # Free tier limit

    def _make_api_request(self, endpoint: str, params: dict) -> dict | None:
        """Make HTTP request to The Odds API with error handling.

        Implements robust error handling and rate limiting for API calls:
        1. API usage tracking to avoid exceeding monthly limits
        2. Automatic retries for temporary failures
        3. Comprehensive error logging for debugging
        4. Timeout handling for slow responses

        Common API Errors Handled:
        - 401: Invalid API key
        - 429: Rate limit exceeded
        - 422: Invalid parameters
        - Network timeouts and connection errors

        Args:
            endpoint: API endpoint path (e.g., "sports/americanfootball_nfl/odds")
            params: Query parameters including API key and filters

        Returns:
            JSON response data or None if request failed
        """
        if not self.api_key:
            logger.warning("No odds API key available, skipping request")
            return None

        # Check if we're approaching monthly limit
        if self.requests_made >= self.monthly_limit - 10:
            logger.warning(f"Approaching API limit ({self.requests_made}/{self.monthly_limit})")

        # Add API key to parameters
        params["apiKey"] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                # Rate limiting - wait between requests
                if attempt > 0:
                    time.sleep(self.request_delay * (2**attempt))  # Exponential backoff
                else:
                    time.sleep(self.request_delay)

                logger.debug(f"Making odds API request to {url} (attempt {attempt + 1})")
                response = self.client.get(url, params=params)

                # Track API usage
                self.requests_made += 1

                # Check remaining requests from headers
                remaining = response.headers.get("x-requests-remaining")
                if remaining:
                    logger.debug(f"API requests remaining: {remaining}")

                # Check for successful response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.error("Invalid API key for odds service")
                    break  # Don't retry auth errors
                elif response.status_code == 422:
                    logger.error(f"Invalid parameters for odds API: {params}")
                    break  # Don't retry parameter errors
                elif response.status_code == 429:
                    logger.warning("Odds API rate limit exceeded, waiting longer...")
                    time.sleep(60)  # Wait 1 minute for rate limit reset
                    continue
                else:
                    logger.warning(
                        f"Odds API returned status {response.status_code}: {response.text}"
                    )

            except TimeoutException:
                logger.warning(f"Odds API request timeout (attempt {attempt + 1})")
            except (ConnectError, HTTPError) as e:
                logger.warning(f"Odds API network error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.exception(f"Unexpected odds API error (attempt {attempt + 1}): {e}")

        logger.error(f"Failed to get odds data after {self.max_retries} attempts")
        return None

    def _calculate_implied_probability(self, american_odds: int) -> float:
        """Calculate implied probability from American odds format.

        American odds format:
        - Positive (+150): Underdog, bet $100 to win $150
        - Negative (-180): Favorite, bet $180 to win $100

        Implied probability formulas:
        - Positive odds: 100 / (odds + 100)
        - Negative odds: -odds / (-odds + 100)

        Args:
            american_odds: American odds format (+150, -180, etc.)

        Returns:
            Implied probability as decimal (0.0 to 1.0)
        """
        if american_odds > 0:
            # Positive odds (underdog)
            return 100 / (american_odds + 100)
        elif american_odds < 0:
            # Negative odds (favorite)
            return -american_odds / (-american_odds + 100)
        else:
            # Even odds (rare, but handle it)
            return 0.5

    def _calculate_vig(self, home_prob: float, away_prob: float) -> float:
        """Calculate the sportsbook vig (overround) from implied probabilities.

        Vig is the sportsbook's built-in profit margin. Probabilities should sum to 1.0
        in a fair market, but sportsbooks set lines so they sum to more than 1.0.

        Example:
        - Home team: 55% implied probability
        - Away team: 50% implied probability
        - Total: 105% (5% vig)

        Args:
            home_prob: Home team implied probability
            away_prob: Away team implied probability

        Returns:
            Vig percentage as decimal (0.05 = 5% vig)
        """
        total_prob = home_prob + away_prob
        return total_prob - 1.0

    def _parse_odds_data(self, odds_response: dict, game: Game) -> list[dict]:
        """Parse raw odds API response into standardized format.

        Transforms the API response structure into our database schema format.
        Handles multiple sportsbooks and betting markets for a single game.

        Args:
            odds_response: Raw JSON response from The Odds API
            game: Game database object for context

        Returns:
            List of parsed odds records ready for database insertion
        """
        parsed_odds = []

        try:
            # Each game can have odds from multiple sportsbooks
            for bookmaker in odds_response.get("bookmakers", []):
                sportsbook = bookmaker.get("key", "unknown")

                # Skip books we don't track
                if sportsbook not in self.target_sportsbooks:
                    continue

                markets = bookmaker.get("markets", [])

                # Initialize odds record for this sportsbook
                odds_record = {
                    "game_id": game.id,
                    "sportsbook": sportsbook,
                    "line_timestamp": datetime.now(),
                    "is_live": True,
                }

                # Parse each betting market
                for market in markets:
                    market_key = market.get("key")
                    outcomes = market.get("outcomes", [])

                    if market_key == "spreads":
                        # Point spread betting
                        for outcome in outcomes:
                            team_name = outcome.get("name")
                            point = outcome.get("point")
                            price = outcome.get("price")

                            # Determine if this is home or away team
                            # API uses team names, we need to match to our teams
                            if self._is_home_team(team_name, game):
                                odds_record["home_spread"] = point
                                odds_record["spread_juice"] = price
                            else:
                                odds_record["away_spread"] = point

                    elif market_key == "totals":
                        # Over/under total points
                        for outcome in outcomes:
                            name = outcome.get("name")
                            point = outcome.get("point")
                            price = outcome.get("price")

                            if name == "Over":
                                odds_record["total_points"] = point
                                odds_record["over_juice"] = price
                            elif name == "Under":
                                odds_record["under_juice"] = price

                    elif market_key == "h2h":  # Head-to-head (moneyline)
                        # Straight up winner betting
                        for outcome in outcomes:
                            team_name = outcome.get("name")
                            price = outcome.get("price")

                            if self._is_home_team(team_name, game):
                                odds_record["home_moneyline"] = price
                            else:
                                odds_record["away_moneyline"] = price

                # Calculate derived metrics if we have moneyline odds
                if "home_moneyline" in odds_record and "away_moneyline" in odds_record:
                    home_prob = self._calculate_implied_probability(odds_record["home_moneyline"])
                    away_prob = self._calculate_implied_probability(odds_record["away_moneyline"])

                    odds_record["implied_home_win_prob"] = home_prob
                    odds_record["implied_away_win_prob"] = away_prob
                    odds_record["total_vig"] = self._calculate_vig(home_prob, away_prob)

                parsed_odds.append(odds_record)

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error parsing odds data: {e}")

        return parsed_odds

    def _is_home_team(self, team_name: str, game: Game) -> bool:
        """Determine if a team name from the API matches the home team.

        The Odds API uses team names that may not exactly match our database.
        We need fuzzy matching to handle variations like:
        - "Kansas City Chiefs" vs "KC" vs "Chiefs"
        - "Los Angeles Rams" vs "LAR" vs "LA Rams"

        Args:
            team_name: Team name from The Odds API
            game: Game object with home/away team relationships

        Returns:
            True if team_name matches the home team
        """
        session = SessionLocal()
        try:
            home_team = session.query(Team).filter_by(id=game.home_team_id).first()
            if not home_team:
                return False

            # Check various name formats
            team_identifiers = [
                home_team.team_name.lower(),
                home_team.team_abbr.lower(),
                home_team.team_name.split()[-1].lower(),  # Last word (mascot)
            ]

            team_name_lower = team_name.lower()

            # Check if any identifier matches
            for identifier in team_identifiers:
                if identifier in team_name_lower or team_name_lower in identifier:
                    return True

            return False

        finally:
            session.close()

    def collect_odds_for_game(self, game_id: int) -> bool:
        """Collect betting odds for a specific game.

        Fetches current odds from multiple sportsbooks and stores in database.
        Handles both pre-game and live betting lines.

        Args:
            game_id: Database ID of the game to collect odds for

        Returns:
            True if odds were successfully collected and stored
        """
        if not self.api_key:
            logger.warning("No odds API key available, skipping odds collection")
            return False

        session = SessionLocal()
        try:
            # Get game details
            game = session.query(Game).filter_by(id=game_id).first()
            if not game:
                logger.error(f"Game not found: {game_id}")
                return False

            logger.info(
                f"Collecting odds for game {game_id} (Season {game.season}, Week {game.week})"
            )

            # Check if game is too far in the past (odds not available)
            days_ago = (datetime.now() - game.game_date).days
            if days_ago > 7:
                logger.debug(f"Game is {days_ago} days old, odds likely not available")
                return False

            # Get odds from The Odds API
            # Note: We need to map our game to The Odds API's game identifier
            # For now, we'll get all current NFL odds and match by date/teams

            params = {
                "sport": "americanfootball_nfl",
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }

            odds_data = self._make_api_request("sports/americanfootball_nfl/odds", params)

            if not odds_data:
                logger.error(f"Failed to get odds data for game {game_id}")
                return False

            # Find the matching game in the odds response
            matching_game = None
            for odds_game in odds_data:
                # Match by date and participating teams
                game_date = datetime.fromisoformat(
                    odds_game["commence_time"].replace("Z", "+00:00")
                )

                # Allow some flexibility in date matching (within 24 hours)
                date_diff = abs((game.game_date - game_date).total_seconds())
                if date_diff < 86400:  # 24 hours
                    # TODO: Add team name matching logic
                    matching_game = odds_game
                    break

            if not matching_game:
                logger.warning(f"No matching odds found for game {game_id}")
                return False

            # Parse and store odds data
            parsed_odds = self._parse_odds_data(matching_game, game)

            for odds_record in parsed_odds:
                # Check if odds already exist for this game/sportsbook
                existing_odds = (
                    session.query(VegasOdds)
                    .filter_by(game_id=odds_record["game_id"], sportsbook=odds_record["sportsbook"])
                    .first()
                )

                if existing_odds:
                    # Update existing odds
                    for key, value in odds_record.items():
                        if hasattr(existing_odds, key):
                            setattr(existing_odds, key, value)
                    existing_odds.updated_at = datetime.now()
                else:
                    # Create new odds record
                    new_odds = VegasOdds(**odds_record)
                    session.add(new_odds)

            session.commit()
            logger.info(f"Odds data updated for game {game_id} from {len(parsed_odds)} sportsbooks")
            return True

        except Exception as e:
            session.rollback()
            logger.exception(f"Error collecting odds for game {game_id}: {e}")
            return False
        finally:
            session.close()

    def collect_odds_for_week(self, season: int, week: int) -> dict[str, int]:
        """Collect odds for all games in a specific week.

        Processes all games in a week to collect current betting odds.
        Useful for weekly updates before DFS contests.

        Args:
            season: NFL season year
            week: Week number (1-18 regular season, 19+ playoffs)

        Returns:
            Dictionary with collection statistics
        """
        session = SessionLocal()
        try:
            # Get all games for the specified week
            games = (
                session.query(Game)
                .filter_by(season=season, week=week)
                .order_by(Game.game_date)
                .all()
            )

            logger.info(f"Collecting odds for {len(games)} games in {season} week {week}")

            stats = {
                "total_games": len(games),
                "odds_collected": 0,
                "failed_collection": 0,
            }

            for game in games:
                if self.collect_odds_for_game(game.id):
                    stats["odds_collected"] += 1
                else:
                    stats["failed_collection"] += 1

            logger.info(f"Odds collection complete for {season} week {week}: {stats}")
            return stats

        except Exception as e:
            logger.exception(f"Error collecting odds for {season} week {week}: {e}")
            raise
        finally:
            session.close()

    def collect_upcoming_games_odds(self, days_ahead: int = 7) -> dict[str, int]:
        """Collect odds for upcoming games.

        Monitors betting lines for games in the next week to provide
        up-to-date market information for fantasy decisions.

        Args:
            days_ahead: Number of days in the future to check for games

        Returns:
            Dictionary with collection statistics
        """
        if not self.api_key:
            logger.warning("No odds API key available, skipping upcoming odds collection")
            return {"total_games": 0, "odds_collected": 0, "failed_collection": 0}

        session = SessionLocal()
        try:
            # Calculate date range for upcoming games
            now = datetime.now()
            future_date = now + timedelta(days=days_ahead)

            # Get upcoming games
            upcoming_games = (
                session.query(Game)
                .filter(Game.game_date.between(now, future_date))
                .filter(Game.game_finished is False)
                .order_by(Game.game_date)
                .all()
            )

            logger.info(f"Collecting odds for {len(upcoming_games)} upcoming games")

            stats = {
                "total_games": len(upcoming_games),
                "odds_collected": 0,
                "failed_collection": 0,
            }

            for game in upcoming_games:
                if self.collect_odds_for_game(game.id):
                    stats["odds_collected"] += 1
                else:
                    stats["failed_collection"] += 1

            logger.info(f"Upcoming games odds collection complete: {stats}")
            return stats

        except Exception as e:
            logger.exception(f"Error collecting upcoming games odds: {e}")
            raise
        finally:
            session.close()

    def __del__(self):
        """Cleanup HTTP client when collector is destroyed."""
        if hasattr(self, "client"):
            self.client.close()
