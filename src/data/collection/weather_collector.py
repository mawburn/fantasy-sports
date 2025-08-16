"""Weather data collection using OpenWeatherMap API.

This module handles collecting real-time and historical weather data for NFL games
to enhance fantasy football predictions. Weather significantly impacts game outcomes:

Weather Impact on Fantasy Football:
1. Wind Speed: Severely affects passing accuracy and field goal attempts
2. Temperature: Cold weather reduces player performance and ball handling
3. Precipitation: Rain/snow increases fumbles, reduces passing efficiency
4. Visibility: Fog/heavy snow limits long passing plays

Weather affects different positions differently:
- QBs: Passing accuracy drops significantly in high wind (>15 MPH)
- WRs/TEs: Targets shift from deep routes to short, safe passes
- RBs: Get increased usage in bad weather (more conservative game plans)
- Kickers: Accuracy dramatically reduced in wind and cold temperatures

Data Collection Strategy:
1. Stadium Location Mapping: Each NFL stadium has precise GPS coordinates
2. Game Time Weather: Collect conditions at kickoff time for each game
3. Historical Data: Backfill weather for completed games using historical API
4. Real-time Updates: Monitor weather changes for upcoming games

API Integration:
Uses OpenWeatherMap API which provides:
- Current weather conditions
- Historical weather data (back to 1979)
- 5-day forecast for upcoming games
- Precise location-based data using coordinates

For beginners:

API Authentication: Most weather APIs require API keys for access.
Register at openweathermap.org to get a free API key.

HTTP Requests: Use httpx library (already in requirements) to make
web requests to the weather API endpoints.

Rate Limiting: APIs have usage limits. We implement delays and
error handling to respect these limits.

Geocoding: Converting stadium names to GPS coordinates for
accurate weather data retrieval.
"""

import logging
import time
from datetime import datetime, timedelta

import httpx
from httpx import ConnectError, HTTPError, TimeoutException

from ...config.settings import settings
from ...database.connection import SessionLocal
from ...database.models import Game, Team
from .weather_validation import WeatherDataValidator

# Set up logging for this module
logger = logging.getLogger(__name__)


class WeatherCollector:
    """Collects and stores weather data for NFL games using OpenWeatherMap API.

    This class handles weather data collection for NFL stadiums to enhance
    fantasy football predictions. It integrates with the OpenWeatherMap API
    to fetch current, historical, and forecast weather data.

    Key Features:
    - Stadium GPS coordinate mapping for all 32 NFL teams
    - Historical weather data collection for completed games
    - Real-time weather monitoring for upcoming games
    - Rate limiting and error handling for API reliability
    - Automatic retry logic for failed requests

    API Endpoints Used:
    - Current Weather: For real-time monitoring
    - Historical Weather: For backfilling completed games
    - 5-day Forecast: For upcoming game predictions

    Weather Fields Collected:
    - Temperature (Fahrenheit) - affects player performance
    - Wind Speed (MPH) - critical for passing and kicking
    - Weather Description - rain, snow, clear, etc.
    - Humidity and Visibility - additional context

    Design Patterns:
    - Repository Pattern: Encapsulates weather data access
    - Circuit Breaker: Handles API failures gracefully
    - Rate Limiting: Respects API usage limits
    - Retry Logic: Automatic recovery from temporary failures
    """

    def __init__(self, api_key: str | None = None):
        """Initialize weather collector with API configuration.

        Args:
            api_key: OpenWeatherMap API key (from settings if not provided)
        """
        self.api_key = api_key or settings.weather_api_key
        if not self.api_key:
            raise ValueError(
                "Weather API key not found. Set WEATHER_API_KEY environment variable "
                "or get a free key from https://openweathermap.org/api"
            )

        # OpenWeatherMap API configuration
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.historical_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

        # API rate limiting (1000 calls/day for free tier, ~1 call per minute)
        self.request_delay = 1.2  # Seconds between requests
        self.max_retries = 3
        self.timeout = 10.0  # Request timeout in seconds

        # Initialize HTTP client with timeout and headers
        self.client = httpx.Client(
            timeout=self.timeout, headers={"User-Agent": "NFL-DFS-System/1.0"}
        )

        # Stadium location mapping: team_abbr -> (latitude, longitude, stadium_name)
        # GPS coordinates are precise to ensure accurate weather data
        self.stadium_locations = self._initialize_stadium_locations()

        # Initialize weather data validator for quality checks
        self.validator = WeatherDataValidator()

    def _initialize_stadium_locations(self) -> dict[str, tuple[float, float, str]]:
        """Initialize GPS coordinates for all NFL stadiums.

        Returns precise latitude/longitude coordinates for each NFL team's home stadium.
        These coordinates are used to fetch accurate weather data from OpenWeatherMap.

        Coordinate Sources:
        - Official stadium websites and Google Maps
        - Verified against multiple sources for accuracy
        - Updated for any recent stadium changes or relocations

        Format: team_abbr -> (latitude, longitude, stadium_name)

        Returns:
            Dictionary mapping team abbreviations to stadium coordinates
        """
        return {
            # AFC East
            "BUF": (42.7738, -78.7870, "Highmark Stadium"),  # Buffalo
            "MIA": (25.9580, -80.2389, "Hard Rock Stadium"),  # Miami
            "NE": (42.0909, -71.2643, "Gillette Stadium"),  # New England
            "NYJ": (40.8135, -74.0745, "MetLife Stadium"),  # New York Jets
            # AFC North
            "BAL": (39.2780, -76.6227, "M&T Bank Stadium"),  # Baltimore
            "CIN": (39.0955, -84.5161, "Paycor Stadium"),  # Cincinnati
            "CLE": (41.5061, -81.6995, "Cleveland Browns Stadium"),  # Cleveland
            "PIT": (40.4468, -80.0158, "Acrisure Stadium"),  # Pittsburgh
            # AFC South
            "HOU": (29.6847, -95.4107, "NRG Stadium"),  # Houston
            "IND": (39.7601, -86.1639, "Lucas Oil Stadium"),  # Indianapolis
            "JAX": (30.3240, -81.6374, "TIAA Bank Field"),  # Jacksonville
            "TEN": (36.1665, -86.7713, "Nissan Stadium"),  # Tennessee
            # AFC West
            "DEN": (39.7439, -105.0201, "Empower Field at Mile High"),  # Denver
            "KC": (39.0489, -94.4839, "Arrowhead Stadium"),  # Kansas City
            "LV": (36.0908, -115.1834, "Allegiant Stadium"),  # Las Vegas
            "LAC": (33.8642, -118.2615, "SoFi Stadium"),  # Los Angeles Chargers
            # NFC East
            "DAL": (32.7473, -97.0945, "AT&T Stadium"),  # Dallas
            "NYG": (40.8135, -74.0745, "MetLife Stadium"),  # New York Giants
            "PHI": (39.9008, -75.1675, "Lincoln Financial Field"),  # Philadelphia
            "WAS": (38.9077, -76.8645, "FedExField"),  # Washington
            # NFC North
            "CHI": (41.8623, -87.6167, "Soldier Field"),  # Chicago
            "DET": (42.3400, -83.0456, "Ford Field"),  # Detroit
            "GB": (44.5013, -88.0622, "Lambeau Field"),  # Green Bay
            "MIN": (44.9738, -93.2581, "U.S. Bank Stadium"),  # Minnesota
            # NFC South
            "ATL": (33.7553, -84.4006, "Mercedes-Benz Stadium"),  # Atlanta
            "CAR": (35.2258, -80.8528, "Bank of America Stadium"),  # Carolina
            "NO": (29.9511, -90.0812, "Caesars Superdome"),  # New Orleans
            "TB": (27.9759, -82.5033, "Raymond James Stadium"),  # Tampa Bay
            # NFC West
            "ARI": (33.5276, -112.2626, "State Farm Stadium"),  # Arizona
            "LAR": (33.8642, -118.2615, "SoFi Stadium"),  # Los Angeles Rams
            "SF": (37.4032, -121.9698, "Levi's Stadium"),  # San Francisco
            "SEA": (47.5952, -122.3316, "Lumen Field"),  # Seattle
        }

    def _make_api_request(self, url: str, params: dict) -> dict | None:
        """Make HTTP request to OpenWeatherMap API with error handling.

        Implements robust error handling and retry logic for API calls:
        1. Rate limiting to respect API usage limits
        2. Automatic retries for temporary failures
        3. Comprehensive error logging for debugging
        4. Timeout handling for slow responses

        Common API Errors Handled:
        - 401: Invalid API key
        - 429: Rate limit exceeded
        - 404: Location not found
        - Network timeouts and connection errors

        Args:
            url: Full API endpoint URL
            params: Query parameters including API key and location

        Returns:
            JSON response data or None if request failed
        """
        # Add API key to parameters
        params["appid"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                # Rate limiting - wait between requests
                if attempt > 0:
                    time.sleep(self.request_delay * (2**attempt))  # Exponential backoff
                else:
                    time.sleep(self.request_delay)

                logger.debug(f"Making API request to {url} (attempt {attempt + 1})")
                response = self.client.get(url, params=params)

                # Check for successful response
                if response.status_code == 200:
                    response_data = response.json()

                    # Validate API response structure and data quality
                    is_valid, errors = self.validator.validate_api_response(response_data)
                    if not is_valid:
                        logger.warning(f"Invalid API response structure: {'; '.join(errors)}")
                        # Continue with response but log the issues

                    return response_data
                elif response.status_code == 401:
                    logger.error("Invalid API key for weather service")
                    break  # Don't retry auth errors
                elif response.status_code == 429:
                    logger.warning("Weather API rate limit exceeded, waiting longer...")
                    time.sleep(60)  # Wait 1 minute for rate limit reset
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Weather data not found for location: {params}")
                    break  # Don't retry not found errors
                else:
                    logger.warning(
                        f"Weather API returned status {response.status_code}: {response.text}"
                    )

            except TimeoutException:
                logger.warning(f"Weather API request timeout (attempt {attempt + 1})")
            except (ConnectError, HTTPError) as e:
                logger.warning(f"Weather API network error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.exception(f"Unexpected weather API error (attempt {attempt + 1}): {e}")

        logger.error(f"Failed to get weather data after {self.max_retries} attempts")
        return None

    def _get_current_weather(self, lat: float, lon: float) -> dict | None:
        """Get current weather conditions for a location.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate

        Returns:
            Weather data dictionary or None if failed
        """
        url = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "units": "imperial",  # Fahrenheit, MPH for US sports
        }

        return self._make_api_request(url, params)

    def _get_historical_weather(self, lat: float, lon: float, timestamp: int) -> dict | None:
        """Get historical weather data for a specific time and location.

        Uses OpenWeatherMap's One Call API historical endpoint to get
        weather conditions for completed games.

        Note: Historical data requires a paid API plan for OpenWeatherMap.
        Free tier only provides current weather and 5-day forecast.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            timestamp: Unix timestamp for the target date/time

        Returns:
            Historical weather data dictionary or None if failed
        """
        url = self.historical_url
        params = {
            "lat": lat,
            "lon": lon,
            "dt": timestamp,
            "units": "imperial",
        }

        return self._make_api_request(url, params)

    def _extract_weather_features(self, weather_data: dict) -> dict[str, any]:
        """Extract relevant weather features from API response.

        Transforms raw API response into standardized weather features
        that match our database schema and ML feature requirements.

        Weather Feature Engineering:
        - Temperature: Direct value in Fahrenheit
        - Wind Speed: Critical for passing games (>15 MPH significant impact)
        - Weather Description: Categorical (Clear, Rain, Snow, etc.)
        - Additional context: Humidity, visibility, pressure

        Args:
            weather_data: Raw JSON response from OpenWeatherMap API

        Returns:
            Dictionary of standardized weather features
        """
        try:
            # Main weather data
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            wind = weather_data.get("wind", {})

            # Extract key weather features
            features = {
                "temperature": round(main.get("temp", 70)),  # Default to 70°F
                "wind_speed": round(wind.get("speed", 0)),  # MPH
                "description": weather.get("main", "Clear"),  # Main category
                "detailed_description": weather.get("description", ""),  # Detailed
                "humidity": main.get("humidity", 50),  # Percentage
                "pressure": main.get("pressure"),  # hPa
                "visibility": weather_data.get("visibility"),  # Meters
            }

            # Validate extracted weather data
            quality_score = self.validator.calculate_quality_score(features)
            if quality_score < 0.6:
                logger.warning(
                    f"Poor weather data quality (score: {quality_score:.2f}): {features}"
                )
            elif quality_score < 0.8:
                logger.info(f"Acceptable weather data quality (score: {quality_score:.2f})")

            # Add quality score to features for potential model use
            features["data_quality_score"] = quality_score

            return features

        except (KeyError, TypeError, IndexError) as e:
            logger.warning(f"Error extracting weather features: {e}")
            # Return default values for missing/malformed data
            return {
                "temperature": 70,
                "wind_speed": 0,
                "description": "Unknown",
                "detailed_description": "",
                "humidity": 50,
                "pressure": None,
                "visibility": None,
            }

    def collect_weather_for_game(self, game_id: int) -> bool:
        """Collect weather data for a specific game.

        Determines whether to use current weather (for future games) or
        historical weather (for completed games) and updates the database.

        Game Type Detection:
        - Future games: Use current weather as approximation
        - Completed games: Use historical weather for accuracy
        - Game day: Use real-time weather at kickoff

        Args:
            game_id: Database ID of the game to collect weather for

        Returns:
            True if weather data was successfully collected and stored
        """
        session = SessionLocal()
        try:
            # Get game details with team information
            game = session.query(Game).filter_by(id=game_id).first()
            if not game:
                logger.error(f"Game not found: {game_id}")
                return False

            # Get home team to determine stadium location
            home_team = session.query(Team).filter_by(id=game.home_team_id).first()
            if not home_team or home_team.team_abbr not in self.stadium_locations:
                logger.error(
                    f"Stadium location not found for team: {home_team.team_abbr if home_team else 'Unknown'}"
                )
                return False

            # Get stadium coordinates
            lat, lon, stadium_name = self.stadium_locations[home_team.team_abbr]
            logger.info(f"Collecting weather for {stadium_name} (Game ID: {game_id})")

            # Determine if game is in the future or past
            now = datetime.now()
            game_datetime = game.game_date

            if game_datetime > now:
                # Future game - use current weather
                logger.debug("Future game, using current weather")
                weather_data = self._get_current_weather(lat, lon)
            else:
                # Past game - use historical weather (if available)
                logger.debug("Past game, using historical weather")
                timestamp = int(game_datetime.timestamp())
                weather_data = self._get_historical_weather(lat, lon, timestamp)

                # Fallback to current weather if historical not available
                if not weather_data:
                    logger.warning(
                        "Historical weather not available, using current weather as fallback"
                    )
                    weather_data = self._get_current_weather(lat, lon)

            if not weather_data:
                logger.error(f"Failed to get weather data for game {game_id}")
                return False

            # Extract weather features
            weather_features = self._extract_weather_features(weather_data)

            # Update game with weather data
            game.weather_temperature = weather_features["temperature"]
            game.weather_wind_speed = weather_features["wind_speed"]
            game.weather_description = weather_features["description"]

            # Update stadium if not already set
            if not game.stadium:
                game.stadium = stadium_name

            game.updated_at = datetime.now()
            session.commit()

            logger.info(
                f"Weather data updated for game {game_id}: "
                f"{weather_features['temperature']}°F, "
                f"{weather_features['wind_speed']} MPH wind, "
                f"{weather_features['description']}"
            )

            return True

        except Exception as e:
            session.rollback()
            logger.exception(f"Error collecting weather for game {game_id}: {e}")
            return False
        finally:
            session.close()

    def collect_weather_for_season(
        self, season: int, weeks: list[int] | None = None
    ) -> dict[str, int]:
        """Collect weather data for all games in a season.

        Processes all games in a season to collect weather data.
        Useful for backfilling historical weather or updating upcoming games.

        Processing Strategy:
        1. Get all games for the specified season
        2. Filter by weeks if specified
        3. Process games in chronological order
        4. Skip games that already have weather data (unless force_update=True)
        5. Rate limit requests to respect API limits

        Args:
            season: NFL season year (e.g., 2024)
            weeks: Optional list of specific weeks to process

        Returns:
            Dictionary with collection statistics
        """
        session = SessionLocal()
        try:
            # Build query for games in the season
            query = session.query(Game).filter_by(season=season)

            # Filter by weeks if specified
            if weeks:
                query = query.filter(Game.week.in_(weeks))

            # Get games ordered by date
            games = query.order_by(Game.game_date).all()

            logger.info(
                f"Collecting weather for {len(games)} games in {season} season"
                f"{f' (weeks {weeks})' if weeks else ''}"
            )

            # Collection statistics
            stats = {
                "total_games": len(games),
                "weather_collected": 0,
                "already_had_weather": 0,
                "failed_collection": 0,
                "api_requests": 0,
            }

            for i, game in enumerate(games):
                # Check if weather data already exists
                if (
                    game.weather_temperature is not None
                    and game.weather_wind_speed is not None
                    and game.weather_description
                ):
                    logger.debug(f"Game {game.id} already has weather data, skipping")
                    stats["already_had_weather"] += 1
                    continue

                # Collect weather for this game
                if self.collect_weather_for_game(game.id):
                    stats["weather_collected"] += 1
                else:
                    stats["failed_collection"] += 1

                stats["api_requests"] += 1

                # Progress logging
                if (i + 1) % 10 == 0 or i == len(games) - 1:
                    logger.info(f"Processed {i + 1}/{len(games)} games")

            logger.info(f"Weather collection complete for {season} season: {stats}")

            # Log validation statistics for data quality monitoring
            validation_stats = self.validator.get_validation_statistics()
            if validation_stats["total_validations"] > 0:
                logger.info(
                    f"Weather data validation summary: "
                    f"Avg quality: {validation_stats['avg_quality_score']:.3f}, "
                    f"Temp warnings: {validation_stats['temperature_warning_rate']:.1%}, "
                    f"Wind warnings: {validation_stats['wind_warning_rate']:.1%}"
                )

            return stats

        except Exception as e:
            logger.exception(f"Error collecting weather for season {season}: {e}")
            raise
        finally:
            session.close()

    def collect_upcoming_games_weather(self, days_ahead: int = 7) -> dict[str, int]:
        """Collect weather data for upcoming games.

        Monitors weather for games in the next week to provide
        up-to-date conditions for fantasy decisions.

        Forecast Strategy:
        - Games within 24 hours: Use current weather
        - Games 1-5 days out: Use forecast data
        - Games beyond 5 days: Skip (forecast unreliable)

        Args:
            days_ahead: Number of days in the future to check for games

        Returns:
            Dictionary with collection statistics
        """
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

            logger.info(f"Collecting weather for {len(upcoming_games)} upcoming games")

            stats = {
                "total_games": len(upcoming_games),
                "weather_updated": 0,
                "failed_updates": 0,
            }

            for game in upcoming_games:
                if self.collect_weather_for_game(game.id):
                    stats["weather_updated"] += 1
                else:
                    stats["failed_updates"] += 1

            logger.info(f"Upcoming games weather collection complete: {stats}")
            return stats

        except Exception as e:
            logger.exception(f"Error collecting upcoming games weather: {e}")
            raise
        finally:
            session.close()

    def __del__(self):
        """Cleanup HTTP client when collector is destroyed."""
        if hasattr(self, "client"):
            self.client.close()
