"""Weather data collection and processing for NFL games.

This module handles weather-related functionality including:
- Stadium location data for outdoor venues
- Historical weather data collection via APIs
- Weather forecast retrieval
- Weather feature engineering for DFS models
"""

import os
import logging
from typing import Dict, Optional, Any
import requests
import time

logger = logging.getLogger(__name__)

# NFL Stadium locations - OUTDOOR ONLY (weather affects gameplay)
OUTDOOR_STADIUMS = {
    "BAL": {"name": "M&T Bank Stadium", "lat": 39.2781, "lon": -76.6227},
    "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870},
    "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8533},
    "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167},
    "CIN": {"name": "Paycor Stadium", "lat": 39.0955, "lon": -84.5160},
    "CLE": {"name": "FirstEnergy Stadium", "lat": 41.5061, "lon": -81.6995},
    "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201},
    "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622},
    "JAX": {"name": "TIAA Bank Field", "lat": 32.0815, "lon": -81.6370},
    "KC": {"name": "Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839},
    "MIA": {"name": "Hard Rock Stadium", "lat": 25.9581, "lon": -80.2389},
    "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643},
    "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
    "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745},
    "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675},
    "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158},
    "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316},
    "TB": {"name": "Raymond James Stadium", "lat": 27.9756, "lon": -82.5034},
    "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713},
    "WAS": {"name": "FedExField", "lat": 38.9077, "lon": -76.8645},
}

# Teams with domed/covered stadiums (weather doesn't affect gameplay)
DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "LV", "MIN", "NO", "SF"}


def is_outdoor_stadium(team_abbr: str) -> bool:
    """Check if a team plays in an outdoor stadium."""
    return team_abbr in OUTDOOR_STADIUMS


def get_stadium_info(team_abbr: str) -> Optional[Dict[str, Any]]:
    """Get stadium information for a team."""
    return OUTDOOR_STADIUMS.get(team_abbr)




def get_historical_weather_batch(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    rate_limit_delay: float = 1.5,
) -> Optional[Dict]:
    """Get historical weather data for a date range from Visual Crossing API."""
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if not api_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"

    params = {
        "unitGroup": "us",
        "key": api_key,
        "elements": "datetime,temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure",
        "include": "days",
    }

    try:
        time.sleep(rate_limit_delay)
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logger.warning("Visual Crossing API rate limit reached")
            return None
        else:
            logger.error(f"Visual Crossing API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching weather batch: {e}")
        return None


def get_historical_weather(
    lat: float, lon: float, date: str, rate_limit_delay: float = 1.0
) -> Optional[Dict]:
    """Get historical weather data for a specific date from Visual Crossing API."""
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if not api_key:
        logger.warning("VISUAL_CROSSING_API_KEY not found in environment")
        return None

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date}"

    params = {
        "unitGroup": "us",
        "key": api_key,
        "elements": "datetime,temp,feelslike,humidity,windspeed,winddir,precipprob,conditions,visibility,pressure",
    }

    try:
        time.sleep(rate_limit_delay)
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("days") and len(data["days"]) > 0:
                return data["days"][0]
        elif response.status_code == 429:
            logger.warning("Visual Crossing API rate limit reached")
            return None
        else:
            logger.error(f"Visual Crossing API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching historical weather: {e}")
        return None


def get_weather_forecast(lat: float, lon: float) -> Optional[Dict]:
    """Get weather forecast from weather.gov API (US only, no API key needed)."""
    try:
        # Get grid coordinates for the location
        point_url = f"https://api.weather.gov/points/{lat},{lon}"
        response = requests.get(point_url, timeout=10)
        if response.status_code != 200:
            logger.error(f"Weather.gov points API error: {response.status_code}")
            return None

        point_data = response.json()
        forecast_url = point_data["properties"]["forecast"]

        # Get the forecast
        forecast_response = requests.get(forecast_url, timeout=10)
        if forecast_response.status_code != 200:
            logger.error(f"Weather.gov forecast API error: {forecast_response.status_code}")
            return None

        forecast_data = forecast_response.json()
        periods = forecast_data["properties"]["periods"]

        # Get the first period (usually next 12 hours)
        if periods:
            period = periods[0]
            # Parse weather.gov format into our standard format
            return {
                "temp": period.get("temperature"),
                "feelslike": period.get("temperature"),  # weather.gov doesn't provide feels like
                "windspeed": int(period.get("windSpeed", "0").split()[0]) if period.get("windSpeed") else 0,
                "winddir": period.get("windDirection"),
                "conditions": period.get("shortForecast"),
                "humidity": None,  # Not provided by weather.gov
                "precipprob": None,  # Would need to parse from detailed forecast
                "visibility": None,
                "pressure": None,
            }
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
        return None


def parse_wind_speed(wind_str: str) -> Optional[int]:
    """Parse wind speed from string like '10 mph'.

    Args:
        wind_str: Wind speed string (e.g., '10 mph', '15 to 20 mph')

    Returns:
        Integer wind speed or None if cannot parse
    """
    try:
        if wind_str and "mph" in wind_str:
            # Handle ranges like "10 to 20 mph" by taking first number
            return int(wind_str.split()[0])
    except (ValueError, AttributeError, IndexError):
        pass
    return None


def calculate_weather_impact_score(weather_data: Dict) -> float:
    """Calculate a weather impact score for fantasy predictions.

    Higher scores indicate worse weather conditions that may negatively impact:
    - Passing games (wind, precipitation)
    - Scoring (extreme cold, precipitation)
    - Visibility (fog, heavy precipitation)

    Returns a score from 0.0 (perfect conditions) to 1.0 (severe impact).
    """
    if not weather_data:
        return 0.0

    score = 0.0

    # Wind impact (0-40+ mph scaled to 0-0.3)
    wind_speed = weather_data.get("windspeed", 0) or 0
    if wind_speed > 15:
        score += min((wind_speed - 15) / 25 * 0.3, 0.3)

    # Temperature impact (extreme cold or heat)
    temp = weather_data.get("temp", 70) or 70
    if temp < 32:  # Freezing
        score += min((32 - temp) / 32 * 0.2, 0.2)
    elif temp > 90:  # Extreme heat
        score += min((temp - 90) / 20 * 0.1, 0.1)

    # Precipitation impact
    precip_prob = weather_data.get("precipprob", 0) or 0
    score += min(precip_prob / 100 * 0.3, 0.3)

    # Visibility impact (if available)
    visibility = weather_data.get("visibility", 10) or 10
    if visibility < 5:
        score += min((5 - visibility) / 5 * 0.2, 0.2)

    return min(score, 1.0)


def get_weather_features(weather_data: Dict) -> Dict[str, float]:
    """Extract normalized weather features for ML models.

    Returns dictionary with normalized features:
    - temperature_norm: 0-1 scale (0-100°F)
    - wind_norm: 0-1 scale (0-40 mph)
    - precipitation_norm: 0-1 scale
    - weather_impact: composite impact score
    """
    if not weather_data:
        return {
            "temperature_norm": 0.5,  # Default to neutral
            "wind_norm": 0.0,
            "precipitation_norm": 0.0,
            "weather_impact": 0.0,
        }

    features = {}

    # Normalize temperature (0-100°F to 0-1)
    temp = weather_data.get("temp", 70) or 70
    features["temperature_norm"] = max(0, min(1, temp / 100))

    # Normalize wind speed (0-40 mph to 0-1)
    wind = weather_data.get("windspeed", 0) or 0
    features["wind_norm"] = max(0, min(1, wind / 40))

    # Normalize precipitation probability
    precip = weather_data.get("precipprob", 0) or 0
    features["precipitation_norm"] = max(0, min(1, precip / 100))

    # Calculate composite impact score
    features["weather_impact"] = calculate_weather_impact_score(weather_data)

    return features
