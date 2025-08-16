"""Weather data validation and quality checks.

This module provides validation functions for weather data collection to ensure:
1. API responses are correctly formatted and complete
2. Weather data values are within realistic ranges
3. Data quality is tracked and reported
4. Invalid or suspicious weather data is flagged for review

Key Validation Areas:
- Temperature range validation (realistic NFL game temperatures)
- Wind speed validation (realistic meteorological ranges)
- Weather description parsing and categorization
- Stadium location coordinate validation
- API response structure validation

For beginners:

Data Validation: The process of checking that data meets certain criteria
before it's stored or used. This prevents bad data from corrupting models.

Range Validation: Checking that numerical values fall within expected bounds.
For example, NFL games don't happen at -50°F or 150°F.

Outlier Detection: Identifying values that are technically valid but
unusually extreme, which might indicate data quality issues.

Data Quality Metrics: Quantitative measures of how complete, accurate,
and reliable the collected data is.
"""

import logging

logger = logging.getLogger(__name__)


class WeatherDataValidator:
    """Validates weather data quality and correctness.

    This class provides comprehensive validation for weather data collected
    from external APIs to ensure data quality and reliability for ML models.

    Validation Categories:
    1. Structure validation - API response format checking
    2. Range validation - Realistic value bounds checking
    3. Consistency validation - Cross-field validation checks
    4. Quality scoring - Overall data quality assessment

    Design Principles:
    - Fail gracefully: Log warnings for suspicious data but don't crash
    - Be conservative: Better to flag good data as suspicious than miss bad data
    - Provide context: Include reasons for validation failures
    - Track metrics: Quantify data quality for monitoring
    """

    def __init__(self):
        """Initialize validator with realistic NFL weather ranges."""
        # Temperature ranges based on historical NFL games
        # Coldest NFL game: -9°F (1967 Ice Bowl)
        # Hottest NFL game: ~120°F field temperature in Arizona
        self.temp_range = (-20, 130)  # Fahrenheit, with some buffer

        # Wind speed ranges based on meteorological data
        # Highest recorded NFL game wind: ~50+ MPH
        self.wind_range = (0, 60)  # MPH, with buffer for extreme weather

        # Valid weather description keywords for categorization
        self.valid_weather_keywords = {
            "clear": ["clear", "sunny", "fair"],
            "cloudy": ["cloudy", "overcast", "partly cloudy"],
            "rain": ["rain", "drizzle", "shower", "storm", "thunderstorm"],
            "snow": ["snow", "sleet", "blizzard", "flurries"],
            "fog": ["fog", "mist", "haze"],
            "wind": ["windy", "breezy", "gusty"],
        }

        # Validation metrics tracking
        self.validation_stats = {
            "total_validations": 0,
            "temperature_warnings": 0,
            "wind_warnings": 0,
            "description_warnings": 0,
            "structure_errors": 0,
            "quality_scores": [],
        }

    def validate_api_response(self, response_data: dict) -> tuple[bool, list[str]]:
        """Validate OpenWeatherMap API response structure.

        Checks that the API response contains all required fields with
        appropriate data types. This catches API changes or malformed responses.

        Required Fields:
        - main.temp: Temperature data
        - main.humidity: Humidity percentage
        - wind.speed: Wind speed data
        - weather[0].main: Weather condition category
        - weather[0].description: Detailed weather description

        Args:
            response_data: Raw JSON response from OpenWeatherMap API

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        self.validation_stats["total_validations"] += 1
        errors = []

        # Check top-level structure
        if not isinstance(response_data, dict):
            errors.append("Response is not a dictionary")
            self.validation_stats["structure_errors"] += 1
            return False, errors

        # Check main weather data section
        main_data = response_data.get("main", {})
        if not isinstance(main_data, dict):
            errors.append("Missing or invalid 'main' section")
            self.validation_stats["structure_errors"] += 1
        else:
            # Temperature validation
            if "temp" not in main_data:
                errors.append("Missing temperature data in 'main' section")
            elif not isinstance(main_data["temp"], int | float):
                errors.append("Temperature is not a number")

            # Humidity validation
            if "humidity" not in main_data:
                errors.append("Missing humidity data in 'main' section")
            elif not isinstance(main_data["humidity"], int | float):
                errors.append("Humidity is not a number")

        # Check wind data section
        wind_data = response_data.get("wind", {})
        if not isinstance(wind_data, dict):
            errors.append("Missing or invalid 'wind' section")
            self.validation_stats["structure_errors"] += 1
        else:
            if "speed" not in wind_data:
                errors.append("Missing wind speed data")
            elif not isinstance(wind_data["speed"], int | float):
                errors.append("Wind speed is not a number")

        # Check weather description section
        weather_data = response_data.get("weather", [])
        if not isinstance(weather_data, list) or len(weather_data) == 0:
            errors.append("Missing or empty 'weather' array")
            self.validation_stats["structure_errors"] += 1
        else:
            weather_main = weather_data[0]
            if not isinstance(weather_main, dict):
                errors.append("Invalid weather data structure")
            else:
                if "main" not in weather_main:
                    errors.append("Missing weather main category")
                if "description" not in weather_main:
                    errors.append("Missing weather description")

        is_valid = len(errors) == 0
        if errors:
            self.validation_stats["structure_errors"] += len(errors)

        return is_valid, errors

    def validate_temperature(self, temperature: float) -> tuple[bool, str | None]:
        """Validate temperature value is within realistic range for NFL games.

        Checks both hard limits (impossible values) and soft limits (unusual values).
        NFL games are played in a wide range of conditions but there are practical
        limits based on historical weather data.

        Args:
            temperature: Temperature in Fahrenheit

        Returns:
            Tuple of (is_valid, warning_message)
        """
        # Hard validation - physically impossible or extremely unlikely
        if temperature < self.temp_range[0] or temperature > self.temp_range[1]:
            self.validation_stats["temperature_warnings"] += 1
            return (
                False,
                f"Temperature {temperature}°F is outside realistic range {self.temp_range}",
            )

        # Soft validation - unusual but possible values
        warning = None
        if temperature < 0:
            warning = f"Extremely cold temperature {temperature}°F (verify data accuracy)"
            self.validation_stats["temperature_warnings"] += 1
        elif temperature > 100:
            warning = f"Extremely hot temperature {temperature}°F (verify data accuracy)"
            self.validation_stats["temperature_warnings"] += 1

        return True, warning

    def validate_wind_speed(self, wind_speed: float) -> tuple[bool, str | None]:
        """Validate wind speed value is within realistic meteorological range.

        Wind speed validation is important because extreme winds significantly
        affect gameplay and fantasy performance. We need to catch both impossible
        values and unusually high winds that might indicate data errors.

        Args:
            wind_speed: Wind speed in MPH

        Returns:
            Tuple of (is_valid, warning_message)
        """
        # Hard validation - impossible values
        if wind_speed < self.wind_range[0] or wind_speed > self.wind_range[1]:
            self.validation_stats["wind_warnings"] += 1
            return (
                False,
                f"Wind speed {wind_speed} MPH is outside realistic range {self.wind_range}",
            )

        # Soft validation - unusual but possible values
        warning = None
        if wind_speed > 30:
            warning = f"Very high wind speed {wind_speed} MPH (major game impact expected)"
            self.validation_stats["wind_warnings"] += 1
        elif wind_speed > 20:
            warning = f"High wind speed {wind_speed} MPH (significant game impact expected)"

        return True, warning

    def validate_weather_description(self, description: str) -> tuple[bool, str | None]:
        """Validate weather description and categorize it.

        Ensures the weather description contains recognizable terms and can be
        properly categorized for feature engineering. Unknown descriptions might
        indicate API changes or data quality issues.

        Args:
            description: Weather description string from API

        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not description or not isinstance(description, str):
            self.validation_stats["description_warnings"] += 1
            return False, "Weather description is missing or not a string"

        description_lower = description.lower()

        # Check if description contains any recognized weather keywords
        found_keywords = []
        for category, keywords in self.valid_weather_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                found_keywords.append(category)

        if not found_keywords:
            warning = (
                f"Unrecognized weather description: '{description}' (may need keyword updates)"
            )
            self.validation_stats["description_warnings"] += 1
            return True, warning  # Valid but suspicious

        return True, None

    def validate_stadium_coordinates(
        self, lat: float, lon: float, stadium_name: str
    ) -> tuple[bool, str | None]:
        """Validate stadium GPS coordinates are within reasonable bounds.

        NFL stadiums are all located within the continental US, Alaska, and Mexico.
        Coordinates outside these regions indicate configuration errors.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            stadium_name: Stadium name for error context

        Returns:
            Tuple of (is_valid, warning_message)
        """
        # North America coordinate bounds (generous to include Mexico games)
        lat_bounds = (15.0, 70.0)  # Southern Mexico to Northern Canada
        lon_bounds = (-180.0, -60.0)  # West Coast to East Coast (negative for Western Hemisphere)

        if not (lat_bounds[0] <= lat <= lat_bounds[1]):
            return (
                False,
                f"Latitude {lat} for {stadium_name} is outside North America bounds {lat_bounds}",
            )

        if not (lon_bounds[0] <= lon <= lon_bounds[1]):
            return (
                False,
                f"Longitude {lon} for {stadium_name} is outside North America bounds {lon_bounds}",
            )

        return True, None

    def calculate_quality_score(self, weather_data: dict) -> float:
        """Calculate overall data quality score for weather data.

        Combines multiple quality factors into a single score (0-1) where:
        - 1.0 = Perfect data quality
        - 0.8+ = Good quality, suitable for ML models
        - 0.6-0.8 = Acceptable quality with some concerns
        - <0.6 = Poor quality, may need manual review

        Quality Factors:
        - Data completeness (all fields present)
        - Value reasonableness (within expected ranges)
        - Description recognizability (known weather terms)
        - Temporal consistency (reasonable for season/location)

        Args:
            weather_data: Processed weather data dictionary

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 1.0
        deductions = []

        # Completeness check (30% of score)
        required_fields = ["temperature", "wind_speed", "description"]
        missing_fields = [field for field in required_fields if not weather_data.get(field)]
        if missing_fields:
            deduction = 0.3 * (len(missing_fields) / len(required_fields))
            score -= deduction
            deductions.append(f"Missing fields: {missing_fields} (-{deduction:.2f})")

        # Value range check (25% of score)
        temp = weather_data.get("temperature")
        if temp is not None:
            temp_valid, temp_warning = self.validate_temperature(temp)
            if not temp_valid:
                score -= 0.15
                deductions.append("Invalid temperature (-0.15)")
            elif temp_warning:
                score -= 0.05
                deductions.append("Unusual temperature (-0.05)")

        # Wind validation (25% of score)
        wind = weather_data.get("wind_speed")
        if wind is not None:
            wind_valid, wind_warning = self.validate_wind_speed(wind)
            if not wind_valid:
                score -= 0.15
                deductions.append("Invalid wind speed (-0.15)")
            elif wind_warning:
                score -= 0.05
                deductions.append("Unusual wind speed (-0.05)")

        # Description validation (20% of score)
        desc = weather_data.get("description")
        if desc is not None:
            desc_valid, desc_warning = self.validate_weather_description(desc)
            if not desc_valid:
                score -= 0.15
                deductions.append("Invalid description (-0.15)")
            elif desc_warning:
                score -= 0.05
                deductions.append("Unrecognized description (-0.05)")

        # Ensure score doesn't go below 0
        score = max(0.0, score)

        # Track quality score for statistics
        self.validation_stats["quality_scores"].append(score)

        # Log quality assessment if score is concerning
        if score < 0.8:
            logger.warning(
                f"Weather data quality score: {score:.2f}. Deductions: {', '.join(deductions)}"
            )

        return score

    def get_validation_statistics(self) -> dict:
        """Get summary statistics of validation results.

        Provides metrics on validation performance and data quality trends.
        Useful for monitoring data collection quality over time.

        Returns:
            Dictionary of validation statistics and quality metrics
        """
        stats = self.validation_stats.copy()

        # Calculate quality score statistics
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
            stats["min_quality_score"] = min(stats["quality_scores"])
            stats["max_quality_score"] = max(stats["quality_scores"])
            stats["poor_quality_count"] = sum(1 for score in stats["quality_scores"] if score < 0.6)
        else:
            stats["avg_quality_score"] = 0.0
            stats["min_quality_score"] = 0.0
            stats["max_quality_score"] = 0.0
            stats["poor_quality_count"] = 0

        # Calculate error rates
        if stats["total_validations"] > 0:
            stats["temperature_warning_rate"] = (
                stats["temperature_warnings"] / stats["total_validations"]
            )
            stats["wind_warning_rate"] = stats["wind_warnings"] / stats["total_validations"]
            stats["description_warning_rate"] = (
                stats["description_warnings"] / stats["total_validations"]
            )
            stats["structure_error_rate"] = stats["structure_errors"] / stats["total_validations"]
        else:
            stats["temperature_warning_rate"] = 0.0
            stats["wind_warning_rate"] = 0.0
            stats["description_warning_rate"] = 0.0
            stats["structure_error_rate"] = 0.0

        return stats

    def reset_statistics(self):
        """Reset validation statistics for fresh tracking period."""
        self.validation_stats = {
            "total_validations": 0,
            "temperature_warnings": 0,
            "wind_warnings": 0,
            "description_warnings": 0,
            "structure_errors": 0,
            "quality_scores": [],
        }
