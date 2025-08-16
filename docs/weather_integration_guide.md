# Weather API Integration Guide

## Overview

The NFL DFS system now includes comprehensive weather data integration using the OpenWeatherMap API. Weather significantly impacts fantasy football performance, and this integration provides real-time and historical weather data for all NFL stadiums.

## Weather Impact on Fantasy Football

### Position-Specific Effects

**Quarterbacks (QB)**

- **Wind > 15 MPH**: Significantly reduces passing accuracy and deep ball attempts
- **Cold < 40°F**: Reduced grip and throwing precision
- **Rain/Snow**: Increased incompletions and conservative play calling

**Wide Receivers/Tight Ends (WR/TE)**

- **Wind**: Affects route running precision and catch difficulty
- **Precipitation**: Increases drop rates and reduces target share
- **Cold**: Reduced concentration and sure-handedness

**Running Backs (RB)**

- **Bad Weather**: Often benefits from increased carries as teams abandon passing
- **Cold**: Minimal direct impact, may see increased usage
- **Wind**: Little direct effect on rushing attempts

**Kickers (K)**

- **Wind > 10 MPH**: Dramatically affects field goal accuracy
- **Cold**: Reduces ball flight distance
- **Precipitation**: Significantly impacts footing and accuracy

## Setup

### 1. Get OpenWeatherMap API Key

```bash
# Get a free API key from OpenWeatherMap
# Visit: https://openweathermap.org/api
# Free tier provides 1000 calls/day (sufficient for our needs)
```

### 2. Configure Environment

```bash
# Add to your .env file
WEATHER_API_KEY=your_openweathermap_api_key_here
```

### 3. Verify Installation

```bash
# Check that weather collection is available
uv run python -m src.cli.collect_data collect-weather --help
```

## Usage

### CLI Commands

#### Collect Weather for Current Season

```bash
# Collect weather for all games in current season
uv run python -m src.cli.collect_data collect-weather

# Collect weather for specific seasons
uv run python -m src.cli.collect_data collect-weather -s 2024 -s 2023

# Collect weather for specific weeks
uv run python -m src.cli.collect_data collect-weather -s 2024 -w 1 -w 2
```

#### Collect Weather for Upcoming Games

```bash
# Collect weather for games in next 7 days
uv run python -m src.cli.collect_data collect-weather --upcoming

# Collect weather for games in next 3 days
uv run python -m src.cli.collect_data collect-weather --upcoming --days 3
```

#### Full Data Collection (including weather)

```bash
# Collect all data including weather
uv run python -m src.cli.collect_data collect-all

# Collect all data but skip weather
uv run python -m src.cli.collect_data collect-all --no-weather
```

#### Check Data Status

```bash
# View database status including weather data coverage
uv run python -m src.cli.collect_data status
```

### Python API Usage

```python
from src.data.collection.weather_collector import WeatherCollector

# Initialize collector
collector = WeatherCollector()

# Collect weather for specific game
success = collector.collect_weather_for_game(game_id=123)

# Collect weather for entire season
stats = collector.collect_weather_for_season(season=2024)
print(f"Collected weather for {stats['weather_collected']} games")

# Collect weather for upcoming games
upcoming_stats = collector.collect_upcoming_games_weather(days_ahead=7)
print(f"Updated weather for {upcoming_stats['weather_updated']} upcoming games")
```

## Weather Features in ML Models

The feature extractor automatically includes weather data when available:

### Weather Features Generated

```python
# Temperature features
"temperature_f": 45,              # Raw temperature in Fahrenheit
"is_cold_weather": 1,             # Temperature < 40°F
"is_very_cold_weather": 0,        # Temperature < 20°F
"is_hot_weather": 0,              # Temperature > 90°F

# Wind features  
"wind_speed_mph": 18,             # Raw wind speed in MPH
"is_windy": 1,                    # Wind >= 15 MPH
"is_very_windy": 0,               # Wind >= 20 MPH

# Condition features
"weather_clear": 0,               # Clear conditions
"weather_rain": 1,                # Rain/storm conditions
"weather_snow": 0,                # Snow conditions
"is_dome_game": 0,                # Indoor/covered stadium

# Composite features
"weather_impact_score": 6,        # Overall impact score (0-10)
"has_weather_data": 1,            # Data availability flag
```

### Using Weather Features in Models

```python
from src.data.processing.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_player_features(
    player_id=player_id,
    target_game_date=game_date
)

# Weather features are automatically included
wind_speed = features.get("wind_speed_mph", 0)
is_bad_weather = features.get("weather_impact_score", 0) > 5

# Use in your model logic
if is_bad_weather and position == "QB":
    # Adjust predictions for bad weather QB performance
    passing_projection *= 0.85
```

## Stadium Coverage

Weather data is collected for all 32 NFL stadiums with precise GPS coordinates:

### Dome/Retractable Roof Stadiums

- Indoor stadiums are automatically detected
- Weather impact is minimized for dome games
- `is_dome_game` feature flags these conditions

### Stadium Locations

All NFL stadiums are mapped with precise coordinates:

- **AFC East**: BUF, MIA, NE, NYJ
- **AFC North**: BAL, CIN, CLE, PIT
- **AFC South**: HOU, IND, JAX, TEN
- **AFC West**: DEN, KC, LV, LAC
- **NFC East**: DAL, NYG, PHI, WAS
- **NFC North**: CHI, DET, GB, MIN
- **NFC South**: ATL, CAR, NO, TB
- **NFC West**: ARI, LAR, SF, SEA

## Data Quality & Validation

### Automatic Validation

- Temperature range validation (-20°F to 130°F)
- Wind speed validation (0-60 MPH)
- Weather description parsing and categorization
- API response structure validation

### Quality Scoring

- Each weather record gets a quality score (0-1)
- Scores < 0.6 are flagged for review
- Quality metrics are logged for monitoring

### Monitoring

```bash
# Check validation statistics in logs
tail -f data/logs/nfl_dfs.log | grep "weather"

# Look for quality warnings
grep "weather data quality" data/logs/nfl_dfs.log
```

## Error Handling

### Common Issues

**API Key Not Configured**

```bash
❌ Weather API key not configured!
Set WEATHER_API_KEY environment variable or get a free key from:
https://openweathermap.org/api
```

**Rate Limit Exceeded**

- Free tier: 1000 calls/day
- System automatically retries with backoff
- Consider upgrading API plan for heavy usage

**Historical Data Limitations**

- Free tier only provides current weather and 5-day forecast
- Historical weather requires paid OpenWeatherMap plan
- System falls back to current weather for past games

### Graceful Degradation

- Missing weather data doesn't break feature extraction
- Default weather values are used when data unavailable
- Models can still function without weather data

## Best Practices

### Data Collection Schedule

```bash
# Collect weather for upcoming games daily
0 6 * * * cd /path/to/project && uv run python -m src.cli.collect_data collect-weather --upcoming

# Collect weather for current season weekly
0 2 * * 0 cd /path/to/project && uv run python -m src.cli.collect_data collect-weather
```

### Performance Tips

- Use `--upcoming` flag for regular updates
- Collect historical data in batches by season
- Monitor API usage to stay within rate limits
- Consider caching weather data for repeated requests

### Model Integration

- Include weather features in all position models
- Weight weather impact by position (QBs > RBs)
- Consider interaction effects (cold + windy)
- Use weather impact score for quick filtering

## Troubleshooting

### Debug Logging

```bash
# Enable debug logging for weather collection
export LOG_LEVEL=DEBUG
uv run python -m src.cli.collect_data collect-weather
```

### Validate Stadium Coordinates

```python
from src.data.collection.weather_collector import WeatherCollector

collector = WeatherCollector()
# Check stadium mapping
for team, (lat, lon, stadium) in collector.stadium_locations.items():
    print(f"{team}: {stadium} ({lat}, {lon})")
```

### Test API Connection

```python
from src.data.collection.weather_collector import WeatherCollector

try:
    collector = WeatherCollector()
    # Test with known coordinates (Kansas City)
    weather = collector._get_current_weather(39.0489, -94.4839)
    print("API connection successful:", weather is not None)
except Exception as e:
    print("API connection failed:", e)
```

## Monitoring and Maintenance

### Regular Checks

- Monitor API usage against daily limits
- Review quality scores for data issues
- Check for new stadium locations or coordinate changes
- Validate weather data accuracy during extreme weather events

### Upgrades

- Consider paid OpenWeatherMap plan for historical data
- Evaluate alternative weather APIs for redundancy
- Add weather alerts for extreme conditions
- Implement weather-based lineup optimization strategies

## Integration with Existing Workflows

The weather integration seamlessly fits into existing workflows:

1. **Data Collection**: Weather is automatically collected with `collect-all`
1. **Feature Engineering**: Weather features are automatically included
1. **Model Training**: Models can use weather features without changes
1. **Predictions**: Weather context enhances prediction accuracy
1. **Optimization**: Weather data informs lineup construction strategies

This comprehensive weather integration provides the fantasy football models with crucial environmental context, improving prediction accuracy and enabling more sophisticated game analysis.
