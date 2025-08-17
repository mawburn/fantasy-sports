# Stadium Data Integration Guide

## Overview

The NFL DFS system now includes comprehensive stadium data integration with detailed venue characteristics and performance factors. Stadium data significantly impacts fantasy football performance through environmental and design factors that affect player performance and game outcomes.

## Fantasy Football Impact

Stadium characteristics provide crucial context for fantasy predictions by accounting for venue-specific factors that traditional statistics may not capture.

### Performance Factors

**Playing Surface**

- **Natural Grass**: Slightly slower surface, may favor power runners over speed
- **Artificial Turf**: Faster surface, benefits speed players and precise route running
- **Injury Rates**: Different surfaces affect injury patterns and player availability

**Roof Type**

- **Dome/Retractable**: Eliminates weather variables, typically increases scoring
- **Outdoor**: Subject to weather effects, more variable conditions
- **Passing Advantage**: Domes generally favor passing games over ground attacks

**Altitude**

- **Denver (Mile High)**: Significantly affects ball flight and player endurance
- **Kicking**: Field goals travel farther at altitude (+15% distance)
- **Passing**: Less air resistance affects deep ball accuracy and distance

**Home Field Advantage**

- **Crowd Noise**: Affects road team communication and penalty frequency
- **Acoustics**: Some stadiums amplify crowd noise through design
- **False Starts**: Louder stadiums cause more road team infractions

### Position-Specific Effects

**Quarterbacks (QB)**

- **Dome Games**: Increased passing attempts and accuracy
- **Loud Stadiums**: More communication issues for road QBs
- **Altitude**: Affects deep ball accuracy and spiral tightness

**Wide Receivers/Tight Ends (WR/TE)**

- **Fast Surfaces**: Benefit speed-based receivers
- **Dome Games**: More targets due to increased passing volume
- **Crowd Noise**: Road teams may struggle with route timing

**Running Backs (RB)**

- **Grass Fields**: May favor power runners over speed backs
- **Altitude**: Affects endurance for high-carry games
- **Home Crowd**: Minimal direct impact on rushing performance

**Kickers (K)**

- **Altitude**: Dramatic increase in field goal range (Denver effect)
- **Dome Games**: Consistent conditions improve accuracy
- **Wind Patterns**: Stadium design affects air flow

## Setup

Stadium data collection requires no external API keys as it uses a comprehensive built-in database compiled from official NFL sources.

### Configure Environment

```bash
# Optional: Configure collection settings in .env
COLLECT_STADIUM_DATA=true
STADIUM_DATA_REFRESH_INTERVAL=2592000  # 30 days in seconds
```

### Verify Installation

```bash
# Check that stadium collection is available
uv run python -m src.cli.collect_data collect-stadiums --help
```

## Usage

### CLI Commands

#### Initial Stadium Data Collection

```bash
# Collect all NFL stadium data (one-time setup)
uv run python -m src.cli.collect_data collect-stadiums
```

#### Enhanced Data Collection (including stadiums)

```bash
# Collect all data including stadiums
uv run python -m src.cli.collect_data collect-enhanced

# Collect all data but skip stadiums
uv run python -m src.cli.collect_data collect-enhanced --no-stadiums
```

#### Check Data Status

```bash
# View database status including stadium data coverage
uv run python -m src.cli.collect_data status
```

### Python API Usage

```python
from src.data.collection.stadium_collector import StadiumDataCollector

# Initialize collector
collector = StadiumDataCollector()

# Collect all stadium data
stats = collector.collect_all_stadiums()
print(f"Added {stats['stadiums_added']} stadiums")
print(f"Updated {stats['stadiums_updated']} stadiums")
print(f"Created {stats['relationships_created']} team relationships")

# Get stadium info for specific team
stadium_info = collector.get_stadium_for_team("KC")
print(f"Chiefs play at {stadium_info['stadium_name']}")
print(f"Home field advantage: {stadium_info['home_field_advantage_score']}")
```

## Stadium Coverage

Comprehensive data for all 32 NFL stadiums with precise GPS coordinates and performance analytics.

### AFC Conference

**AFC East**

- **Buffalo Bills**: Highmark Stadium (Orchard Park, NY) - Cold weather, natural grass
- **Miami Dolphins**: Hard Rock Stadium (Miami Gardens, FL) - Partial canopy, turf
- **New England Patriots**: Gillette Stadium (Foxborough, MA) - Modern turf, excellent drainage
- **New York Jets**: MetLife Stadium (East Rutherford, NJ) - Shared with Giants, turf

**AFC North**

- **Baltimore Ravens**: M&T Bank Stadium (Baltimore, MD) - Natural grass, passionate fans
- **Cincinnati Bengals**: Paycor Stadium (Cincinnati, OH) - Turf, riverfront location
- **Cleveland Browns**: Cleveland Browns Stadium (Cleveland, OH) - Lakefront, wind effects
- **Pittsburgh Steelers**: Acrisure Stadium (Pittsburgh, PA) - Natural grass, loud crowd

**AFC South**

- **Houston Texans**: NRG Stadium (Houston, TX) - Retractable roof, natural grass
- **Indianapolis Colts**: Lucas Oil Stadium (Indianapolis, IN) - Dome, excellent acoustics
- **Jacksonville Jaguars**: TIAA Bank Field (Jacksonville, FL) - Outdoor, unique amenities
- **Tennessee Titans**: Nissan Stadium (Nashville, TN) - Outdoor, riverfront location

**AFC West**

- **Denver Broncos**: Empower Field at Mile High (Denver, CO) - High altitude effects
- **Kansas City Chiefs**: Arrowhead Stadium (Kansas City, MO) - Loudest stadium (142.2 dB)
- **Las Vegas Raiders**: Allegiant Stadium (Las Vegas, NV) - New dome, natural grass
- **Los Angeles Chargers**: SoFi Stadium (Los Angeles, CA) - Shared with Rams, modern

### NFC Conference

**NFC East**

- **Dallas Cowboys**: AT&T Stadium (Arlington, TX) - Massive dome, retractable roof
- **New York Giants**: MetLife Stadium (East Rutherford, NJ) - Shared with Jets
- **Philadelphia Eagles**: Lincoln Financial Field (Philadelphia, PA) - Outdoor, passionate fans
- **Washington Commanders**: FedExField (Landover, MD) - Outdoor, natural grass

**NFC North**

- **Chicago Bears**: Soldier Field (Chicago, IL) - Historic, lakefront winds
- **Detroit Lions**: Ford Field (Detroit, MI) - Dome, consistent conditions
- **Green Bay Packers**: Lambeau Field (Green Bay, WI) - Historic frozen tundra
- **Minnesota Vikings**: U.S. Bank Stadium (Minneapolis, MN) - Modern dome

**NFC South**

- **Atlanta Falcons**: Mercedes-Benz Stadium (Atlanta, GA) - Modern dome, retractable roof
- **Carolina Panthers**: Bank of America Stadium (Charlotte, NC) - Outdoor, moderate climate
- **New Orleans Saints**: Caesars Superdome (New Orleans, LA) - Historic dome
- **Tampa Bay Buccaneers**: Raymond James Stadium (Tampa, FL) - Outdoor, warm climate

**NFC West**

- **Arizona Cardinals**: State Farm Stadium (Glendale, AZ) - Retractable roof, desert
- **Los Angeles Rams**: SoFi Stadium (Los Angeles, CA) - Shared with Chargers
- **San Francisco 49ers**: Levi's Stadium (Santa Clara, CA) - Outdoor, mild climate
- **Seattle Seahawks**: Lumen Field (Seattle, WA) - 12th Man crowd noise

### Shared Stadiums

- **MetLife Stadium**: New York Giants and New York Jets
- **SoFi Stadium**: Los Angeles Rams and Los Angeles Chargers

## Stadium Features in ML Models

Stadium characteristics are automatically included in feature extraction when available:

### Stadium Features Generated

```python
# Physical characteristics
"is_dome_game": 1,                # Indoor/covered stadium
"is_grass_field": 0,              # Natural grass surface
"is_turf_field": 1,               # Artificial turf
"stadium_elevation": 5280,        # Elevation in feet (Denver = 5,280)

# Performance factors
"home_field_advantage": 0.85,     # Stadium-specific advantage score
"scoring_factor": 1.05,           # Relative scoring vs league average
"passing_factor": 1.12,           # Pass-friendly rating
"kicking_factor": 1.22,           # Kicking advantage factor (altitude boost)
"injury_rate_factor": 0.95,       # Injury rate vs league average

# Environmental
"noise_level": 142.2,             # Decibel level during games
"climate_type": "Temperate",      # Regional climate classification
"typical_wind": 8.5,              # Average wind speed
"drainage_quality": "Excellent",  # Field drainage rating
```

### Using Stadium Features in Models

```python
from src.data.processing.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_player_features(
    player_id=player_id,
    target_game_date=game_date
)

# Stadium features are automatically included
is_dome = features.get("is_dome_game", 0)
elevation = features.get("stadium_elevation", 0)
home_advantage = features.get("home_field_advantage", 0.5)

# Use in your model logic
if position == "K" and elevation > 4000:
    # Boost kicker projections at high altitude
    field_goal_projection *= features.get("kicking_factor", 1.0)

elif is_dome and position in ["QB", "WR", "TE"]:
    # Boost passing game in dome conditions
    passing_projection *= features.get("passing_factor", 1.0)
```

## Data Quality & Validation

### Comprehensive Stadium Database

- **Verified Coordinates**: GPS coordinates validated against known locations
- **Performance Analytics**: Quantified impact factors based on historical analysis
- **Relationship Management**: Proper team-stadium associations including shared venues
- **Historical Tracking**: Opening dates, renovations, surface changes

### Data Sources

Stadium information compiled from:

- Official NFL stadium specifications
- Team websites and press releases
- Historical performance analysis
- Geographic and environmental databases
- Stadium engineering reports

### Quality Assurance

```python
# Stadium data validation example
def validate_stadium_data(stadium):
    errors = []
    
    # Coordinate validation
    if not (-180 <= stadium.longitude <= 180):
        errors.append("Invalid longitude")
    if not (-90 <= stadium.latitude <= 90):
        errors.append("Invalid latitude")
    
    # Performance factor validation
    if not (0.5 <= stadium.scoring_factor <= 1.5):
        errors.append("Scoring factor out of range")
    
    return errors
```

## Stadium Impact Analysis

### Performance Factor Calculation

Stadium impact factors are calculated based on historical analysis:

```python
# Example impact factor calculations
def calculate_stadium_factors(stadium_id):
    """Calculate performance factors for a stadium."""
    
    # Scoring factor: points per game vs league average
    stadium_scoring = get_average_points_at_stadium(stadium_id)
    league_average = get_league_average_scoring()
    scoring_factor = stadium_scoring / league_average
    
    # Home field advantage: home win percentage vs expected
    home_wins = get_home_win_percentage(stadium_id)
    expected_home_wins = 0.57  # League average
    home_advantage = (home_wins - expected_home_wins) + 0.5
    
    return {
        "scoring_factor": scoring_factor,
        "home_field_advantage": home_advantage
    }
```

### Altitude Effects (Denver Analysis)

```python
# Denver altitude impact analysis
DENVER_EFFECTS = {
    "kicking_boost": 1.22,      # 22% increase in field goal range
    "passing_depth": 1.08,      # 8% increase in deep ball distance
    "endurance_factor": 0.95,   # 5% decrease in endurance for visiting teams
    "ball_flight": 1.12,        # 12% increase in punt distance
}

def apply_altitude_effects(player_projection, position, stadium_elevation):
    if stadium_elevation > 4000:  # High altitude threshold
        if position == "K":
            return player_projection * DENVER_EFFECTS["kicking_boost"]
        elif position == "QB":
            return player_projection * DENVER_EFFECTS["passing_depth"]
    
    return player_projection
```

### Dome vs Outdoor Analysis

```python
# Dome game analysis
def analyze_dome_effect(stadium_roof_type, position):
    """Analyze dome effect on different positions."""
    
    if stadium_roof_type in ["Dome", "Retractable"]:
        dome_effects = {
            "QB": 1.08,   # 8% boost for passing consistency
            "WR": 1.06,   # 6% boost for route precision
            "TE": 1.04,   # 4% boost for targets
            "RB": 0.98,   # 2% decrease (less rushing in shootouts)
            "K": 1.12,    # 12% boost for consistent conditions
        }
        return dome_effects.get(position, 1.0)
    
    return 1.0  # No dome effect
```

## Advanced Usage

### Stadium-Specific Strategies

```python
from src.database.connection import SessionLocal
from src.database.models import Stadium, Game

session = SessionLocal()

# Find dome games (good for passing offense)
dome_games = session.query(Game).join(Stadium).filter(
    Stadium.roof_type.in_(["Dome", "Retractable"])
).all()

# Find high-altitude games (good for kickers)
altitude_games = session.query(Game).join(Stadium).filter(
    Stadium.elevation_feet > 4000
).all()

# Find loudest stadiums (bad for road teams)
loud_stadiums = session.query(Stadium).filter(
    Stadium.noise_level_db > 130
).order_by(Stadium.noise_level_db.desc()).all()

session.close()
```

### Team-Stadium Performance Analysis

```python
def analyze_team_stadium_performance(team_abbr):
    """Analyze team performance at their home stadium."""
    
    collector = StadiumDataCollector()
    stadium = collector.get_stadium_for_team(team_abbr)
    
    analysis = {
        "team": team_abbr,
        "stadium": stadium["stadium_name"],
        "surface_type": stadium["playing_surface"],
        "roof_type": stadium["roof_type"],
        "home_advantage": stadium["home_field_advantage_score"],
        "scoring_boost": stadium["scoring_factor"],
        "passing_friendly": stadium["passing_factor"] > 1.0,
        "kicking_advantage": stadium["kicking_factor"] > 1.0,
    }
    
    return analysis
```

### Weather-Stadium Interaction

```python
def stadium_weather_interaction(stadium_info, weather_data):
    """Analyze interaction between stadium design and weather."""
    
    # Dome stadiums eliminate weather effects
    if stadium_info["roof_type"] in ["Dome", "Retractable"]:
        return {"weather_impact": 0, "explanation": "Indoor stadium"}
    
    # Outdoor stadiums affected by weather
    weather_impact = 0
    explanations = []
    
    if weather_data.get("wind_speed_mph", 0) > 15:
        if stadium_info.get("typical_wind", 0) > 10:
            weather_impact += 2  # Stadium amplifies wind
            explanations.append("Stadium design amplifies wind effects")
        else:
            weather_impact += 1
            explanations.append("High wind conditions")
    
    return {
        "weather_impact": weather_impact,
        "explanations": explanations
    }
```

## Error Handling

### Common Issues

**Missing Team Associations**

- Stadium collector validates team relationships
- Warnings logged for unmatched teams
- System continues with available data

**Coordinate Validation**

- GPS coordinates validated against known ranges
- Invalid coordinates logged and flagged
- Fallback to approximate locations if needed

### Data Consistency

```python
def validate_stadium_consistency():
    """Validate stadium data consistency."""
    
    issues = []
    
    # Check for teams without stadiums
    teams_without_stadiums = find_teams_without_stadiums()
    if teams_without_stadiums:
        issues.append(f"Teams without stadiums: {teams_without_stadiums}")
    
    # Check for invalid performance factors
    invalid_factors = find_invalid_performance_factors()
    if invalid_factors:
        issues.append(f"Invalid performance factors: {invalid_factors}")
    
    return issues
```

### Graceful Degradation

- Missing stadium data doesn't break feature extraction
- Default values used when specific data unavailable
- Models function without complete stadium information
- Feature flags indicate data availability levels

## Best Practices

### Data Collection Schedule

```bash
# Stadium data collection monthly (rarely changes)
0 3 1 * * cd /path/to/project && uv run python -m src.cli.collect_data collect-stadiums

# Include in weekly enhanced collection
0 2 * * 0 cd /path/to/project && uv run python -m src.cli.collect_data collect-enhanced
```

### Performance Tips

- Stadium data rarely changes, collect infrequently
- Cache stadium lookups for repeated queries
- Pre-calculate stadium impact factors for common scenarios
- Use stadium features in conjunction with weather data

### Model Integration

- Include stadium features in all position models
- Weight stadium effects by position relevance (kickers > QBs > RBs)
- Consider interaction effects (dome + high total = shootout)
- Use venue factors for lineup construction strategies

## Troubleshooting

### Debug Logging

```bash
# Enable debug logging for stadium collection
export LOG_LEVEL=DEBUG
uv run python -m src.cli.collect_data collect-stadiums
```

### Validate Stadium Data

```python
from src.data.collection.stadium_collector import StadiumDataCollector

collector = StadiumDataCollector()

# Check stadium database
for stadium_id, data in collector.stadium_data.items():
    print(f"{data['stadium_name']}: {data['city']}, {data['state']}")
    print(f"  Surface: {data['playing_surface']}, Roof: {data['roof_type']}")
    print(f"  Coordinates: {data['latitude']}, {data['longitude']}")
    print(f"  Home advantage: {data['home_field_advantage_score']}")
```

### Test Stadium Lookups

```python
from src.database.connection import SessionLocal
from src.database.models import Stadium, Team

session = SessionLocal()

# Verify team-stadium relationships
teams = session.query(Team).all()
for team in teams:
    stadiums = team.stadiums
    if not stadiums:
        print(f"Warning: {team.team_abbr} has no stadium assigned")
    else:
        for stadium in stadiums:
            print(f"{team.team_abbr}: {stadium.stadium_name}")

session.close()
```

## Stadium Data Updates

### Tracking Changes

Stadium characteristics can change due to:

- **Renovations**: New playing surfaces, roof installations
- **Team Relocations**: New cities, new stadiums
- **Technology Upgrades**: Better drainage, heating systems
- **Capacity Changes**: Seating expansions or reductions

### Update Process

```python
def update_stadium_data(stadium_id, updates):
    """Update stadium data with new information."""
    
    session = SessionLocal()
    try:
        stadium = session.query(Stadium).filter_by(stadium_id=stadium_id).first()
        
        if stadium:
            for field, value in updates.items():
                if hasattr(stadium, field):
                    setattr(stadium, field, value)
            
            stadium.updated_at = datetime.now()
            session.commit()
            
            logger.info(f"Updated stadium {stadium_id}: {list(updates.keys())}")
        
    finally:
        session.close()
```

## Integration with Existing Workflows

The stadium data integration seamlessly fits into existing workflows:

1. **Data Collection**: Stadium data automatically collected with `collect-enhanced`
1. **Feature Engineering**: Venue features automatically included
1. **Model Training**: Models can use stadium features without changes
1. **Predictions**: Venue context enhances prediction accuracy
1. **Optimization**: Stadium factors inform lineup construction strategies

### Lineup Construction Examples

```python
# Use stadium data for lineup construction
def build_venue_aware_lineup(players, stadium_data):
    """Build lineup considering venue factors."""
    
    for player in players:
        stadium = get_stadium_for_player(player)
        
        # Apply stadium-specific adjustments
        if player.position == "K" and stadium.elevation_feet > 4000:
            player.projection *= stadium.kicking_factor
        
        elif player.position in ["QB", "WR"] and stadium.roof_type == "Dome":
            player.projection *= stadium.passing_factor
        
        elif player.is_home_team and stadium.home_field_advantage_score > 0.7:
            player.projection *= 1.02  # Small home field boost
```

This comprehensive stadium data integration provides fantasy football models with crucial venue-specific context, enhancing prediction accuracy and enabling sophisticated game analysis strategies that account for the physical and environmental factors that affect player performance.
