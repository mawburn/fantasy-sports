# Vegas Odds Integration Guide

## Overview

The NFL DFS system now includes comprehensive Vegas odds integration using The Odds API. Vegas odds provide crucial market intelligence that significantly enhances fantasy football predictions by incorporating all available information including injuries, weather, and professional handicapper analysis.

## Fantasy Football Impact

Vegas odds are among the most powerful predictors for fantasy performance because they represent the collective wisdom of professional handicappers and incorporate all public information.

### Position-Specific Effects

**Game Totals (Over/Under)**

- **High Totals (50+ points)**: Favor all skill positions, especially QBs and WRs
- **Low Totals (\<42 points)**: Favor defensive players and RBs in clock-control games
- **Shootout Games**: Increased targets for WRs/TEs, more passing attempts

**Point Spreads**

- **Large Spreads (>7 points)**: Favorites get early leads, more rushing attempts
- **Close Games (\<3 points)**: More balanced play calling, higher target share for skill positions
- **Blowout Potential**: Garbage time for losing team's skill positions

**Moneyline Odds**

- **Heavy Favorites**: Positive game script, more red zone opportunities
- **Heavy Underdogs**: Likely to abandon run game, increased passing volume

### Market Efficiency

Vegas odds incorporate all available information including:

- Injury reports and player availability
- Weather forecasts and conditions
- Historical matchup data and trends
- Public betting sentiment and sharp money movement

This makes odds data extremely valuable for fantasy predictions as it represents the most accurate available assessment of game outcomes.

## Setup

### 1. Get The Odds API Key

```bash
# Get a free API key from The Odds API
# Visit: https://the-odds-api.com/
# Free tier provides 500 requests/month (sufficient for personal use)
```

### 2. Configure Environment

```bash
# Add to your .env file
ODDS_API_KEY=your_the_odds_api_key_here

# Optional: Configure collection settings
COLLECT_VEGAS_ODDS=true
ODDS_COLLECTION_INTERVAL=14400  # 4 hours in seconds
```

### 3. Verify Installation

```bash
# Check that odds collection is available
uv run python -m src.cli.collect_data collect-odds --help
```

## Usage

### CLI Commands

#### Collect Odds for Current Week

```bash
# Collect odds for current week
uv run python -m src.cli.collect_data collect-odds

# Collect odds for specific season/week
uv run python -m src.cli.collect_data collect-odds -s 2024 -w 5
```

#### Collect Odds for Upcoming Games

```bash
# Collect odds for games in next 7 days
uv run python -m src.cli.collect_data collect-odds --upcoming

# Collect odds for games in next 3 days
uv run python -m src.cli.collect_data collect-odds --upcoming --days 3
```

#### Enhanced Data Collection (including odds)

```bash
# Collect all data including odds
uv run python -m src.cli.collect_data collect-enhanced

# Collect all data but skip odds
uv run python -m src.cli.collect_data collect-enhanced --no-odds
```

#### Check Data Status

```bash
# View database status including odds data coverage
uv run python -m src.cli.collect_data status
```

### Python API Usage

```python
from src.data.collection.vegas_odds_collector import VegasOddsCollector

# Initialize collector
collector = VegasOddsCollector()

# Collect odds for specific game
success = collector.collect_odds_for_game(game_id=123)

# Collect odds for entire week
stats = collector.collect_odds_for_week(season=2024, week=5)
print(f"Collected odds for {stats['odds_collected']} games")

# Collect odds for upcoming games
upcoming_stats = collector.collect_upcoming_games_odds(days_ahead=7)
print(f"Updated odds for {upcoming_stats['odds_collected']} upcoming games")
```

## Sportsbook Coverage

The odds collector integrates with major sportsbooks for comprehensive line shopping:

### Supported Sportsbooks

- **DraftKings**: Primary sportsbook, excellent API coverage
- **FanDuel**: Major competitor, consistent odds
- **Caesars**: Traditional Vegas book, sharp lines
- **BetMGM**: MGM Resorts book, good market coverage
- **PointsBet**: Australian book, unique spread options

### Betting Markets

- **Point Spreads**: Team handicaps and point differentials
- **Totals**: Over/under game point totals
- **Moneylines**: Straight-up winner odds
- **Player Props**: Individual player performance bets (future enhancement)

## Odds Features in ML Models

The feature extractor automatically includes odds data when available:

### Odds Features Generated

```python
# Game total features
"game_total_points": 47.5,        # Over/under total
"is_high_total": 1,               # Total >= 50 points
"is_low_total": 0,                # Total <= 42 points

# Spread features
"home_spread": -3.5,              # Home team spread
"away_spread": 3.5,               # Away team spread
"is_large_spread": 0,             # Absolute spread >= 7
"is_close_game": 1,               # Absolute spread <= 3

# Moneyline features
"home_win_probability": 0.65,     # Implied probability
"away_win_probability": 0.35,     # Implied probability
"is_heavy_favorite": 0,           # Win probability >= 70%
"is_heavy_underdog": 0,           # Win probability <= 30%

# Market features
"total_vig": 0.045,               # Sportsbook edge
"line_movement": -1.0,            # Movement from opening line
"has_odds_data": 1,               # Data availability flag
```

### Using Odds Features in Models

```python
from src.data.processing.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_player_features(
    player_id=player_id,
    target_game_date=game_date
)

# Odds features are automatically included
game_total = features.get("game_total_points", 0)
is_shootout = features.get("is_high_total", 0)

# Use in your model logic
if is_shootout and position == "QB":
    # Adjust predictions for high-scoring games
    passing_projection *= 1.15
```

## Data Quality & Validation

### Automatic Validation

- Real-time odds collection with timestamps
- Multiple sportsbooks for line shopping and validation
- Implied probability calculation and consistency checks
- Vig calculation for market efficiency analysis
- Line movement tracking from opening to game time

### Quality Scoring

- Each odds record gets validation scores
- Inconsistent odds across books are flagged
- Stale odds (>24 hours old) are marked for refresh
- Quality metrics are logged for monitoring

### Monitoring

```bash
# Check validation statistics in logs
tail -f data/logs/nfl_dfs.log | grep "odds"

# Look for quality warnings
grep "odds.*quality" data/logs/nfl_dfs.log
```

## Error Handling

### Common Issues

**API Key Not Configured**

```bash
❌ Odds API key not configured!
Set ODDS_API_KEY environment variable or get a free key from:
https://the-odds-api.com/
```

**Rate Limit Exceeded**

- Free tier: 500 requests/month
- System automatically retries with exponential backoff
- Consider upgrading API plan for heavy usage

**Stale Odds Data**

- Odds older than 24 hours are flagged for refresh
- System attempts to collect fresh data automatically
- Games without odds continue with default predictions

### Graceful Degradation

- Missing odds data doesn't break feature extraction
- Default odds values are used when data unavailable
- Models can still function without odds data
- Feature flags indicate data availability

## Best Practices

### Data Collection Schedule

```bash
# Collect odds for upcoming games daily (before rate limit)
0 6 * * * cd /path/to/project && uv run python -m src.cli.collect_data collect-odds --upcoming

# Full odds collection weekly
0 2 * * 0 cd /path/to/project && uv run python -m src.cli.collect_data collect-odds
```

### Performance Tips

- Use `--upcoming` flag for regular updates
- Collect historical odds in batches by week
- Monitor API usage to stay within rate limits
- Consider caching odds data for repeated requests

### Model Integration

- Include odds features in all position models
- Weight market intelligence by position relevance
- Consider interaction effects (high total + dome = shootout)
- Use market sentiment for lineup construction

## Advanced Usage

### Line Movement Analysis

```python
from src.database.connection import SessionLocal
from src.database.models import VegasOdds, Game

session = SessionLocal()

# Find games with significant line movement
big_moves = session.query(VegasOdds).filter(
    VegasOdds.line_movement_spread.abs() >= 3.0
).all()

# Analyze high-total games
shootouts = session.query(VegasOdds).filter(
    VegasOdds.total_points >= 50
).all()

session.close()
```

### Market Efficiency Analysis

```python
from src.data.collection.vegas_odds_collector import VegasOddsCollector

collector = VegasOddsCollector()

# Analyze vig across sportsbooks
def analyze_market_efficiency(game_id):
    odds_records = get_odds_for_game(game_id)
    
    min_vig = min(record.total_vig for record in odds_records)
    max_vig = max(record.total_vig for record in odds_records)
    
    # Lower vig indicates more efficient market
    return {"min_vig": min_vig, "max_vig": max_vig}
```

### Fantasy Impact Scoring

```python
def calculate_fantasy_impact_score(odds_data):
    """Calculate fantasy relevance score from odds data."""
    score = 0
    
    # High totals favor skill positions
    if odds_data.total_points >= 50:
        score += 3
    elif odds_data.total_points >= 47:
        score += 2
    elif odds_data.total_points <= 42:
        score -= 2
    
    # Close spreads favor balanced game scripts
    spread = abs(odds_data.home_spread or 0)
    if spread <= 3:
        score += 2
    elif spread >= 10:
        score -= 1
    
    return max(0, min(10, score))
```

## Troubleshooting

### Debug Logging

```bash
# Enable debug logging for odds collection
export LOG_LEVEL=DEBUG
uv run python -m src.cli.collect_data collect-odds
```

### Test API Connection

```python
from src.data.collection.vegas_odds_collector import VegasOddsCollector

try:
    collector = VegasOddsCollector()
    # Test API connection
    test_response = collector._make_api_request(
        "sports", 
        {"sport": "americanfootball_nfl"}
    )
    print("Odds API connection successful:", test_response is not None)
except Exception as e:
    print("Odds API connection failed:", e)
```

### Validate Odds Data

```python
from src.database.connection import SessionLocal
from src.database.models import VegasOdds

session = SessionLocal()

# Check recent odds coverage
recent_odds = session.query(VegasOdds).filter(
    VegasOdds.line_timestamp >= datetime.now() - timedelta(days=7)
).count()

print(f"Recent odds records: {recent_odds}")

# Check sportsbook coverage
sportsbooks = session.query(VegasOdds.sportsbook).distinct().all()
print(f"Active sportsbooks: {[book[0] for book in sportsbooks]}")

session.close()
```

## API Usage and Limits

### The Odds API Tiers

**Free Tier**

- 500 requests per month
- Current and upcoming games
- Major sportsbooks included
- Sufficient for personal use

**Paid Tiers**

- More requests per month
- Historical odds data
- Additional markets and props
- Premium support

### Request Optimization

```python
# Efficient odds collection
collector = VegasOddsCollector()

# Collect for upcoming games only (saves requests)
collector.collect_upcoming_games_odds(days_ahead=3)

# Monitor usage
print(f"Requests used: {collector.requests_made}/{collector.monthly_limit}")
```

## Integration with Existing Workflows

The Vegas odds integration seamlessly fits into existing workflows:

1. **Data Collection**: Odds automatically collected with `collect-enhanced`
1. **Feature Engineering**: Market features automatically included
1. **Model Training**: Models can use odds features without changes
1. **Predictions**: Market intelligence enhances prediction accuracy
1. **Optimization**: Odds data informs lineup construction strategies

### Lineup Construction Examples

```python
# Use odds data for lineup construction
def build_optimal_lineup(players, odds_data):
    # Favor players in high-total games
    high_total_games = [g for g in odds_data if g.total_points >= 50]
    
    # Weight players by game script
    for player in players:
        game_odds = get_odds_for_player_game(player)
        
        if player.position == "QB" and game_odds.total_points >= 50:
            player.projection *= 1.1  # Boost for shootouts
        
        elif player.position == "RB" and game_odds.home_spread >= 7:
            if player.team == game_odds.favorite:
                player.projection *= 1.05  # Boost for clock control
```

## Monitoring and Maintenance

### Regular Checks

- Monitor API usage against monthly limits
- Review line movement patterns for unusual activity
- Validate odds accuracy against actual game outcomes
- Check for new sportsbooks or market types

### Upgrades

- Consider paid Odds API plan for more requests
- Evaluate additional sportsbooks for line shopping
- Add player prop betting lines for individual projections
- Implement automated line movement alerts

This comprehensive Vegas odds integration provides fantasy football models with crucial market intelligence, significantly enhancing prediction accuracy and enabling sophisticated game analysis strategies.
