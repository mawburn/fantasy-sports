# NFL DFS API Specifications

## Overview

This document provides comprehensive API specifications for the NFL DFS system, including REST endpoints, request/response schemas, authentication, and integration patterns. The API is built using FastAPI and follows RESTful principles.

## API Architecture

### Base Configuration

```yaml
base_url: http://localhost:8000
api_version: v1
api_prefix: /api/v1
content_type: application/json
authentication: Optional (local deployment)
rate_limiting: Configurable per endpoint
```

### API Standards

- RESTful design principles
- JSON request/response bodies
- ISO 8601 date formats
- HTTP status codes for responses
- Pagination for list endpoints
- Consistent error responses

## Core API Endpoints

### Player Predictions

#### Get Weekly Predictions

```http
GET /api/v1/nfl/predictions/{season}/{week}
```

**Parameters:**

- `season` (path, integer): NFL season year
- `week` (path, integer): Week number (1-22)
- `slate_type` (query, string): "classic" | "showdown" | "all"
- `position` (query, string): "QB" | "RB" | "WR" | "TE" | "DST" | "ALL"
- `min_salary` (query, integer): Minimum DraftKings salary
- `max_salary` (query, integer): Maximum DraftKings salary
- `team` (query, string): Team abbreviation
- `game_id` (query, string): Specific game ID
- `sort_by` (query, string): "projected_points" | "value" | "salary"
- `order` (query, string): "asc" | "desc"
- `limit` (query, integer): Number of results (default: 100)
- `offset` (query, integer): Pagination offset

**Response:**

```json
{
  "data": [
    {
      "player_id": "JAllen_17",
      "player_name": "Josh Allen",
      "team": "BUF",
      "opponent": "MIA",
      "position": "QB",
      "game_info": {
        "game_id": "2024_01_BUF_MIA",
        "game_time": "2024-09-08T13:00:00Z",
        "is_home": true,
        "spread": -3.5,
        "total": 48.5,
        "implied_total": 26.0
      },
      "salary_info": {
        "draftkings_salary": 8200,
        "salary_change": 300,
        "value_rating": 3.2
      },
      "projections": {
        "median": 24.5,
        "floor": 18.2,
        "ceiling": 31.8,
        "confidence_interval": [20.1, 28.9],
        "boom_probability": 0.35,
        "bust_probability": 0.12
      },
      "ownership": {
        "projected_cash": 22.5,
        "projected_gpp": 18.3,
        "leverage_score": 1.2
      },
      "model_metadata": {
        "model_version": "v2.1.0",
        "model_confidence": 0.78,
        "features_count": 127
      }
    }
  ],
  "pagination": {
    "total": 250,
    "limit": 100,
    "offset": 0,
    "has_more": true
  },
  "metadata": {
    "season": 2024,
    "week": 1,
    "generated_at": "2024-09-07T10:00:00Z",
    "cache_ttl": 3600
  }
}
```

#### Get Individual Player Projection

```http
GET /api/v1/nfl/projections/{player_id}
```

**Parameters:**

- `player_id` (path, string): Player identifier
- `season` (query, integer): Season year
- `week` (query, integer): Week number

**Response:**

```json
{
  "player": {
    "player_id": "JAllen_17",
    "full_name": "Josh Allen",
    "position": "QB",
    "team": "BUF"
  },
  "current_projection": {
    "season": 2024,
    "week": 1,
    "median_points": 24.5,
    "floor_points": 18.2,
    "ceiling_points": 31.8,
    "confidence": 0.78
  },
  "historical_performance": {
    "season_average": 22.3,
    "last_3_average": 25.1,
    "last_5_average": 23.8,
    "vs_opponent_average": 21.5
  },
  "feature_breakdown": {
    "passing_yards": 285.5,
    "passing_tds": 2.1,
    "rushing_yards": 38.2,
    "rushing_tds": 0.4,
    "interceptions": 0.8
  },
  "correlations": {
    "top_stacks": [
      { "player_id": "SDiggs_14", "correlation": 0.72 },
      { "player_id": "GDavis_13", "correlation": 0.58 }
    ]
  }
}
```

#### Generate New Predictions

```http
POST /api/v1/nfl/predictions/generate
```

**Request Body:**

```json
{
  "season": 2024,
  "week": 1,
  "slate_type": "classic",
  "force_refresh": false,
  "include_injured": false,
  "model_version": "latest",
  "optimization_settings": {
    "use_time_decay": true,
    "decay_factor": 0.95,
    "min_games_played": 3
  }
}
```

**Response:**

```json
{
  "status": "success",
  "job_id": "pred_2024_01_abc123",
  "predictions_generated": 245,
  "execution_time_ms": 3450,
  "model_versions": {
    "QB": "v2.1.0",
    "RB": "v2.0.3",
    "WR": "v2.1.1",
    "TE": "v2.0.2",
    "DST": "v1.9.0"
  },
  "cache_key": "predictions_2024_01_classic"
}
```

### DraftKings Lineup Optimization

#### Optimize Classic Lineup

```http
POST /api/v1/dk/lineups/optimize/classic
```

**Request Body:**

```json
{
  "slate_id": "DK_MAIN_2024_01",
  "contest_type": "GPP",
  "optimization_settings": {
    "num_lineups": 20,
    "unique_players": 3,
    "max_exposure": 0.6,
    "min_salary_used": 49000
  },
  "player_settings": {
    "locked_players": ["JAllen_17"],
    "excluded_players": ["TBrady_12"],
    "player_groups": [
      {
        "players": ["SDiggs_14", "GDavis_13"],
        "min_from_group": 1,
        "max_from_group": 1
      }
    ]
  },
  "stacking_rules": {
    "qb_stack": {
      "enabled": true,
      "min_receivers": 1,
      "max_receivers": 3,
      "allow_opposing": true
    },
    "game_stack": {
      "enabled": true,
      "min_players": 3,
      "max_players": 5
    },
    "team_stack": {
      "max_from_team": 4
    }
  },
  "correlation_settings": {
    "use_correlations": true,
    "correlation_weight": 0.3,
    "negative_correlation_penalty": 0.2
  }
}
```

**Response:**

```json
{
  "lineups": [
    {
      "lineup_id": "LU_2024_01_001",
      "rank": 1,
      "roster": {
        "QB": {
          "player_id": "JAllen_17",
          "name": "Josh Allen",
          "salary": 8200
        },
        "RB1": {
          "player_id": "CMcCaffrey_22",
          "name": "Christian McCaffrey",
          "salary": 9500
        },
        "RB2": {
          "player_id": "APollard_20",
          "name": "Tony Pollard",
          "salary": 6200
        },
        "WR1": {
          "player_id": "THill_10",
          "name": "Tyreek Hill",
          "salary": 9000
        },
        "WR2": {
          "player_id": "SDiggs_14",
          "name": "Stefon Diggs",
          "salary": 8000
        },
        "WR3": {
          "player_id": "CKupp_10",
          "name": "Cooper Kupp",
          "salary": 7500
        },
        "TE": {
          "player_id": "TKelce_87",
          "name": "Travis Kelce",
          "salary": 7200
        },
        "FLEX": {
          "player_id": "KWalker_9",
          "name": "Kenneth Walker",
          "salary": 5800
        },
        "DST": {
          "player_id": "DST_BUF",
          "name": "Buffalo Bills",
          "salary": 3600
        }
      },
      "metrics": {
        "total_salary": 49800,
        "projected_points": 142.3,
        "floor_points": 118.5,
        "ceiling_points": 168.2,
        "projected_ownership": 12.4,
        "correlation_score": 0.68,
        "leverage_score": 2.1
      },
      "stacks": [
        {
          "type": "QB_STACK",
          "players": ["JAllen_17", "SDiggs_14"],
          "correlation": 0.72
        }
      ]
    }
  ],
  "optimization_metadata": {
    "total_combinations_evaluated": 125000,
    "optimization_time_ms": 4200,
    "algorithm": "mixed_integer_programming",
    "solver": "PuLP_CBC"
  }
}
```

#### Optimize Showdown Lineup

```http
POST /api/v1/dk/lineups/optimize/showdown
```

**Request Body:**

```json
{
  "game_id": "2024_01_BUF_MIA",
  "optimization_strategy": "balanced",
  "captain_settings": {
    "force_qb_captain": false,
    "min_captain_ownership": 5.0,
    "leverage_captain": true
  },
  "num_lineups": 10,
  "unique_lineups": 3
}
```

### Game Selection

#### Get Game Recommendations

```http
GET /api/v1/games/recommendations/{season}/{week}
```

**Parameters:**

- `season` (path, integer): Season year
- `week` (path, integer): Week number
- `min_entertainment_score` (query, float): Minimum score (0-100)
- `contest_type` (query, string): "cash" | "gpp" | "both"
- `include_primetime` (query, boolean): Include SNF/MNF/TNF

**Response:**

```json
{
  "recommendations": [
    {
      "game_id": "2024_01_BUF_MIA",
      "matchup": "BUF @ MIA",
      "game_time": "2024-09-08T13:00:00Z",
      "scores": {
        "entertainment": 85.5,
        "shootout_potential": 78.2,
        "star_power": 92.0,
        "rivalry_factor": 88.0,
        "dfs_complexity": 76.5
      },
      "reasons": [
        "Division rivalry with playoff implications",
        "High-scoring potential (O/U 52.5)",
        "Multiple elite fantasy options"
      ],
      "recommended_contests": ["GPP", "Single-Entry"],
      "key_players": ["JAllen_17", "THill_10", "TTagovailoa_1"]
    }
  ],
  "summary": {
    "total_games": 16,
    "recommended_games": 5,
    "average_entertainment_score": 72.3
  }
}
```

### Model Management

#### Learn from Results

```http
POST /api/v1/models/learn-from-results
```

**Request Body:**

```json
{
  "season": 2024,
  "week": 1,
  "learning_settings": {
    "update_weights": true,
    "adjust_features": true,
    "retrain_threshold": 0.1
  }
}
```

**Response:**

```json
{
  "learning_summary": {
    "predictions_evaluated": 245,
    "average_error": 2.8,
    "improved_features": 12,
    "deprecated_features": 3,
    "weight_adjustments": 45
  },
  "position_analysis": {
    "QB": { "mae": 3.2, "improvement": -0.3 },
    "RB": { "mae": 2.5, "improvement": -0.2 },
    "WR": { "mae": 2.1, "improvement": -0.1 }
  },
  "model_updates": {
    "retrained": ["QB", "WR"],
    "next_retrain_week": 3
  }
}
```

#### Get Error Analysis

```http
GET /api/v1/models/error-analysis/{season}/{week}
```

**Response:**

```json
{
  "error_analysis": {
    "overall_mae": 2.8,
    "overall_rmse": 4.2,
    "systematic_errors": [
      {
        "pattern": "Overestimating mobile QBs in bad weather",
        "frequency": 0.75,
        "average_error": 4.5,
        "affected_players": ["JAllen_17", "LJackson_8"]
      }
    ],
    "biggest_misses": [
      {
        "player_id": "DHenry_22",
        "predicted": 18.5,
        "actual": 8.2,
        "error": -10.3,
        "likely_cause": "Injury not factored"
      }
    ]
  },
  "feature_performance": {
    "best_features": ["implied_total", "target_share", "red_zone_touches"],
    "worst_features": ["weather_wind", "primetime_bonus", "revenge_game"]
  }
}
```

### Data Management

#### Upload DraftKings Salaries

```http
POST /api/v1/dk/salaries/upload
```

**Request Body (multipart/form-data):**

```
file: <CSV file>
slate_id: "DK_MAIN_2024_01"
slate_type: "classic"
season: 2024
week: 1
```

**Response:**

```json
{
  "upload_status": "success",
  "players_processed": 450,
  "new_players_added": 3,
  "salary_updates": 447,
  "validation_errors": [],
  "processing_time_ms": 850
}
```

#### Refresh NFL Data

```http
POST /api/v1/nfl/data/refresh
```

**Request Body:**

```json
{
  "data_types": ["rosters", "stats", "schedules"],
  "season": 2024,
  "weeks": [1, 2, 3],
  "force_update": false
}
```

### Backtesting

#### Run Backtest

```http
POST /api/v1/models/backtest
```

**Request Body:**

```json
{
  "backtest_config": {
    "start_season": 2022,
    "end_season": 2023,
    "positions": ["QB", "RB", "WR"],
    "validation_split": 0.2,
    "walk_forward": true
  },
  "parameter_search": {
    "enabled": true,
    "search_space": {
      "learning_rate": [0.01, 0.1],
      "max_depth": [3, 10],
      "n_estimators": [100, 500]
    },
    "optimization_metric": "mae",
    "n_trials": 100
  }
}
```

**Response:**

```json
{
  "backtest_id": "BT_2024_001",
  "status": "completed",
  "results": {
    "best_parameters": {
      "learning_rate": 0.05,
      "max_depth": 7,
      "n_estimators": 300
    },
    "performance": {
      "train_mae": 2.1,
      "validation_mae": 2.8,
      "test_mae": 3.0
    },
    "improvement_over_baseline": 0.15
  },
  "execution_time_seconds": 420
}
```

## WebSocket Endpoints

### Live Updates

```websocket
ws://localhost:8000/ws/live-updates
```

**Connection Message:**

```json
{
  "type": "subscribe",
  "channels": ["predictions", "lineups", "injuries"]
}
```

**Update Messages:**

```json
{
  "type": "prediction_update",
  "data": {
    "player_id": "JAllen_17",
    "field": "projected_points",
    "old_value": 24.5,
    "new_value": 22.1,
    "reason": "Weather update"
  },
  "timestamp": "2024-09-08T12:30:00Z"
}
```

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid player ID format",
    "details": {
      "field": "player_id",
      "provided": "Josh Allen",
      "expected_format": "{first_initial}{last_name}_{number}"
    },
    "timestamp": "2024-09-08T10:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Codes

- `VALIDATION_ERROR` (400): Invalid request parameters
- `NOT_FOUND` (404): Resource not found
- `CONFLICT` (409): Resource conflict
- `RATE_LIMITED` (429): Too many requests
- `INTERNAL_ERROR` (500): Server error
- `SERVICE_UNAVAILABLE` (503): Service temporarily unavailable

## Rate Limiting

### Default Limits

```yaml
endpoints:
  predictions:
    rate: 60/minute
    burst: 10
  optimization:
    rate: 20/minute
    burst: 5
  data_upload:
    rate: 10/minute
    burst: 2
  backtesting:
    rate: 5/minute
    burst: 1
```

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1693584000
```

## Caching Strategy

### Cache Configuration

```yaml
cache_ttl:
  predictions: 3600 # 1 hour
  player_stats: 86400 # 24 hours
  game_info: 300 # 5 minutes
  lineups: 60 # 1 minute

cache_headers:
  - Cache-Control
  - ETag
  - Last-Modified
```

### Cache Invalidation

```http
POST /api/v1/cache/invalidate
```

**Request Body:**

```json
{
  "cache_keys": ["predictions_2024_01", "lineups_classic_2024_01"],
  "pattern": "predictions_*",
  "invalidate_all": false
}
```

## API Versioning

### Version Strategy

- URL path versioning: `/api/v1/`, `/api/v2/`
- Backward compatibility for 2 major versions
- Deprecation notices in headers
- Migration guides for breaking changes

### Version Headers

```http
API-Version: 1.0
API-Deprecated: false
API-Sunset-Date: 2025-01-01
```

## Authentication & Security

### Local Authentication (Optional)

```yaml
authentication:
  enabled: false # Local deployment
  type: api_key
  header: X-API-Key
  storage: environment_variable
```

### Security Headers

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
```

## Performance Specifications

### Response Time SLAs

- Prediction endpoints: < 200ms (p95)
- Optimization endpoints: < 5000ms (p95)
- Data upload endpoints: < 1000ms (p95)
- Backtest endpoints: < 30000ms (p95)

### Throughput Targets

- Concurrent connections: 100
- Requests per second: 100
- Database queries per second: 500

## Monitoring & Metrics

### Health Check Endpoint

```http
GET /api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "database": "connected",
  "cache": "connected",
  "last_data_update": "2024-09-08T09:00:00Z"
}
```

### Metrics Endpoint

```http
GET /api/v1/metrics
```

**Response:**

```json
{
  "requests": {
    "total": 10000,
    "success": 9950,
    "error": 50
  },
  "performance": {
    "avg_response_time_ms": 125,
    "p95_response_time_ms": 450,
    "p99_response_time_ms": 1200
  },
  "resources": {
    "cpu_usage_percent": 45,
    "memory_usage_mb": 2048,
    "database_connections": 10
  }
}
```

## SDK Support

### Python SDK Example

```python
from nfl_dfs_client import NFLDFSClient

client = NFLDFSClient(base_url="http://localhost:8000")

# Get predictions
predictions = client.predictions.get_weekly(
    season=2024,
    week=1,
    position="QB"
)

# Optimize lineup
lineup = client.lineups.optimize_classic(
    slate_id="DK_MAIN_2024_01",
    contest_type="GPP",
    num_lineups=20
)
```

## API Testing

### Test Endpoints

```http
POST /api/v1/test/generate-mock-data
GET /api/v1/test/validate-models
POST /api/v1/test/simulate-week
```

### Integration Testing

- Postman collection available
- OpenAPI/Swagger documentation
- Automated API tests with pytest
- Load testing with locust
