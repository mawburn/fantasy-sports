"""
Main FastAPI application for NFL DFS System.

This module creates the main FastAPI application that serves as the web API
for the NFL Daily Fantasy Sports prediction system. FastAPI is a modern,
high-performance web framework for building APIs with Python.

Key FastAPI Features Used:
- Automatic API documentation (OpenAPI/Swagger)
- Data validation with Pydantic models
- Async support for high performance
- Type hints for better code quality
- Dependency injection system

The API provides endpoints for:
- Data access (NFL stats, player info, etc.)
- ML predictions (fantasy point projections)
- System health and configuration
"""

# Import FastAPI core components
from fastapi import FastAPI  # Main framework class
from fastapi.middleware.cors import CORSMiddleware  # Cross-Origin Resource Sharing

# Import our custom routers that define API endpoints
from src.api.routers import data, predictions

# Import application settings/configuration
from src.config import settings

# Create the main FastAPI application instance
# This is the core object that handles all HTTP requests
app = FastAPI(
    title="NFL DFS System API",  # Appears in API documentation
    description="NFL Daily Fantasy Sports prediction and optimization system",
    version="0.1.0",  # API version for client compatibility
    # Note: FastAPI automatically generates interactive docs at /docs and /redoc
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# CORS allows web browsers to make requests from one domain to another
# This is essential for frontend applications running on different ports/domains
app.add_middleware(
    CORSMiddleware,  # FastAPI's built-in CORS middleware
    allow_origins=["*"],  # Allow requests from any origin - configure restrictively for production!
    allow_credentials=True,  # Allow cookies/auth headers in requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all request headers
    # SECURITY NOTE: In production, replace "*" with specific allowed origins
    # Example: allow_origins=["http://localhost:3000", "https://myapp.com"]
)


@app.get("/")  # HTTP GET decorator - responds to GET requests at root "/"
async def root():
    """
    Root endpoint - basic API information.

    This is the main landing page for the API. It provides basic information
    about the service and links to documentation.

    The 'async' keyword makes this function asynchronous, allowing FastAPI
    to handle many concurrent requests efficiently without blocking.

    Returns:
        dict: Basic API information including version and documentation links
    """
    return {
        "message": "NFL DFS System API",
        "version": "0.1.0",
        # Dynamically construct documentation URL from settings
        "docs": f"http://{settings.api_host}:{settings.api_port}/docs",
    }


@app.get("/health")  # Health check endpoint for monitoring/load balancers
async def health_check():
    """
    Health check endpoint for system monitoring.

    This endpoint is commonly used by:
    - Load balancers to check if the service is running
    - Monitoring systems to track service health
    - Container orchestrators (Docker, Kubernetes)
    - Development debugging

    It returns system status and configuration information that's useful
    for diagnosing issues or confirming system state.

    Returns:
        dict: Service health status and system configuration
    """
    return {
        "status": "healthy",  # Simple health indicator
        "service": "NFL DFS System",
        # Include CPU optimization settings since they affect performance
        "cpu_optimization": settings.enable_cpu_optimization,
        "num_threads": settings.num_cpu_threads,
    }


@app.get("/api/config")  # Configuration endpoint for runtime settings
async def get_config():
    """
    Get current system configuration (non-sensitive values only).

    This endpoint exposes configuration settings that are safe to share
    publicly. It's useful for:
    - Frontend applications to adapt their behavior
    - Debugging configuration issues
    - Confirming system settings after deployment

    SECURITY NOTE: Only non-sensitive configuration values are exposed.
    Database passwords, API keys, etc. are never included in responses.

    Returns:
        dict: Public configuration settings for the NFL DFS system
    """
    return {
        # NFL data collection settings
        "nfl_seasons_to_load": settings.nfl_seasons_to_load,
        # DraftKings contest configuration
        "dk_classic_salary_cap": settings.dk_classic_salary_cap,
        "dk_showdown_salary_cap": settings.dk_showdown_salary_cap,
        # System behavior flags
        "enable_self_tuning": settings.enable_self_tuning,  # ML model auto-improvement
        "enable_monitoring": settings.enable_monitoring,  # Performance tracking
        "enable_caching": settings.enable_caching,  # Response caching
        "cache_backend": settings.cache_backend,  # Caching technology used
    }


# Register API routers - this is how FastAPI organizes endpoints into logical groups
# Each router contains related endpoints and can have its own prefix and tags

# Data access router - handles NFL stats, player info, team data, etc.
app.include_router(
    data.router,  # The router object containing data endpoints
    prefix="/api/data",  # URL prefix - all routes will start with /api/data
    tags=["data"],  # OpenAPI tags for documentation grouping
)

# ML predictions router - handles fantasy point projections and model operations
app.include_router(
    predictions.router,  # The router object containing prediction endpoints
    prefix="/api",  # URL prefix - routes will start with /api
    tags=["predictions"],  # OpenAPI tags for documentation grouping
)

# Future routers to be added when those features are implemented:
# Lineup optimization router for generating optimal DFS lineups
# from src.api.routers import optimization
# app.include_router(optimization.router, prefix="/api/optimize", tags=["optimization"])

# The FastAPI application is now fully configured and ready to handle requests!
# When run with uvicorn, it will serve on the configured host:port with automatic
# API documentation available at /docs (Swagger UI) and /redoc (ReDoc)
