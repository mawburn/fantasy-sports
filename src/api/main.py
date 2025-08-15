"""Main FastAPI application for NFL DFS System."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import data, predictions
from src.config import settings

# Create FastAPI app
app = FastAPI(
    title="NFL DFS System API",
    description="NFL Daily Fantasy Sports prediction and optimization system",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NFL DFS System API",
        "version": "0.1.0",
        "docs": f"http://{settings.api_host}:{settings.api_port}/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NFL DFS System",
        "cpu_optimization": settings.enable_cpu_optimization,
        "num_threads": settings.num_cpu_threads,
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration (non-sensitive values only)."""
    return {
        "nfl_seasons_to_load": settings.nfl_seasons_to_load,
        "dk_classic_salary_cap": settings.dk_classic_salary_cap,
        "dk_showdown_salary_cap": settings.dk_showdown_salary_cap,
        "enable_self_tuning": settings.enable_self_tuning,
        "enable_monitoring": settings.enable_monitoring,
        "enable_caching": settings.enable_caching,
        "cache_backend": settings.cache_backend,
    }


app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])

# Future routers to be added:
# from src.api.routers import optimization
# app.include_router(optimization.router, prefix="/api/optimize", tags=["optimization"])
