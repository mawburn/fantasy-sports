"""
API endpoints for fantasy sports predictions.

This module provides REST API endpoints for generating fantasy football point
predictions using trained machine learning models. It demonstrates several
advanced patterns:

- Service Layer Pattern: PredictionService encapsulates business logic
- Model Loading and Caching: Efficiently manages ML model instances
- Batch Prediction: Handles both single player and full slate predictions
- Error Handling: Robust error handling for ML operations
- Feature Engineering Integration: Coordinates with data preparation pipeline

Machine Learning Concepts:
- Point Estimates: Primary prediction value (expected fantasy points)
- Confidence Scores: Model's certainty in the prediction (0-1)
- Floor/Ceiling: Lower and upper bounds for predictions
- Model Versioning: Tracks which model version made each prediction
"""

import logging  # For error tracking and debugging
from datetime import UTC, datetime  # Date/time handling for predictions
from typing import Any  # Type hints for complex return types

# FastAPI components for building REST API endpoints
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session  # Database session management

# Import Pydantic schemas that define API request/response structure
from src.api.schemas import PlayerPredictionRequest, PredictionResponse, SlatePredictionResponse

# Database connection and ORM models
from src.database.connection import get_db
from src.database.models import ModelMetadata, Player

# ML training infrastructure
from src.ml.training.model_trainer import ModelTrainer

# Set up logging for debugging and monitoring prediction operations
logger = logging.getLogger(__name__)

# Create router with prefix - all routes will start with /api/predictions
router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionService:
    """
    Service class for managing ML predictions and model operations.

    This class demonstrates the Service Layer Pattern, which encapsulates
    business logic away from API endpoints. It provides:

    - Model loading and caching for performance
    - Single player predictions
    - Batch slate predictions (all players in games on a date)
    - Error handling and logging

    The service handles the complex orchestration between:
    - Model loading from disk/database
    - Feature engineering for new data
    - ML model inference
    - Response formatting
    """

    def __init__(self, db: Session):
        """
        Initialize the prediction service.

        Args:
            db: SQLAlchemy database session for data access
        """
        self.db = db
        # ModelTrainer handles loading trained models from storage
        self.trainer = ModelTrainer(db)
        # Cache loaded models to avoid repeated disk I/O
        # Key: position (str), Value: loaded ML model
        self._model_cache = {}

    def _create_http_exception(self, status_code: int, detail: str) -> HTTPException:
        """
        Helper method to create HTTPException instances.

        This method exists to avoid linting issues with direct HTTPException
        creation inside exception handlers (TRY301 rule).

        Args:
            status_code: HTTP status code (400, 404, 500, etc.)
            detail: Error message for the client

        Returns:
            HTTPException: Configured exception ready to be raised
        """
        return HTTPException(status_code=status_code, detail=detail)

    def get_active_model(self, position: str) -> Any:
        """
        Get the active trained model for a given position.

        This method implements model loading with caching and fallback logic:
        1. Check in-memory cache first (fastest)
        2. Query database for active model
        3. Fallback to most recent trained model if no active model
        4. Load model from disk and cache it

        Args:
            position: NFL position code (QB, RB, WR, TE, DEF)

        Returns:
            Loaded ML model ready for inference

        Raises:
            HTTPException: 404 if no trained model exists, 500 if loading fails
        """
        # Check cache first - avoids expensive disk I/O for repeated requests
        if position in self._model_cache:
            logger.debug("Using cached model for position %s", position)
            return self._model_cache[position]

        # Query database for active model metadata
        # is_active flag allows manual control over which model to use
        active_model_meta = (
            self.db.query(ModelMetadata)
            .filter(
                ModelMetadata.position == position,  # Position-specific model
                ModelMetadata.status == "trained",  # Only fully trained models
                ModelMetadata.is_active,  # Manually activated model
            )
            .first()
        )

        # Fallback strategy: use most recent trained model if no active model
        if not active_model_meta:
            logger.warning("No active model for %s, falling back to most recent", position)
            active_model_meta = (
                self.db.query(ModelMetadata)
                .filter(
                    ModelMetadata.position == position,
                    ModelMetadata.status == "trained",
                )
                .order_by(ModelMetadata.created_at.desc())  # Most recent first
                .first()
            )

        # Error if no trained models exist at all
        if not active_model_meta:
            logger.error("No trained model found for position %s", position)
            raise HTTPException(
                status_code=404, detail=f"No trained model found for position {position}"
            )

        # Load model from disk using the trainer
        try:
            logger.info("Loading model %s for position %s", active_model_meta.model_id, position)
            # ModelTrainer.load_model() handles deserialization from disk
            model = self.trainer.load_model(active_model_meta.model_id)

            # Cache the loaded model for future requests
            self._model_cache[position] = model
            logger.debug("Model cached successfully for position %s", position)

            return model

        except Exception as e:
            # Log the full exception for debugging
            logger.exception("Failed to load model %s", active_model_meta.model_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to load model for position {position}"
            ) from e

    def predict_player(
        self, player_id: int, game_date: datetime, position: str
    ) -> PredictionResponse:
        """
        Generate fantasy point prediction for a single player.

        This method orchestrates the complete ML prediction pipeline:
        1. Load the appropriate trained model for the position
        2. Extract features from historical data for the player
        3. Run inference to get predictions
        4. Format results into API response

        Args:
            player_id: Database ID of the player to predict
            game_date: Date of the game for context (affects opponent, etc.)
            position: Player's position (QB, RB, WR, TE, DEF)

        Returns:
            PredictionResponse: Structured prediction with point estimate,
                               confidence, floor/ceiling, and metadata

        Raises:
            HTTPException: 400 for feature extraction issues, 404 for missing player,
                          500 for model errors
        """
        # Load the trained model for this position
        model = self.get_active_model(position)

        # Import here to avoid circular imports
        from src.ml.training.data_preparation import DataPreparator

        # DataPreparator handles feature engineering from raw data
        data_prep = DataPreparator(self.db)

        try:
            # Extract features for ML prediction
            # Returns: X (feature matrix), valid_player_ids (successfully processed players)
            X, valid_player_ids = data_prep.prepare_prediction_data(
                [player_id], game_date, position
            )

            # Validate that features were successfully extracted
            if not valid_player_ids:
                logger.warning("No valid features extracted for player %s", player_id)
                raise self._create_http_exception(
                    400, f"Unable to extract features for player {player_id}"
                )

            # Run ML model inference
            # Returns prediction object with point estimate, confidence, bounds
            prediction_result = model.predict(X)
            logger.debug(
                "Generated prediction for player %s: %.2f points",
                player_id,
                prediction_result.point_estimate[0],
            )

            # Get player information for response context
            player = self.db.query(Player).filter(Player.id == player_id).first()
            if not player:
                raise self._create_http_exception(404, f"Player {player_id} not found")

            # Construct structured API response
            return PredictionResponse(
                player_id=player_id,
                player_name=player.display_name,
                position=player.position,
                # Convert numpy types to Python floats for JSON serialization
                predicted_points=float(prediction_result.point_estimate[0]),
                # Use default confidence if model doesn't provide it
                confidence_score=(
                    float(prediction_result.confidence_score[0])
                    if prediction_result.confidence_score is not None
                    else 0.8  # Default confidence score
                ),
                # Floor/ceiling might be None depending on model type
                floor=(
                    float(prediction_result.floor[0])
                    if prediction_result.floor is not None
                    else None
                ),
                ceiling=(
                    float(prediction_result.ceiling[0])
                    if prediction_result.ceiling is not None
                    else None
                ),
                # Metadata for tracking and debugging
                model_version=prediction_result.model_version,
                prediction_date=prediction_result.prediction_date,
            )

        except Exception as e:
            # Log full exception details for debugging
            logger.exception("Prediction failed for player %s", player_id)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e

    def predict_slate(
        self, game_date: datetime, positions: list[str] | None = None
    ) -> SlatePredictionResponse:
        """
        Generate predictions for all active players in a game slate.

        A "slate" in daily fantasy sports refers to all the games happening
        on a particular date. This method finds all active players from teams
        playing on the given date and generates predictions for them.

        This demonstrates batch processing patterns:
        - Efficient database queries to find relevant players
        - Grouping players by position for optimized prediction
        - Error handling that doesn't fail the entire batch
        - Comprehensive logging and error reporting

        Args:
            game_date: Date to find games and players for
            positions: List of positions to include (defaults to main fantasy positions)

        Returns:
            SlatePredictionResponse: All predictions with summary statistics and errors
        """
        if positions is None:
            positions = ["QB", "RB", "WR", "TE", "DEF"]

        # Get all active players for the game date
        from src.database.models import Game

        # Find games for the date
        games = (
            self.db.query(Game)
            .filter(
                Game.game_date >= game_date,
                Game.game_date < game_date.replace(hour=23, minute=59, second=59),
            )
            .all()
        )

        if not games:
            raise HTTPException(
                status_code=404, detail=f"No games found for date {game_date.date()}"
            )

        # Get team IDs playing on this date
        team_ids = []
        for game in games:
            team_ids.extend([game.home_team_id, game.away_team_id])

        # Get all active players from these teams
        players = (
            self.db.query(Player)
            .filter(
                Player.team_id.in_(team_ids),
                Player.position.in_(positions),
                Player.status == "Active",
            )
            .all()
        )

        if not players:
            raise HTTPException(
                status_code=404, detail=f"No active players found for date {game_date.date()}"
            )

        predictions = []
        errors = []

        # Group players by position for efficient prediction
        players_by_position = {}
        for player in players:
            if player.position not in players_by_position:
                players_by_position[player.position] = []
            players_by_position[player.position].append(player)

        # Generate predictions for each position
        for position, position_players in players_by_position.items():
            try:
                # Get model for position
                model = self.get_active_model(position)

                # Prepare data
                from src.ml.training.data_preparation import DataPreparator

                data_prep = DataPreparator(self.db)
                player_ids = [p.id for p in position_players]

                X, valid_player_ids = data_prep.prepare_prediction_data(
                    player_ids, game_date, position
                )

                if not valid_player_ids:
                    errors.append(f"No valid features for {position} players")
                    continue

                # Generate predictions
                prediction_result = model.predict(X)

                # Create response objects
                for i, player_id in enumerate(valid_player_ids):
                    player = next(p for p in position_players if p.id == player_id)

                    prediction = PredictionResponse(
                        player_id=player_id,
                        player_name=player.display_name,
                        position=player.position,
                        predicted_points=float(prediction_result.point_estimate[i]),
                        confidence_score=(
                            float(prediction_result.confidence_score[i])
                            if prediction_result.confidence_score is not None
                            else 0.8
                        ),
                        floor=(
                            float(prediction_result.floor[i])
                            if prediction_result.floor is not None
                            else None
                        ),
                        ceiling=(
                            float(prediction_result.ceiling[i])
                            if prediction_result.ceiling is not None
                            else None
                        ),
                        model_version=prediction_result.model_version,
                        prediction_date=prediction_result.prediction_date,
                    )
                    predictions.append(prediction)

            except Exception as e:
                logger.exception("Failed to predict %s players", position)
                errors.append(f"{position}: {e!s}")

        return SlatePredictionResponse(
            game_date=game_date,
            total_predictions=len(predictions),
            predictions=predictions,
            errors=errors if errors else None,
        )


# ========== API ENDPOINTS ==========


@router.get("/player/{player_id}", response_model=PredictionResponse)
async def predict_player(
    player_id: int,  # Path parameter from URL
    game_date: datetime = Query(..., description="Game date for prediction (ISO format)"),
    db: Session = Depends(get_db),
):
    """
    Generate fantasy point prediction for a single player.

    This endpoint provides the primary prediction functionality for individual
    players. The game_date is crucial because it affects:
    - Opponent matchup difficulty
    - Weather conditions
    - Recent form and trends

    Example: GET /api/predictions/player/123?game_date=2024-12-15T13:00:00

    Args:
        player_id: Database ID of the player (from URL path)
        game_date: ISO format datetime for the game
        db: Database session (injected by FastAPI)

    Returns:
        PredictionResponse: Complete prediction with confidence and bounds

    Raises:
        HTTPException: 404 if player not found, 400/500 for prediction errors
    """
    # Look up player to determine their position (needed for model selection)
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    # Create service instance and delegate to business logic
    service = PredictionService(db)
    return service.predict_player(player_id, game_date, player.position)


@router.post("/player", response_model=PredictionResponse)
async def predict_player_batch(
    request: PlayerPredictionRequest,  # Pydantic model from request body
    db: Session = Depends(get_db),
):
    """
    Generate prediction for a player using POST request with JSON body.

    This endpoint accepts the same parameters as the GET version but via
    POST request body. This pattern is useful for:
    - Complex request parameters
    - Avoiding URL length limits
    - Future extensibility with additional parameters

    Request body example:
    {
        "player_id": 123,
        "game_date": "2024-12-15T13:00:00"
    }

    Args:
        request: PlayerPredictionRequest with player_id and game_date
        db: Database session (injected by FastAPI)

    Returns:
        PredictionResponse: Complete prediction with confidence and bounds
    """
    # Look up player to determine position
    player = db.query(Player).filter(Player.id == request.player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {request.player_id} not found")

    # Delegate to service layer
    service = PredictionService(db)
    return service.predict_player(request.player_id, request.game_date, player.position)


@router.get("/slate", response_model=SlatePredictionResponse)
async def predict_slate(
    game_date: datetime = Query(..., description="Game date for predictions (ISO format)"),
    positions: list[str] = Query(
        ["QB", "RB", "WR", "TE", "DEF"], description="Positions to include in predictions"
    ),
    db: Session = Depends(get_db),
):
    """
    Generate predictions for all active players in a game slate.

    This is the most comprehensive prediction endpoint, generating forecasts
    for all eligible players across all games on a given date. It's typically
    used for:
    - Daily fantasy lineup optimization
    - Full slate analysis and research
    - Batch prediction processing

    Example: GET /api/predictions/slate?game_date=2024-12-15T13:00:00&positions=QB&positions=RB

    Args:
        game_date: Date to find games and generate predictions for
        positions: List of positions to include (can specify multiple)
        db: Database session (injected by FastAPI)

    Returns:
        SlatePredictionResponse: All predictions with summary stats and error info
    """
    service = PredictionService(db)
    return service.predict_slate(game_date, positions)


@router.get("/models", response_model=dict[str, Any])
async def list_active_models(db: Session = Depends(get_db)):
    """
    List all active/trained models by position with performance metrics.

    This endpoint provides visibility into the current model inventory,
    including:
    - Which models are available for each position
    - Model performance metrics (MAE, R²)
    - Model versions and creation dates
    - Active status flags

    Useful for:
    - Monitoring model health and performance
    - Debugging prediction issues
    - Model management and deployment decisions

    Returns:
        Dict with active_models by position and summary statistics
    """
    # Get all trained models ordered by position and recency
    models = (
        db.query(ModelMetadata)
        .filter(ModelMetadata.status == "trained")  # Only fully trained models
        .order_by(ModelMetadata.position, ModelMetadata.created_at.desc())
        .all()
    )

    # Group by position and keep only the most recent for each
    # This shows the "current" model for each position
    active_models = {}
    for model in models:
        if model.position not in active_models:  # First (most recent) for this position
            active_models[model.position] = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "created_at": model.created_at,
                # Performance metrics for model quality assessment
                "mae_validation": model.mae_validation,  # Mean Absolute Error
                "r2_validation": model.r2_validation,  # R-squared score
                "is_active": model.is_active,  # Manual activation flag
            }

    return {"active_models": active_models, "total_positions": len(active_models)}


@router.get("/health")
async def prediction_health_check():
    """
    Health check endpoint specifically for the prediction service.

    This endpoint allows monitoring systems to verify that the prediction
    service is operational. It's separate from the main API health check
    to provide service-specific status information.

    Returns:
        Dict with service status, timestamp, and operational message
    """
    return {
        "status": "healthy",
        "service": "predictions",
        # UTC timestamp for consistent logging across time zones
        "timestamp": datetime.now(UTC).replace(tzinfo=None),
        "message": "Prediction service is operational",
    }
