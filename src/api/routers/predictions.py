"""API endpoints for fantasy sports predictions."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import PlayerPredictionRequest, PredictionResponse, SlatePredictionResponse
from src.database.connection import get_db
from src.database.models import ModelMetadata, Player
from src.ml.training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionService:
    """Service for managing predictions and model loading."""

    def __init__(self, db: Session):
        """Initialize prediction service."""
        self.db = db
        self.trainer = ModelTrainer(db)
        self._model_cache = {}  # Cache loaded models

    def _create_http_exception(self, status_code: int, detail: str) -> HTTPException:
        """Create HTTPException to avoid TRY301 issues."""
        return HTTPException(status_code=status_code, detail=detail)

    def get_active_model(self, position: str) -> Any:
        """Get the active model for a position."""
        # Check cache first
        if position in self._model_cache:
            return self._model_cache[position]

        # Load from database
        active_model_meta = (
            self.db.query(ModelMetadata)
            .filter(
                ModelMetadata.position == position,
                ModelMetadata.status == "trained",
                ModelMetadata.is_active,
            )
            .first()
        )

        if not active_model_meta:
            # Fallback to most recent model
            active_model_meta = (
                self.db.query(ModelMetadata)
                .filter(
                    ModelMetadata.position == position,
                    ModelMetadata.status == "trained",
                )
                .order_by(ModelMetadata.created_at.desc())
                .first()
            )

        if not active_model_meta:
            raise HTTPException(
                status_code=404, detail=f"No trained model found for position {position}"
            )

        # Load model
        try:
            model = self.trainer.load_model(active_model_meta.model_id)
            self._model_cache[position] = model
            return model
        except Exception as e:
            logger.exception("Failed to load model %s", active_model_meta.model_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to load model for position {position}"
            ) from e

    def predict_player(
        self, player_id: int, game_date: datetime, position: str
    ) -> PredictionResponse:
        """Generate prediction for a single player."""
        # Get model
        model = self.get_active_model(position)

        # Prepare data for prediction
        from src.ml.training.data_preparation import DataPreparator

        data_prep = DataPreparator(self.db)

        try:
            # Extract features for the player
            X, valid_player_ids = data_prep.prepare_prediction_data(
                [player_id], game_date, position
            )

            if not valid_player_ids:
                raise self._create_http_exception(
                    400, f"Unable to extract features for player {player_id}"
                )

            # Generate prediction
            prediction_result = model.predict(X)

            # Get player info
            player = self.db.query(Player).filter(Player.id == player_id).first()
            if not player:
                raise self._create_http_exception(404, f"Player {player_id} not found")

            return PredictionResponse(
                player_id=player_id,
                player_name=player.display_name,
                position=player.position,
                predicted_points=float(prediction_result.point_estimate[0]),
                confidence_score=(
                    float(prediction_result.confidence_score[0])
                    if prediction_result.confidence_score is not None
                    else 0.8
                ),
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
                model_version=prediction_result.model_version,
                prediction_date=prediction_result.prediction_date,
            )

        except Exception as e:
            logger.exception("Prediction failed for player %s", player_id)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e

    def predict_slate(
        self, game_date: datetime, positions: list[str] | None = None
    ) -> SlatePredictionResponse:
        """Generate predictions for all players in a slate."""
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


@router.get("/player/{player_id}", response_model=PredictionResponse)
async def predict_player(
    player_id: int,
    game_date: datetime = Query(..., description="Game date for prediction (ISO format)"),
    db: Session = Depends(get_db),
):
    """Generate prediction for a single player."""
    # Get player to determine position
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    service = PredictionService(db)
    return service.predict_player(player_id, game_date, player.position)


@router.post("/player", response_model=PredictionResponse)
async def predict_player_batch(
    request: PlayerPredictionRequest,
    db: Session = Depends(get_db),
):
    """Generate prediction for a player with custom parameters."""
    # Get player to determine position
    player = db.query(Player).filter(Player.id == request.player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {request.player_id} not found")

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
    """Generate predictions for all players in a game slate."""
    service = PredictionService(db)
    return service.predict_slate(game_date, positions)


@router.get("/models", response_model=dict[str, Any])
async def list_active_models(db: Session = Depends(get_db)):
    """List all active models by position."""
    models = (
        db.query(ModelMetadata)
        .filter(ModelMetadata.status == "trained")
        .order_by(ModelMetadata.position, ModelMetadata.created_at.desc())
        .all()
    )

    # Group by position and get most recent for each
    active_models = {}
    for model in models:
        if model.position not in active_models:
            active_models[model.position] = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "created_at": model.created_at,
                "mae_validation": model.mae_validation,
                "r2_validation": model.r2_validation,
                "is_active": model.is_active,
            }

    return {"active_models": active_models, "total_positions": len(active_models)}


@router.get("/health")
async def prediction_health_check():
    """Health check endpoint for prediction service."""
    return {
        "status": "healthy",
        "service": "predictions",
        "timestamp": datetime.now(UTC).replace(tzinfo=None),
        "message": "Prediction service is operational",
    }
