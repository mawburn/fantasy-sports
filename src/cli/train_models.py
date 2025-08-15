"""CLI commands for training ML models."""

import logging
from datetime import datetime
from pathlib import Path

import typer

from src.database.connection import get_db
from src.ml.models.base import ModelConfig
from src.ml.models.evaluation import ModelEvaluator
from src.ml.registry import DeploymentPipeline, ModelRegistry
from src.ml.training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

app = typer.Typer(help="Train and evaluate ML models for fantasy sports predictions")


@app.command()
def train_position(
    position: str = typer.Argument(..., help="Position to train (QB, RB, WR, TE, DEF)"),
    start_date: str = typer.Option("2020-09-01", help="Training data start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2023-12-31", help="Training data end date (YYYY-MM-DD)"),
    model_name: str = typer.Option(None, help="Custom model name"),
    save_model: bool = typer.Option(True, help="Save trained model"),
    evaluate: bool = typer.Option(True, help="Evaluate model performance"),
    backtest: bool = typer.Option(False, help="Run backtesting analysis"),
) -> None:
    """Train a model for a specific position."""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Create model configuration
        config = ModelConfig(
            model_name=model_name or f"{position}_model",
            position=position,
            model_dir=Path("data/models"),
        )

        # Initialize trainer
        trainer = ModelTrainer()

        typer.echo(f"🏈 Training {position} model from {start_date} to {end_date}")

        # Train model
        results = trainer.train_position_model(
            position=position,
            start_date=start_dt,
            end_date=end_dt,
            config=config,
            save_model=save_model,
        )

        # Display results
        test_metrics = results["test_metrics"]
        typer.echo("\n📊 Training Results:")
        typer.echo(f"  Test MAE: {test_metrics.mae:.3f} points")
        typer.echo(f"  Test RMSE: {test_metrics.rmse:.3f} points")
        typer.echo(f"  Test R²: {test_metrics.r2:.3f}")
        typer.echo(f"  Test MAPE: {test_metrics.mape:.1f}%")

        # Run evaluation if requested
        if evaluate:
            evaluator = ModelEvaluator()
            report = evaluator.generate_evaluation_report(results["model"], position, test_metrics)
            typer.echo("\n📋 Evaluation Report:")
            typer.echo(report)

        # Run backtest if requested
        if backtest:
            typer.echo("\n🔄 Running backtest analysis...")
            from src.ml.models.evaluation import BacktestConfiguration

            backtest_config = BacktestConfiguration(
                start_date=start_dt,
                end_date=end_dt,
            )

            backtest_results = evaluator.backtest_model(results["model"], position, backtest_config)

            financial = backtest_results["financial_metrics"]
            typer.echo(f"  Hit Rate: {financial.get('hit_rate', 0):.1f}%")
            typer.echo(f"  Simulated ROI: {financial.get('simulated_roi', 0):.2f}")

        typer.echo("\n✅ Training completed successfully!")

    except Exception as e:
        logger.exception("Training failed")
        typer.echo(f"❌ Training failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def train_all(
    start_date: str = typer.Option("2020-09-01", help="Training data start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2023-12-31", help="Training data end date (YYYY-MM-DD)"),
    ensemble: bool = typer.Option(False, help="Train ensemble models"),
    _save_models: bool = typer.Option(True, help="Save trained models"),
) -> None:
    """Train models for all positions."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        trainer = ModelTrainer()

        typer.echo(f"🏈 Training all position models from {start_date} to {end_date}")

        # Train all positions
        results = trainer.train_all_positions(
            start_date=start_dt, end_date=end_dt, use_ensemble=ensemble
        )

        # Display summary
        typer.echo("\n📊 Training Summary:")
        for position, result in results.items():
            if "error" in result:
                typer.echo(f"  {position}: ❌ Failed - {result['error']}")
            else:
                if ensemble:
                    mae = result["test_mae"]
                    r2 = result["test_r2"]
                else:
                    test_metrics = result["test_metrics"]
                    mae = test_metrics.mae
                    r2 = test_metrics.r2

                typer.echo(f"  {position}: ✅ MAE={mae:.3f}, R²={r2:.3f}")

        typer.echo("\n✅ All models training completed!")

    except Exception as e:
        logger.exception("Training failed")
        typer.echo(f"❌ Training failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def evaluate_model(
    model_id: str = typer.Argument(..., help="Model ID to evaluate"),
    start_date: str = typer.Option("2023-09-01", help="Evaluation start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2023-12-31", help="Evaluation end date (YYYY-MM-DD)"),
    backtest: bool = typer.Option(True, help="Run backtest analysis"),
) -> None:
    """Evaluate a trained model."""
    try:
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()

        # Load model
        typer.echo(f"📁 Loading model: {model_id}")
        model = trainer.load_model(model_id)

        # Get position from model config
        position = model.config.position

        if backtest:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            typer.echo(f"🔄 Running backtest for {position} model...")

            from src.ml.models.evaluation import BacktestConfiguration

            config = BacktestConfiguration(start_date=start_dt, end_date=end_dt)

            results = evaluator.backtest_model(model, position, config)

            # Display results
            metrics = results["metrics"]
            financial = results["financial_metrics"]

            typer.echo("\n📊 Backtest Results:")
            typer.echo(f"  MAE: {metrics.mae:.3f} points")
            typer.echo(f"  RMSE: {metrics.rmse:.3f} points")
            typer.echo(f"  R²: {metrics.r2:.3f}")
            typer.echo(f"  Accuracy (±5 pts): {metrics.accuracy_within_5:.1f}%")
            typer.echo(f"  Accuracy (±10 pts): {metrics.accuracy_within_10:.1f}%")
            typer.echo(f"  Consistency Score: {metrics.consistency_score:.3f}")
            typer.echo(f"  Total Predictions: {metrics.total_predictions}")

            typer.echo("\n💰 Financial Simulation:")
            typer.echo(f"  Hit Rate: {financial.get('hit_rate', 0):.1f}%")
            typer.echo(f"  Simulated ROI: {financial.get('simulated_roi', 0):.2f}")

        typer.echo("\n✅ Evaluation completed!")

    except Exception as e:
        logger.exception("Evaluation failed")
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def compare_models(
    position: str = typer.Argument(..., help="Position to compare models for"),
    _start_date: str = typer.Option("2023-09-01", help="Evaluation start date (YYYY-MM-DD)"),
    _end_date: str = typer.Option("2023-12-31", help="Evaluation end date (YYYY-MM-DD)"),
) -> None:
    """Compare all models for a specific position."""
    try:
        # This would need to be implemented to load all models for a position
        # and compare their performance
        typer.echo(f"🔍 Comparing all {position} models...")
        typer.echo("⚠️  Model comparison not yet implemented")
        typer.echo("    Use 'evaluate-model' to analyze individual models")

    except Exception as e:
        logger.exception("Comparison failed")
        typer.echo(f"❌ Comparison failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def predict_slate(
    game_date: str = typer.Argument(..., help="Game date to predict (YYYY-MM-DD)"),
    positions: str = typer.Option("QB,RB,WR,TE,DEF", help="Comma-separated positions"),
    _output_file: str = typer.Option(None, help="Save predictions to CSV file"),
) -> None:
    """Generate predictions for a game slate."""
    try:
        _game_dt = datetime.strptime(game_date, "%Y-%m-%d")
        position_list = [p.strip() for p in positions.split(",")]

        typer.echo(f"🎯 Generating predictions for {game_date}")
        typer.echo(f"📋 Positions: {', '.join(position_list)}")

        # This would need to be implemented to load trained models
        # and generate predictions for all players in the slate
        typer.echo("⚠️  Slate prediction not yet implemented")
        typer.echo("    This will generate predictions for all players in active games")

    except Exception as e:
        logger.exception("Prediction failed")
        typer.echo(f"❌ Prediction failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def list_models() -> None:
    """List all trained models."""
    try:
        db = next(get_db())

        from src.database.models import ModelMetadata

        models = (
            db.query(ModelMetadata)
            .filter(ModelMetadata.status == "trained")
            .order_by(ModelMetadata.position, ModelMetadata.created_at.desc())
            .all()
        )

        if not models:
            typer.echo("📭 No trained models found")
            return

        typer.echo("📚 Trained Models:")
        typer.echo()

        current_position = None
        for model in models:
            if model.position != current_position:
                current_position = model.position
                typer.echo(f"  {model.position}:")

            # Format model info
            created = model.created_at.strftime("%Y-%m-%d %H:%M")
            mae = model.mae_validation or 0
            r2 = model.r2_validation or 0

            typer.echo(
                f"    {model.model_id:<30} | MAE: {mae:.3f} | R²: {r2:.3f} | Created: {created}"
            )

        typer.echo()

    except Exception as e:
        logger.exception("Failed to list models")
        typer.echo(f"❌ Failed to list models: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def model_info(
    model_id: str = typer.Argument(..., help="Model ID to get info for"),
) -> None:
    """Get detailed information about a specific model."""
    try:
        db = next(get_db())

        from src.database.models import ModelMetadata

        model = db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()

        if not model:
            typer.echo(f"❌ Model not found: {model_id}")
            raise typer.Exit(1) from None

        typer.echo(f"📊 Model Information: {model_id}")
        typer.echo()
        typer.echo(f"  Name: {model.model_name}")
        typer.echo(f"  Position: {model.position}")
        typer.echo(f"  Type: {model.model_type}")
        typer.echo(f"  Version: {model.version}")
        typer.echo(f"  Status: {model.status}")
        typer.echo()
        typer.echo("  Training Data:")
        typer.echo(f"    Period: {model.training_start_date} to {model.training_end_date}")
        typer.echo(f"    Training Samples: {model.training_data_size:,}")
        typer.echo(f"    Validation Samples: {model.validation_data_size:,}")
        typer.echo(f"    Features: {model.feature_count}")
        typer.echo()
        typer.echo("  Performance Metrics:")
        typer.echo(f"    MAE (Validation): {model.mae_validation:.3f}")
        typer.echo(f"    RMSE (Validation): {model.rmse_validation:.3f}")
        typer.echo(f"    R² (Validation): {model.r2_validation:.3f}")
        typer.echo(f"    MAPE: {model.mape_validation:.1f}%")
        typer.echo()
        typer.echo(f"  Created: {model.created_at}")
        if model.deployment_date:
            typer.echo(f"  Deployed: {model.deployment_date}")

    except Exception as e:
        logger.exception("Failed to get model info")
        typer.echo(f"❌ Failed to get model info: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def deploy_model(
    model_id: str = typer.Argument(..., help="Model ID to deploy"),
    make_active: bool = typer.Option(True, help="Make this the active model"),
    retire_previous: bool = typer.Option(True, help="Retire previous active models"),
) -> None:
    """Deploy a trained model to production."""
    try:
        registry = ModelRegistry()

        typer.echo(f"🚀 Deploying model: {model_id}")

        # Validate model first
        validation = registry.validate_model_compatibility(model_id)
        if not validation["valid"]:
            typer.echo("❌ Model validation failed:")
            for check, result in validation["checks"].items():
                status = "✅" if result else "❌"
                typer.echo(f"  {status} {check}: {result}")
            raise typer.Exit(1) from None

        # Deploy model
        success = registry.deploy_model(model_id, make_active, retire_previous)

        if success:
            typer.echo("✅ Model deployed successfully!")
            if make_active:
                typer.echo(f"  Model {model_id} is now active")
        else:
            typer.echo("❌ Deployment failed")
            raise typer.Exit(1) from None

    except Exception as e:
        logger.exception("Deployment failed")
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def auto_deploy(
    position: str = typer.Argument(..., help="Position to auto-deploy best model for"),
    min_improvement: float = typer.Option(0.05, help="Minimum improvement threshold"),
) -> None:
    """Automatically deploy the best performing model for a position."""
    try:
        registry = ModelRegistry()
        pipeline = DeploymentPipeline(registry)

        typer.echo(f"🤖 Auto-deploying best model for {position}")

        result = pipeline.auto_deploy_best_model(position, min_improvement)

        if result["deployed"]:
            typer.echo("✅ Auto-deployment successful!")
            typer.echo(f"  Deployed: {result['model_id']}")
            typer.echo(f"  Improvement: {result['improvement']:.3f}")
            typer.echo(f"  New MAE: {result['new_mae']:.3f}")
            if result.get("previous_active"):
                typer.echo(f"  Previous: {result['previous_active']}")
        else:
            typer.echo(f"⚠️  Auto-deployment skipped: {result['reason']}")

    except Exception as e:
        logger.exception("Auto-deployment failed")
        typer.echo(f"❌ Auto-deployment failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def retire_model(
    model_id: str = typer.Argument(..., help="Model ID to retire"),
) -> None:
    """Retire a deployed model."""
    try:
        registry = ModelRegistry()

        typer.echo(f"🏁 Retiring model: {model_id}")

        success = registry.retire_model(model_id)

        if success:
            typer.echo("✅ Model retired successfully!")
        else:
            typer.echo("❌ Retirement failed")
            raise typer.Exit(1) from None

    except Exception as e:
        logger.exception("Retirement failed")
        typer.echo(f"❌ Retirement failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def rollback(
    position: str = typer.Argument(..., help="Position to rollback deployment for"),
) -> None:
    """Rollback to the previous active model."""
    try:
        registry = ModelRegistry()
        pipeline = DeploymentPipeline(registry)

        typer.echo(f"⏪ Rolling back {position} model deployment")

        result = pipeline.rollback_deployment(position)

        if result["success"]:
            typer.echo("✅ Rollback successful!")
            typer.echo(f"  Rolled back from: {result['rolled_back_from']}")
            typer.echo(f"  Rolled back to: {result['rolled_back_to']}")
        else:
            typer.echo(f"❌ Rollback failed: {result['reason']}")
            raise typer.Exit(1) from None

    except Exception as e:
        logger.exception("Rollback failed")
        typer.echo(f"❌ Rollback failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def cleanup_models(
    keep_recent: int = typer.Option(5, help="Number of recent models to keep per position"),
    keep_active: bool = typer.Option(True, help="Always keep active models"),
) -> None:
    """Clean up old models to save storage space."""
    try:
        registry = ModelRegistry()

        typer.echo("🧹 Cleaning up old models...")

        cleaned_count = registry.cleanup_old_models(keep_recent, keep_active)

        typer.echo(f"✅ Cleanup complete! Removed {cleaned_count} old models")

    except Exception as e:
        logger.exception("Cleanup failed")
        typer.echo(f"❌ Cleanup failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def validate_model(
    model_id: str = typer.Argument(..., help="Model ID to validate"),
) -> None:
    """Validate model compatibility and integrity."""
    try:
        registry = ModelRegistry()

        typer.echo(f"🔍 Validating model: {model_id}")

        validation = registry.validate_model_compatibility(model_id)

        if validation["valid"]:
            typer.echo("✅ Model validation passed!")
        else:
            typer.echo("❌ Model validation failed!")

        typer.echo("\nValidation checks:")
        for check, result in validation["checks"].items():
            status = "✅" if result else "❌"
            typer.echo(f"  {status} {check}: {result}")

    except Exception as e:
        logger.exception("Validation failed")
        typer.echo(f"❌ Validation failed: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
