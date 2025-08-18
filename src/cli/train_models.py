"""CLI commands for training ML models.

This file provides command-line interface tools for training, evaluating, and managing
ML models for fantasy sports predictions. It uses the Typer library to create a
user-friendly CLI with proper argument parsing and help text.

Key Benefits of CLI Training:

1. Automation: Enable scheduled model retraining via cron jobs or CI/CD
2. Reproducibility: Consistent training parameters and logging
3. Experimentation: Easy parameter sweeps and model comparison
4. Production Integration: Simple deployment pipeline integration
5. User Experience: Non-programmers can retrain models

Typer Library: Modern CLI framework for Python that provides:
- Automatic help generation from docstrings and type hints
- Type validation and conversion
- Rich terminal output with colors and emojis
- Intuitive command structure

Commands Overview:
- train_position: Train model for specific position (QB, RB, WR, TE, DEF)
- train_all: Batch training for all positions
- evaluate_model: Comprehensive model evaluation
- deploy_model: Deploy trained models to production
- backtest: Historical performance analysis

For beginners: CLI (Command Line Interface) allows users to interact with
the program through text commands rather than graphical interfaces.
"""

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
    # Neural networks are always used now
) -> None:
    """Train a model for a specific position.

    This is the main command for training individual position models. It provides
    a complete training pipeline from data preparation through model evaluation.

    Training Pipeline:
    1. Parse and validate input parameters
    2. Configure model training settings
    3. Initialize trainer and execute training
    4. Display core performance metrics
    5. Optionally run comprehensive evaluation
    6. Optionally run backtesting analysis

    Parameter Explanation:

    position: Which NFL position to train (QB, RB, WR, TE, DEF)
    Each position has different statistical patterns and gets its own model.

    start_date/end_date: Training data time range
    - Use historical data for training (typically 2+ seasons)
    - Ensure sufficient data volume for reliable training
    - Leave recent data for testing if doing manual evaluation

    model_name: Optional custom name for model identification
    - Defaults to "{position}_model" if not specified
    - Useful for experiments: "QB_model_v2" or "QB_model_ensemble"

    save_model: Whether to persist trained model to disk
    - True: Model saved for production use or later evaluation
    - False: Training-only (useful for quick experiments)

    evaluate: Whether to run comprehensive evaluation
    - Generates detailed performance report beyond basic metrics
    - Includes quality assessments and recommendations

    backtest: Whether to run historical backtesting
    - Tests model on historical predictions vs actual results
    - More realistic performance assessment than holdout validation

    All models use neural networks:
    - Deep learning models with PyTorch for all positions
    - Position-specific architectures optimized for each role
    - Can capture complex patterns and player interactions

    Usage Examples:
    python -m src.cli.train_models train-position QB
    python -m src.cli.train_models train-position RB --start-date 2021-01-01 --backtest
    python -m src.cli.train_models train-position WR --model-name WR_experiment_v1
    """
    try:
        # Step 1: Parse and validate date parameters
        # Convert string dates to datetime objects for model training
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            typer.echo(f"❌ Invalid date format: {e}", err=True)
            typer.echo("Please use YYYY-MM-DD format (e.g., 2020-09-01)", err=True)
            raise typer.Exit(1) from e

        # Validate date range makes sense
        if start_dt >= end_dt:
            typer.echo("❌ Start date must be before end date", err=True)
            raise typer.Exit(1)

        # Step 2: Create model configuration
        # This object contains all training parameters and settings
        config = ModelConfig(
            model_name=model_name or f"{position}_model",  # Default or custom name
            position=position,  # Position being trained
            model_dir=Path("data/models"),  # Where to save model files
        )

        # Step 3: Initialize trainer with database connection
        trainer = ModelTrainer()

        # User feedback: Show what's happening
        model_type = "neural network"
        typer.echo(f"🏈 Training {position} {model_type} model from {start_date} to {end_date}")
        typer.echo(f"📝 Model name: {config.model_name}")

        typer.echo("🧠 Using PyTorch deep learning architecture")
        typer.echo("   (Training may take longer but can capture more complex patterns)")

        # Step 4: Execute main training pipeline
        # This includes data preparation, training, and validation
        results = trainer.train_position_model(
            position=position,
            start_date=start_dt,
            end_date=end_dt,
            config=config,
            save_model=save_model,
        )

        # Step 5: Display core training results
        test_metrics = results["test_metrics"]
        typer.echo("\n📊 Training Results:")

        # Core performance metrics with explanations
        typer.echo(f"  Test MAE: {test_metrics.mae:.3f} points (avg prediction error)")
        typer.echo(f"  Test RMSE: {test_metrics.rmse:.3f} points (error magnitude)")
        typer.echo(f"  Test R²: {test_metrics.r2:.3f} (variance explained, higher = better)")
        typer.echo(f"  Test MAPE: {test_metrics.mape:.1f}% (percentage error)")

        # Additional context for interpreting results
        typer.echo("\n📈 Model Performance:")
        if test_metrics.mae < 5.0:
            typer.echo("  ✅ Excellent accuracy (MAE < 5.0)")
        elif test_metrics.mae < 7.0:
            typer.echo("  ✅ Good accuracy (MAE < 7.0)")
        else:
            typer.echo("  ⚠️  Consider improvements (MAE ≥ 7.0)")

        if test_metrics.r2 > 0.3:
            typer.echo("  ✅ Strong predictive power (R² > 0.3)")
        elif test_metrics.r2 > 0.1:
            typer.echo("  ✅ Moderate predictive power (R² > 0.1)")
        else:
            typer.echo("  ⚠️  Low predictive power (R² ≤ 0.1)")

        # Step 6: Run comprehensive evaluation if requested
        if evaluate:
            typer.echo("\n🔍 Running comprehensive evaluation...")
            evaluator = ModelEvaluator()

            # Generate detailed performance report
            report = evaluator.generate_evaluation_report(results["model"], position, test_metrics)
            typer.echo("\n📋 Comprehensive Evaluation Report:")
            typer.echo(report)

        # Step 7: Run backtesting analysis if requested
        if backtest:
            typer.echo("\n🔄 Running backtest analysis...")
            typer.echo("   (Testing model on historical predictions vs actual results)")

            from src.ml.models.evaluation import BacktestConfiguration

            # Configure backtesting parameters
            backtest_config = BacktestConfiguration(
                start_date=start_dt,
                end_date=end_dt,
                save_detailed_results=True,  # Store detailed results for analysis
            )

            # Execute backtesting (requires historical prediction data)
            try:
                backtest_results = evaluator.backtest_model(
                    results["model"], position, backtest_config
                )

                # Display key business metrics
                financial = backtest_results["financial_metrics"]
                typer.echo("\n💰 Backtesting Results:")
                typer.echo(f"  Hit Rate: {financial.get('hit_rate', 0):.1f}% (successful picks)")
                typer.echo(
                    f"  Simulated ROI: {financial.get('simulated_roi', 0):.2f} (return on investment)"
                )
                typer.echo(
                    f"  Total Weeks: {financial.get('total_weeks_analyzed', 0)} weeks analyzed"
                )

            except Exception as backtest_error:
                typer.echo(f"⚠️  Backtesting failed: {backtest_error}")
                typer.echo("   This usually means no historical prediction data is available")

        # Success message with next steps
        typer.echo("\n✅ Training completed successfully!")

        if save_model:
            typer.echo(f"💾 Model saved as: {config.model_name}")
            typer.echo("   Use 'deploy-model' command to activate for predictions")

        typer.echo("\n🎯 Next steps:")
        typer.echo("   1. Review evaluation metrics above")
        typer.echo("   2. Compare with previous model versions")
        typer.echo("   3. Deploy if performance is satisfactory")
        typer.echo("   4. Monitor performance in production")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        typer.echo("\n⚠️  Training interrupted by user", err=True)
        raise typer.Exit(1) from None
    except Exception as e:
        # Log detailed error for debugging
        logger.exception("Training failed with unexpected error")

        # User-friendly error message
        typer.echo(f"\n❌ Training failed: {e}", err=True)
        typer.echo("\n🔧 Troubleshooting tips:")
        typer.echo("   1. Check database connection and data availability")
        typer.echo("   2. Verify date range contains sufficient training data")
        typer.echo("   3. Ensure position is valid (QB, RB, WR, TE, DEF)")
        typer.echo("   4. Check logs for detailed error information")

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
