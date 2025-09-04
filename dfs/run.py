#!/usr/bin/env python3
"""
Optimized CLI for DFS optimization system with faster predictions.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest import BacktestConfig, BacktestRunner, run_quick_backtest
from data import (
    collect_injury_data,
    collect_nfl_data,
    collect_odds_data,
    collect_weather_data_optimized,
    get_current_week_players,
    get_db_connection,
    get_player_features,
    get_training_data,
    import_spreadspoke_data,
    load_draftkings_csv,
)
from calculate_snaps import calculate_snap_counts_from_pbp
from models import ModelConfig, create_model
from optimize import (
    LineupConstraints,
    Player,
    export_lineup_to_csv,
    generate_multiple_lineups,
    optimize_cash_game_lineup,
    optimize_tournament_lineup,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "data/nfl_dfs.db"
DEFAULT_MODELS_DIR = "models"
DEFAULT_OUTPUT_DIR = "lineups"


def get_available_seasons(db_path: str = DEFAULT_DB_PATH) -> List[int]:
    """Get all seasons with completed games available in the database."""
    with get_db_connection(db_path) as conn:
        # Only return seasons that have finished games with player stats
        seasons = conn.execute("""
            SELECT DISTINCT g.season
            FROM games g
            WHERE g.game_finished = 1
            AND EXISTS (
                SELECT 1 FROM player_stats ps
                WHERE ps.game_id = g.id
            )
            ORDER BY g.season
        """).fetchall()
        return [season[0] for season in seasons]


def get_position_specific_seasons(
    position: str, available_seasons: List[int]
) -> List[int]:
    """Get optimal training seasons for each position based on recent NFL evolution."""
    current_year = max(available_seasons) if available_seasons else 2025

    if position == "DST":
        # DST: Use 2019, 2021-2024+ (skip 2020 COVID year)
        target_seasons = [2019] + list(range(2021, current_year + 1))
    else:
        # QB, RB, WR, TE: Use 2018-2019, 2021-2024+ (ALWAYS skip 2020 COVID year)
        target_seasons = list(range(2018, 2020)) + list(range(2021, current_year + 1))

    # Filter to only seasons we actually have data for
    valid_seasons = [s for s in target_seasons if s in available_seasons]
    logger.info(f"Using {len(valid_seasons)} seasons for {position}: {valid_seasons}")
    return valid_seasons


def train_models(
    seasons: List[int] = None,
    positions: List[str] = None,
    tune_lr: bool = False,
    tune_batch_size: bool = False,
    tune_all: bool = False,
    trials: int = 20,
    epochs: int = 100,
    override_lr: float = None,
    override_batch_size: int = None,
    simplified_dst: bool = False,
):
    """Train prediction models for specified positions using position-specific seasons.

    Args:
        seasons: List of seasons to use for training (if None, uses position-specific optimal ranges)
        positions: Specific positions to train (default: all)
        tune_lr: Whether to find optimal learning rate
        tune_batch_size: Whether to find optimal batch size
        tune_all: Whether to perform full hyperparameter optimization
        trials: Number of trials for hyperparameter optimization
        epochs: Number of epochs for hyperparameter tuning trials
        override_lr: Override learning rate (if not tuning)
        override_batch_size: Override batch size (if not tuning)
    """
    available_seasons = get_available_seasons()

    if not available_seasons:
        logger.error("No data available for training. Run 'collect' first.")
        return

    if positions is None:
        positions = ["QB", "RB", "WR", "TE", "DST"]

    # Create models directory
    models_dir = Path(DEFAULT_MODELS_DIR)
    models_dir.mkdir(exist_ok=True)

    # Train each position model
    for position in positions:
        logger.info(f"Training {position} model...")

        try:
            # Get position-specific seasons or use provided seasons
            if seasons is None:
                position_seasons = get_position_specific_seasons(
                    position, available_seasons
                )
            else:
                position_seasons = seasons
                logger.info(
                    f"Using provided seasons for {position}: {position_seasons}"
                )

            # Get training data
            X, y, feature_names = get_training_data(
                position, position_seasons, DEFAULT_DB_PATH
            )

            if len(X) == 0:
                logger.warning(f"No training data available for {position}")
                continue

            # Data quality validation
            import numpy as np

            # Check for NaN/Inf values
            nan_count_X = np.isnan(X).sum()
            inf_count_X = np.isinf(X).sum()
            nan_count_y = np.isnan(y).sum()
            inf_count_y = np.isinf(y).sum()

            if nan_count_X > 0 or inf_count_X > 0:
                logger.warning(
                    f"Found {nan_count_X} NaN and {inf_count_X} Inf values in features. Cleaning..."
                )
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if nan_count_y > 0 or inf_count_y > 0:
                logger.warning(
                    f"Found {nan_count_y} NaN and {inf_count_y} Inf values in targets. Cleaning..."
                )
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Log target range for fantasy points (clipping now handled in models.py)
            y_min, y_max = np.min(y), np.max(y)
            if y_min < -10 or y_max > 100:
                logger.info(
                    f"Target values have wide range: min={y_min:.2f}, max={y_max:.2f} (this is ok, models.py handles position-specific clipping)"
                )

            # Check feature variance
            feature_variance = np.var(X, axis=0)
            zero_var_features = np.sum(feature_variance < 1e-10)
            if zero_var_features > 0:
                logger.warning(
                    f"Found {zero_var_features} zero-variance features out of {len(feature_names)}"
                )

            logger.info(
                f"Training {position} with {len(X)} samples, {len(feature_names)} features"
            )
            logger.info(
                f"Target stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}, min={np.min(y):.2f}, max={np.max(y):.2f}"
            )

            # Time-based split to prevent data leakage FIRST
            # Use last 20% of games chronologically for validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"Train/Val split: {len(X_train)} training samples, {len(X_val)} validation samples")
            logger.info(f"Using chronological split - training on earlier games, validating on later games")

            # Create and train model (use ensemble for QB and RB)
            config = ModelConfig(position=position, features=feature_names)

            # Use simplified model for DST if enabled
            if position == "DST" and simplified_dst:
                logger.info("Using Simplified DST Model (Vegas-focused)")
                from models_dst_simple import SimplifiedDSTModel
                model = SimplifiedDSTModel()
                # Train the simplified model
                results = model.train(X_train, y_train, X_val, y_val, feature_names)
                # Create a result object compatible with the rest of the pipeline
                from models import TrainingResult
                result = TrainingResult(
                    model=model,
                    training_time=0,
                    best_iteration=0,
                    feature_importance=None,
                    train_mae=results['train_mae'],
                    val_mae=results['val_mae'],
                    train_rmse=0,
                    val_rmse=0,
                    train_r2=0,
                    val_r2=0,
                    training_samples=len(X_train),
                    validation_samples=len(X_val),
                    feature_count=len(feature_names),
                    train_spearman=results['train_spearman'],
                    val_spearman=results['val_spearman']
                )
                # Skip normal training flow
                model.save_model(str(models_dir / "dst_model_simplified.pkl"))
                logger.info(
                    f"Simplified DST model saved. Val MAE: {result.val_mae:.2f}, Val Spearman: {result.val_spearman:.3f}"
                )
                continue  # Skip to next position
            else:
                use_ensemble = (
                    position in ["QB", "WR", "TE"]
                )  # Enable ensemble for QB, WR, TE (RB neural-only, DST CatBoost-only perform better)
                model = create_model(position, config, use_ensemble=use_ensemble)

            # Split already done above for simplified DST, skip if already split
            if not (position == "DST" and simplified_dst):
                # Time-based split to prevent data leakage
                # Use last 20% of games chronologically for validation
                # This ensures we're always predicting future games, not past ones
                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # Log split information for transparency
                logger.info(
                    f"Train/Val split: {len(X_train)} training samples, {len(X_val)} validation samples"
                )
                logger.info(
                    "Using chronological split - training on earlier games, validating on later games"
                )

            # Apply hyperparameter tuning if requested
            if tune_all:
                logger.info(
                    f"Running full hyperparameter optimization for {position} ({trials} trials)..."
                )
                best_params = model.tune_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=trials, epochs=epochs
                )
                logger.info(f"Best hyperparameters for {position}: {best_params}")

                # CRITICAL FIX: Save the best model from tuning, don't skip
                # The model now contains the best model from hyperparameter tuning
                model_path = models_dir / f"{position.lower()}_model.pth"
                model.save_model(str(model_path))
                logger.info(
                    f"Best tuned model saved to {model_path}. Hyperparameters: {best_params}"
                )
                continue  # Skip additional training since we already have the best model
            else:
                # Apply individual tuning options
                if tune_lr:
                    logger.info(f"Finding optimal learning rate for {position}...")
                    optimal_lr = model.find_optimal_lr(X_train, y_train)
                    model.learning_rate = optimal_lr
                    logger.info(f"Using optimal LR for {position}: {optimal_lr:.2e}")

                if tune_batch_size:
                    logger.info(f"Finding optimal batch size for {position}...")
                    optimal_batch_size = model.optimize_batch_size(
                        X_train, y_train, X_val, y_val
                    )
                    model.batch_size = optimal_batch_size
                    logger.info(
                        f"Using optimal batch size for {position}: {optimal_batch_size}"
                    )

                # Apply manual overrides
                if override_lr is not None:
                    model.learning_rate = override_lr
                    logger.info(f"Using override LR for {position}: {override_lr:.2e}")

                if override_batch_size is not None:
                    model.batch_size = override_batch_size
                    logger.info(
                        f"Using override batch size for {position}: {override_batch_size}"
                    )

            # Train model
            result = model.train(X_train, y_train, X_val, y_val)

            # Feature importance analysis
            logger.info(f"Analyzing feature importance for {position}...")
            feature_importance = analyze_feature_importance(
                model, X_val, y_val, feature_names
            )

            # Log top 20 most important features
            if feature_importance:
                logger.info(f"Top 20 features for {position}:")
                for i, (feat, importance) in enumerate(feature_importance[:20], 1):
                    logger.info(f"  {i:2}. {feat:40} {importance:.4f}")

                # Save feature importance to file
                import json
                importance_path = models_dir / f"{position.lower()}_feature_importance.json"
                with open(importance_path, "w") as f:
                    json.dump(
                        {feat: float(imp) for feat, imp in feature_importance},
                        f,
                        indent=2
                    )
                logger.info(f"Feature importance saved to {importance_path}")

            # Save model
            model_path = models_dir / f"{position.lower()}_model.pth"
            model.save_model(str(model_path))

            # Log appropriate metrics based on position
            if position == "DST" and hasattr(result, 'val_spearman') and result.val_spearman is not None:
                logger.info(
                    f"{position} model saved. Val MAE: {result.val_mae:.2f}, Val Spearman: {result.val_spearman:.3f}, Val R²: {result.val_r2:.3f}"
                )
            else:
                logger.info(
                    f"{position} model saved. Val MAE: {result.val_mae:.2f}, Val R²: {result.val_r2:.3f}"
                )

        except Exception as e:
            logger.error(f"Failed to train {position} model: {e}")
            import traceback

            logger.debug(traceback.format_exc())


def analyze_feature_importance(model, X_val, y_val, feature_names):
    """Analyze feature importance using permutation importance.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        feature_names: List of feature names

    Returns:
        List of (feature_name, importance) tuples sorted by importance
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    # Get baseline predictions
    baseline_pred = model.predict(X_val).point_estimate
    baseline_mae = mean_absolute_error(y_val, baseline_pred)

    feature_importance = []

    for i, feature_name in enumerate(feature_names):
        # Permute feature values
        X_val_permuted = X_val.copy()
        np.random.shuffle(X_val_permuted[:, i])

        # Get predictions with permuted feature
        permuted_pred = model.predict(X_val_permuted).point_estimate
        permuted_mae = mean_absolute_error(y_val, permuted_pred)

        # Calculate importance as increase in error
        importance = permuted_mae - baseline_mae
        feature_importance.append((feature_name, importance))

    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def backtest_models(
    seasons: List[int] = None,
    positions: List[str] = None,
    walk_forward_windows: int = 3,
):
    """Run comprehensive backtesting with walk-forward validation.

    Args:
        seasons: List of seasons to test on
        positions: Positions to backtest (default: all)
        walk_forward_windows: Number of walk-forward validation windows
    """
    if positions is None:
        positions = ["QB", "RB", "WR", "TE", "DST"]

    if seasons is None:
        # Use last 3 seasons for backtesting
        available_seasons = get_available_seasons()
        if len(available_seasons) >= 3:
            seasons = available_seasons[-3:]
        else:
            seasons = available_seasons

    logger.info(f"Running backtesting for positions: {positions}")
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Walk-forward windows: {walk_forward_windows}")

    backtest_results = {}

    for position in positions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting {position} model")
        logger.info(f"{'='*60}")

        position_results = {
            "walk_forward_results": [],
            "out_of_time_results": None,
            "summary": {}
        }

        # Walk-forward validation
        if len(seasons) >= 2:
            window_size = max(1, len(seasons) // walk_forward_windows)

            for window_idx in range(walk_forward_windows):
                train_end = min((window_idx + 1) * window_size, len(seasons) - 1)
                train_seasons = seasons[:train_end]
                test_seasons = [seasons[train_end]] if train_end < len(seasons) else []

                if not test_seasons:
                    continue

                logger.info(f"\nWalk-forward window {window_idx + 1}:")
                logger.info(f"  Train: {train_seasons}")
                logger.info(f"  Test: {test_seasons}")

                # Get training and test data
                X_train, y_train, feature_names = get_training_data(
                    position, train_seasons, DEFAULT_DB_PATH
                )
                X_test, y_test, _ = get_training_data(
                    position, test_seasons, DEFAULT_DB_PATH
                )

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Insufficient data for window {window_idx + 1}")
                    continue

                # Train model on training window
                config = ModelConfig(position=position, features=feature_names)
                model = create_model(position, config, use_ensemble=False)

                # Use 80/20 split within training data for validation
                split_idx = int(0.8 * len(X_train))
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

                # Quick training for backtesting (fewer epochs)
                model.epochs = 50
                result = model.train(X_tr, y_tr, X_val, y_val)

                # Test on held-out season
                from sklearn.metrics import mean_absolute_error, r2_score
                from scipy.stats import spearmanr

                predictions = model.predict(X_test).point_estimate
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                spearman, _ = spearmanr(y_test, predictions)

                window_result = {
                    "train_seasons": train_seasons,
                    "test_seasons": test_seasons,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "mae": mae,
                    "r2": r2,
                    "spearman": spearman,
                    "val_mae": result.val_mae,
                    "val_r2": result.val_r2,
                }

                position_results["walk_forward_results"].append(window_result)

                logger.info(f"  Test MAE: {mae:.3f}")
                logger.info(f"  Test R²: {r2:.3f}")
                logger.info(f"  Test Spearman: {spearman:.3f}")

        # Calculate summary statistics
        if position_results["walk_forward_results"]:
            maes = [r["mae"] for r in position_results["walk_forward_results"]]
            r2s = [r["r2"] for r in position_results["walk_forward_results"]]
            spearmans = [r["spearman"] for r in position_results["walk_forward_results"]
                        if r["spearman"] is not None and not np.isnan(r["spearman"])]

            position_results["summary"] = {
                "avg_mae": np.mean(maes),
                "std_mae": np.std(maes),
                "avg_r2": np.mean(r2s),
                "std_r2": np.std(r2s),
                "avg_spearman": np.mean(spearmans) if spearmans else 0,
                "std_spearman": np.std(spearmans) if spearmans else 0,
                "worst_mae": np.max(maes),
                "best_mae": np.min(maes),
            }

            logger.info(f"\n{position} Backtest Summary:")
            logger.info(f"  Average MAE: {position_results['summary']['avg_mae']:.3f} ± {position_results['summary']['std_mae']:.3f}")
            logger.info(f"  Average R²: {position_results['summary']['avg_r2']:.3f} ± {position_results['summary']['std_r2']:.3f}")
            logger.info(f"  Average Spearman: {position_results['summary']['avg_spearman']:.3f} ± {position_results['summary']['std_spearman']:.3f}")
            logger.info(f"  Best/Worst MAE: {position_results['summary']['best_mae']:.3f} / {position_results['summary']['worst_mae']:.3f}")

        backtest_results[position] = position_results

    # Save backtest results
    import json
    results_path = Path(DEFAULT_MODELS_DIR) / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(backtest_results, f, indent=2, default=str)

    logger.info(f"\nBacktest results saved to {results_path}")

    return backtest_results


def update_injury_statuses(
    injury_file: str = None,
    manual_updates: Dict[str, str] = None,
    db_path: str = DEFAULT_DB_PATH,
    season: int = None,
    week: int = None,
):
    """Update player injury statuses from CSV file or manual input.

    Args:
        injury_file: Path to CSV file with columns: player_name, injury_status
        manual_updates: Dictionary of player_name -> injury_status
        db_path: Database path
        season: Season to update injuries for
        week: Week to update injuries for

    Injury status codes:
        Q: Questionable (75% likely to play)
        D: Doubtful (25% likely to play)
        OUT: Out (will not play)
        IR: Injured Reserve (will not play)
        PUP: Physically Unable to Perform (will not play)
        PPD: Game Postponed
    """
    conn = get_db_connection(db_path)
    updated_count = 0

    # Get current season/week if not provided
    if not season or not week:
        latest = conn.execute(
            "SELECT MAX(season), MAX(week) FROM games WHERE game_finished = 0"
        ).fetchone()
        if latest and latest[0]:
            season = season or latest[0]
            week = week or latest[1]
        else:
            # Fall back to most recent data
            latest = conn.execute(
                "SELECT season, MAX(week) FROM games GROUP BY season ORDER BY season DESC LIMIT 1"
            ).fetchone()
            if latest:
                season = season or latest[0]
                week = week or latest[1]

    if not season or not week:
        logger.error("Could not determine season/week for injury updates")
        return 0

    try:
        report_date = datetime.now().strftime("%Y-%m-%d")

        # Update from CSV file if provided
        if injury_file and Path(injury_file).exists():
            import pandas as pd

            df = pd.read_csv(injury_file)
            for _, row in df.iterrows():
                player_name = row["player_name"]
                injury_status = row["injury_status"].upper()
                injury_description = row.get("injury_description", "")
                practice_status = row.get("practice_status", "")

                # Find player
                player_record = conn.execute(
                    "SELECT id, team_id FROM players WHERE display_name = ? OR player_name = ?",
                    (player_name, player_name),
                ).fetchone()

                if player_record:
                    player_id, team_id = player_record

                    # Find game for this week
                    game_record = conn.execute(
                        """SELECT id FROM games
                           WHERE season = ? AND week = ?
                           AND (home_team_id = ? OR away_team_id = ?)""",
                        (season, week, team_id, team_id)
                    ).fetchone()

                    game_id = game_record[0] if game_record else None

                    # Insert or update injury report
                    conn.execute(
                        """INSERT OR REPLACE INTO injury_reports
                           (player_id, season, week, game_id, report_date, injury_status,
                            injury_designation, injury_body_part, practice_status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (player_id, season, week, game_id, report_date, injury_status,
                         injury_status, injury_description, practice_status),
                    )
                    updated_count += 1

        # Update from manual dictionary if provided
        if manual_updates:
            for player_name, injury_status in manual_updates.items():
                # Find player
                player_record = conn.execute(
                    "SELECT id, team_id FROM players WHERE display_name = ? OR player_name = ?",
                    (player_name, player_name),
                ).fetchone()

                if player_record:
                    player_id, team_id = player_record

                    # Find game for this week
                    game_record = conn.execute(
                        """SELECT id FROM games
                           WHERE season = ? AND week = ?
                           AND (home_team_id = ? OR away_team_id = ?)""",
                        (season, week, team_id, team_id)
                    ).fetchone()

                    game_id = game_record[0] if game_record else None

                    # Insert or update injury report
                    conn.execute(
                        """INSERT OR REPLACE INTO injury_reports
                           (player_id, season, week, game_id, report_date, injury_status,
                            injury_designation, injury_body_part, practice_status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (player_id, season, week, game_id, report_date, injury_status.upper(),
                         injury_status.upper(), "", ""),
                    )
                    updated_count += 1

        conn.commit()
        logger.info(f"Updated injury status for {updated_count} players for season {season} week {week}")

    finally:
        conn.close()

    return updated_count


def extract_primary_position(roster_position: str) -> str:
    """Extract primary position from DraftKings roster position string.

    Examples:
        'QB' -> 'QB'
        'RB/FLEX' -> 'RB'
        'WR/FLEX' -> 'WR'
        'TE/FLEX' -> 'TE'
        'DST' -> 'DST'
    """
    if not roster_position:
        return "FLEX"

    # Split by '/' and take the first part
    primary = roster_position.split("/")[0].strip()

    # Handle special cases
    if primary in ["QB", "RB", "WR", "TE", "DST", "DEF"]:
        return primary
    elif primary == "FLEX":
        # Pure FLEX should not happen, but handle it
        return "FLEX"
    else:
        return primary


def get_injury_multiplier(injury_status: str) -> Tuple[float, float, float]:
    """Get projection multipliers based on injury status.

    Returns:
        Tuple of (projection_multiplier, floor_multiplier, ceiling_multiplier)
    """
    if not injury_status:
        return (1.0, 1.0, 1.0)

    status = injury_status.upper()

    # Injury status multipliers
    multipliers = {
        "Q": (
            0.85,
            0.75,
            0.90,
        ),  # Questionable: reduce projection by 15%, floor by 25%, ceiling by 10%
        "D": (
            0.40,
            0.20,
            0.50,
        ),  # Doubtful: reduce projection by 60%, floor by 80%, ceiling by 50%
        "OUT": (0.0, 0.0, 0.0),  # Out: zero projections
        "IR": (0.0, 0.0, 0.0),  # Injured Reserve: zero projections
        "PUP": (0.0, 0.0, 0.0),  # PUP: zero projections
        "PPD": (0.0, 0.0, 0.0),  # Postponed: zero projections
    }

    return multipliers.get(status, (1.0, 1.0, 1.0))


def predict_players_optimized(
    contest_id: str = None, output_file: str = None, injury_file: str = None
):
    """Optimized version of predict_players with batch processing and injury status support.

    Args:
        contest_id: DraftKings contest ID
        output_file: Output CSV file for predictions
        injury_file: Optional CSV file with injury statuses
    """
    logger.info(f"Generating predictions for contest: {contest_id}")

    # Update injury statuses if file provided
    if injury_file:
        update_injury_statuses(injury_file=injury_file)

    # Get players for contest
    players_data = get_current_week_players(contest_id, DEFAULT_DB_PATH)

    if not players_data:
        logger.error("No players found for contest")
        return []

    # Load trained models and cache feature metadata
    models_dir = Path(DEFAULT_MODELS_DIR)
    models = {}
    position_feature_names = {}

    # Load feature metadata once from a small sample
    logger.info("Loading model metadata...")
    for position in ["QB", "RB", "WR", "TE", "DST"]:
        # For DST, check for multiple file formats
        if position == "DST":
            # Check for simplified model first (.pkl), then regular CatBoost (.cbm), then neural (.pth)
            model_path = None
            for ext in ['.pkl', '.cbm', '.pth']:
                candidate_path = models_dir / f"{position.lower()}_model{ext}"
                if candidate_path.exists():
                    model_path = candidate_path
                    break

            if model_path and model_path.suffix == '.pkl':
                # Load simplified DST model
                try:
                    from models_dst_simple import SimplifiedDSTModel
                    model = SimplifiedDSTModel()
                    model.load(str(model_path))
                    models[position] = model
                    # Simplified model has its own feature set
                    position_feature_names[position] = None  # Will be handled by the model
                    logger.info(f"Loaded simplified DST model from {model_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load simplified DST model: {e}")
                continue
        else:
            model_path = models_dir / f"{position.lower()}_model.pth"

        if model_path and model_path.exists():
            try:
                # Get feature names from a minimal sample (just for metadata)
                available_seasons = get_available_seasons()[
                    :1
                ]  # Use only 1 season for metadata
                X_sample, _, feature_names = get_training_data(
                    position, available_seasons, DEFAULT_DB_PATH
                )
                if len(X_sample) > 0:
                    config = ModelConfig(position=position, features=feature_names)
                    # Use ensemble for QB, WR, TE (matches training configuration, RB neural-only, DST CatBoost-only)
                    use_ensemble = position in ["QB", "WR", "TE"]
                    model = create_model(position, config, use_ensemble=use_ensemble)
                    # Let load_model determine the input size from the saved model
                    model.load_model(str(model_path))
                    models[position] = model
                    position_feature_names[position] = feature_names
                    logger.info(
                        f"Loaded {position} model with {len(feature_names)} features"
                    )
            except Exception as e:
                logger.warning(f"Failed to load {position} model: {e}")

    # Batch fetch recent games for all players
    logger.info("Fetching player game data...")
    conn = get_db_connection(DEFAULT_DB_PATH)

    player_ids = [p["player_id"] for p in players_data]
    placeholders = ",".join("?" * len(player_ids))

    recent_games_query = f"""
        SELECT ps.player_id, MAX(g.id) as game_id
        FROM games g
        JOIN player_stats ps ON g.id = ps.game_id
        WHERE ps.player_id IN ({placeholders})
        GROUP BY ps.player_id
    """

    recent_games = {}
    for row in conn.execute(recent_games_query, player_ids).fetchall():
        recent_games[row[0]] = row[1]

    # Fetch injury statuses for all players
    # Get current season/week for injury lookups
    current_game = conn.execute(
        """SELECT DISTINCT g.season, g.week
           FROM draftkings_salaries dk
           JOIN players p ON dk.player_id = p.id
           JOIN teams t ON p.team_id = t.id
           JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id)
           WHERE dk.contest_id = ?
           LIMIT 1""",
        (contest_id,)
    ).fetchone()

    if current_game:
        season, week = current_game
        injury_status_query = f"""
            SELECT ir.player_id, ir.injury_status
            FROM injury_reports ir
            WHERE ir.player_id IN ({placeholders})
            AND ir.season = ? AND ir.week = ?
            ORDER BY ir.report_date DESC
        """

        injury_statuses = {}
        for row in conn.execute(injury_status_query, player_ids + [season, week]).fetchall():
            if row[1]:  # Only store if injury status is not NULL
                injury_statuses[row[0]] = row[1]

        if injury_statuses:
            logger.info(f"Found injury statuses for {len(injury_statuses)} players for season {season} week {week}")
    else:
        injury_statuses = {}
        logger.warning(f"Could not determine season/week for contest {contest_id}")

    conn.close()

    # Generate predictions
    predictions = []
    players_by_position = {}

    # Group players by DraftKings roster position for batch processing
    for player_data in players_data:
        # Use DraftKings roster position instead of database position
        dk_position = extract_primary_position(player_data.get("roster_position", ""))

        # Store the DK position for later use
        player_data["dk_position"] = dk_position

        # Group by DK position for model selection
        if dk_position not in players_by_position:
            players_by_position[dk_position] = []
        players_by_position[dk_position].append(player_data)

    # Process each position in batches
    for position, position_players in players_by_position.items():
        if position in models:
            model = models[position]
            feature_names = position_feature_names.get(position)

            # Handle simplified DST model separately
            if position == "DST" and feature_names is None:
                # This is the simplified DST model
                from models_dst_simple import SimplifiedDSTModel
                if isinstance(model, SimplifiedDSTModel):
                    for player_data in position_players:
                        try:
                            # SimplifiedDSTModel handles its own feature extraction and prediction
                            pred = model.predict_single(player_data)
                            if pred is not None:
                                player_id = player_data["player_id"]
                                injury_status = injury_statuses.get(player_id)

                                # Apply injury adjustments if needed
                                if injury_status:
                                    if injury_status == "Questionable":
                                        pred *= 0.85
                                    elif injury_status == "Doubtful":
                                        pred *= 0.5

                                predictions.append({
                                    "player_id": player_id,
                                    "name": player_data["name"],
                                    "team": player_data["team"],
                                    "opponent": player_data.get("opponent", ""),
                                    "position": position,
                                    "dk_position": player_data["dk_position"],
                                    "salary": player_data["salary"],
                                    "projection": max(0.0, pred),
                                    "floor": max(0.0, pred * 0.8),
                                    "ceiling": max(0.0, pred * 1.3),
                                    "value": max(0.0, pred / (player_data["salary"] / 1000)) if player_data["salary"] > 0 else 0,
                                    "injury_status": injury_status
                                })
                        except Exception as e:
                            logger.debug(f"Failed to predict for {player_data['name']}: {e}")
                    continue

            if feature_names and len(feature_names) > 0:
                # Batch extract features for all players in this position
                batch_features = []
                valid_players = []
                fallback_players = []

                for player_data in position_players:
                    player_id = player_data["player_id"]

                    if player_id in recent_games:
                        try:
                            features_dict = get_player_features(
                                player_id, recent_games[player_id]
                            )
                            if features_dict:
                                # CRITICAL: Filter out players with 0 FPPG or very few games
                                avg_fantasy_points = features_dict.get(
                                    "avg_fantasy_points", 0
                                )
                                games_played = features_dict.get("games_played", 0)

                                if avg_fantasy_points <= 0.1:
                                    logger.debug(
                                        f"Skipping {player_data['name']} - FPPG is {avg_fantasy_points:.1f}"
                                    )
                                    fallback_players.append(player_data)
                                elif games_played < 3:
                                    logger.debug(
                                        f"Skipping {player_data['name']} - only {games_played} games played"
                                    )
                                    fallback_players.append(player_data)
                                else:
                                    feature_vector = [
                                        features_dict.get(name, 0)
                                        for name in feature_names
                                    ]
                                    batch_features.append(feature_vector)
                                    valid_players.append(player_data)
                            else:
                                fallback_players.append(player_data)
                        except Exception as e:
                            logger.debug(
                                f"Failed to get features for {player_data['name']}: {e}"
                            )
                            fallback_players.append(player_data)
                    else:
                        fallback_players.append(player_data)

                # Batch predict if we have valid features
                if batch_features:
                    try:
                        X_pred = np.array(batch_features, dtype=np.float32)

                        # FORCE exact input size to match model expectations
                        if (
                            hasattr(model, "input_size")
                            and model.input_size
                            and X_pred.shape[1] != model.input_size
                        ):
                            logger.warning(
                                f"Feature mismatch for {position}: {X_pred.shape[1]} -> {model.input_size} (fixing)"
                            )
                            # Pad with zeros if we have fewer features than expected
                            if X_pred.shape[1] < model.input_size:
                                padding = np.zeros(
                                    (
                                        X_pred.shape[0],
                                        model.input_size - X_pred.shape[1],
                                    ),
                                    dtype=np.float32,
                                )
                                X_pred = np.concatenate([X_pred, padding], axis=1)
                            # Truncate if we have more features than expected
                            elif X_pred.shape[1] > model.input_size:
                                X_pred = X_pred[:, : model.input_size]
                            logger.info(
                                f"Fixed feature dimensions for {position}: {X_pred.shape[1]} features"
                            )

                        prediction_result = model.predict(X_pred)

                        for i, player_data in enumerate(valid_players):
                            player_id = player_data["player_id"]
                            injury_status = injury_statuses.get(player_id)

                            # Get base predictions from model
                            base_proj = float(prediction_result.point_estimate[i])
                            base_floor = float(prediction_result.floor[i])
                            base_ceiling = float(prediction_result.ceiling[i])

                            # Apply injury adjustments using FeatureEngineer if injury data available
                            if season and week:
                                try:
                                    from features import FeatureEngineer
                                    fe = FeatureEngineer(DEFAULT_DB_PATH)
                                    injury_features = fe.get_injury_features(player_id, season, week)

                                    # Use sophisticated injury adjustments
                                    proj_points, floor_points, ceiling_points = fe.get_injury_adjusted_projections(
                                        base_proj, base_floor, base_ceiling, injury_features
                                    )
                                except Exception as e:
                                    logger.debug(f"Could not get injury features for {player_data['name']}: {e}")
                                    # Fall back to simple multipliers
                                    proj_mult, floor_mult, ceil_mult = get_injury_multiplier(injury_status)
                                    proj_points = base_proj * proj_mult
                                    floor_points = base_floor * floor_mult
                                    ceiling_points = base_ceiling * ceil_mult
                            else:
                                # Fall back to simple multipliers if no season/week context
                                proj_mult, floor_mult, ceil_mult = get_injury_multiplier(injury_status)
                                proj_points = base_proj * proj_mult
                                floor_points = base_floor * floor_mult
                                ceiling_points = base_ceiling * ceil_mult

                            pred_dict = {
                                "player_id": player_id,
                                "name": player_data["name"],
                                "position": player_data[
                                    "dk_position"
                                ],  # Use DK position
                                "salary": player_data["salary"],
                                "projected_points": round(proj_points, 1),
                                "floor": round(floor_points, 1),
                                "ceiling": round(ceiling_points, 1),
                                "team": player_data["team_abbr"],
                                "roster_position": player_data["roster_position"],
                            }

                            # Add injury status to output if present
                            if injury_status:
                                pred_dict["injury_status"] = injury_status

                            predictions.append(pred_dict)
                            logger.debug(
                                f"Model prediction for {player_data['name']}: {prediction_result.point_estimate[i]:.1f}"
                            )
                    except Exception as e:
                        logger.warning(f"Batch prediction failed for {position}: {e}")
                        fallback_players.extend(valid_players)

                # Handle fallback players
                for player_data in fallback_players:
                    injury_status = injury_statuses.get(player_data["player_id"])
                    predictions.append(
                        generate_fallback_prediction(player_data, injury_status, season, week)
                    )
            else:
                # No features available, use fallback for all
                for player_data in position_players:
                    injury_status = injury_statuses.get(player_data["player_id"])
                    predictions.append(
                        generate_fallback_prediction(player_data, injury_status, season, week)
                    )

        elif position in ["DST", "DEF"]:
            # Handle DST separately
            for player_data in position_players:
                if "DST" in models:
                    try:
                        # Simplified DST features (in production would extract real defensive stats)
                        dst_features = generate_dst_features(player_data)
                        X_dst = np.array([dst_features], dtype=np.float32)
                        model = models["DST"]
                        prediction_result = model.predict(X_dst)

                        injury_status = injury_statuses.get(player_data["player_id"])
                        proj_mult, floor_mult, ceil_mult = get_injury_multiplier(
                            injury_status
                        )

                        pred_dict = {
                            "player_id": player_data["player_id"],
                            "name": player_data["name"],
                            "position": player_data["dk_position"],  # Use DK position
                            "salary": player_data["salary"],
                            "projected_points": round(
                                float(prediction_result.point_estimate[0]) * proj_mult,
                                1,
                            ),
                            "floor": round(
                                float(prediction_result.floor[0]) * floor_mult, 1
                            ),
                            "ceiling": round(
                                float(prediction_result.ceiling[0]) * ceil_mult, 1
                            ),
                            "team": player_data["team_abbr"],
                            "roster_position": player_data["roster_position"],
                        }

                        if injury_status:
                            pred_dict["injury_status"] = injury_status

                        predictions.append(pred_dict)
                    except Exception as e:
                        logger.debug(
                            f"DST prediction failed for {player_data['name']}: {e}"
                        )
                        injury_status = injury_statuses.get(player_data["player_id"])
                        predictions.append(
                            generate_fallback_prediction(player_data, injury_status, season, week)
                        )
                else:
                    injury_status = injury_statuses.get(player_data["player_id"])
                    predictions.append(
                        generate_fallback_prediction(player_data, injury_status, season, week)
                    )
        else:
            # Fallback for positions without models
            for player_data in position_players:
                injury_status = injury_statuses.get(player_data["player_id"])
                predictions.append(
                    generate_fallback_prediction(player_data, injury_status, season, week)
                )

    # Save predictions if requested
    if output_file:
        import pandas as pd

        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

    logger.info(f"Generated predictions for {len(predictions)} players")
    return predictions


def generate_fallback_prediction(player_data, injury_status=None, season=None, week=None):
    """Generate fallback prediction based on salary and position."""
    # Use DK position if available, otherwise fall back to database position
    position = player_data.get("dk_position", player_data["position"])

    # CRITICAL: For players with no track record, use very conservative projections
    # These are likely practice squad or rarely-used players
    # Position-based MINIMUM points for established players
    pos_averages = {
        "QB": 18.0,
        "RB": 12.0,
        "WR": 11.0,
        "TE": 9.0,
        "DST": 8.0,
        "DEF": 8.0,
    }

    # Check if this is a minimum salary player (likely no track record)
    if player_data["salary"] <= 3000:
        # Very conservative for minimum salary players
        base_projection = pos_averages.get(position, 10.0) * 0.2  # Only 20% of normal
    else:
        base_projection = pos_averages.get(position, 10.0)
        salary_factor = player_data["salary"] / 6000
        base_projection = (
            base_projection * salary_factor * 0.5
        )  # Conservative 50% for fallbacks

    # Base floor and ceiling
    base_floor = base_projection * 0.7
    base_ceiling = base_projection * 1.5

    # Apply injury adjustments using FeatureEngineer if available
    if season and week and player_data.get("player_id"):
        try:
            from features import FeatureEngineer
            fe = FeatureEngineer(DEFAULT_DB_PATH)
            injury_features = fe.get_injury_features(player_data["player_id"], season, week)
            projected_points, floor, ceiling = fe.get_injury_adjusted_projections(
                base_projection, base_floor, base_ceiling, injury_features
            )
        except Exception:
            # Fall back to simple multipliers
            proj_mult, floor_mult, ceil_mult = get_injury_multiplier(injury_status)
            projected_points = base_projection * proj_mult
            floor = base_floor * floor_mult
            ceiling = base_ceiling * ceil_mult
    else:
        # Use simple multipliers if no season/week context
        proj_mult, floor_mult, ceil_mult = get_injury_multiplier(injury_status)
        projected_points = base_projection * proj_mult
        floor = base_floor * floor_mult
        ceiling = base_ceiling * ceil_mult

    result = {
        "player_id": player_data["player_id"],
        "name": player_data["name"],
        "position": position,  # This is now the DK position
        "salary": player_data["salary"],
        "projected_points": round(projected_points, 1),
        "floor": round(floor, 1),
        "ceiling": round(ceiling, 1),
        "team": player_data["team_abbr"],
        "roster_position": player_data["roster_position"],
    }

    if injury_status:
        result["injury_status"] = injury_status

    return result


def generate_dst_features(player_data):
    """Generate simplified DST features."""
    # This is a simplified version - in production would extract real defensive stats
    return [
        20,  # avg points_allowed (league average)
        2.5,  # avg sacks
        1.0,  # avg interceptions
        0.8,  # avg fumbles_recovered
        0.1,  # avg safeties
        0.1,  # avg defensive_tds
        8.0,  # avg_recent_points
        22,  # avg_recent_points_allowed
        2.3,  # avg_recent_sacks
        10,  # week (mid-season)
        2024,  # season
    ]


def optimize_lineups(
    contest_id: str = None,
    strategy: str = "balanced",
    num_lineups: int = 1,
    output_dir: str = "lineups",
    save_predictions: str = None,
    injury_file: str = None,
):
    """Build optimal lineups for a contest.

    Args:
        contest_id: DraftKings contest ID
        strategy: Optimization strategy (balanced, tournament, cash)
        num_lineups: Number of lineups to generate
        output_dir: Directory to save lineup files
        save_predictions: Optional file to save player predictions CSV
        injury_file: Optional CSV file with injury statuses
    """
    logger.info(f"Optimizing lineups for contest: {contest_id}")
    logger.info(f"Strategy: {strategy}, Count: {num_lineups}")

    # Get player predictions (optimized version)
    try:
        predictions = predict_players_optimized(
            contest_id, save_predictions, injury_file
        )

        if not predictions:
            logger.error("No predictions generated")
            return

    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return

    # Use real odds from database to enrich player pool
    conn = get_db_connection(DEFAULT_DB_PATH)

    def get_real_odds_for_team(team_abbr: str) -> dict:
        """Fetch betting odds for the team's next game, prioritizing live odds."""
        try:
            # Find next upcoming game for this team
            row = conn.execute(
                """
                SELECT g.id, g.game_date,
                       ht.team_abbr AS home_abbr, at.team_abbr AS away_abbr
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE (ht.team_abbr = ? OR at.team_abbr = ?)
                  AND (g.game_finished = 0 OR date(g.game_date) >= date('now'))
                ORDER BY date(g.game_date) ASC
                LIMIT 1
                """,
                (team_abbr, team_abbr),
            ).fetchone()

            if not row:
                return {"implied_total": 21.0, "game_total": 44.0, "spread": 0.0}

            game_id, game_date, home_abbr, away_abbr = row

            odds = conn.execute(
                """
                SELECT favorite_team, spread_favorite, over_under_line,
                       home_team_spread, away_team_spread
                FROM betting_odds
                WHERE game_id = ?
                ORDER BY CASE WHEN source = 'odds_api' THEN 1 ELSE 2 END
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

            if not odds:
                return {"implied_total": 21.0, "game_total": 44.0, "spread": 0.0}

            favorite_team, spread_fav, total, home_spread, away_spread = odds

            is_home = team_abbr == home_abbr
            team_spread = home_spread if is_home else away_spread
            total = total or 44.0
            team_spread = team_spread or 0.0
            implied_total = total / 2.0 - team_spread / 2.0

            return {
                "implied_total": float(implied_total),
                "game_total": float(total),
                "spread": float(team_spread),
            }
        except Exception:
            return {"implied_total": 21.0, "game_total": 44.0, "spread": 0.0}

    # Convert to Player objects
    player_pool = []
    for pred in predictions:
        odds_data = get_real_odds_for_team(pred["team"])
        player = Player(
            player_id=pred["player_id"],
            name=pred["name"],
            position=pred["position"],
            salary=pred["salary"],
            projected_points=pred["projected_points"],
            floor=pred["floor"],
            ceiling=pred["ceiling"],
            team_abbr=pred["team"],
            roster_position=pred.get("roster_position", ""),
            injury_status=pred.get("injury_status"),
            # Add odds data for stacking optimization
            implied_team_total=odds_data["implied_total"],
            game_total=odds_data["game_total"],
            spread=odds_data["spread"],
        )
        player_pool.append(player)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate lineups based on strategy
    if strategy == "cash":
        logger.info("Building cash game lineup...")
        result = optimize_cash_game_lineup(player_pool)
        lineups = [result] if result.is_valid else []
    elif strategy == "tournament":
        logger.info("Building tournament lineup...")
        result = optimize_tournament_lineup(player_pool)
        lineups = [result] if result.is_valid else []
    else:  # balanced
        logger.info("Building balanced lineups...")
        lineups = generate_multiple_lineups(
            player_pool, LineupConstraints(), num_lineups=num_lineups, strategy=strategy
        )

    # Save valid lineups
    valid_count = 0
    for i, lineup in enumerate(lineups):
        if lineup.is_valid:
            filename = f"{strategy}_lineup_{i + 1}.csv"
            filepath = output_path / filename
            export_lineup_to_csv(lineup, str(filepath))
            logger.info(f"Saved lineup {i + 1} to {filepath}")
            valid_count += 1
        else:
            logger.warning(f"Lineup {i + 1} is invalid: {lineup.constraint_violations}")

    logger.info(f"Generated {valid_count} valid lineups")

    if valid_count == 0:
        logger.error(
            "No valid lineups could be generated. Check player pool and constraints."
        )

    # Close DB connection used for odds
    try:
        conn.close()
    except Exception:
        pass


def run_backtest_command(args):
    """Run backtesting with the specified arguments."""
    from datetime import datetime

    logger.info(f"🚀 Starting backtest from {args.start_date} to {args.end_date}")

    if args.quick:
        # Quick backtest
        results = run_quick_backtest(DEFAULT_DB_PATH, args.start_date, args.end_date)

        print("\n📈 BACKTEST RESULTS")
        print("=" * 50)
        print(f"Mean ROI: {results.get('mean_roi', 0):.2%}")
        print(f"Cash Rate: {results.get('mean_cash_rate', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Total Contests: {results.get('total_contests', 0)}")

    else:
        # Full backtest with custom settings
        config = BacktestConfig(
            start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
            slate_types=["main"],
            contest_types=args.contest_types,
            include_ownership=True,
            include_injuries=True,
        )

        # Create model prediction function using existing models
        def model_prediction_func(features, position):
            # Use the existing get_player_features approach
            base_prediction = features.get("fantasy_points_avg", 8.0)

            # Position-specific adjustments (simplified)
            multipliers = {"QB": 1.2, "RB": 1.0, "WR": 0.9, "TE": 0.8, "DST": 1.1}
            multiplier = multipliers.get(position, 1.0)

            # Add trend and injury risk
            trend = features.get("fantasy_points_trend", 0) * 0.5
            injury_risk = features.get("injury_risk", 1.0)

            return (base_prediction * multiplier + trend) * injury_risk

        # Run comprehensive backtest
        runner = BacktestRunner(DEFAULT_DB_PATH, config)
        results = runner.run_backtest(model_prediction_func)

        # Print detailed results
        print("\n📈 COMPREHENSIVE BACKTEST RESULTS")
        print("=" * 50)

        # Core Performance
        print("💰 Performance:")
        print(f"  Mean ROI: {results.get('mean_roi', 0):.2%}")
        print(f"  Cash Rate: {results.get('mean_cash_rate', 0):.2%}")
        print(f"  Win Rate: {results.get('win_rate', 0):.2%}")

        # Risk Metrics
        print("\n⚠️  Risk:")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"  Consistency: {results.get('consistency_score', 0):.3f}")

        print(f"\n📊 Total Contests: {results.get('total_contests', 0)}")

        # Optional correlation testing
        if args.test_correlations:
            print("\n🎯 Testing Correlation Strategies...")
            # Would implement correlation testing here

        # Optional portfolio testing
        if args.test_portfolio:
            print(f"\n💼 Testing Portfolio ({args.entries} entries)...")
            # Would implement portfolio testing here

    logger.info("✅ Backtesting completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DFS Optimization System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect NFL data")
    collect_parser.add_argument(
        "--seasons", nargs="+", type=int, help="Seasons to collect (e.g., 2022 2023)"
    )
    collect_parser.add_argument(
        "--csv", help="Path to DraftKings CSV file for salary integration"
    )
    collect_parser.add_argument(
        "--injuries", action="store_true", help="Collect injury data from NFL library"
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import external data sources")
    import_parser.add_argument(
        "--spreadspoke",
        help="Path to spreadspoke CSV file with weather and betting data",
    )
    import_parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Optional list of seasons to import (e.g., 2021 2022 2023 2024)",
    )

    # Odds command
    odds_parser = subparsers.add_parser(
        "odds", help="Collect betting odds from The Odds API"
    )
    odds_parser.add_argument(
        "--date",
        help="Target date for odds collection (YYYY-MM-DD format). If not provided, collects all upcoming games.",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction models")
    train_parser.add_argument(
        "--seasons", nargs="+", type=int, help="Seasons to use for training"
    )
    train_parser.add_argument(
        "--positions",
        nargs="+",
        choices=["QB", "RB", "WR", "TE", "DST"],
        help="Positions to train",
    )
    train_parser.add_argument(
        "--tune-lr",
        action="store_true",
        help="Find optimal learning rate using LR range test",
    )
    train_parser.add_argument(
        "--tune-batch-size",
        action="store_true",
        help="Find optimal batch size considering memory constraints",
    )
    train_parser.add_argument(
        "--tune-all",
        action="store_true",
        help="Perform full hyperparameter optimization with Optuna",
    )
    train_parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter optimization (default: 20)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for hyperparameter tuning trials (default: 100)",
    )
    train_parser.add_argument(
        "--lr", type=float, help="Override learning rate (if not tuning)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, help="Override batch size (if not tuning)"
    )
    train_parser.add_argument(
        "--simplified-dst",
        action="store_true",
        help="Use simplified Vegas-focused DST model instead of neural network"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Generate player predictions"
    )
    predict_parser.add_argument("--contest-id", help="DraftKings contest ID")
    predict_parser.add_argument("--output", help="Output CSV file for predictions")
    predict_parser.add_argument(
        "--injury-file",
        help="CSV file with injury statuses (columns: player_name, injury_status)",
    )

    # Injury command
    injury_parser = subparsers.add_parser(
        "injury", help="Update player injury statuses"
    )
    injury_parser.add_argument(
        "--csv",
        help="CSV file with injury statuses (columns: player_name, injury_status)",
    )
    injury_parser.add_argument(
        "--player",
        nargs=2,
        metavar=("NAME", "STATUS"),
        action="append",
        help="Update single player injury status (e.g., --player 'Patrick Mahomes' Q)",
    )

    # Weather command
    weather_parser = subparsers.add_parser("weather", help="Collect weather data")
    weather_parser.add_argument(
        "--historical",
        action="store_true",
        help="Collect all missing historical weather data"
    )
    weather_parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Collect weather for upcoming games only"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Build optimal lineups")
    optimize_parser.add_argument("--contest-id", help="DraftKings contest ID")
    optimize_parser.add_argument(
        "--strategy",
        choices=["cash", "tournament", "balanced"],
        default="balanced",
        help="Optimization strategy",
    )
    optimize_parser.add_argument(
        "--count", type=int, default=1, help="Number of lineups to generate"
    )
    optimize_parser.add_argument(
        "--output-dir", default="lineups", help="Directory to save lineups"
    )
    optimize_parser.add_argument(
        "--save-predictions", help="Optional: Save player predictions to CSV"
    )
    optimize_parser.add_argument(
        "--injury-file",
        help="CSV file with injury statuses (columns: player_name, injury_status)",
    )

    # Backtest command
    # Comprehensive model backtesting
    model_backtest_parser = subparsers.add_parser(
        "model-backtest", help="Run comprehensive model backtesting with walk-forward validation"
    )
    model_backtest_parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Seasons to use for backtesting (default: last 3 available)",
    )
    model_backtest_parser.add_argument(
        "--positions",
        nargs="+",
        choices=["QB", "RB", "WR", "TE", "DST"],
        help="Positions to backtest (default: all)",
    )
    model_backtest_parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="Number of walk-forward validation windows (default: 3)",
    )

    # Original backtest parser for lineup optimization testing
    backtest_parser = subparsers.add_parser("backtest", help="Run model backtesting")
    backtest_parser.add_argument(
        "--start-date", required=True, help="Backtest start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end-date", required=True, help="Backtest end date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--quick", action="store_true", help="Run quick backtest with default settings"
    )
    backtest_parser.add_argument(
        "--contest-types",
        nargs="+",
        choices=["cash", "gpp", "satellite"],
        default=["gpp"],
        help="Contest types to test",
    )
    backtest_parser.add_argument(
        "--test-correlations",
        action="store_true",
        help="Include correlation strategy testing",
    )
    backtest_parser.add_argument(
        "--test-portfolio",
        action="store_true",
        help="Include portfolio strategy testing",
    )
    backtest_parser.add_argument(
        "--entries",
        type=int,
        default=20,
        help="Number of entries for portfolio testing",
    )

    args = parser.parse_args()

    if args.command == "collect":
        seasons = args.seasons or [2023, 2024]
        logger.info(f"Collecting data for seasons: {seasons}")

        # Step 1: Collect base NFL data with enhanced columns
        collect_nfl_data(seasons, DEFAULT_DB_PATH)

        # Step 2: Calculate snap counts and usage metrics from play-by-play
        logger.info("Calculating snap counts and usage metrics from play-by-play data...")
        for season in seasons:
            try:
                calculate_snap_counts_from_pbp(season, DEFAULT_DB_PATH)
                logger.info(f"✓ Calculated snap counts for {season}")
            except Exception as e:
                logger.warning(f"Could not calculate snap counts for {season}: {e}")

        # Step 3: Load DraftKings salaries if provided
        if args.csv:
            logger.info(f"Integrating DraftKings salaries from {args.csv}")
            load_draftkings_csv(args.csv, None, DEFAULT_DB_PATH)

        # Note: Injuries option removed - only current injury data available
        if args.injuries:
            logger.info("Collecting injury data from NFL library...")
            collect_injury_data(seasons, DEFAULT_DB_PATH)

    elif args.command == "import":
        if args.spreadspoke:
            logger.info(f"Importing spreadspoke data from {args.spreadspoke}")
            import_spreadspoke_data(args.spreadspoke, DEFAULT_DB_PATH, args.seasons)
            logger.info("Spreadspoke data import completed")
        else:
            logger.error("Please specify a data source to import (--spreadspoke)")

    elif args.command == "odds":
        try:
            target_date = args.date
            if target_date:
                logger.info(f"Collecting odds for date: {target_date}")
            else:
                logger.info("Collecting odds for all upcoming games")
            collect_odds_data(target_date, DEFAULT_DB_PATH)
            logger.info("Odds collection completed successfully")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.error("Make sure to set ODDS_API_KEY in your .env file")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error collecting odds: {e}")
            sys.exit(1)

    elif args.command == "train":
        train_models(
            seasons=args.seasons,
            positions=args.positions,
            tune_lr=args.tune_lr,
            tune_batch_size=args.tune_batch_size,
            tune_all=args.tune_all,
            trials=args.trials,
            epochs=args.epochs,
            override_lr=args.lr,
            override_batch_size=args.batch_size,
            simplified_dst=getattr(args, 'simplified_dst', False),
        )

    elif args.command == "injury":
        manual_updates = {}
        if args.player:
            for name, status in args.player:
                manual_updates[name] = status

        count = update_injury_statuses(
            injury_file=args.csv,
            manual_updates=manual_updates if manual_updates else None,
        )
        logger.info(f"Updated injury status for {count} player(s)")

    elif args.command == "weather":
        if args.historical:
            logger.info("Collecting all historical weather data...")
            collect_weather_data_optimized(db_path=DEFAULT_DB_PATH)
        elif args.upcoming:
            logger.info("Collecting weather for upcoming games...")
            # Import the upcoming weather function
            from collect_weather_today import main as collect_upcoming
            collect_upcoming()
        else:
            # Default: collect both
            logger.info("Collecting all weather data (historical + upcoming)...")
            collect_weather_data_optimized(db_path=DEFAULT_DB_PATH)
            from collect_weather_today import main as collect_upcoming
            collect_upcoming()

    elif args.command == "predict":
        predict_players_optimized(args.contest_id, args.output, args.injury_file)

    elif args.command == "optimize":
        optimize_lineups(
            args.contest_id,
            args.strategy,
            args.count,
            args.output_dir,
            args.save_predictions,
            args.injury_file,
        )

    elif args.command == "model-backtest":
        backtest_models(
            seasons=args.seasons,
            positions=args.positions,
            walk_forward_windows=args.windows,
        )

    elif args.command == "backtest":
        run_backtest_command(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
