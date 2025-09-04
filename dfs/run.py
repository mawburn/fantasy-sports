#!/usr/bin/env python3
"""
Optimized CLI for DFS optimization system with faster predictions.
"""

import argparse
import logging
import sys
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

            logger.info(
                f"Training {position} with {len(X)} samples, {len(feature_names)} features"
            )

            # Create and train model (use ensemble for QB and RB)
            config = ModelConfig(position=position, features=feature_names)
            use_ensemble = (
                position in ["QB", "WR", "TE"]
            )  # Enable ensemble for QB, WR, TE (RB neural-only, DST CatBoost-only perform better)
            model = create_model(position, config, use_ensemble=use_ensemble)

            # Split data (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Apply hyperparameter tuning if requested
            if tune_all:
                logger.info(
                    f"Running full hyperparameter optimization for {position} ({trials} trials)..."
                )
                best_params = model.tune_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=trials, epochs=epochs
                )
                logger.info(f"Best hyperparameters for {position}: {best_params}")
                logger.info(
                    f"Hyperparameter tuning complete for {position}. Run training without --tune-all to train with optimized hyperparameters."
                )
                continue  # Skip training and saving when tuning
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

            # Save model
            model_path = models_dir / f"{position.lower()}_model.pth"
            model.save_model(str(model_path))

            logger.info(
                f"{position} model saved. Val MAE: {result.val_mae:.2f}, Val R²: {result.val_r2:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to train {position} model: {e}")
            import traceback

            logger.debug(traceback.format_exc())


def update_injury_statuses(
    injury_file: str = None,
    manual_updates: Dict[str, str] = None,
    db_path: str = DEFAULT_DB_PATH,
):
    """Update player injury statuses from CSV file or manual input.

    Args:
        injury_file: Path to CSV file with columns: player_name, injury_status
        manual_updates: Dictionary of player_name -> injury_status
        db_path: Database path

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

    try:
        # Clear existing injury statuses
        conn.execute("UPDATE players SET injury_status = NULL")

        # Update from CSV file if provided
        if injury_file and Path(injury_file).exists():
            import pandas as pd

            df = pd.read_csv(injury_file)
            for _, row in df.iterrows():
                player_name = row["player_name"]
                injury_status = row["injury_status"].upper()

                cursor = conn.execute(
                    "UPDATE players SET injury_status = ? WHERE display_name = ? OR player_name = ?",
                    (injury_status, player_name, player_name),
                )
                updated_count += cursor.rowcount

        # Update from manual dictionary if provided
        if manual_updates:
            for player_name, injury_status in manual_updates.items():
                cursor = conn.execute(
                    "UPDATE players SET injury_status = ? WHERE display_name = ? OR player_name = ?",
                    (injury_status.upper(), player_name, player_name),
                )
                updated_count += cursor.rowcount

        conn.commit()
        logger.info(f"Updated injury status for {updated_count} players")

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
        model_path = models_dir / f"{position.lower()}_model.pth"
        if model_path.exists():
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
    injury_status_query = f"""
        SELECT id, injury_status
        FROM players
        WHERE id IN ({placeholders})
    """

    injury_statuses = {}
    for row in conn.execute(injury_status_query, player_ids).fetchall():
        if row[1]:  # Only store if injury status is not NULL
            injury_statuses[row[0]] = row[1]

    if injury_statuses:
        logger.info(f"Found injury statuses for {len(injury_statuses)} players")

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
        if position in models and position in position_feature_names:
            model = models[position]
            feature_names = position_feature_names[position]

            if len(feature_names) > 0:
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

                            # Get injury multipliers
                            proj_mult, floor_mult, ceil_mult = get_injury_multiplier(
                                injury_status
                            )

                            # Apply injury multipliers to predictions
                            proj_points = (
                                float(prediction_result.point_estimate[i]) * proj_mult
                            )
                            floor_points = (
                                float(prediction_result.floor[i]) * floor_mult
                            )
                            ceiling_points = (
                                float(prediction_result.ceiling[i]) * ceil_mult
                            )

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
                        generate_fallback_prediction(player_data, injury_status)
                    )
            else:
                # No features available, use fallback for all
                for player_data in position_players:
                    injury_status = injury_statuses.get(player_data["player_id"])
                    predictions.append(
                        generate_fallback_prediction(player_data, injury_status)
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
                            generate_fallback_prediction(player_data, injury_status)
                        )
                else:
                    injury_status = injury_statuses.get(player_data["player_id"])
                    predictions.append(
                        generate_fallback_prediction(player_data, injury_status)
                    )
        else:
            # Fallback for positions without models
            for player_data in position_players:
                injury_status = injury_statuses.get(player_data["player_id"])
                predictions.append(
                    generate_fallback_prediction(player_data, injury_status)
                )

    # Save predictions if requested
    if output_file:
        import pandas as pd

        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

    logger.info(f"Generated predictions for {len(predictions)} players")
    return predictions


def generate_fallback_prediction(player_data, injury_status=None):
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

    projected_points = base_projection

    # Apply injury multipliers
    proj_mult, floor_mult, ceil_mult = get_injury_multiplier(injury_status)
    projected_points *= proj_mult
    floor = projected_points * 0.7 * floor_mult
    ceiling = projected_points * 1.5 * ceil_mult

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

    elif args.command == "backtest":
        run_backtest_command(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
