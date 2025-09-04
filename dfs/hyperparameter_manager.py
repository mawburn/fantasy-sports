"""
Hyperparameter Configuration Manager

Manages loading, saving, and updating hyperparameters from YAML configuration files.
Provides a clean interface between the model training code and hyperparameter storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class HyperparameterManager:
    """Manages hyperparameter configuration using YAML files."""

    def __init__(self, config_path: str = "hyperparameters.yaml"):
        """Initialize hyperparameter manager.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load hyperparameters from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Hyperparameter config not found at {self.config_path}")
            return self._create_default_config()

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded hyperparameters from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load hyperparameters: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration if file doesn't exist."""
        return {
            "defaults": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 500,
                "hidden_size": 256,
                "num_layers": 3,
                "dropout_rate": 0.3,
                "weight_decay": 0.001,
                "patience": 50,
            },
            "positions": {},
        }

    def save_config(self):
        """Save current configuration to YAML file."""
        try:
            # Convert numpy types to Python native types before saving
            import numpy as np

            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                else:
                    return obj

            # Convert config before saving
            clean_config = convert_numpy_types(self.config)

            with open(self.config_path, "w") as f:
                yaml.dump(
                    clean_config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    allow_unicode=True,
                    sort_keys=False,
                )
            logger.info(f"Saved hyperparameters to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save hyperparameters: {e}")

    def get_hyperparameters(self, position: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific position.

        Args:
            position: Position code (QB, RB, WR, TE, DST)

        Returns:
            Dictionary of hyperparameters for the position
        """
        position = position.upper()

        # Start with defaults
        params = self.config.get("defaults", {}).copy()

        # Override with position-specific parameters
        if "positions" in self.config and position in self.config["positions"]:
            position_params = self.config["positions"][position].copy()
            # Remove metadata fields
            metadata_keys = ["last_tuned", "best_validation_r2", "tuning_trials"]
            for key in metadata_keys:
                position_params.pop(key, None)
            params.update(position_params)

        logger.info(
            f"Retrieved hyperparameters for {position}: {len(params)} parameters"
        )
        return params

    def update_hyperparameters(
        self,
        position: str,
        new_params: Dict[str, Any],
        validation_r2: float = None,
        validation_mae: float = None,
        validation_spearman: float = None,
        validation_ndcg: float = None,
        trials: int = None,
    ):
        """Update hyperparameters for a position.

        Args:
            position: Position code (QB, RB, WR, TE, DST)
            new_params: New hyperparameter values
            validation_r2: Best validation R² achieved
            validation_mae: Best validation MAE achieved
            validation_spearman: Best validation Spearman correlation achieved
            validation_ndcg: Best validation NDCG@20 achieved
            trials: Number of trials used in optimization
        """
        position = position.upper()

        # Initialize position if it doesn't exist
        if "positions" not in self.config:
            self.config["positions"] = {}

        if position not in self.config["positions"]:
            self.config["positions"][position] = {}

        # Check if we should update based on position-specific optimization metric
        should_update = True

        # First, check if new metrics meet position-specific quality guardrails
        mae_guardrails = {
            "QB": 8.0,  # QBs have higher variance in fantasy points
            "RB": 6.0,  # RBs have moderate variance
            "WR": 5.0,  # WRs have moderate variance
            "TE": 4.0,  # TEs have lower variance
            "DST": 4.0,  # DST has moderate variance, adjusted from 3.0
            "DEF": 4.0,  # Alternative DST name
        }
        mae_threshold = mae_guardrails.get(position, 6.0)

        if validation_mae is not None and validation_mae >= mae_threshold:
            should_update = False
            logger.warning(
                f"Not updating {position} hyperparameters: "
                f"MAE guardrail failed ({validation_mae:.4f} >= {mae_threshold})"
            )
        elif validation_spearman is not None and validation_spearman <= 0:
            should_update = False
            logger.warning(
                f"Not updating {position} hyperparameters: "
                f"Spearman guardrail failed ({validation_spearman:.4f} <= 0)"
            )
        elif validation_r2 is not None and validation_r2 <= 0:
            should_update = False
            logger.warning(
                f"Not updating {position} hyperparameters: "
                f"R² guardrail failed ({validation_r2:.4f} <= 0)"
            )

        # If guardrails pass, then check metric-specific comparisons
        if should_update:
            # For neural network positions (QB, RB, WR, TE), prioritize NDCG@20 if available
            if position in ["QB", "RB", "WR", "TE"] and validation_ndcg is not None:
                current_best = self.config["positions"][position].get(
                    "best_validation_ndcg_at_k"
                )
                if current_best is not None:
                    try:
                        current_best_float = float(current_best)
                        validation_ndcg_float = float(validation_ndcg)
                        if (
                            validation_ndcg_float < current_best_float
                        ):  # Lower NDCG is worse
                            should_update = False
                            logger.warning(
                                f"Not updating {position} hyperparameters: "
                                f"new NDCG@20 ({validation_ndcg_float:.4f}) is worse than "
                                f"current best ({current_best_float:.4f})"
                            )
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Could not compare NDCG@20 values for {position}, updating anyway"
                        )
                else:
                    # No existing NDCG@20 value, so any valid NDCG@20 is an improvement
                    logger.info(
                        f"No existing NDCG@20 for {position}, updating with new NDCG@20: {validation_ndcg:.4f}"
                    )
            # For DST, prioritize Spearman correlation if available
            elif position == "DST" and validation_spearman is not None:
                current_best = self.config["positions"][position].get(
                    "best_validation_spearman"
                )
                if current_best is not None:
                    try:
                        current_best_float = float(current_best)
                        validation_spearman_float = float(validation_spearman)
                        if (
                            validation_spearman_float < current_best_float
                        ):  # Lower Spearman is worse
                            should_update = False
                            logger.warning(
                                f"Not updating {position} hyperparameters: "
                                f"new Spearman ({validation_spearman_float:.4f}) is worse than "
                                f"current best ({current_best_float:.4f})"
                            )
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Could not compare Spearman values for {position}, updating anyway"
                        )
            elif validation_mae is not None:
                current_best = self.config["positions"][position].get(
                    "best_validation_mae"
                )
                if current_best is not None:
                    try:
                        current_best_float = float(current_best)
                        validation_mae_float = float(validation_mae)
                        if (
                            validation_mae_float > current_best_float
                        ):  # Higher MAE is worse
                            should_update = False
                            logger.warning(
                                f"Not updating {position} hyperparameters: "
                                f"new MAE ({validation_mae_float:.4f}) is worse than "
                                f"current best ({current_best_float:.4f})"
                            )
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Could not compare MAE values for {position}, updating anyway"
                        )
            elif validation_r2 is not None:
                # Fallback to R² comparison if MAE not available
                current_best = self.config["positions"][position].get(
                    "best_validation_r2"
                )
                if current_best is not None:
                    try:
                        current_best_float = float(current_best)
                        validation_r2_float = float(validation_r2)
                        if validation_r2_float < current_best_float:
                            should_update = False
                            logger.warning(
                                f"Not updating {position} hyperparameters: "
                                f"new R² ({validation_r2_float:.4f}) is worse than "
                                f"current best ({current_best_float:.4f})"
                            )
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Could not compare R² values for {position}, updating anyway"
                        )

        if should_update:
            # Update hyperparameters
            self.config["positions"][position].update(new_params)

            # Update metadata
            self.config["positions"][position]["last_tuned"] = (
                datetime.now().isoformat()
            )

            # Store all metrics as floats to avoid numpy serialization issues
            if validation_r2 is not None:
                self.config["positions"][position]["best_validation_r2"] = float(
                    validation_r2
                )
            if validation_mae is not None:
                self.config["positions"][position]["best_validation_mae"] = float(
                    validation_mae
                )
            if validation_spearman is not None:
                self.config["positions"][position]["best_validation_spearman"] = float(
                    validation_spearman
                )
            if validation_ndcg is not None:
                self.config["positions"][position]["best_validation_ndcg_at_k"] = float(
                    validation_ndcg
                )
            if trials is not None:
                self.config["positions"][position]["tuning_trials"] = trials

            logger.info(
                f"Updated hyperparameters for {position} with {len(new_params)} parameters"
            )

            # Log all available metrics
            metrics_log = []
            if validation_mae is not None:
                metrics_log.append(f"MAE: {validation_mae:.4f}")
            if validation_spearman is not None:
                metrics_log.append(f"Spearman: {validation_spearman:.4f}")
            if validation_r2 is not None:
                metrics_log.append(f"R²: {validation_r2:.4f}")
            if validation_ndcg is not None:
                metrics_log.append(f"NDCG@20: {validation_ndcg:.4f}")

            if metrics_log:
                logger.info(f"Best metrics for {position}: {', '.join(metrics_log)}")

            # Save to file
            self.save_config()

        return should_update

    def get_search_ranges(self) -> Dict[str, Any]:
        """Get hyperparameter search ranges for optimization.

        Returns:
            Dictionary of search ranges for Optuna
        """
        return self.config.get("search_ranges", {})

    def get_position_history(self, position: str) -> Dict[str, Any]:
        """Get tuning history for a position.

        Args:
            position: Position code (QB, RB, WR, TE, DST)

        Returns:
            Dictionary with tuning metadata
        """
        position = position.upper()

        if "positions" in self.config and position in self.config["positions"]:
            pos_config = self.config["positions"][position]
            return {
                "last_tuned": pos_config.get("last_tuned"),
                "best_validation_r2": pos_config.get("best_validation_r2"),
                "best_validation_mae": pos_config.get("best_validation_mae"),
                "best_validation_spearman": pos_config.get("best_validation_spearman"),
                "tuning_trials": pos_config.get("tuning_trials", 0),
            }

        return {
            "last_tuned": None,
            "best_validation_r2": None,
            "best_validation_mae": None,
            "best_validation_spearman": None,
            "tuning_trials": 0,
        }

    def list_positions(self) -> list:
        """Get list of positions with saved hyperparameters."""
        if "positions" not in self.config:
            return []
        return list(self.config["positions"].keys())

    def reset_position(self, position: str):
        """Reset a position to default hyperparameters.

        Args:
            position: Position code (QB, RB, WR, TE, DST)
        """
        position = position.upper()

        if "positions" in self.config and position in self.config["positions"]:
            del self.config["positions"][position]
            self.save_config()
            logger.info(f"Reset hyperparameters for {position} to defaults")


# Global instance for easy access
_global_manager = None


def get_hyperparameter_manager(
    config_path: str = "hyperparameters.yaml",
) -> HyperparameterManager:
    """Get global hyperparameter manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = HyperparameterManager(config_path)
    return _global_manager
