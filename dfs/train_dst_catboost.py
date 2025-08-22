#!/usr/bin/env python3
"""
Enhanced DST Model Training with CatBoost
Implements component models and advanced feature engineering
"""

import json
import logging
import sqlite3
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DSTModelPipeline:
    """Complete pipeline for DST prediction using component models."""

    def __init__(self, db_path: str = "data/nfl_dfs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.models = {}
        self.feature_means = {}
        self.feature_stds = {}

    def create_enhanced_features(self) -> pd.DataFrame:
        """Create comprehensive feature set for DST modeling."""

        query = """
        WITH vegas_features AS (
            SELECT
                game_id,
                favorite_team,
                spread_favorite,
                over_under_line as game_total,
                -- Calculate implied totals
                (over_under_line / 2.0) - (ABS(spread_favorite) / 2.0) as favorite_implied,
                (over_under_line / 2.0) + (ABS(spread_favorite) / 2.0) as underdog_implied
            FROM betting_odds
        ),
        game_info AS (
            SELECT
                game_id,
                home_team,
                away_team,
                season,
                week
            FROM games
        ),
        rolling_defense AS (
            SELECT
                ds.*,
                -- Rolling 5-game averages (excluding current game)
                AVG(sacks) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as sacks_l5,
                AVG(interceptions) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as int_l5,
                AVG(fumbles_recovered) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as fr_l5,
                AVG(points_allowed) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as pa_l5,
                AVG(defensive_tds + return_tds + special_teams_tds) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as td_l5,
                AVG(fantasy_points) OVER (
                    PARTITION BY team_abbr
                    ORDER BY season, week
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as fp_l5,
                -- Season averages
                AVG(sacks) OVER (
                    PARTITION BY team_abbr, season
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as sacks_season,
                AVG(fantasy_points) OVER (
                    PARTITION BY team_abbr, season
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as fp_season
            FROM dst_stats ds
        ),
        weather_data AS (
            SELECT
                game_id,
                temp,
                wind,
                CASE WHEN precip > 0 THEN 1 ELSE 0 END as has_precip,
                CASE WHEN roof = 'dome' OR roof = 'closed' THEN 1 ELSE 0 END as is_dome
            FROM weather
        )
        SELECT
            rd.*,
            gi.home_team,
            gi.away_team,
            CASE WHEN rd.team_abbr = gi.home_team THEN 1 ELSE 0 END as is_home,
            vf.game_total,
            vf.spread_favorite,
            -- Determine if team is favorite and opponent implied total
            CASE
                WHEN vf.favorite_team = rd.team_abbr THEN 1
                ELSE 0
            END as is_favorite,
            CASE
                WHEN vf.favorite_team = rd.team_abbr THEN vf.underdog_implied
                ELSE vf.favorite_implied
            END as opponent_implied_total,
            -- Spread from team perspective
            CASE
                WHEN vf.favorite_team = rd.team_abbr THEN -ABS(vf.spread_favorite)
                ELSE ABS(vf.spread_favorite)
            END as team_spread,
            -- Weather features
            COALESCE(wd.temp, 70) as temperature,
            COALESCE(wd.wind, 5) as wind_speed,
            COALESCE(wd.has_precip, 0) as has_precipitation,
            COALESCE(wd.is_dome, 0) as is_dome,
            -- Create target components for modeling
            rd.sacks as target_sacks,
            rd.interceptions as target_int,
            rd.fumbles_recovered as target_fr,
            rd.points_allowed as target_pa,
            CASE WHEN rd.defensive_tds + rd.return_tds + rd.special_teams_tds > 0 THEN 1 ELSE 0 END as target_has_td,
            rd.fantasy_points as target_fantasy_points
        FROM rolling_defense rd
        JOIN game_info gi ON rd.game_id = gi.game_id
        LEFT JOIN vegas_features vf ON rd.game_id = vf.game_id
        LEFT JOIN weather_data wd ON rd.game_id = wd.game_id
        WHERE rd.sacks_l5 IS NOT NULL  -- Ensure we have rolling history
        AND vf.game_total IS NOT NULL  -- Ensure we have Vegas data
        ORDER BY rd.season, rd.week
        """

        df = pd.read_sql(query, self.conn)

        # Add derived features
        df["pressure_expectation"] = df["sacks_l5"] * (
            1 + df["team_spread"] / 50
        )  # More pressure when favored
        df["turnover_expectation"] = (df["int_l5"] + df["fr_l5"]) * (
            1 + df["wind_speed"] / 20
        )  # Wind increases TOs
        df["scoring_defense_strength"] = 20 - df["pa_l5"]  # Inverse of points allowed
        df["game_script_advantage"] = (
            df["is_favorite"] * df["is_home"] * (1 + abs(df["team_spread"]) / 10)
        )

        # Create points allowed buckets for classification
        def get_pa_bucket(pa):
            if pa == 0:
                return 0
            elif pa <= 6:
                return 1
            elif pa <= 13:
                return 2
            elif pa <= 20:
                return 3
            elif pa <= 27:
                return 4
            elif pa <= 34:
                return 5
            else:
                return 6

        df["target_pa_bucket"] = df["target_pa"].apply(get_pa_bucket)

        logger.info(
            f"Created feature dataset with {len(df)} samples and {len(df.columns)} columns"
        )
        logger.info(
            f"Fantasy points range: [{df['target_fantasy_points'].min():.1f}, {df['target_fantasy_points'].max():.1f}]"
        )
        logger.info(
            f"Average fantasy points: {df['target_fantasy_points'].mean():.2f} ± {df['target_fantasy_points'].std():.2f}"
        )

        return df

    def prepare_train_val_split(
        self, df: pd.DataFrame, val_season: int = 2023, val_week_start: int = 10
    ) -> Tuple:
        """Create time-based train/validation split."""

        # Training data: everything before validation period
        train_mask = (df["season"] < val_season) | (
            (df["season"] == val_season) & (df["week"] < val_week_start)
        )
        val_mask = ~train_mask

        feature_cols = [
            "opponent_implied_total",
            "game_total",
            "team_spread",
            "is_home",
            "is_favorite",
            "sacks_l5",
            "int_l5",
            "fr_l5",
            "pa_l5",
            "td_l5",
            "fp_l5",
            "sacks_season",
            "fp_season",
            "temperature",
            "wind_speed",
            "has_precipitation",
            "is_dome",
            "pressure_expectation",
            "turnover_expectation",
            "scoring_defense_strength",
            "game_script_advantage",
        ]

        X_train = df[train_mask][feature_cols].values
        X_val = df[val_mask][feature_cols].values

        # Store feature names for later use
        self.feature_names = feature_cols

        # Prepare all targets
        targets = {
            "fantasy_points": (
                df[train_mask]["target_fantasy_points"].values,
                df[val_mask]["target_fantasy_points"].values,
            ),
            "sacks": (
                df[train_mask]["target_sacks"].values,
                df[val_mask]["target_sacks"].values,
            ),
            "turnovers": (
                df[train_mask]["target_int"].values
                + df[train_mask]["target_fr"].values,
                df[val_mask]["target_int"].values + df[val_mask]["target_fr"].values,
            ),
            "pa_bucket": (
                df[train_mask]["target_pa_bucket"].values,
                df[val_mask]["target_pa_bucket"].values,
            ),
            "has_td": (
                df[train_mask]["target_has_td"].values,
                df[val_mask]["target_has_td"].values,
            ),
        }

        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

        return X_train, X_val, targets

    def train_direct_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> CatBoostRegressor:
        """Train direct fantasy points prediction model."""

        logger.info("Training direct DST fantasy points model...")

        model = CatBoostRegressor(
            iterations=4000,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=6,
            loss_function="MAE",
            eval_metric="MAE",
            use_best_model=True,
            verbose=200,
            random_seed=42,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=300,
            verbose=True,
        )

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_corr = spearmanr(y_val, val_pred)[0]

        logger.info(
            f"Direct Model - Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}"
        )
        logger.info(f"Direct Model - Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}")
        logger.info(f"Direct Model - Val Spearman: {val_corr:.3f}")
        logger.info(
            f"Direct Model - Prediction range: [{val_pred.min():.1f}, {val_pred.max():.1f}]"
        )

        return model

    def train_component_models(
        self, X_train: np.ndarray, X_val: np.ndarray, targets: Dict
    ) -> Dict[str, Any]:
        """Train individual component models."""

        models = {}

        # 1. Sacks Model (Poisson)
        logger.info("Training sacks component model...")
        models["sacks"] = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=6,
            loss_function="Poisson",
            eval_metric="Poisson",
            use_best_model=True,
            verbose=False,
            random_seed=42,
        )
        models["sacks"].fit(
            X_train,
            targets["sacks"][0],
            eval_set=(X_val, targets["sacks"][1]),
            early_stopping_rounds=200,
            verbose=False,
        )

        # 2. Turnovers Model (Poisson)
        logger.info("Training turnovers component model...")
        models["turnovers"] = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=6,
            loss_function="Poisson",
            eval_metric="Poisson",
            use_best_model=True,
            verbose=False,
            random_seed=42,
        )
        models["turnovers"].fit(
            X_train,
            targets["turnovers"][0],
            eval_set=(X_val, targets["turnovers"][1]),
            early_stopping_rounds=200,
            verbose=False,
        )

        # 3. Points Allowed Bucket Model (Multiclass)
        logger.info("Training points allowed bucket model...")
        models["pa_bucket"] = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            use_best_model=True,
            verbose=False,
            random_seed=42,
        )
        models["pa_bucket"].fit(
            X_train,
            targets["pa_bucket"][0],
            eval_set=(X_val, targets["pa_bucket"][1]),
            early_stopping_rounds=200,
            verbose=False,
        )

        # 4. TD Probability Model (Binary)
        logger.info("Training TD probability model...")
        # Calculate class weights for imbalanced TD data
        td_rate = targets["has_td"][0].mean()
        class_weights = {0: 1.0, 1: 1.0 / td_rate if td_rate > 0 else 1.0}

        models["td_prob"] = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            use_best_model=True,
            class_weights=class_weights,
            verbose=False,
            random_seed=42,
        )
        models["td_prob"].fit(
            X_train,
            targets["has_td"][0],
            eval_set=(X_val, targets["has_td"][1]),
            early_stopping_rounds=200,
            verbose=False,
        )

        return models

    def combine_component_predictions(self, models: Dict, X: np.ndarray) -> np.ndarray:
        """Combine component model predictions into fantasy points."""

        # Get predictions from each component
        sacks_pred = models["sacks"].predict(X)
        turnovers_pred = models["turnovers"].predict(X)
        pa_bucket_pred = models["pa_bucket"].predict(X)
        td_prob = models["td_prob"].predict_proba(X)[:, 1]

        # PA bucket to points mapping (DraftKings scoring)
        pa_points_map = {0: 10, 1: 7, 2: 4, 3: 1, 4: 0, 5: -1, 6: -4}
        pa_points = np.array([pa_points_map[int(bucket)] for bucket in pa_bucket_pred])

        # Combine using DraftKings scoring formula
        fantasy_points = (
            sacks_pred * 1.0  # 1 point per sack
            + turnovers_pred * 2.0  # 2 points per TO (INT + FR)
            + pa_points  # Points based on PA bucket
            + td_prob * 6.0  # Expected value of TD (6 points)
        )

        return fantasy_points

    def evaluate_models(
        self,
        direct_model: Any,
        component_models: Dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Evaluate both direct and component models."""

        # Direct model predictions
        direct_preds = direct_model.predict(X_val)

        # Component model predictions
        component_preds = self.combine_component_predictions(component_models, X_val)

        # Ensemble (simple average for now)
        ensemble_preds = (direct_preds + component_preds) / 2

        results = {
            "direct": {
                "mae": mean_absolute_error(y_val, direct_preds),
                "r2": r2_score(y_val, direct_preds),
                "spearman": spearmanr(y_val, direct_preds)[0],
                "pred_range": (direct_preds.min(), direct_preds.max()),
                "pred_std": direct_preds.std(),
            },
            "component": {
                "mae": mean_absolute_error(y_val, component_preds),
                "r2": r2_score(y_val, component_preds),
                "spearman": spearmanr(y_val, component_preds)[0],
                "pred_range": (component_preds.min(), component_preds.max()),
                "pred_std": component_preds.std(),
            },
            "ensemble": {
                "mae": mean_absolute_error(y_val, ensemble_preds),
                "r2": r2_score(y_val, ensemble_preds),
                "spearman": spearmanr(y_val, ensemble_preds)[0],
                "pred_range": (ensemble_preds.min(), ensemble_preds.max()),
                "pred_std": ensemble_preds.std(),
            },
        }

        return results

    def save_models(self, direct_model: Any, component_models: Dict):
        """Save trained models to disk."""

        # Save CatBoost models
        direct_model.save_model("models/dst_catboost_direct.cbm")
        component_models["sacks"].save_model("models/dst_catboost_sacks.cbm")
        component_models["turnovers"].save_model("models/dst_catboost_turnovers.cbm")
        component_models["pa_bucket"].save_model("models/dst_catboost_pa_bucket.cbm")
        component_models["td_prob"].save_model("models/dst_catboost_td_prob.cbm")

        # Save feature names and other metadata
        metadata = {
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
        }

        with open("models/dst_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Models saved successfully!")

    def run_full_pipeline(self):
        """Execute the complete training pipeline."""

        logger.info("=" * 60)
        logger.info("Starting DST Model Training Pipeline")
        logger.info("=" * 60)

        # Step 1: Create features
        df = self.create_enhanced_features()

        # Step 2: Prepare train/validation split
        X_train, X_val, targets = self.prepare_train_val_split(df)

        # Normalize features
        self.feature_means = X_train.mean(axis=0)
        self.feature_stds = X_train.std(axis=0)
        self.feature_stds[self.feature_stds == 0] = 1  # Prevent division by zero

        X_train_norm = (X_train - self.feature_means) / self.feature_stds
        X_val_norm = (X_val - self.feature_means) / self.feature_stds

        # Step 3: Train direct model
        direct_model = self.train_direct_model(
            X_train_norm,
            X_val_norm,
            targets["fantasy_points"][0],
            targets["fantasy_points"][1],
        )

        # Step 4: Train component models
        component_models = self.train_component_models(
            X_train_norm, X_val_norm, targets
        )

        # Step 5: Evaluate all models
        results = self.evaluate_models(
            direct_model, component_models, X_val_norm, targets["fantasy_points"][1]
        )

        # Step 6: Print results summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 60)

        for model_type, metrics in results.items():
            logger.info(f"\n{model_type.upper()} Model:")
            logger.info(f"  MAE: {metrics['mae']:.3f}")
            logger.info(f"  R²: {metrics['r2']:.3f}")
            logger.info(f"  Spearman: {metrics['spearman']:.3f}")
            logger.info(
                f"  Prediction Range: [{metrics['pred_range'][0]:.1f}, {metrics['pred_range'][1]:.1f}]"
            )
            logger.info(f"  Prediction Std: {metrics['pred_std']:.2f}")

        # Step 7: Save models
        self.save_models(direct_model, component_models)

        # Select best model for production
        best_model = min(results.keys(), key=lambda x: results[x]["mae"])
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"BEST MODEL: {best_model.upper()} (MAE: {results[best_model]['mae']:.3f})"
        )
        logger.info(f"{'=' * 60}")

        return direct_model, component_models, results


def main():
    """Main execution function."""
    pipeline = DSTModelPipeline()
    direct_model, component_models, results = pipeline.run_full_pipeline()

    # Compare with current model performance
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVEMENT OVER CURRENT MODEL")
    logger.info("=" * 60)
    logger.info("Current Model: R² = 0.008, MAE = 3.905")

    best_result = min(results.values(), key=lambda x: x["mae"])
    logger.info(
        f"New Best Model: R² = {best_result['r2']:.3f}, MAE = {best_result['mae']:.3f}"
    )
    logger.info(
        f"R² Improvement: {(best_result['r2'] - 0.008):.3f} ({(best_result['r2'] / 0.008 - 1) * 100:.0f}% better)"
    )
    logger.info(
        f"MAE Improvement: {(3.905 - best_result['mae']):.3f} ({(1 - best_result['mae'] / 3.905) * 100:.1f}% better)"
    )


if __name__ == "__main__":
    main()
