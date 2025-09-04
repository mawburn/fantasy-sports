"""
Simplified DST Model for DFS
Focuses on the most predictive features: Vegas lines, recent form, and matchups
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class SimpleDSTConfig:
    """Configuration for simplified DST model."""
    learning_rate: float = 0.05
    n_estimators: int = 100
    max_depth: int = 3
    min_samples_split: int = 20
    subsample: float = 0.8


class SimplifiedDSTModel:
    """
    Simplified DST model focusing on proven predictive features:
    1. Vegas opponent implied total (most predictive)
    2. Recent defensive performance (3-game average)
    3. Home/away splits
    4. Spread (are they favored?)
    5. Weather conditions
    """

    def __init__(self, config: Optional[SimpleDSTConfig] = None):
        self.config = config or SimpleDSTConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def extract_key_features(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        """Extract only the most predictive features for DST."""

        # Define the key features we want
        key_features = [
            # Vegas lines (MOST IMPORTANT)
            'opp_implied_total',         # Lower is better for DST
            'game_total_ou',              # Lower total = defensive game
            'spread_signed',              # Negative = team is favored
            'spread_magnitude',           # How much they're favored by
            'is_favorite',                # Binary favored indicator

            # Recent form (3-game averages)
            'def_fantasy_points_l3',     # Recent DST fantasy performance
            'def_points_allowed_l3',     # Recent points allowed
            'def_turnovers_per_game_l3', # Recent turnover generation
            'def_sacks_per_game_l3',     # Recent pass rush success

            # Opponent vulnerability
            'opp_turnover_rate_l3',      # How often opponent turns it over
            'opp_sack_rate_l3',          # How often opponent gets sacked
            'opp_scoring_rate_l3',       # Opponent scoring efficiency

            # Situational
            'is_home',                   # Home field advantage
            'week',                      # Week of season (fatigue/weather)

            # Weather (for outdoor games)
            'wind_speed',                # High wind = more turnovers
            'precipitation',             # Rain/snow = more turnovers
            'temperature',               # Extreme temps affect scoring
        ]

        # Get indices of key features
        feature_indices = []
        features_found = []
        for feat in key_features:
            if feat in feature_names:
                idx = feature_names.index(feat)
                feature_indices.append(idx)
                features_found.append(feat)
            else:
                logger.debug(f"Feature {feat} not found in input features")

        if not feature_indices:
            logger.warning("No key features found, using all features")
            self.feature_names = feature_names
            return X

        logger.info(f"Using {len(features_found)} key DST features: {features_found[:5]}...")
        self.feature_names = features_found

        # Extract only the key features
        return X[:, feature_indices]

    def create_composite_features(self, X: np.ndarray) -> np.ndarray:
        """Create composite features that capture DST value."""

        X_composite = X.copy()

        # Find feature indices (if they exist)
        def get_feat_idx(name):
            try:
                return self.feature_names.index(name)
            except ValueError:
                return None

        # Create composite features if base features exist
        opp_total_idx = get_feat_idx('opp_implied_total')
        spread_idx = get_feat_idx('spread_signed')

        if opp_total_idx is not None and spread_idx is not None:
            # Game script score: Low opponent total + big favorite = good for DST
            game_script_score = (25 - X[:, opp_total_idx]) + (-X[:, spread_idx])
            X_composite = np.column_stack([X_composite, game_script_score.reshape(-1, 1)])
            self.feature_names.append('game_script_score')

        # Turnover opportunity score
        turnover_idx = get_feat_idx('opp_turnover_rate_l3')
        wind_idx = get_feat_idx('wind_speed')

        if turnover_idx is not None:
            turnover_opp = X[:, turnover_idx]
            if wind_idx is not None:
                # High wind increases turnover chances
                turnover_opp = turnover_opp * (1 + X[:, wind_idx] / 20)
            X_composite = np.column_stack([X_composite, turnover_opp.reshape(-1, 1)])
            self.feature_names.append('turnover_opportunity')

        return X_composite

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: list) -> Dict:
        """Train simplified DST model."""

        logger.info("Training Simplified DST Model focused on Vegas lines and matchups")

        # Extract key features
        X_key = self.extract_key_features(X, feature_names)
        X_val_key = self.extract_key_features(X_val, feature_names)

        # Create composite features
        X_enhanced = self.create_composite_features(X_key)
        X_val_enhanced = self.create_composite_features(X_val_key)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)
        X_val_scaled = self.scaler.transform(X_val_enhanced)

        # Train gradient boosting model (more stable than neural nets for small data)
        self.model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            subsample=self.config.subsample,
            random_state=42,
            loss='huber',  # Robust to outliers
            alpha=0.9,     # Huber loss parameter
        )

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_scaled)
        val_pred = self.model.predict(X_val_scaled)

        train_mae = np.mean(np.abs(y - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))

        # Calculate Spearman (most important for DST)
        from scipy.stats import spearmanr
        train_spearman, _ = spearmanr(y, train_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)

        # Feature importance
        importance = self.model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(f"Simplified DST Model - Val MAE: {val_mae:.3f}, Val Spearman: {val_spearman:.3f}")
        logger.info(f"Top 5 features: {feature_importance[:5]}")

        return {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_spearman': train_spearman,
            'val_spearman': val_spearman,
            'feature_importance': feature_importance
        }

    def predict(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        """Generate predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract same features used in training
        X_key = self.extract_key_features(X, feature_names)
        X_enhanced = self.create_composite_features(X_key)
        X_scaled = self.scaler.transform(X_enhanced)

        predictions = self.model.predict(X_scaled)

        # DST scoring range is typically -4 to 24
        # Clip predictions to reasonable range
        return np.clip(predictions, -4, 24)

    def save_model(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
        logger.info(f"Saved simplified DST model to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        logger.info(f"Loaded simplified DST model from {path}")

    def load(self, path: str):
        """Alias for load_model to match interface."""
        self.load_model(path)

    def predict_single(self, player_data: dict) -> float:
        """Predict for a single DST given player data dict.

        This method extracts features from player_data dict and generates prediction.
        Used during prediction phase when we don't have pre-extracted features.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Get team abbreviation from player data (DST name format: "Team DST")
            team_name = player_data.get('name', '').replace(' DST', '')
            team_abbr = player_data.get('team_abbr', player_data.get('team', ''))

            # Get Vegas data and recent performance from database
            from data import get_db_connection
            conn = get_db_connection('data/nfl_dfs.db')

            features = {}

            # Get Vegas data for upcoming game
            vegas_query = """
                SELECT bo.over_under_line, bo.home_team_spread, bo.away_team_spread,
                       CASE WHEN g.home_team_id = t.id THEN 1 ELSE 0 END as is_home,
                       CASE WHEN g.home_team_id = t.id THEN at.team_abbr ELSE ht.team_abbr END as opponent
                FROM teams t
                JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id)
                LEFT JOIN teams ht ON g.home_team_id = ht.id
                LEFT JOIN teams at ON g.away_team_id = at.id
                LEFT JOIN betting_odds bo ON g.id = bo.game_id
                WHERE t.team_abbr = ?
                AND g.game_finished = 0
                ORDER BY g.game_date ASC
                LIMIT 1
            """

            vegas_data = conn.execute(vegas_query, (team_abbr,)).fetchone()

            if vegas_data:
                game_total = vegas_data[0] or 45.0
                home_spread = vegas_data[1] or 0.0
                away_spread = vegas_data[2] or 0.0
                is_home = vegas_data[3]

                # Calculate team spread and opponent implied total
                team_spread = home_spread if is_home else away_spread
                opp_implied_total = (game_total / 2.0) - (team_spread / 2.0)

                features['opp_implied_total'] = opp_implied_total
                features['game_total_ou'] = game_total
                features['spread_signed'] = team_spread
                features['is_home'] = float(is_home)
            else:
                # Use defaults if no Vegas data found
                features['opp_implied_total'] = 22.5  # NFL avg
                features['game_total_ou'] = 45.0  # NFL avg
                features['spread_signed'] = 0.0
                features['is_home'] = 0.0

            # Get recent defensive performance
            recent_query = """
                SELECT AVG(ps.fantasy_points) as avg_fp
                FROM player_stats ps
                JOIN games g ON ps.game_id = g.id
                WHERE ps.player_id IN (
                    SELECT id FROM players
                    WHERE display_name LIKE ?
                    AND position = 'DST'
                )
                AND g.game_finished = 1
                ORDER BY g.game_date DESC
                LIMIT 3
            """

            recent_data = conn.execute(recent_query, (f'%{team_name}%',)).fetchone()
            features['def_fantasy_points_l3'] = recent_data[0] if recent_data and recent_data[0] else 7.0

            conn.close()

            # Create feature array in the same order as training
            key_feature_names = [
                'opp_implied_total',
                'game_total_ou',
                'spread_signed',
                'is_home',
                'def_fantasy_points_l3'
            ]

            X_key = np.array([[features.get(name, 0.0) for name in key_feature_names]])

            # Create composite features
            X_enhanced = self.create_composite_features(X_key)

            # Scale features
            X_scaled = self.scaler.transform(X_enhanced)

            # Generate prediction
            pred = self.model.predict(X_scaled)[0]

            # Clip to reasonable DST range
            return np.clip(pred, -4, 24)

        except Exception as e:
            logger.warning(f"Failed to predict for DST {player_data.get('name', 'unknown')}: {e}")
            # Return a conservative default prediction
            return 7.0  # Average DST score


def train_simplified_dst(X_train, y_train, X_val, y_val, feature_names):
    """Convenience function to train simplified DST model."""

    model = SimplifiedDSTModel()
    results = model.train(X_train, y_train, X_val, y_val, feature_names)

    return model, results
