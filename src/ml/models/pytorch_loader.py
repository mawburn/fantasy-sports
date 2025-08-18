"""PyTorch model loader for the API."""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import joblib
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PyTorchModelWrapper:
    """Wrapper to make PyTorch models compatible with the API."""
    
    def __init__(self, position: str, model_path: Path, preprocessor_path: Optional[Path] = None):
        """Initialize the PyTorch model wrapper."""
        self.position = position.upper()
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.network = None
        self.preprocessor = None
        self.state_dict = None
        self.is_trained = False
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the PyTorch model state_dict and instantiate the network."""
        try:
            # Load the state_dict
            self.state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Try to load preprocessor first to get input dimensions
            if self.preprocessor_path and self.preprocessor_path.exists():
                try:
                    self.preprocessor = joblib.load(self.preprocessor_path)
                    logger.info(f"Loaded preprocessor for {self.position}")
                except Exception as e:
                    logger.warning(f"Could not load preprocessor: {e}")
            
            # Instantiate the correct neural network architecture
            self._instantiate_network()
            
            if self.network is not None:
                # Load the state dict into the network
                self.network.load_state_dict(self.state_dict)
                self.network.eval()  # Set to evaluation mode
                self.is_trained = True
                logger.info(f"Loaded PyTorch model for {self.position} from {self.model_path}")
            else:
                logger.error(f"Failed to instantiate network for {self.position}")
                
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _instantiate_network(self):
        """Instantiate the correct network architecture based on position."""
        from .neural_models import QBNetwork, RBNetwork, WRNetwork, TENetwork, DEFNetwork
        
        # Estimate input size from state_dict or use default
        input_size = self._estimate_input_size()
        
        network_classes = {
            'QB': QBNetwork,
            'RB': RBNetwork, 
            'WR': WRNetwork,
            'TE': TENetwork,
            'DEF': DEFNetwork,
            'DST': DEFNetwork  # DST is same as DEF
        }
        
        network_class = network_classes.get(self.position)
        if network_class:
            try:
                self.network = network_class(input_size)
                logger.info(f"Instantiated {network_class.__name__} with input_size={input_size}")
            except Exception as e:
                logger.error(f"Failed to instantiate {network_class.__name__}: {e}")
        else:
            logger.error(f"Unknown position: {self.position}")
    
    def _estimate_input_size(self) -> int:
        """Estimate input size from state_dict or preprocessor."""
        if self.state_dict:
            # Find the first layer's input size
            for key, tensor in self.state_dict.items():
                if 'shared_layers.0.weight' in key or 'feature_layers.0.weight' in key or 'main_layers.0.weight' in key:
                    return tensor.shape[1]  # Input dimension
        
        # Default to expected feature count (143 base + 18 correlation = 161)
        return 161
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the actual PyTorch model."""
        if not self.is_trained or self.network is None:
            logger.error(f"Model not properly loaded for {self.position}")
            return self._fallback_predict(X)
        
        try:
            # Ensure input is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Apply preprocessor if available
            if self.preprocessor is not None:
                try:
                    X = self.preprocessor.transform(X)
                except Exception as e:
                    logger.warning(f"Preprocessor transform failed: {e}, using raw features")
            
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Make predictions
            with torch.no_grad():
                predictions = self.network(X_tensor)
                
                # Handle different output shapes
                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                
                # Convert to numpy
                predictions_np = predictions.cpu().numpy()
                
                # Ensure we have reasonable values (clip extreme outliers)
                from .neural_models import POSITION_RANGES
                max_range = POSITION_RANGES.get(self.position, 30.0)
                predictions_np = np.clip(predictions_np, 0, max_range * 1.2)  # Allow 20% over max
                
                return predictions_np
                
        except Exception as e:
            logger.error(f"Prediction failed for {self.position}: {e}")
            return self._fallback_predict(X)
    
    def _fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction method when neural network fails."""
        from .neural_models import POSITION_RANGES
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        base_range = POSITION_RANGES.get(self.position, 15.0)
        
        # Use position-specific typical scores
        typical_scores = {
            'QB': 20.0,
            'RB': 12.0, 
            'WR': 10.0,
            'TE': 8.0,
            'DEF': 8.0,
            'DST': 8.0
        }
        
        base_prediction = typical_scores.get(self.position, 10.0)
        
        # Add some variance based on features if available
        predictions = []
        for i in range(n_samples):
            pred = base_prediction
            if X.shape[1] > 0:
                # Use first few features to add variance
                feature_factor = np.mean(X[i, :min(5, X.shape[1])])
                pred *= (0.8 + 0.4 * (1 + feature_factor))  # Scale between 0.8x and 1.2x
            
            predictions.append(min(pred, base_range))  # Cap at position max
            
        return np.array(predictions)
    
    def predict_single(self, features: np.ndarray) -> float:
        """Predict for a single sample."""
        result = self.predict(features.reshape(1, -1))
        return float(result[0])


def load_pytorch_model(position: str, model_dir: Path = Path("models")) -> Optional[PyTorchModelWrapper]:
    """Load a PyTorch model for a specific position.
    
    Args:
        position: Player position (QB, RB, WR, TE, DEF/DST)
        model_dir: Directory containing model files
        
    Returns:
        PyTorchModelWrapper or None if not found
    """
    # Handle DEF vs DST naming
    search_pos = 'DEF' if position.upper() in ['DEF', 'DST'] else position.upper()
    
    # Find the latest model file
    model_pattern = f"{search_pos}_{search_pos}_model_*.pkl"
    model_files = list(model_dir.glob(model_pattern))
    
    # Filter out preprocessor files
    model_files = [f for f in model_files if '_preprocessor' not in str(f)]
    
    if not model_files:
        logger.warning(f"No PyTorch model found for {position}")
        return None
        
    # Get the latest model
    latest_model = sorted(model_files)[-1]
    
    # Find corresponding preprocessor
    preproc_pattern = str(latest_model).replace('.pkl', '_preprocessor.pkl')
    preproc_path = Path(preproc_pattern) if Path(preproc_pattern).exists() else None
    
    try:
        return PyTorchModelWrapper(position, latest_model, preproc_path)
    except Exception as e:
        logger.error(f"Failed to load PyTorch model for {position}: {e}")
        return None