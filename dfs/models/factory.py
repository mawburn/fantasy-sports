"""Model factory for creating position-specific models."""

from typing import Dict, Any
from dfs.core.logging import get_logger
from dfs.core.exceptions import ModelError
from .base.utils import ModelConfig
from .networks.qb import QBNeuralModel
from .networks.rb import RBNeuralModel  
from .networks.wr import WRNeuralModel
from .networks.te import TENeuralModel
from .networks.dst import DSTNeuralModel

logger = get_logger("models.factory")

# Model registry
MODEL_REGISTRY = {
    'QB': QBNeuralModel,
    'RB': RBNeuralModel,
    'WR': WRNeuralModel, 
    'TE': TENeuralModel,
    'DST': DSTNeuralModel,
    'DEF': DSTNeuralModel  # Alias for DST
}


def create_model(position: str, config: ModelConfig, use_ensemble: bool = False):
    """Create a model for the specified position.
    
    Args:
        position: Player position (QB, RB, WR, TE, DST)
        config: Model configuration
        use_ensemble: Whether to use ensemble model (future implementation)
        
    Returns:
        Model instance for the position
        
    Raises:
        ModelError: If position is not supported
    """
    position = position.upper()
    
    if position not in MODEL_REGISTRY:
        raise ModelError(f"Unsupported position: {position}")
    
    model_class = MODEL_REGISTRY[position]
    
    try:
        model = model_class(config)
        logger.info(f"Created {position} model with {len(config.features)} features")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create {position} model: {e}")
        raise ModelError(f"Model creation failed: {e}")


def get_supported_positions() -> list:
    """Get list of supported positions."""
    return list(MODEL_REGISTRY.keys())


def register_model(position: str, model_class: type) -> None:
    """Register a new model class for a position."""
    MODEL_REGISTRY[position.upper()] = model_class
    logger.info(f"Registered model class for position: {position}")


# Backward compatibility function
def QBNeuralModel_create(config: ModelConfig):
    """Backward compatibility wrapper for QB model creation."""
    return create_model('QB', config)


def RBNeuralModel_create(config: ModelConfig):
    """Backward compatibility wrapper for RB model creation."""
    return create_model('RB', config)


def WRNeuralModel_create(config: ModelConfig):
    """Backward compatibility wrapper for WR model creation."""
    return create_model('WR', config)


def TENeuralModel_create(config: ModelConfig):
    """Backward compatibility wrapper for TE model creation."""
    return create_model('TE', config)


def DSTNeuralModel_create(config: ModelConfig):
    """Backward compatibility wrapper for DST model creation."""
    return create_model('DST', config)