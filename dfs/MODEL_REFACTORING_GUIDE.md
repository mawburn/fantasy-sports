# DFS Application Refactoring Guide: Breaking Up the Monolith

## Current State Analysis

Your DFS application has grown into several large, complex files:

- **data.py**: 4774 lines - Data collection, DB ops, feature engineering
- **models.py**: 2070 lines - All ML models and training logic
- **run.py**: 1063 lines - CLI interface with all commands
- **optimize.py**: 1049 lines - Lineup optimization and constraints
- **backtest.py**: 939 lines - Backtesting and validation
- **utils.py**: 509 lines - Mixed utilities

**Total**: 12,000+ lines in 6 monolithic files

**Problems**: Large files make it hard to:
- Navigate and understand code
- Test individual components  
- Add new features without conflicts
- Onboard new developers
- Debug specific functionality
- Deploy individual components

## Proposed Modular Structure

```
models/
├── __init__.py                 # Public API exports
├── base.py                     # BaseNeuralModel, TrainingResult, etc.
├── loss_functions.py           # DFSLoss, quantile_loss
├── networks/
│   ├── __init__.py
│   ├── qb.py                  # QBNetwork + QBNeuralModel
│   ├── rb.py                  # RBNetwork + RBNeuralModel
│   ├── wr.py                  # WRNetwork + WRNeuralModel
│   ├── te.py                  # TENetwork + TENeuralModel
│   └── dst.py                 # DEFNetwork + DEFNeuralModel
├── ensemble.py                 # EnsembleModel, CorrelatedFantasyModel
├── correlation.py              # CorrelationFeatureExtractor
├── utils.py                    # Device detection, validation
└── factory.py                  # create_model(), load_trained_model()
```

## Benefits of This Structure

### 1. **Logical Separation**

- Each position gets its own file
- Related functionality grouped together
- Clear import paths

### 2. **Easier Maintenance**

- Modify QB model without affecting RB model
- Test individual positions in isolation
- Smaller files = easier navigation

### 3. **Team Collaboration**

- Different developers can work on different positions
- Reduced merge conflicts
- Position-specific expertise can be developed

### 4. **Future Extensibility**

- Easy to add new positions (K, LB, etc.)
- Can experiment with different architectures per position
- Modular hyperparameter tuning per position

## Migration Plan

### Phase 1: Extract Base Components (30 min)

```python
# models/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class ModelConfig: ...

@dataclass
class TrainingResult: ...

@dataclass
class PredictionResult: ...

class BaseNeuralModel(ABC):
    # All the common training/prediction logic
    @abstractmethod
    def build_network(self, input_size: int) -> nn.Module:
        pass
```

### Phase 2: Extract Position Networks (45 min)

```python
# models/networks/qb.py
from ..base import BaseNeuralModel, ModelConfig
import torch.nn as nn

class QBNetwork(nn.Module):
    # Current QBNetwork implementation

class QBNeuralModel(BaseNeuralModel):
    # Current QBNeuralModel implementation
```

### Phase 3: Extract Utilities (15 min)

```python
# models/utils.py
import torch
import os

def get_optimal_device():
    # Current device detection logic

POSITION_RANGES = {
    'QB': 45.0,
    # ... existing ranges
}
```

### Phase 4: Update Imports (15 min)

```python
# models/__init__.py
from .factory import create_model, load_trained_model
from .base import ModelConfig, TrainingResult, PredictionResult
from .networks.qb import QBNeuralModel
from .networks.rb import RBNeuralModel
# ... etc

# Maintain backward compatibility
__all__ = [
    'create_model', 'load_trained_model',
    'QBNeuralModel', 'RBNeuralModel', 'WRNeuralModel',
    'TENeuralModel', 'DEFNeuralModel',
    'ModelConfig', 'TrainingResult', 'PredictionResult'
]
```

### Phase 5: Update run.py Imports (5 min)

```python
# run.py - Change this:
# from models import create_model

# To this:
from models import create_model  # Same import, works via __init__.py
```

## File-by-File Breakdown

### models/base.py (~300 lines)

- `ModelConfig`, `TrainingResult`, `PredictionResult` dataclasses
- `BaseNeuralModel` abstract class with all common methods
- Training loops, validation, prediction logic
- Device management and normalization

### models/networks/qb.py (~200 lines)

- `QBNetwork` with multi-head architecture
- `QBNeuralModel` with QB-specific parameters
- QB validation methods
- QB-specific loss function integration

### models/networks/rb.py (~150 lines)

- `RBNetwork` architecture
- `RBNeuralModel` with optimized parameters
- RB-specific training parameters

### models/networks/wr.py (~150 lines)

- `WRNetwork` with target/big-play branches
- `WRNeuralModel` implementation

### models/networks/te.py (~120 lines)

- `TENetwork` simpler architecture
- `TENeuralModel` implementation

### models/networks/dst.py (~200 lines)

- `DEFNetwork` with attention mechanism
- `DEFCatBoostModel` CatBoost implementation
- `DEFNeuralModel` alias

### models/correlation.py (~400 lines)

- `CorrelationFeatureExtractor` class
- All position-specific correlation methods
- Database query logic

### models/ensemble.py (~300 lines)

- `EnsembleModel` combining neural + XGBoost
- `CorrelatedFantasyModel` for multi-position
- Game context encoding

### models/loss_functions.py (~100 lines)

- `DFSLoss` custom loss function
- Quantile loss implementations
- Loss utilities

### models/utils.py (~100 lines)

- Device detection (`get_optimal_device`)
- Position ranges constants
- Validation utilities
- ResidualBlock component

### models/factory.py (~50 lines)

- `create_model()` function
- `load_trained_model()` function
- Model registry logic

## Backward Compatibility Strategy

### Keep Existing API

```python
# This still works after refactoring:
from models import create_model, QBNeuralModel
model = create_model('QB')
```

### Gradual Migration

1. **Phase 1**: Split files but keep old imports working
2. **Phase 2**: Update internal imports gradually
3. **Phase 3**: Remove old monolithic file once everything works

### Import Mapping

```python
# Old way (still works):
from models import QBNeuralModel

# New way (equivalent):
from models.networks.qb import QBNeuralModel
```

## Testing Strategy

### 1. **Smoke Tests**

```bash
# Verify imports work
python -c "from models import create_model; print('✓ Imports work')"

# Verify model creation
python -c "from models import create_model; m = create_model('QB'); print('✓ QB model created')"
```

### 2. **Functionality Tests**

```bash
# Test training still works
uv run python run.py train --positions QB

# Test prediction still works
uv run python run.py predict --contest-id 123
```

### 3. **File-by-File Validation**

- Import each new module independently
- Verify no circular imports
- Check all classes/functions are accessible

## Implementation Timeline

- **Total Time**: ~2 hours
- **Risk Level**: Low (backward compatible)
- **Benefits**: High (maintainability, collaboration)

### Hour 1: Core Extraction

- Extract `base.py`, `utils.py`, `loss_functions.py`
- Create directory structure
- Set up `__init__.py` exports

### Hour 2: Position Networks

- Split position models into separate files
- Update imports and test
- Verify backward compatibility

## Post-Refactor Benefits

### Development Workflow

```bash
# Work on QB model only
code models/networks/qb.py

# Test QB model in isolation
python -m pytest tests/test_qb_model.py

# Add new position easily
cp models/networks/te.py models/networks/k.py
# Edit for kicker-specific logic
```

### Code Organization

- **Before**: Find QB logic in 2070-line file
- **After**: QB logic in dedicated 200-line file

### Team Collaboration

- **Before**: Merge conflicts on massive file
- **After**: Clean separation, parallel development

This refactoring maintains all existing functionality while dramatically improving code organization and maintainability.
