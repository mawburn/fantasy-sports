"""Custom exceptions for ML components.

This file defines domain-specific exceptions for the fantasy sports ML system.
Custom exceptions provide better error handling and debugging by:

1. Clear Error Communication: Specific exception types immediately indicate what went wrong
2. Targeted Error Handling: Code can catch and handle specific errors appropriately
3. Better Debugging: Stack traces clearly show the type of failure
4. Professional Error Messages: User-friendly error descriptions for common issues

For beginners: Exceptions are Python's way of handling errors gracefully.
Instead of crashing, programs can "catch" exceptions and respond appropriately.
Custom exceptions help organize and categorize different types of errors.

Inheritance Pattern:
Most exceptions inherit from ValueError (for invalid values) or FileNotFoundError
(for missing files). This allows code to catch either the specific exception
or the broader base exception as needed.

Usage Examples:
- raise ModelNotTrainedError("Cannot predict with untrained QB model")
- raise InsufficientDataError("Need at least 100 samples, got 23")
- raise UnsupportedPositionError(f"Position '{pos}' not supported")
"""


class ModelNotTrainedError(ValueError):
    """Raised when trying to use an untrained model.

    This exception occurs when code attempts to use a model for predictions
    or evaluation before it has been properly trained. It's a safety mechanism
    to prevent meaningless results from untrained models.

    Common Scenarios:
    - Calling predict() on a newly instantiated model
    - Calling evaluate() before training is complete
    - Loading a model file that doesn't contain trained weights
    - Skipping the training step in a pipeline

    Prevention:
    - Always check model.is_trained before using model
    - Ensure training completes successfully before prediction
    - Validate loaded models have trained status

    Example:
    ```python
    if not model.is_trained:
        raise ModelNotTrainedError(f"Model {model.config.model_name} must be trained before prediction")
    ```
    """


class ModelNotFoundError(ValueError):
    """Raised when a model is not found.

    This exception occurs when the system cannot locate a requested model,
    either in the model registry database or in the file system.

    Common Scenarios:
    - Requesting a model_id that doesn't exist in the registry
    - Model file was deleted from disk but registry record remains
    - Typos in model identifiers
    - Models that were cleaned up or archived

    Context Information:
    - Include the missing model_id in the error message
    - Suggest similar model names if available
    - Provide list of available models for the position

    Example:
    ```python
    available = [m.model_id for m in registry.list_models(position=position)]
    raise ModelNotFoundError(f"Model '{model_id}' not found. Available: {available}")
    ```
    """


class ModelNotReadyError(ValueError):
    """Raised when a model is not ready for the requested operation.

    This exception covers situations where a model exists and may be trained,
    but is not in the correct state for the requested operation.

    Common Scenarios:
    - Trying to deploy a model that hasn't passed validation
    - Attempting to activate a model that's in 'retired' status
    - Loading a model with incompatible feature schema
    - Using a model during maintenance or update operations

    State Validation:
    - Check model status (trained, deployed, retired)
    - Verify model compatibility with current system version
    - Ensure all required dependencies are available

    Example:
    ```python
    if model.status != 'trained':
        raise ModelNotReadyError(f"Model {model_id} status is '{model.status}', expected 'trained'")
    ```
    """


class InsufficientDataError(ValueError):
    """Raised when there is insufficient data for an operation.

    Machine learning operations require minimum amounts of data to produce
    reliable results. This exception prevents operations that would produce
    meaningless or unreliable results due to insufficient sample sizes.

    Common Scenarios:
    - Training with fewer samples than minimum threshold
    - Player has too few games for reliable feature extraction
    - Validation set too small for meaningful metrics
    - Backtesting period contains insufficient predictions

    Data Requirements:
    - Model training: typically 100+ samples per position
    - Player features: 3+ recent games for trend analysis
    - Model evaluation: 50+ test samples for reliable metrics
    - Statistical analysis: varies by operation

    Example:
    ```python
    if len(training_data) < min_samples:
        raise InsufficientDataError(
            f"Need at least {min_samples} samples for training, got {len(training_data)}"
        )
    ```
    """


class InvalidInputError(ValueError):
    """Raised when input data is invalid.

    This exception covers various input validation failures that prevent
    ML operations from proceeding. It helps catch data quality issues
    early before they cause downstream failures.

    Common Scenarios:
    - Feature arrays with wrong dimensions (samples × features)
    - NaN or infinite values in critical features
    - Categorical values outside expected ranges
    - Date ranges that don't make sense (end_date before start_date)
    - Player IDs that don't exist in the database

    Validation Categories:
    - Shape validation: correct array dimensions
    - Value validation: no NaN/inf, reasonable ranges
    - Type validation: correct data types
    - Business logic validation: dates, IDs, constraints

    Example:
    ```python
    if X.ndim != 2:
        raise InvalidInputError(f"Features must be 2D array, got shape {X.shape}")
    if np.any(np.isnan(X)):
        raise InvalidInputError("Features contain NaN values")
    ```
    """


class UnsupportedPositionError(ValueError):
    """Raised when an unsupported position is requested.

    The fantasy sports system supports specific NFL positions. This exception
    prevents operations on unsupported positions and provides clear feedback
    about what positions are available.

    Supported Positions:
    - QB (Quarterback)
    - RB (Running Back)
    - WR (Wide Receiver)
    - TE (Tight End)
    - DEF (Defense/Special Teams)

    Common Scenarios:
    - User requests model for 'K' (Kicker) position
    - Data contains players with position 'FB' (Fullback)
    - API receives invalid position parameter
    - Import data has non-standard position codes

    Prevention:
    - Validate position parameters at API boundaries
    - Use constants/enums for supported positions
    - Provide clear error messages with supported options

    Example:
    ```python
    SUPPORTED_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'DEF']
    if position not in SUPPORTED_POSITIONS:
        raise UnsupportedPositionError(
            f"Position '{position}' not supported. Supported: {SUPPORTED_POSITIONS}"
        )
    ```
    """


class ModelFileError(FileNotFoundError):
    """Raised when a model file is not found or corrupted.

    This exception handles file system issues related to model artifacts.
    It extends FileNotFoundError to provide ML-specific context about
    missing or corrupted model files.

    Common Scenarios:
    - Model file deleted from disk but registry record remains
    - Model file corrupted during save/load operations
    - File permissions prevent model file access
    - Disk full during model saving
    - Network storage unavailable

    File Types:
    - .pkl files: serialized model objects
    - .json files: model configuration and metadata
    - .txt files: feature names and preprocessing info

    Recovery Strategies:
    - Check if backup copies exist
    - Retrain model from scratch if possible
    - Load previous version of model
    - Verify disk space and permissions

    Example:
    ```python
    try:
        model_data = joblib.load(model_path)
    except (FileNotFoundError, EOFError, PickleError) as e:
        raise ModelFileError(f"Cannot load model from {model_path}: {e}") from e
    ```
    """


class EnsembleError(ValueError):
    """Raised when ensemble operations fail.

    Ensemble models combine multiple base models for improved performance.
    This exception handles failures in ensemble construction, training,
    or prediction that don't fall under other categories.

    Common Scenarios:
    - Base models have incompatible output dimensions
    - Ensemble weights don't sum to 1.0
    - Mix of trained and untrained base models
    - Conflicting model configurations
    - Meta-learner training failures

    Ensemble Requirements:
    - All base models must be trained
    - Models must have compatible input/output shapes
    - Ensemble weights must be valid probabilities
    - Sufficient diversity among base models

    Example:
    ```python
    if not all(model.is_trained for model in base_models):
        untrained = [m.config.model_name for m in base_models if not m.is_trained]
        raise EnsembleError(f"Base models not trained: {untrained}")
    ```
    """


class FeatureExtractionError(ValueError):
    """Raised when feature extraction fails.

    Feature extraction converts raw player and game data into ML-ready
    numerical features. This exception handles failures in this critical
    data preprocessing step.

    Common Scenarios:
    - Missing historical data for player feature calculation
    - Database query failures during feature extraction
    - Calculation errors (division by zero, etc.)
    - Inconsistent data schemas between expected and actual
    - Memory errors when processing large datasets

    Feature Categories:
    - Player statistics (passing yards, receptions, etc.)
    - Team metrics (offensive ranking, pace, etc.)
    - Matchup data (opponent strength, weather, etc.)
    - Historical trends (recent form, season patterns)

    Debugging Information:
    - Include player_id and game_date in error messages
    - Log the specific feature that failed to calculate
    - Provide information about available data ranges

    Example:
    ```python
    try:
        features = calculate_player_features(player_id, game_date)
    except (ZeroDivisionError, KeyError, IndexError) as e:
        raise FeatureExtractionError(
            f"Failed to extract features for player {player_id} on {game_date}: {e}"
        ) from e
    ```
    """
