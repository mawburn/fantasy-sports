"""Custom exceptions for ML components."""


class ModelNotTrainedError(ValueError):
    """Raised when trying to use an untrained model."""


class ModelNotFoundError(ValueError):
    """Raised when a model is not found."""


class ModelNotReadyError(ValueError):
    """Raised when a model is not ready for the requested operation."""


class InsufficientDataError(ValueError):
    """Raised when there is insufficient data for an operation."""


class InvalidInputError(ValueError):
    """Raised when input data is invalid."""


class UnsupportedPositionError(ValueError):
    """Raised when an unsupported position is requested."""


class ModelFileError(FileNotFoundError):
    """Raised when a model file is not found or corrupted."""


class EnsembleError(ValueError):
    """Raised when ensemble operations fail."""


class FeatureExtractionError(ValueError):
    """Raised when feature extraction fails."""
