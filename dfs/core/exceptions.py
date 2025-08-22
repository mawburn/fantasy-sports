"""Custom exceptions for the DFS application."""


class DFSError(Exception):
    """Base exception for DFS application errors."""
    pass


class ModelError(DFSError):
    """Exception raised for model-related errors."""
    pass


class OptimizationError(DFSError):
    """Exception raised for optimization-related errors."""
    pass


class DataError(DFSError):
    """Exception raised for data-related errors."""
    pass


class ValidationError(DFSError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(DFSError):
    """Exception raised for configuration errors."""
    pass