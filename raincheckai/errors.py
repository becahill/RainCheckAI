"""RainCheckAI domain exceptions."""


class RainCheckAIError(Exception):
    """Base exception for recoverable application errors."""


class DatasetValidationError(RainCheckAIError):
    """Raised when an input dataset does not satisfy the required schema."""


class ArtifactNotAvailableError(RainCheckAIError):
    """Raised when model artifacts are not available on disk."""


class PredictionFailureError(RainCheckAIError):
    """Raised when a prediction cannot be produced from valid inputs."""
