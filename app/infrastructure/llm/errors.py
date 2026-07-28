"""Provider-neutral LLM exceptions."""


class LLMError(Exception):
    """Base exception for all LLM-related failures."""


class LLMRetryableError(LLMError):
    """Infrastructure failure that may succeed on retry."""


class LLMNonRetryableError(LLMError):
    """Permanent request or authentication failure."""


class LLMInvalidOutputError(LLMError):
    """LLM output violated its required schema."""


class LLMTimeoutError(LLMRetryableError):
    """Request exceeded timeout threshold."""


class LLMRateLimitError(LLMRetryableError):
    """Provider rate limit exceeded."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthenticationError(LLMNonRetryableError):
    """Authentication failed."""


class LLMModelNotFoundError(LLMNonRetryableError):
    """Requested model is unavailable."""


class LLMProviderError(LLMRetryableError):
    """Generic provider server error."""


class LLMProviderUnavailableError(LLMRetryableError):
    """Provider service is temporarily unavailable."""
