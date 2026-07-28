"""Provider-neutral LLM configuration, clients, routing, and errors."""

from app.infrastructure.llm.config import LLMConfig, LLMRoute
from app.infrastructure.llm.errors import (
    LLMError,
    LLMInvalidOutputError,
    LLMNonRetryableError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMRetryableError,
    LLMTimeoutError,
)
from app.infrastructure.llm.factory import make_llm_client
from app.infrastructure.llm.fallback import LLMRoutedClient
from app.infrastructure.llm.protocol import LLMClient

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMRoute",
    "LLMError",
    "LLMRetryableError",
    "LLMNonRetryableError",
    "LLMInvalidOutputError",
    "LLMTimeoutError",
    "LLMProviderError",
    "LLMProviderUnavailableError",
    "make_llm_client",
    "LLMRoutedClient",
]
