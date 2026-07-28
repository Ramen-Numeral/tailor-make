"""Ollama LLM provider adapter.

Wraps LangChain's ChatOllama behind the application LLMClient protocol.
"""

import json
import logging
from typing import Any, TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.infrastructure.llm.config import LLMConfig
from app.infrastructure.llm.errors import (
    LLMInvalidOutputError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)
from app.infrastructure.llm.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter(ProviderAdapter):
    """Ollama LLM client implementing the application LLMClient protocol."""

    def __init__(self, config: LLMConfig):
        """Initialize Ollama adapter.

        Args:
            config: LLMConfig with model and parameters.
        """
        if config.provider.lower() != "ollama":
            raise ValueError(
                f"OllamaAdapter only supports 'ollama' provider, got {config.provider}"
            )

        super().__init__(config, api_key=None)

        try:
            self._client = ChatOllama(
                model=config.model,
                temperature=config.temperature,
                num_ctx=2048,
            )
        except Exception as exc:
            logger.error(f"Failed to initialize Ollama client: {exc}")
            raise LLMProviderUnavailableError(
                f"Failed to initialize Ollama: {exc}"
            ) from exc

    def _call_llm(self, messages: list, **kwargs: Any) -> str:
        """Call Ollama API and return string response.

        Args:
            messages: LangChain message list.
            **kwargs: Extracted overrides (ignored—Ollama doesn't support per-call overrides).

        Returns:
            Response string.
        """
        response = self._client.invoke(messages)
        return response.content

    def _call_llm_structured(
        self,
        messages: list,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Call Ollama API with structured output.

        Args:
            messages: LangChain message list.
            schema: Pydantic model for adv_val.
            **kwargs: Extracted overrides (ignored).

        Returns:
            Validated response instance.
        """
        structured_client = self._client.with_structured_output(schema)
        response = structured_client.invoke(messages)
        return response

    def _handle_error(self, exc: Exception) -> None:
        """Translate Ollama/LangChain exceptions to application errors.

        Args:
            exc: The exception from Ollama or LangChain.

        Raises:
            Appropriate LLMError subclass.
        """
        exc_str = str(exc).lower()

        # Connection failures (Ollama server not running)
        if "connection" in exc_str or "refused" in exc_str or "unreachable" in exc_str:
            logger.error(f"Ollama connection error: {exc}")
            raise LLMProviderUnavailableError(
                f"Ollama not available: {exc}"
            ) from exc

        # Timeout
        if "timeout" in exc_str or "timed out" in exc_str:
            logger.error(f"Ollama timeout: {exc}")
            raise LLMTimeoutError(f"Ollama request timeout: {exc}") from exc

        # Model not found
        if "model" in exc_str and ("not found" in exc_str or "404" in exc_str):
            logger.error("Ollama model not found: %s", exc)
            raise LLMModelNotFoundError(
                f"Ollama model not found: {exc}"
            ) from exc

        # Invalid output / JSON
        if isinstance(exc, (json.JSONDecodeError, ValueError)):
            logger.error(f"Ollama invalid output: {exc}")
            raise LLMInvalidOutputError(f"Invalid Ollama output: {exc}") from exc

        # Generic provider error
        logger.error(f"Ollama provider error: {exc}")
        raise LLMProviderError(f"Ollama error: {exc}") from exc
