"""Groq LLM provider adapter.

Wraps LangChain's ChatGroq behind the application LLMClient protocol.
"""

import json
import logging
import re
from typing import Any, TypeVar

import httpx
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.infrastructure.llm.config import LLMConfig
from app.infrastructure.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidOutputError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.infrastructure.llm.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GroqAdapter(ProviderAdapter):
    """Groq LLM client implementing the application LLMClient protocol."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: str | None = None,
    ):
        """Initialize Groq adapter.

        Args:
            config: LLMConfig with model and parameters.
            api_key: Groq API key (uses environment variable if not provided).
        """
        if config.provider.lower() != "groq":
            raise ValueError(
                f"GroqAdapter only supports 'groq' provider, got {config.provider}"
            )

        super().__init__(config, api_key)

        try:
            reasoning_effort = (
                "low" if config.model.startswith("openai/gpt-oss-") else None
            )
            self._client = ChatGroq(
                model=config.model,
                temperature=config.temperature,
                timeout=httpx.Timeout(config.timeout, connect=10.0),
                api_key=api_key,
                max_tokens=config.max_tokens,
                reasoning_effort=reasoning_effort,
                # Retry routing belongs to LLMRoutedClient. Hidden SDK retries
                # make one logged attempt last multiple timeout windows.
                max_retries=0,
            )
        except Exception as exc:
            raise LLMAuthenticationError(
                f"Failed to initialize Groq client: {exc}"
            ) from exc

    def _call_llm(self, messages: list, **kwargs: Any) -> str:
        """Call Groq API and return string response.

        Args:
            messages: LangChain message list.
            **kwargs: Extracted overrides (temperature, max_tokens).

        Returns:
            Response string.
        """
        response = self._client.invoke(messages, **kwargs)
        return response.content

    def _call_llm_structured(
        self,
        messages: list,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Call Groq API with structured output.

        Args:
            messages: LangChain message list.
            schema: Pydantic model for adv_val.
            **kwargs: Extracted overrides.

        Returns:
            Validated response instance.
        """
        strict = kwargs.pop("structured_strict", False)
        json_schema_models = (
            "openai/gpt-oss-",
            "meta-llama/llama-4-scout-",
        )
        json_mode_models = (
            "qwen/",
            "llama-3.1-",
            "llama-3.3-",
        )
        if self.config.model.startswith(json_mode_models):
            method = "json_mode"
        elif self.config.model.startswith(json_schema_models):
            method = "json_schema"
        else:
            method = "function_calling"
        structured_client = self._client.with_structured_output(
            schema,
            method=method,
            strict=strict if method == "json_schema" else None,
        )
        response = structured_client.invoke(messages, **kwargs)
        return response

    def _handle_error(self, exc: Exception) -> None:
        """Translate Groq/LangChain exceptions to application errors.

        Args:
            exc: The exception from Groq or LangChain.

        Raises:
            Appropriate LLMError subclass.
        """
        exc_str = str(exc).lower()

        # Rate limiting. Groq commonly includes a retry duration in the error
        # body (for example, "try again in 929ms").
        status_code = getattr(exc, "status_code", None)
        if status_code == 429 or "rate limit" in exc_str or "429" in exc_str:
            retry_after = _retry_after_seconds(exc, exc_str)
            logger.warning("Groq rate limit: %s", exc)
            raise LLMRateLimitError(
                f"Groq rate limit exceeded: {exc}",
                retry_after_seconds=retry_after,
            ) from exc

        # Timeout
        if "timeout" in exc_str or "timed out" in exc_str:
            logger.error(f"Groq timeout: {exc}")
            raise LLMTimeoutError(f"Groq request timeout: {exc}") from exc

        # Authentication
        if "authentication" in exc_str or "unauthorized" in exc_str or "401" in exc_str:
            logger.error(f"Groq authentication error: {exc}")
            raise LLMAuthenticationError(
                f"Groq authentication failed: {exc}"
            ) from exc

        # Model not found
        if "model" in exc_str and ("not found" in exc_str or "404" in exc_str):
            logger.error("Groq model not found: %s", exc)
            raise LLMModelNotFoundError(
                f"Groq model not found: {exc}"
            ) from exc

        # Invalid output / JSON
        if isinstance(exc, (json.JSONDecodeError, ValueError)):
            logger.error(f"Groq invalid output: {exc}")
            raise LLMInvalidOutputError(f"Invalid Groq output: {exc}") from exc

        # Generic provider error
        logger.error(f"Groq provider error: {exc}")
        raise LLMProviderError(f"Groq error: {exc}") from exc


def _retry_after_seconds(exc: Exception, message: str) -> float | None:
    """Extract Groq's requested retry delay from headers or error text."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    header_value = headers.get("retry-after") if headers else None

    if header_value:
        try:
            return max(0.0, float(header_value))
        except (TypeError, ValueError):
            pass

    match = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)",
        message,
    )
    if not match:
        return None

    delay = float(match.group(1))
    return delay / 1000 if match.group(2) == "ms" else delay
