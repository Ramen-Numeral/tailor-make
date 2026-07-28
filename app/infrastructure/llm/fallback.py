"""Fallback routing for LLM requests.

Implements ordered fallback logic: try primary, then fallbacks in order.
Falls back only on retryable errors (infrastructure failures).
Does not fall back on non-retryable errors (programmer errors, auth, etc.).
"""

from functools import lru_cache
from time import perf_counter, sleep
from typing import Any, Callable, TypeVar
from uuid import uuid4

from app.infrastructure.llm.config import LLMRoute
from app.infrastructure.llm.errors import (
    LLMError,
    LLMNonRetryableError,
    LLMRateLimitError,
)
from app.infrastructure.llm.factory import make_llm_client
from app.infrastructure.llm.protocol import LLMClient
from app.infrastructure.llm.usage import record_llm_call

@lru_cache(maxsize=1)
def _get_loggers():
    """Load application loggers lazily to avoid configuration cycles."""
    from app.infrastructure.logging import (
        get_llm_calls_logger,
        get_llm_errors_logger,
    )
    return get_llm_calls_logger(), get_llm_errors_logger()

T = TypeVar("T")
RATE_LIMIT_ATTEMPTS = 3
MAX_INTERACTIVE_RATE_LIMIT_WAIT_SECONDS = 10.0


class LLMRoutedClient:
    """LLM client that implements fallback routing.

    Tries the primary configuration, and on retryable failures,
    attempts each fallback in order.
    """

    def __init__(self, route: LLMRoute, api_key: str | None = None):
        """Initialize with a route and optional API key.

        Args:
            route: LLMRoute defining primary and fallbacks.
            api_key: Optional API key (used for Groq and other providers).
        """
        self.route = route
        self.api_key = api_key
        self._clients: dict[str, LLMClient] = {}

    def _get_client(self, index: int) -> LLMClient:
        """Get or create a client for the config at index."""
        config = self.route.all_configs()[index]
        key = (config.provider, config.model)

        if key not in self._clients:
            self._clients[key] = make_llm_client(config, api_key=self.api_key)

        return self._clients[key]

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke with fallback support.

        Tries primary, then fallbacks in order. Falls back only on
        retryable errors.

        Args:
            prompt: The prompt to send.
            **kwargs: Optional parameter overrides.

        Returns:
            The LLM response.

        Raises:
            LLMError: If all options (primary + fallbacks) are exhausted.
        """
        trace_context = kwargs.pop("trace_context", None)
        return self._try_route(
            lambda client: client.invoke(prompt, **kwargs),
            operation_name="invoke",
            prompt_length=len(prompt),
            trace_context=trace_context,
        )

    def invoke_structured(
        self,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Invoke structured with fallback support.

        Tries primary, then fallbacks in order. Falls back only on
        retryable errors.

        Args:
            prompt: The prompt to send.
            schema: The Pydantic schema for adv_val.
            **kwargs: Optional parameter overrides.

        Returns:
            Validated response as an instance of schema.

        Raises:
            LLMError: If all options (primary + fallbacks) are exhausted.
        """
        trace_context = kwargs.pop("trace_context", None)
        return self._try_route(
            lambda client: client.invoke_structured(prompt, schema, **kwargs),
            operation_name="invoke_structured",
            prompt_length=len(prompt),
            schema_name=schema.__name__,
            trace_context=trace_context,
        )

    def _try_route(
        self,
        operation: Callable[[LLMClient], T],
        *,
        operation_name: str,
        prompt_length: int,
        schema_name: str | None = None,
        trace_context: str | None = None,
    ) -> T:
        """Try operation across route (primary + fallbacks).

        Falls back only on retryable errors. Raises non-retryable errors
        immediately without trying fallbacks.

        Args:
            operation: A callable that takes an LLMClient and returns a result.

        Returns:
            The result from the first successful client.

        Raises:
            LLMError: If all options exhausted or non-retryable error occurs.
        """
        call_logger, error_logger = _get_loggers()
        all_configs = self.route.all_configs()
        last_error: LLMError | None = None
        request_id = uuid4().hex
        total_attempts = len(all_configs)

        call_logger.info(
            "request_started request_id=%s operation=%s schema=%s "
            "prompt_chars=%d total_attempts=%d context=%s",
            request_id,
            operation_name,
            schema_name or "none",
            prompt_length,
            total_attempts,
            trace_context or "none",
        )

        for index, config in enumerate(all_configs):
            attempt = index + 1
            started = perf_counter()
            call_logger.info(
                "attempt_started request_id=%s attempt=%d/%d provider=%s "
                "model=%s operation=%s",
                request_id,
                attempt,
                total_attempts,
                config.provider,
                config.model,
                operation_name,
            )
            try:
                client = self._get_client(index)
                result = self._invoke_with_rate_limit_retries(
                    operation,
                    client,
                    request_id=request_id,
                    provider=config.provider,
                    model=config.model,
                    operation_name=operation_name,
                )
                record_llm_call(prompt_length, result)
                call_logger.info(
                    "attempt_succeeded request_id=%s attempt=%d/%d provider=%s "
                    "model=%s operation=%s elapsed_ms=%.1f",
                    request_id,
                    attempt,
                    total_attempts,
                    config.provider,
                    config.model,
                    operation_name,
                    (perf_counter() - started) * 1000,
                )
                return result

            except LLMNonRetryableError as exc:
                # Non-retryable: give up immediately
                error_logger.error(
                    "attempt_failed request_id=%s attempt=%d/%d provider=%s "
                    "model=%s operation=%s retryable=false elapsed_ms=%.1f "
                    "error_type=%s error=%s",
                    request_id, attempt, total_attempts, config.provider,
                    config.model, operation_name,
                    (perf_counter() - started) * 1000,
                    type(exc).__name__, exc,
                )
                raise

            except LLMError as exc:
                error_logger.warning(
                    "attempt_failed request_id=%s attempt=%d/%d provider=%s "
                    "model=%s operation=%s retryable=true elapsed_ms=%.1f "
                    "error_type=%s error=%s",
                    request_id, attempt, total_attempts, config.provider,
                    config.model, operation_name,
                    (perf_counter() - started) * 1000,
                    type(exc).__name__, exc,
                )
                last_error = exc
                continue

        # All options exhausted
        if last_error is not None:
            error_logger.error(
                "request_exhausted request_id=%s operation=%s attempts=%d "
                "last_error_type=%s last_error=%s",
                request_id, operation_name, total_attempts,
                type(last_error).__name__, last_error,
            )
            raise LLMError(
                f"All LLM options exhausted. Last error: {last_error}"
            ) from last_error
        raise LLMError("All LLM options exhausted (no error recorded)")

    @staticmethod
    def _invoke_with_rate_limit_retries(
        operation: Callable[[LLMClient], T],
        client: LLMClient,
        *,
        request_id: str,
        provider: str,
        model: str,
        operation_name: str,
    ) -> T:
        """Retry one provider call only when it returns a rate limit."""
        for rate_attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
            try:
                return operation(client)
            except LLMRateLimitError as exc:
                if rate_attempt == RATE_LIMIT_ATTEMPTS:
                    raise
                if (
                    exc.retry_after_seconds is not None
                    and exc.retry_after_seconds
                    > MAX_INTERACTIVE_RATE_LIMIT_WAIT_SECONDS
                ):
                    # A provider can return waits measured in many minutes.
                    # Holding an HTTP request open that long makes the app look
                    # frozen; let the route try its next configured model.
                    raise

                error_logger = _get_loggers()[1]
                exponential_delay = float(2 ** (rate_attempt - 1))
                delay = max(
                    exponential_delay,
                    exc.retry_after_seconds or 0.0,
                )
                error_logger.warning(
                    "rate_limit_retry request_id=%s rate_attempt=%d/%d "
                    "provider=%s model=%s operation=%s wait_seconds=%.3f "
                    "provider_retry_after=%s",
                    request_id,
                    rate_attempt,
                    RATE_LIMIT_ATTEMPTS,
                    provider,
                    model,
                    operation_name,
                    delay,
                    exc.retry_after_seconds,
                )
                sleep(delay)

        raise RuntimeError("unreachable")
