"""Shared LLM provider adapter behavior."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from app.infrastructure.llm.config import LLMConfig
from app.infrastructure.llm.errors import LLMError
from app.infrastructure.llm.messages import build_messages

T = TypeVar("T", bound=BaseModel)


class ProviderAdapter(ABC):
    """Translate the application LLM protocol to a provider client."""

    def __init__(self, config: LLMConfig, api_key: str | None = None):
        self.config = config
        self.api_key = api_key

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt and return text."""
        try:
            messages = build_messages(prompt, self.config.system_prompt)
            extracted_kwargs = self._extract_kwargs(kwargs)
            return self._call_llm(messages, **extracted_kwargs)
        except LLMError:
            raise
        except Exception as exc:
            self._handle_error(exc)

    def invoke_structured(
        self,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Send a prompt and return a schema-validated response."""
        try:
            messages = build_messages(prompt, self.config.system_prompt)
            extracted_kwargs = self._extract_kwargs(kwargs)
            return self._call_llm_structured(messages, schema, **extracted_kwargs)
        except LLMError:
            raise
        except Exception as exc:
            self._handle_error(exc)

    @staticmethod
    def _extract_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only provider-supported per-call overrides."""
        return {
            key: kwargs[key]
            for key in ("temperature", "max_tokens", "structured_strict")
            if key in kwargs
        }

    @abstractmethod
    def _call_llm(self, messages: list, **kwargs: Any) -> str:
        """Make a provider-specific text call."""
        ...

    @abstractmethod
    def _call_llm_structured(
        self,
        messages: list,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Make a provider-specific structured call."""
        ...

    @abstractmethod
    def _handle_error(self, exc: Exception) -> None:
        """Raise the application error corresponding to a provider error."""
        ...
