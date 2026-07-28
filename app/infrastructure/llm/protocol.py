"""Application-owned LLM client protocol."""

from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class LLMClient(Protocol):
    """Provider-independent text and structured invocation."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt and return text."""
        ...

    def invoke_structured(
        self,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Send a prompt and return a schema-validated response."""
        ...
