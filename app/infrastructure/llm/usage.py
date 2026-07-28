"""Request-scoped LLM usage metering for UI summaries."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from typing import Iterator


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens_estimated: int = 0
    completion_tokens_estimated: int = 0

    @property
    def total_tokens_estimated(self) -> int:
        return self.prompt_tokens_estimated + self.completion_tokens_estimated

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens_estimated,
            "completion_tokens": self.completion_tokens_estimated,
            "total_tokens": self.total_tokens_estimated,
            "tokens_are_estimated": True,
        }


_active_usage: ContextVar[LLMUsage | None] = ContextVar(
    "active_llm_usage",
    default=None,
)


@contextmanager
def meter_llm_usage() -> Iterator[LLMUsage]:
    usage = LLMUsage()
    token = _active_usage.set(usage)
    try:
        yield usage
    finally:
        _active_usage.reset(token)


def record_llm_call(prompt_characters: int, result: object) -> None:
    """Record a conservative tokenizer-independent estimate."""
    usage = _active_usage.get()
    if usage is None:
        return
    usage.calls += 1
    usage.prompt_tokens_estimated += _estimate_character_count(prompt_characters)
    if hasattr(result, "model_dump_json"):
        rendered = result.model_dump_json()
    elif isinstance(result, str):
        rendered = result
    else:
        rendered = json.dumps(result, default=str)
    usage.completion_tokens_estimated += _estimate_tokens(rendered)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _estimate_character_count(characters: int) -> int:
    return max(1, (characters + 3) // 4)
