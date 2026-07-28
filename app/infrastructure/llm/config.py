"""Provider-neutral LLM configuration.

Defines immutable configuration objects for LLMs and routing rules.
These objects describe LLM models but do not construct clients.
Client construction is handled by the factory module.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Configuration for callable LLM models.


    Attributes:
        provider: Provider name (e.g., "groq", "ollama").
        model: Model identifier (e.g., "llama-3.1-8b-instant").
        system_prompt: Optional system prompt to send with every request.
        temperature: Sampling temperature (0.0 to 2.0, typically). Lower = deterministic.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens in response (optional).
    """

    provider: str
    model: str
    system_prompt: str | None = None
    temperature: float = 0.2
    timeout: float = 30.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class LLMRoute:
    """Ordered routing for LLM requests with fallback support.

    Defines a primary and zero or more fallback models.
    If the primary fails with a retryable error, fallbacks are tried in order.

    Attributes:
        primary: The primary LLMConfig to use.
        fallbacks: Ordered tuple of fallback LLMConfig objects.
    """

    primary: LLMConfig
    fallbacks: tuple[LLMConfig, ...] = field(default_factory=tuple)

    def all_configs(self) -> tuple[LLMConfig, ...]:
        """Return all configurations (primary + fallbacks) in order."""
        return (self.primary,) + self.fallbacks
