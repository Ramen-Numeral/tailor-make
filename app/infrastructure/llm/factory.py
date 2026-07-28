"""LLM client factory.

Constructs provider adapters from configuration and secrets.
This is the only module that imports provider-specific LangChain classes.
"""

from app.infrastructure.llm.config import LLMConfig
from app.infrastructure.llm.errors import LLMNonRetryableError
from app.infrastructure.llm.protocol import LLMClient


def make_llm_client(
    config: LLMConfig,
    api_key: str | None = None,
) -> LLMClient:
    """Construct an LLM client from configuration.

    Selects the appropriate provider adapter based on config.provider.

    Args:
        config: LLMConfig describing the ai_detection and parameters.
        api_key: API key for the provider (optional; may be read from environment).

    Returns:
        An LLMClient instance implementing the application protocol.

    Raises:
        LLMNonRetryableError: If the provider is unknown or unsupported.
    """
    provider = config.provider.lower()

    if provider == "groq":
        from app.infrastructure.llm.providers.groq import GroqAdapter

        return GroqAdapter(config=config, api_key=api_key)

    if provider == "ollama":
        from app.infrastructure.llm.providers.ollama import OllamaAdapter

        return OllamaAdapter(config=config)

    raise LLMNonRetryableError(
        f"Unknown LLM provider: {provider}. Supported: groq, ollama"
    )
