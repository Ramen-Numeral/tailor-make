
"""Environment-backed non-filesystem runtime settings."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeConfig(BaseSettings):
    """Runtime values that may vary by machine or deployment."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Secrets
    groq_api_key: SecretStr | None = None

    # Environment
    app_env: str = "development"
    device: str = "cpu"
    rand_seed: int = Field(default=42, ge=0)
    log_level: str = "INFO"
    ai_detection_enabled: bool = True
    keyword_embeddings_enabled: bool = True
    keyword_embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    keyword_embedding_revision: str = (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )

    def require_groq_api_key(self) -> str:
        """Return the Groq key for an operation that actually needs Groq."""
        if self.groq_api_key is None:
            raise RuntimeError(
                "GROQ_API_KEY is required for Groq-backed LLM operations"
            )
        return self.groq_api_key.get_secret_value()
