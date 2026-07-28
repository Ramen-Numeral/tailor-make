from app.infrastructure.llm.config import LLMConfig, LLMRoute
from app.infrastructure.llm.errors import LLMRateLimitError
from app.infrastructure.llm.fallback import LLMRoutedClient
from app.infrastructure.llm.providers.groq import _retry_after_seconds


class RateLimitedClient:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls < 3:
            raise LLMRateLimitError(
                "limited",
                retry_after_seconds=0.25,
            )
        return "done"


class LongRateLimitedClient:
    def invoke(self, prompt: str, **kwargs) -> str:
        raise LLMRateLimitError("limited", retry_after_seconds=1468)


class HealthyClient:
    def invoke(self, prompt: str, **kwargs) -> str:
        return "fallback"


def test_rate_limit_retries_with_exponential_minimum(monkeypatch) -> None:
    client = RateLimitedClient()
    waits = []
    route = LLMRoute(primary=LLMConfig(provider="groq", model="test"))

    monkeypatch.setattr(
        "app.infrastructure.llm.fallback.make_llm_client",
        lambda config, api_key=None: client,
    )
    monkeypatch.setattr("app.infrastructure.llm.fallback.sleep", waits.append)

    assert LLMRoutedClient(route).invoke("hello") == "done"
    assert client.calls == 3
    assert waits == [1.0, 2.0]


def test_retry_delay_is_read_from_groq_error_text() -> None:
    error = Exception("Please try again in 929.999999ms")
    assert _retry_after_seconds(error, str(error).lower()) == 0.929999999


def test_long_rate_limit_wait_fails_over_without_sleep(monkeypatch) -> None:
    waits = []
    route = LLMRoute(
        primary=LLMConfig(provider="groq", model="limited"),
        fallbacks=(LLMConfig(provider="groq", model="healthy"),),
    )
    clients = {
        "limited": LongRateLimitedClient(),
        "healthy": HealthyClient(),
    }
    monkeypatch.setattr(
        "app.infrastructure.llm.fallback.make_llm_client",
        lambda config, api_key=None: clients[config.model],
    )
    monkeypatch.setattr("app.infrastructure.llm.fallback.sleep", waits.append)

    assert LLMRoutedClient(route).invoke("hello") == "fallback"
    assert waits == []
