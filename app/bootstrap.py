"""Application dependency assembly."""

import logging
from functools import cached_property, lru_cache

from app.infrastructure.llm import LLMRoutedClient
from app.infrastructure.runtime import prepare_runtime_directories
from config.llm import (
    job_parser_route,
    judge_route,
    content_matcher_route,
    evidence_judge_route,
    resume_critic_route,
    resume_parser_route,
    resume_writer_route,
    style_alignment_route,
)
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClients:
    """Lazy container for application LLM clients."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _build_client(self, route) -> LLMRoutedClient:
        key = self._settings.runtime.groq_api_key
        return LLMRoutedClient(
            route,
            api_key=key.get_secret_value() if key is not None else None,
        )

    @cached_property
    def judge(self) -> LLMRoutedClient:
        return self._build_client(judge_route)

    @cached_property
    def job_parser(self) -> LLMRoutedClient:
        return self._build_client(job_parser_route)

    @cached_property
    def resume_parser(self) -> LLMRoutedClient:
        return self._build_client(resume_parser_route)

    @cached_property
    def content_matcher(self) -> LLMRoutedClient:
        return self._build_client(content_matcher_route)

    @cached_property
    def evidence_judge(self) -> LLMRoutedClient:
        return self._build_client(evidence_judge_route)

    @cached_property
    def resume_writer(self) -> LLMRoutedClient:
        return self._build_client(resume_writer_route)

    @cached_property
    def resume_critic(self) -> LLMRoutedClient:
        return self._build_client(resume_critic_route)

    @cached_property
    def style_alignment(self) -> LLMRoutedClient:
        return self._build_client(style_alignment_route)


class Application:
    """Root application dependency container."""

    def __init__(self) -> None:
        self.settings = get_settings()

        prepare_runtime_directories(self.settings.io)

        self.llm_clients = LLMClients(self.settings)
        logger.info("Application dependencies initialized")


@lru_cache(maxsize=1)
def get_application() -> Application:
    return Application()


def get_llm_clients() -> LLMClients:
    return get_application().llm_clients


@lru_cache(maxsize=1)
def get_ai_detector():
    """Load and cache the completed AI-detection ensemble."""
    from app.features.ai_detection.detector import AIDetector
    from app.infrastructure.ai_detection import (
        load_cb_model,
        load_dbert_model,
        load_rubric_regressor,
        load_tfidf_model,
    )

    catboost_model = load_cb_model()
    distilbert_model, distilbert_tokenizer = load_dbert_model()
    tfidf_svm_model = load_tfidf_model()
    rubric_regression_model, _ = load_rubric_regressor()

    return AIDetector(
        catboost_model=catboost_model,
        distilbert_model=distilbert_model,
        distilbert_tokenizer=distilbert_tokenizer,
        tfidf_svm_model=tfidf_svm_model,
        rubric_regression_model=rubric_regression_model,
    )


@lru_cache(maxsize=1)
def get_keyword_embedder():
    """Load and cache the free local sentence-embedding model."""
    from sentence_transformers import SentenceTransformer

    runtime = get_settings().runtime
    if not runtime.keyword_embeddings_enabled:
        return None
    return SentenceTransformer(
        runtime.keyword_embedding_model,
        device=runtime.device,
        revision=runtime.keyword_embedding_revision,
    )


def reset_for_tests() -> None:
    from app.infrastructure.cache import clear_stage_cache

    clear_stage_cache()
    get_application.cache_clear()
    get_ai_detector.cache_clear()
    get_keyword_embedder.cache_clear()
    get_settings.cache_clear()
