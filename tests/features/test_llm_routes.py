from config.llm import (
    EXTRACTION_MODEL,
    LLAMA_LARGE_FALLBACK,
    LLAMA_SMALL_FALLBACK,
    MATCHING_MODEL,
    OPENAI_SMALL_FALLBACK,
    WRITING_MODEL,
    content_matcher_route,
    evidence_judge_route,
    job_parser_route,
    judge_route,
    resume_critic_route,
    resume_parser_route,
    resume_writer_route,
    style_alignment_route,
)


def test_mvp_routes_use_role_specific_models() -> None:
    assert job_parser_route.primary.model == EXTRACTION_MODEL
    assert job_parser_route.fallbacks[0].model == LLAMA_SMALL_FALLBACK
    assert resume_parser_route.primary.model == EXTRACTION_MODEL
    assert resume_parser_route.fallbacks[0].model == LLAMA_LARGE_FALLBACK

    assert content_matcher_route.primary.model == MATCHING_MODEL
    assert content_matcher_route.fallbacks[0].model == OPENAI_SMALL_FALLBACK
    assert evidence_judge_route.primary.model == MATCHING_MODEL
    assert evidence_judge_route.fallbacks[0].model == OPENAI_SMALL_FALLBACK

    assert resume_writer_route.primary.model == WRITING_MODEL
    assert resume_writer_route.fallbacks[0].model == LLAMA_LARGE_FALLBACK


def test_writing_support_routes_use_large_fallback() -> None:
    for route in (resume_critic_route, style_alignment_route):
        assert route.primary.model == WRITING_MODEL
        assert route.fallbacks[0].model == LLAMA_LARGE_FALLBACK


def test_judge_uses_distinct_fallback_models() -> None:
    assert judge_route.primary.model == WRITING_MODEL
    assert judge_route.fallbacks[0].model == OPENAI_SMALL_FALLBACK
    assert judge_route.fallbacks[1].model == LLAMA_SMALL_FALLBACK
