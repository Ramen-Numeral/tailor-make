"""Role-specific LLM routes."""

from app.infrastructure.llm import LLMConfig, LLMRoute


EXTRACTION_MODEL = "openai/gpt-oss-20b"
MATCHING_MODEL = "llama-3.3-70b-versatile"
WRITING_MODEL = "openai/gpt-oss-120b"
OPENAI_SMALL_FALLBACK = "openai/gpt-oss-20b"
LLAMA_SMALL_FALLBACK = "llama-3.1-8b-instant"
LLAMA_LARGE_FALLBACK = "llama-3.3-70b-versatile"


def groq_config(
    model: str,
    *,
    temperature: float,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> LLMConfig:
    return LLMConfig(
        provider="groq",
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )

RUBRIC_JUDGE_PRIMARY = groq_config(
    WRITING_MODEL,
    temperature=0.2,
    max_tokens=1200,
)

RUBRIC_JUDGE_FALLBACK = LLMConfig(
    provider="groq",
    model=OPENAI_SMALL_FALLBACK,
    temperature=0.2,
    timeout=30.0,
    max_tokens=1200,
)

judge_route = LLMRoute(
    primary=RUBRIC_JUDGE_PRIMARY,
    fallbacks=(
        RUBRIC_JUDGE_FALLBACK,
        groq_config(
            LLAMA_SMALL_FALLBACK,
            temperature=0.2,
            max_tokens=1200,
        ),
    ),
)

JOB_PARSER_PRIMARY = groq_config(
    EXTRACTION_MODEL,
    temperature=0.2,
    max_tokens=900,
)

job_parser_route = LLMRoute(
    primary=JOB_PARSER_PRIMARY,
    fallbacks=(
        groq_config(
            LLAMA_SMALL_FALLBACK,
            temperature=0.2,
            max_tokens=900,
        ),
    ),
)

RESUME_PARSER_PRIMARY = groq_config(
    EXTRACTION_MODEL,
    temperature=0,
    max_tokens=4000,
    timeout=20.0,
)

resume_parser_route = LLMRoute(
    primary=RESUME_PARSER_PRIMARY,
    fallbacks=(
        groq_config(
            LLAMA_LARGE_FALLBACK,
            temperature=0,
            max_tokens=4000,
            timeout=20.0,
        ),
    ),
)


CONTENT_MATCHER_PRIMARY = groq_config(
    MATCHING_MODEL,
    temperature=0,
)

content_matcher_route = LLMRoute(
    primary=CONTENT_MATCHER_PRIMARY,
    fallbacks=(
        groq_config(OPENAI_SMALL_FALLBACK, temperature=0),
    ),
)

EVIDENCE_JUDGE_PRIMARY = groq_config(
    MATCHING_MODEL,
    temperature=0,
    max_tokens=2400,
)

evidence_judge_route = LLMRoute(
    primary=EVIDENCE_JUDGE_PRIMARY,
    fallbacks=(
        groq_config(
            OPENAI_SMALL_FALLBACK,
            temperature=0,
            max_tokens=2400,
        ),
    ),
)

RESUME_WRITER_PRIMARY = groq_config(
    WRITING_MODEL,
    temperature=0,
)

resume_writer_route = LLMRoute(
    primary=RESUME_WRITER_PRIMARY,
    fallbacks=(
        groq_config(LLAMA_LARGE_FALLBACK, temperature=0),
    ),
)

RESUME_CRITIC_PRIMARY = groq_config(
    WRITING_MODEL,
    temperature=0.3,
)

resume_critic_route = LLMRoute(
    primary=RESUME_CRITIC_PRIMARY,
    fallbacks=(
        groq_config(LLAMA_LARGE_FALLBACK, temperature=0.3),
    ),
)

STYLE_ALIGNMENT_PRIMARY = groq_config(
    WRITING_MODEL,
    temperature=0.1,
)

style_alignment_route = LLMRoute(
    primary=STYLE_ALIGNMENT_PRIMARY,
    fallbacks=(
        groq_config(LLAMA_LARGE_FALLBACK, temperature=0.1),
    ),
)
