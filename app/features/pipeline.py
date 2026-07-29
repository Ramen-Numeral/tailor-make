"""End-to-end MVP resume tailoring pipeline."""

import logging
from collections.abc import Callable
from typing import Protocol

from app.bootstrap import get_keyword_embedder, get_llm_clients
from app.features.content_extractor.content_extractor import match_resume
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.job_listing_parser.parse_listing import parse_listing
from app.features.keyword_evidence import CoveragePlan
from app.features.keyword_evidence.planner import (
    apply_coverage_plan,
    build_coverage_plan,
)
from app.features.writer.writer import apply_resume_limits, global_resume_rewrite
from app.infrastructure.llm.errors import LLMError
from app.infrastructure.logging import get_llm_errors_logger
from app.resume_schema.resume_schema import MutableResume, Resume

logger = logging.getLogger(__name__)


class ResumeQualityHook(Protocol):
    """Extension point for optional AI detection or other quality checks."""

    def __call__(
        self,
        job_listing: JobListing,
        resume: MutableResume,
    ) -> MutableResume: ...


def tailor_resume(
    resume: Resume,
    job_listing_text: str,
    *,
    special_instructions: str | None = None,
    quality_hook: ResumeQualityHook | None = None,
    coverage_hook: Callable[[CoveragePlan], None] | None = None,
    fail_on_rewrite_error: bool = False,
) -> MutableResume:
    """Parse, globally select, globally rewrite, constrain, and return a resume.

    The MVP performs exactly one feature-level model call for each of parsing,
    candidate selection, and rewriting. Provider fallbacks remain internal to
    each call. If rewriting fails, the selected factual resume is returned by
    default so rendering can still complete.
    """
    if not job_listing_text.strip():
        raise ValueError("job_listing_text cannot be empty")

    job_listing = parse_listing(
        job_listing_text,
        max_attempts=3,
        minimum_attempts=2,
    )
    selected = match_resume(job_listing, resume)
    try:
        embedder = get_keyword_embedder()
    except Exception as error:
        logger.warning(
            "keyword_embedder_unavailable using_exact_and_bm25=true error=%s",
            error,
        )
        embedder = None
    try:
        coverage_plan = build_coverage_plan(
            job_listing,
            selected,
            embedder=embedder,
            adjudicator=getattr(
                get_llm_clients(),
                "evidence_judge",
                None,
            ),
        )
    except Exception:
        logger.exception(
            "keyword_coverage_planning_failed using_dynamic_keywords=false"
        )
        coverage_plan = CoveragePlan()
    selected = apply_coverage_plan(selected, coverage_plan)
    if coverage_hook is not None:
        coverage_hook(coverage_plan)

    try:
        rewritten = global_resume_rewrite(
            job_listing,
            selected,
            special_instructions=special_instructions,
        )
    except LLMError:
        if fail_on_rewrite_error:
            raise
        get_llm_errors_logger().exception(
            "global_rewrite_failed using_selected_resume=true"
        )
        rewritten = selected

    constrained = apply_resume_limits(rewritten, reference=selected)
    return (
        quality_hook(job_listing, constrained)
        if quality_hook is not None
        else constrained
    )
