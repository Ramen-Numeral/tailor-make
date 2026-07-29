"""Stateful, bounded orchestration for explainable resume tailoring."""

import logging
from collections.abc import Callable

from app.bootstrap import get_keyword_embedder, get_llm_clients
from app.features.agent.page_fit import fit_resume_to_page_limit
from app.features.agent.positioning import build_positioning_brief, brief_instruction
from app.features.agent.hiring_quality import (
    evaluate_bullets,
    evaluate_recruiter_quality,
)
from app.features.agent.actions import actions_for_section
from app.features.agent.schema import (
    AgentBudget,
    TailoringRunResult,
    TailoringRunState,
)
from app.features.content_extractor.content_extractor import match_resume
from app.features.job_listing_parser.parse_listing import parse_listing
from app.features.keyword_evidence.planner import (
    apply_coverage_plan,
    build_coverage_plan,
)
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.keyword_evidence.scoring import score_coverage_plan
from app.features.resume_diff.differ import build_resume_diffs
from app.features.validator.schema import AgentTraceEvent, TraceCallback
from app.features.validator.validator import validate_resume_with_trace
from app.features.writer.writer import apply_resume_limits, global_resume_rewrite
from app.infrastructure.llm.errors import LLMError
from app.infrastructure.logging import get_llm_errors_logger
from app.resume_schema.resume_schema import (
    MutableResume,
    ProfessionalSummaryItem,
    Resume,
)
from config.resume.layout import LAYOUT_STYLE

logger = logging.getLogger(__name__)
CoverageCallback = Callable[[CoveragePlan], None]
HumanInputCallback = Callable[[CoveragePlan], str | None]


def tailor_resume_agent(
    resume: Resume,
    job_listing_text: str,
    *,
    special_instructions: str | None = None,
    budget: AgentBudget | None = None,
    maximum_pages: int = LAYOUT_STYLE.max_pages,
    trace_callback: TraceCallback | None = None,
    coverage_callback: CoverageCallback | None = None,
    human_input_callback: HumanInputCallback | None = None,
    fail_on_rewrite_error: bool = False,
    include_summary: bool = True,
) -> TailoringRunResult:
    """Execute the complete bounded workflow and return its state."""
    if not job_listing_text.strip():
        raise ValueError("job_listing_text cannot be empty")
    budget = budget or AgentBudget()
    events: list[AgentTraceEvent] = []

    _emit(
        events,
        AgentTraceEvent(
            event_type="run_started",
            title="Tailoring run started",
            summary="Preparing the job and candidate evidence.",
        ),
        trace_callback,
    )
    job_listing = parse_listing(
        job_listing_text,
        max_attempts=3,
        minimum_attempts=1,
    )
    _emit(
        events,
        AgentTraceEvent(
            event_type="job_parsed",
            title="Job listing parsed",
            summary=(
                f"Extracted {len(job_listing.requirements)} requirement(s)."
            ),
            observations=[
                requirement.text
                for requirement in job_listing.requirements
            ],
            job_requirements=job_listing.requirements,
        ),
        trace_callback,
    )

    embedder = _load_embedder()
    adjudicator = getattr(
        get_llm_clients(),
        "evidence_judge",
        None,
    )
    coverage_plan = build_coverage_plan(
        job_listing,
        resume,
        embedder=embedder,
        adjudicator=adjudicator,
    )
    initial_match_score = score_coverage_plan(
        coverage_plan,
        stage="initial",
    )
    _emit_match_score(
        events,
        initial_match_score,
        coverage_plan,
        trace_callback,
    )
    positioning_brief = build_positioning_brief(job_listing, coverage_plan)
    _emit(
        events,
        AgentTraceEvent(
            event_type="positioning_brief_created",
            title="Positioning and section plan created",
            summary=(
                f"Positioning the candidate for {positioning_brief.target_identity} "
                f"using {len(positioning_brief.primary_evidence)} primary "
                "evidence themes."
            ),
            observations=[
                *[
                    f"Lead with: {item}"
                    for item in positioning_brief.primary_evidence
                ],
                *[
                    f"Do not claim: {item}"
                    for item in positioning_brief.gaps
                ],
            ],
            action="Use this evidence hierarchy for drafting and ordering.",
            decision="preserve",
        ),
        trace_callback,
    )

    working_resume = _prepare_summary(resume, include_summary)
    selected = match_resume(job_listing, working_resume)
    _emit(
        events,
        AgentTraceEvent(
            event_type="content_selected",
            title="Candidate evidence selected",
            summary="Resume items are ordered from most to least relevant.",
        ),
        trace_callback,
    )

    selected = apply_coverage_plan(selected, coverage_plan)
    if coverage_callback is not None:
        coverage_callback(coverage_plan)
    _emit(
        events,
        AgentTraceEvent(
            event_type="coverage_plan_created",
            title="Supported keyword plan created",
            summary=(
                f"{len(coverage_plan.assignments)} section assignment(s), "
                f"{len(coverage_plan.unsupported_requirements)} unsupported "
                "requirement(s)."
            ),
            observations=[
                (
                    f"{match.requirement_text}: {match.support}"
                    + (
                        f" — {match.adjudication_reason}"
                        if match.adjudication_reason
                        else ""
                    )
                )
                for match in coverage_plan.requirement_matches
            ],
            coverage_plan=coverage_plan,
        ),
        trace_callback,
    )
    supplemental_evidence: str | None = None
    if (
        human_input_callback is not None
        and any(
            match.support != "supported"
            for match in coverage_plan.requirement_matches
        )
    ):
        missing = [
            match
            for match in coverage_plan.requirement_matches
            if match.support != "supported"
        ]
        _emit(
            events,
            AgentTraceEvent(
                event_type="human_input_requested",
                title="Candidate input requested",
                summary=(
                    "The initial profile does not support these requirements. "
                    "Add omitted experience or continue without adding evidence."
                ),
                observations=[
                    (
                        f"{match.requirement_text} — do you have a concrete "
                        "example, scope, tool, collaborator, or outcome that "
                        "was omitted from the profile?"
                    )
                    for match in missing
                ],
                action="Waiting for candidate-supplied factual notes.",
            ),
            trace_callback,
        )
        supplemental_evidence = (
            human_input_callback(coverage_plan) or ""
        ).strip() or None
        _emit(
            events,
            AgentTraceEvent(
                event_type="human_input_received",
                title="Candidate input received",
                summary=(
                    "Supplemental candidate evidence was added to the rewrite plan."
                    if supplemental_evidence
                    else "The candidate continued without supplemental evidence."
                ),
                observations=(
                    [supplemental_evidence] if supplemental_evidence else []
                ),
                action="Resume the evidence-grounded rewrite plan.",
            ),
            trace_callback,
        )

    global_rewrite_completed = False
    try:
        rewrite_instructions = _rewrite_instructions(
            special_instructions,
            supplemental_evidence,
            positioning_brief=positioning_brief,
        )
        rewritten = global_resume_rewrite(
            job_listing,
            selected,
            special_instructions=rewrite_instructions,
        )
        global_rewrite_completed = True
    except LLMError:
        if fail_on_rewrite_error:
            raise
        get_llm_errors_logger().exception(
            "global_rewrite_failed using_selected_resume=true"
        )
        rewritten = selected
    current = apply_resume_limits(rewritten, reference=selected)
    _emit(
        events,
        AgentTraceEvent(
            event_type="draft_generated",
            title=(
                "Global tailored draft generated"
                if global_rewrite_completed
                else "Global rewrite unavailable"
            ),
            summary=(
                "The global writer completed. Starting section-specific "
                "factual and writing checks."
                if global_rewrite_completed
                else "The selected factual resume was preserved. Starting "
                "section-specific factual and writing checks."
            ),
        ),
        trace_callback,
    )

    validation = validate_resume_with_trace(
        job_listing,
        current,
        max_attempts=budget.max_section_rewrites,
        total_rewrite_budget=budget.max_total_rewrites,
        source_resume=selected,
        supplemental_evidence=supplemental_evidence,
        trace_callback=trace_callback,
    )
    events.extend(validation.events)
    current = _as_mutable(validation.resume)
    total_rewrites = sum(
        max(0, len(section.attempts) - 1)
        for section in validation.sections
    )

    _emit(
        events,
        AgentTraceEvent(
            event_type="global_validation_started",
            title="Final rendered-page check started",
            summary=f"Maximum allowed pages: {maximum_pages}.",
        ),
        trace_callback,
    )
    fitted, page_check = fit_resume_to_page_limit(
        current,
        maximum_pages=maximum_pages,
        max_trims=budget.max_page_trims,
    )
    for action in page_check.trim_actions:
        _emit(
            events,
            AgentTraceEvent(
                event_type="content_trimmed",
                section=action.section,
                title="Least-relevant content trimmed",
                summary=f"Removed {action.field}: {action.removed}",
                action=action.reason,
                decision="trim",
            ),
            trace_callback,
        )
    _emit(
        events,
        AgentTraceEvent(
            event_type="page_count_completed",
            title="Rendered-page check completed",
            summary=(
                f"Rendered pages: {page_check.final_pages}/"
                f"{page_check.maximum_pages}."
            ),
            decision="accept" if page_check.passed else "preserve",
            decision_reasons=[
                (
                    "The final PDF fits the configured page limit."
                    if page_check.passed
                    else "No additional safe deterministic trim was available."
                )
            ],
        ),
        trace_callback,
    )

    has_warnings = (
        validation.status != "completed"
        or not page_check.passed
    )
    status = "completed_with_warnings" if has_warnings else "completed"
    diffs = build_resume_diffs(
        resume,
        selected,
        fitted,
        page_trim_actions=page_check.trim_actions,
    )
    final_coverage_plan = build_coverage_plan(
        job_listing,
        fitted,
        embedder=embedder,
        adjudicator=adjudicator,
    )
    final_match_score = score_coverage_plan(
        final_coverage_plan,
        stage="final",
    )
    _emit_match_score(
        events,
        final_match_score,
        final_coverage_plan,
        trace_callback,
    )
    bullet_quality = evaluate_bullets(fitted)
    recruiter_evaluation = evaluate_recruiter_quality(
        fitted,
        positioning_brief,
        critic=getattr(get_llm_clients(), "resume_critic", None),
    )
    if not recruiter_evaluation.ready:
        has_warnings = True
        status = "completed_with_warnings"
    _emit(
        events,
        AgentTraceEvent(
            event_type="recruiter_evaluation_completed",
            title="Final recruiter-quality check completed",
            summary=(
                f"{'Ready for review' if recruiter_evaluation.ready else 'Needs review'}: "
                f"{len(recruiter_evaluation.strengths)} strength(s), "
                f"{len(recruiter_evaluation.gaps)} remaining gap(s)."
            ),
            observations=[
                recruiter_evaluation.summary,
                *recruiter_evaluation.recommendations,
            ],
            decision=(
                "accept" if recruiter_evaluation.ready
                else "accept_with_warnings"
            ),
        ),
        trace_callback,
    )
    _emit(
        events,
        AgentTraceEvent(
            event_type="run_completed",
            title="Tailoring run completed",
            summary=(
                (
                    "Global draft completed; "
                    if global_rewrite_completed
                    else "Global rewrite unavailable; selected content "
                    "preserved; "
                )
                + (
                    f"{total_rewrites} corrective section "
                    f"{'retry' if total_rewrites == 1 else 'retries'}; "
                )
                + (
                    f"{len(page_check.trim_actions)} page-fit "
                    f"{'trim' if len(page_check.trim_actions) == 1 else 'trims'}."
                )
            ),
            decision="accept_with_warnings" if has_warnings else "accept",
        ),
        trace_callback,
    )
    state = TailoringRunState(
        source_resume=resume,
        job_listing=job_listing,
        evidence_plan=coverage_plan,
        positioning_brief=positioning_brief,
        initial_match_score=initial_match_score,
        final_match_score=final_match_score,
        selected_resume=selected,
        current_resume=fitted,
        section_results=validation.sections,
        actions=[
            action
            for section in validation.sections
            for action in actions_for_section(section)
        ],
        diffs=diffs,
        events=events,
        total_rewrites=total_rewrites,
        supplemental_evidence=supplemental_evidence,
        bullet_quality=bullet_quality,
        recruiter_evaluation=recruiter_evaluation,
        status=status,
    )
    return TailoringRunResult(
        resume=fitted,
        state=state,
        page_check=page_check,
        diffs=diffs,
    )


def _rewrite_instructions(
    instructions: str | None,
    supplemental_evidence: str | None,
    positioning_brief=None,
) -> str | None:
    parts = [instructions.strip()] if instructions and instructions.strip() else []
    if supplemental_evidence:
        parts.append(
            "Candidate-supplied supplemental evidence (treat as factual, but "
            "do not infer beyond the exact notes):\n"
            f"{supplemental_evidence}"
        )
    if positioning_brief is not None:
        parts.append(brief_instruction(positioning_brief))
    return "\n\n".join(parts) or None


def _load_embedder():
    try:
        return get_keyword_embedder()
    except Exception as error:
        logger.warning(
            "keyword_embedder_unavailable using_exact_and_bm25=true error=%s",
            error,
        )
        return None


def _as_mutable(resume: Resume) -> MutableResume:
    return (
        resume
        if isinstance(resume, MutableResume)
        else MutableResume.model_validate(resume.model_dump())
    )


def _emit(
    events: list[AgentTraceEvent],
    event: AgentTraceEvent,
    callback: TraceCallback | None,
) -> None:
    events.append(event)
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        logger.exception(
            "agent_trace_callback_failed event_type=%s",
            event.event_type,
        )


def _emit_match_score(
    events: list[AgentTraceEvent],
    match_score,
    coverage_plan: CoveragePlan,
    callback: TraceCallback | None,
) -> None:
    label = "Initial profile" if match_score.stage == "initial" else "Final resume"
    _emit(
        events,
        AgentTraceEvent(
            event_type="match_score_completed",
            title=f"{label} match score",
            summary=(
                f"{match_score.score}/100 weighted job match: "
                f"{match_score.supported} supported, "
                f"{match_score.partial} partial, "
                f"{match_score.unsupported} unsupported."
            ),
            score=match_score.score,
            match_score=match_score,
            coverage_plan=coverage_plan,
            observations=[
                *(
                    [
                        (
                            f"Evidence coverage: "
                            f"{match_score.evidence_coverage_score}/100"
                        ),
                        (
                            f"Holistic fit: "
                            f"{match_score.holistic_fit_score}/100"
                        ),
                    ]
                    if match_score.holistic_fit_score is not None
                    else [
                        (
                            "Holistic fit judge unavailable; this score uses "
                            "evidence coverage only."
                        )
                    ]
                ),
                *match_score.rubric_observations,
                *[
                    (
                        f"{item.kind}: {item.score}/100 "
                        f"({item.supported} supported, {item.partial} partial)"
                    )
                    for item in match_score.breakdown
                ],
                *[
                    f"Gap: {gap}"
                    for gap in match_score.largest_gaps
                ],
            ],
            decision="preserve",
        ),
        callback,
    )


def _prepare_summary(
    resume: Resume,
    include_summary: bool,
) -> MutableResume:
    working = MutableResume.model_validate(resume.model_dump())
    if not include_summary:
        working.summary = working.summary.model_copy(
            update={"items": []},
            deep=True,
        )
    elif not working.summary.items:
        working.summary = working.summary.model_copy(
            update={"items": [ProfessionalSummaryItem(content=None)]},
            deep=True,
        )
    return working
