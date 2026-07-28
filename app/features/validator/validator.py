"""Iteratively rewrite resume sections using AI-detection feedback."""

import logging

from app.features.ai_detection.config import ai_detection_config
from app.features.ai_detection.schema import (
    AIDetectionResult,
    ComponentDelta,
    CounterfactualComparison,
    EvaluationPolicy,
)
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.validator.constraints import evaluate_constraints
from app.features.validator.decision import attempt_rank, evaluate_attempt
from app.features.validator.factual import evaluate_factual_integrity
from app.features.validator.policy import policy_for_section
from app.features.validator.schema import (
    AgentTraceEvent,
    AttemptEvaluation,
    ResumeValidationResult,
    SectionValidationResult,
    TraceCallback,
)
from app.features.writer.writer import write_section
from app.resume_schema.resume_schema import (
    MutableResume,
    Section,
    WRITABLE_SECTION_FIELDS,
)

logger = logging.getLogger(__name__)


def validate_resume(
    job_listing: JobListing,
    resume: MutableResume,
    max_attempts: int = ai_detection_config.rewrite_attempts,
    max_ai_probability: float = ai_detection_config.rewrite_threshold,
    policy: EvaluationPolicy | None = None,
) -> MutableResume:
    """Compatibility API returning only the final validated resume."""
    return validate_resume_with_trace(
        job_listing=job_listing,
        resume=resume,
        max_attempts=max_attempts,
        max_ai_probability=max_ai_probability,
        policy=policy,
    ).resume


def validate_resume_with_trace(
    job_listing: JobListing,
    resume: MutableResume,
    max_attempts: int = ai_detection_config.rewrite_attempts,
    max_ai_probability: float = ai_detection_config.rewrite_threshold,
    policy: EvaluationPolicy | None = None,
    trace_callback: TraceCallback | None = None,
    source_resume: MutableResume | None = None,
    total_rewrite_budget: int | None = None,
    supplemental_evidence: str | None = None,
) -> ResumeValidationResult:
    """Validate the resume and retain a frontend-safe decision audit trail."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    from app.bootstrap import get_ai_detector

    events: list[AgentTraceEvent] = []
    try:
        detector = get_ai_detector()
    except Exception as error:
        logger.exception(
            "ai_detector_initialization_failed using_rewritten_resume=true"
        )
        event = AgentTraceEvent(
            event_type="section_failed",
            title="Writing evaluation unavailable",
            summary="The rewritten resume was preserved without evaluation.",
            action="Preserve the rewritten resume.",
            decision="preserve",
            decision_reasons=[str(error)],
        )
        _emit(events, event, trace_callback)
        return ResumeValidationResult(
            resume=resume,
            events=events,
            status="detector_unavailable",
        )

    section_results: list[SectionValidationResult] = []
    completed_with_warnings = False
    remaining_rewrites = total_rewrite_budget
    for field_name in WRITABLE_SECTION_FIELDS:
        section = getattr(resume, field_name)

        if section is None or not section.items:
            event = AgentTraceEvent(
                event_type="section_skipped",
                section=field_name,
                title="Section skipped",
                summary="The section was empty or unavailable.",
                decision="skip",
            )
            _emit(events, event, trace_callback)
            continue

        try:
            section_attempts = (
                max_attempts
                if remaining_rewrites is None
                else min(max_attempts, remaining_rewrites)
            )
            result = validate_section_with_trace(
                job_listing=job_listing,
                section=section,
                detector=detector,
                max_attempts=section_attempts,
                max_ai_probability=max_ai_probability,
                policy=policy,
                trace_callback=trace_callback,
                section_name=field_name,
                source_section=(
                    getattr(source_resume, field_name, None)
                    if source_resume is not None
                    else None
                ),
                supplemental_evidence=supplemental_evidence,
            )
        except Exception as error:
            logger.exception(
                "ai_section_validation_failed section=%s "
                "using_rewritten_section=true",
                field_name,
            )
            event = AgentTraceEvent(
                event_type="section_failed",
                section=field_name,
                title="Section evaluation failed",
                summary="The current section was preserved.",
                action="Preserve the current section.",
                decision="preserve",
                decision_reasons=[str(error)],
            )
            _emit(events, event, trace_callback)
            completed_with_warnings = True
            continue

        setattr(resume, field_name, result.final_section)
        section_results.append(result)
        events.extend(result.events)
        if remaining_rewrites is not None:
            remaining_rewrites -= max(0, len(result.attempts) - 1)
        if result.status in {
            "accepted_with_warnings",
            "best_attempt_selected",
        }:
            completed_with_warnings = True

    event = AgentTraceEvent(
        event_type="workflow_completed",
        title="Resume evaluation completed",
        summary=f"Evaluated {len(section_results)} writable section(s).",
        decision=(
            "accept_with_warnings"
            if completed_with_warnings
            else "accept"
        ),
    )
    _emit(events, event, trace_callback)
    return ResumeValidationResult(
        resume=resume,
        sections=section_results,
        events=events,
        status=(
            "completed_with_warnings"
            if completed_with_warnings
            else "completed"
        ),
    )


def validate_section(
    job_listing: JobListing,
    section: Section,
    detector,
    max_attempts: int = ai_detection_config.rewrite_attempts,
    max_ai_probability: float = ai_detection_config.rewrite_threshold,
    policy: EvaluationPolicy | None = None,
) -> Section:
    """Compatibility API returning only the final validated section."""
    return validate_section_with_trace(
        job_listing=job_listing,
        section=section,
        detector=detector,
        max_attempts=max_attempts,
        max_ai_probability=max_ai_probability,
        policy=policy,
    ).final_section


def validate_section_with_trace(
    job_listing: JobListing,
    section: Section,
    detector,
    max_attempts: int = ai_detection_config.rewrite_attempts,
    max_ai_probability: float = ai_detection_config.rewrite_threshold,
    policy: EvaluationPolicy | None = None,
    trace_callback: TraceCallback | None = None,
    section_name: str | None = None,
    source_section: Section | None = None,
    supplemental_evidence: str | None = None,
) -> SectionValidationResult:
    """Validate one section while retaining and streaming every decision."""
    if max_attempts < 0:
        raise ValueError("max_attempts cannot be negative")

    original = section.model_copy(deep=True)
    factual_source = (
        source_section.model_copy(deep=True)
        if source_section is not None
        else None
    )
    name = section_name or section.heading
    events: list[AgentTraceEvent] = []
    original_text = section_to_text(original)
    if not original_text:
        event = AgentTraceEvent(
            event_type="section_skipped",
            section=name,
            title="Section skipped",
            summary="The section contains no writable text.",
            decision="skip",
        )
        _emit(events, event, trace_callback)
        return SectionValidationResult(
            section_name=name,
            original_section=original,
            final_section=original,
            events=events,
            status="unchanged",
        )

    active_policy = policy or policy_for_section(
        name,
        ai_likeness_retry_threshold=max_ai_probability,
    )
    detection = _classify(detector, original_text, active_policy)
    constraints = [
        *evaluate_constraints(original),
        *(
            evaluate_factual_integrity(
                factual_source,
                original,
                supplemental_evidence=supplemental_evidence,
            )
            if factual_source is not None
            else []
        ),
    ]
    decision = evaluate_attempt(
        detection,
        constraints,
        active_policy,
    )
    initial = AttemptEvaluation(
        attempt=0,
        text=original_text,
        detection=detection,
        constraints=constraints,
        decision=decision,
    )
    attempts = [initial]
    _emit_evaluation(events, name, initial, trace_callback)
    if decision.outcome != "retry":
        selected = initial.model_copy(update={"selected": True})
        attempts[0] = selected
        _emit_acceptance(events, name, selected, trace_callback)
        return SectionValidationResult(
            section_name=name,
            original_section=original,
            final_section=original,
            attempts=attempts,
            events=events,
            status=(
                "accepted_with_warnings"
                if decision.outcome == "accept_with_warnings"
                else "accepted"
            ),
        )

    best_section = original
    best_index = 0
    best_rank = attempt_rank(decision, detection)
    rewrite_base = original
    feedback = build_retry_feedback(
        decision.retry_instructions or detection.feedback
    )
    previous_failure_signature = tuple(sorted(decision.hard_failures))
    repeated_failure_count = 0
    no_improvement_count = 0
    stop_reason: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        event = AgentTraceEvent(
            event_type="rewrite_started",
            section=name,
            attempt=attempt_number,
            title=f"Rewrite attempt {attempt_number} started",
            summary="Applying targeted guidance from the failed checks.",
            observations=decision.hard_failures,
            action=feedback,
            decision="retry",
            decision_reasons=decision.reasons,
        )
        _emit(events, event, trace_callback)
        target_item_indices = (
            sorted(
                {
                    check.item_index
                    for check in constraints
                    if not check.passed and check.item_index is not None
                }
            )
            if (
                decision.failed_constraint_count > 0
                and decision.failed_rubric_count == 0
                and not decision.ai_likeness_failed
            )
            else None
        )
        candidate = write_section(
            job_listing=job_listing,
            section=rewrite_base,
            special_instructions=feedback,
            target_item_indices=target_item_indices,
        )

        detection = _classify(
            detector,
            section_to_text(candidate),
            active_policy,
        )
        constraints = [
            *evaluate_constraints(candidate),
            *(
                evaluate_factual_integrity(
                    factual_source,
                    candidate,
                    supplemental_evidence=supplemental_evidence,
                )
                if factual_source is not None
                else []
            ),
        ]
        decision = evaluate_attempt(
            detection,
            constraints,
            active_policy,
        )
        evaluated = AttemptEvaluation(
            attempt=attempt_number,
            text=section_to_text(candidate),
            detection=detection,
            constraints=constraints,
            decision=decision,
            counterfactual=_counterfactual(
                attempts[-1].detection,
                detection,
            ),
        )
        attempts.append(evaluated)
        rewrite_base = candidate
        _emit_evaluation(events, name, evaluated, trace_callback)
        rank = attempt_rank(decision, detection)
        if rank < best_rank:
            best_section = candidate
            best_index = attempt_number
            best_rank = rank
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if decision.outcome != "retry":
            selected = evaluated.model_copy(update={"selected": True})
            attempts[-1] = selected
            _emit_acceptance(events, name, selected, trace_callback)
            return SectionValidationResult(
                section_name=name,
                original_section=original,
                final_section=candidate,
                attempts=attempts,
                events=events,
                status=(
                    "accepted_with_warnings"
                    if decision.outcome == "accept_with_warnings"
                    else "accepted"
                ),
            )

        event = AgentTraceEvent(
            event_type="attempt_rejected",
            section=name,
            attempt=attempt_number,
            title=f"Rewrite attempt {attempt_number} needs another pass",
            summary="One or more blocking rules still failed.",
            observations=decision.hard_failures,
            action="Retry with updated targeted guidance.",
            decision="retry",
            decision_reasons=decision.reasons,
            evaluation=evaluated,
        )
        _emit(events, event, trace_callback)
        failure_signature = tuple(sorted(decision.hard_failures))
        repeated_failure_count = (
            repeated_failure_count + 1
            if failure_signature == previous_failure_signature
            else 0
        )
        previous_failure_signature = failure_signature
        previous_attempt = attempts[-2]
        if evaluated.text == previous_attempt.text:
            stop_reason = "The rewrite produced no textual change."
        elif (
            detection.ai_probability
            > previous_attempt.detection.ai_probability
            and rank >= attempt_rank(
                previous_attempt.decision,
                previous_attempt.detection,
            )
        ):
            stop_reason = (
                "The rewrite increased AI-likeness without improving the "
                "deterministic constraint and rubric ranking."
            )
        elif repeated_failure_count >= 2:
            stop_reason = (
                "The same blocking failures remained across two rewrites."
            )
        elif no_improvement_count >= 2:
            stop_reason = "Two consecutive rewrites did not improve the ranking."
        if stop_reason is not None:
            break

        best_probability = attempts[best_index].detection.ai_probability
        regression_guidance = []
        if detection.ai_probability > best_probability:
            component_scores = ", ".join(
                f"{component.model_name}={component.ai_probability:.3f}"
                for component in detection.components
                if component.available
            )
            regression_guidance.append(
                "The previous rewrite regressed ensemble AI-likeness from "
                f"{best_probability:.3f} to {detection.ai_probability:.3f}. "
                "Correct the flagged patterns without making the prose more "
                "uniform, generic, polished, or formulaic."
            )
            if component_scores:
                regression_guidance.append(
                    f"Previous component scores: {component_scores}."
                )
        feedback = build_retry_feedback(
            [
                *regression_guidance,
                *(decision.retry_instructions or detection.feedback),
            ]
        )

    selected = attempts[best_index].model_copy(update={"selected": True})
    attempts[best_index] = selected
    event = AgentTraceEvent(
        event_type="best_attempt_selected",
        section=name,
        attempt=best_index,
        title="Best available attempt selected",
        summary=(
            (
                f"Rewriting stopped early: {stop_reason} "
                "The strongest available version was preserved."
            )
            if stop_reason
            else (
                "The retry limit was reached, so the deterministic ranking "
                "selected the strongest available version."
            )
        ),
        observations=selected.decision.warnings,
        action="Preserve the highest-ranked attempt.",
        decision="preserve",
        decision_reasons=[
            "Fewest required-constraint failures.",
            "Then prefer attempts below the AI-likeness blocking threshold.",
            "Then fewest rubric failures and strongest rubric scores.",
            "Then lowest AI-likeness as the final tie-breaker.",
        ],
        evaluation=selected,
    )
    _emit(events, event, trace_callback)
    return SectionValidationResult(
        section_name=name,
        original_section=original,
        final_section=best_section,
        attempts=attempts,
        events=events,
        status="best_attempt_selected",
    )


def _emit_evaluation(
    events: list[AgentTraceEvent],
    section_name: str,
    evaluation: AttemptEvaluation,
    callback: TraceCallback | None,
) -> None:
    decision = evaluation.decision
    event = AgentTraceEvent(
        event_type="evaluation_completed",
        section=section_name,
        attempt=evaluation.attempt,
        title=f"Attempt {evaluation.attempt} evaluated",
        summary=(
            f"{len(decision.passed_checks)} checks passed, "
            f"{len(decision.hard_failures)} blocking rules failed, and "
            f"{len(decision.warnings)} warnings remain."
        ),
        observations=[
            *decision.hard_failures,
            *decision.warnings,
        ],
        action=(
            "Retry with targeted guidance."
            if decision.outcome == "retry"
            else "Accept this version."
        ),
        decision=decision.outcome,
        decision_reasons=decision.reasons,
        evaluation=evaluation,
    )
    _emit(events, event, callback)


def _classify(
    detector,
    text: str,
    policy: EvaluationPolicy,
) -> AIDetectionResult:
    if policy.evaluate_writing:
        return detector.classify(text)
    return AIDetectionResult(
        ai_probability=0.0,
        threshold=1.0,
        feedback=[],
        scoring_status="skipped",
        skipped_reason=(
            "Prose scoring was skipped because this section contains "
            "structured skills rather than prose."
        ),
    )


def _counterfactual(
    before: AIDetectionResult,
    after: AIDetectionResult,
) -> CounterfactualComparison:
    before_components = {
        component.model_name: component
        for component in before.components
        if component.available
    }
    deltas = [
        ComponentDelta(
            model_name=component.model_name,
            before=before_components[component.model_name].ai_probability,
            after=component.ai_probability,
            delta=(
                component.ai_probability
                - before_components[component.model_name].ai_probability
            ),
        )
        for component in after.components
        if component.available and component.model_name in before_components
    ]
    return CounterfactualComparison(
        before_probability=before.ai_probability,
        after_probability=after.ai_probability,
        delta=after.ai_probability - before.ai_probability,
        components=deltas,
    )


def _emit_acceptance(
    events: list[AgentTraceEvent],
    section_name: str,
    evaluation: AttemptEvaluation,
    callback: TraceCallback | None,
) -> None:
    decision = evaluation.decision
    event = AgentTraceEvent(
        event_type="attempt_accepted",
        section=section_name,
        attempt=evaluation.attempt,
        title=f"Attempt {evaluation.attempt} accepted",
        summary=(
            "All blocking rules passed."
            if decision.outcome == "accept"
            else "All blocking rules passed with non-blocking warnings."
        ),
        observations=[
            *decision.passed_checks,
            *decision.warnings,
        ],
        action="Use this version in the final resume.",
        decision=decision.outcome,
        decision_reasons=decision.reasons,
        evaluation=evaluation,
    )
    _emit(events, event, callback)


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
            "ai_trace_callback_failed event_type=%s section=%s",
            event.event_type,
            event.section,
        )


def section_to_text(section: Section) -> str:
    """Extract only fields exposed by each item's WritableForm."""
    text: list[str] = []

    for item in section.items:
        for field_name in type(item).WritableForm.model_fields:
            value = getattr(item, field_name, None)

            if isinstance(value, str) and value.strip():
                text.append(value.strip())

            elif isinstance(value, list):
                text.extend(
                    str(entry).strip()
                    for entry in value
                    if str(entry).strip()
                )

    return "\n".join(text)


def build_retry_feedback(feedback: list[str]) -> str:
    """Turn detector feedback into section rewrite instructions."""
    guidance = feedback or [
        "Use more natural sentence variation and less repetitive phrasing."
    ]

    return "\n".join(
        [
            "Revise the writing using the style feedback below.",
            "Preserve all facts and modify only writable fields.",
            *guidance,
        ]
    )
