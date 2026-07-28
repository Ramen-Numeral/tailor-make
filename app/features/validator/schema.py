"""Typed audit trail contracts for explainable resume validation."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, SerializeAsAny

from app.features.ai_detection.schema import (
    AIDetectionResult,
    ConstraintCheck,
    CounterfactualComparison,
    EvaluationDecision,
)
from app.resume_schema.resume_schema import Resume, Section
from app.features.job_listing_parser.listing_schema import Requirement
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.keyword_evidence.scoring import ResumeMatchScore


class AttemptEvaluation(BaseModel):
    """All observable evidence and the decision for one candidate."""

    attempt: int = Field(ge=0)
    text: str
    detection: AIDetectionResult
    constraints: list[ConstraintCheck] = Field(default_factory=list)
    decision: EvaluationDecision
    selected: bool = False
    counterfactual: CounterfactualComparison | None = None


class AgentTraceEvent(BaseModel):
    """One frontend-safe event emitted by the deterministic agent loop."""

    event_type: Literal[
        "section_skipped",
        "evaluation_completed",
        "rewrite_started",
        "attempt_rejected",
        "attempt_accepted",
        "best_attempt_selected",
        "section_failed",
        "workflow_completed",
        "run_started",
        "job_parsed",
        "match_score_completed",
        "content_selected",
        "coverage_plan_created",
        "positioning_brief_created",
        "draft_generated",
        "global_validation_started",
        "content_trimmed",
        "page_count_completed",
        "run_completed",
        "recruiter_evaluation_completed",
        "human_input_requested",
        "human_input_received",
    ]
    section: str | None = None
    attempt: int | None = Field(default=None, ge=0)
    title: str
    summary: str
    observations: list[str] = Field(default_factory=list)
    action: str | None = None
    decision: Literal[
        "accept",
        "retry",
        "accept_with_warnings",
        "preserve",
        "skip",
        "trim",
    ] | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    evaluation: AttemptEvaluation | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    job_requirements: list[Requirement] = Field(default_factory=list)
    coverage_plan: CoveragePlan | None = None
    match_score: ResumeMatchScore | None = None


TraceCallback = Callable[[AgentTraceEvent], None]


class SectionValidationResult(BaseModel):
    """Final section plus its complete evaluation and event history."""

    section_name: str
    original_section: SerializeAsAny[Section]
    final_section: SerializeAsAny[Section]
    attempts: list[AttemptEvaluation] = Field(default_factory=list)
    events: list[AgentTraceEvent] = Field(default_factory=list)
    status: Literal[
        "accepted",
        "accepted_with_warnings",
        "best_attempt_selected",
        "unchanged",
    ]


class ResumeValidationResult(BaseModel):
    """Validated resume and section-level audit trails."""

    resume: SerializeAsAny[Resume]
    sections: list[SectionValidationResult] = Field(default_factory=list)
    events: list[AgentTraceEvent] = Field(default_factory=list)
    status: Literal[
        "completed",
        "completed_with_warnings",
        "detector_unavailable",
    ]
