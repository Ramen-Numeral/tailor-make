"""State and outcomes for the bounded resume-tailoring agent."""

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SerializeAsAny

from app.features.job_listing_parser.listing_schema import JobListing
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.keyword_evidence.scoring import ResumeMatchScore
from app.features.resume_diff.schema import FieldDiff
from app.features.validator.schema import (
    AgentTraceEvent,
    SectionValidationResult,
)
from app.resume_schema.resume_schema import Resume


class RequirementPosition(BaseModel):
    requirement: str
    support: Literal["supported", "partial", "unsupported"]
    evidence_strength: Literal["strong", "moderate", "weak", "none"]
    destination_sections: list[str] = Field(default_factory=list)


class PositioningBrief(BaseModel):
    target_identity: str
    primary_evidence: list[str] = Field(default_factory=list)
    transferable_narrative: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    writing_priorities: list[str] = Field(default_factory=list)
    requirement_positions: list[RequirementPosition] = Field(default_factory=list)
    section_plan: dict[str, list[str]] = Field(default_factory=dict)


class BulletQualityResult(BaseModel):
    section: str
    item_index: int
    bullet_index: int
    text: str
    score: int = Field(ge=0, le=100)
    passed_dimensions: list[str] = Field(default_factory=list)
    improvement_dimensions: list[str] = Field(default_factory=list)


class RecruiterAxis(BaseModel):
    axis: Literal[
        "target_clarity",
        "evidence_visibility",
        "coherence",
        "seniority_credibility",
        "specificity",
        "scanability",
    ]
    score: int = Field(ge=1, le=5)
    reason: str


class RecruiterEvaluation(BaseModel):
    axes: list[RecruiterAxis] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""
    ready: bool = False


class AgentBudget(BaseModel):
    max_section_rewrites: int = Field(default=2, ge=1)
    max_total_rewrites: int = Field(default=5, ge=0)
    max_page_trims: int = Field(default=20, ge=0)


class AgentAction(BaseModel):
    action: Literal[
        "accept_section",
        "rewrite_section",
        "restore_facts",
        "trim_content",
        "keep_best_attempt",
    ]
    section: str | None = None
    reason: str


class PageTrimAction(BaseModel):
    section: str
    item_id: UUID | None = None
    field: str
    removed: str
    reason: str


class GlobalPageCheck(BaseModel):
    maximum_pages: int = Field(ge=1)
    initial_pages: int = Field(ge=1)
    final_pages: int = Field(ge=1)
    passed: bool
    trim_actions: list[PageTrimAction] = Field(default_factory=list)


class TailoringRunState(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    source_resume: SerializeAsAny[Resume]
    job_listing: JobListing
    evidence_plan: CoveragePlan
    positioning_brief: PositioningBrief | None = None
    initial_match_score: ResumeMatchScore
    final_match_score: ResumeMatchScore
    selected_resume: SerializeAsAny[Resume]
    current_resume: SerializeAsAny[Resume]
    section_results: list[SectionValidationResult] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    diffs: list[FieldDiff] = Field(default_factory=list)
    events: list[AgentTraceEvent] = Field(default_factory=list)
    total_rewrites: int = Field(default=0, ge=0)
    supplemental_evidence: str | None = None
    bullet_quality: list[BulletQualityResult] = Field(default_factory=list)
    recruiter_evaluation: RecruiterEvaluation | None = None
    status: Literal[
        "planning",
        "drafting",
        "evaluating",
        "fitting_pages",
        "completed",
        "completed_with_warnings",
        "failed",
    ] = "planning"


class TailoringRunResult(BaseModel):
    resume: SerializeAsAny[Resume]
    state: TailoringRunState
    page_check: GlobalPageCheck
    diffs: list[FieldDiff] = Field(default_factory=list)
