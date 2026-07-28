"""Typed provenance for job requirements matched to resume evidence."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ResumeEvidence(BaseModel):
    """One atomic, immutable fact-bearing fragment from the source resume."""

    evidence_id: str
    section: str
    item_id: UUID
    field: str
    text: str


class EvidenceMatch(BaseModel):
    """Retrieval evidence for one requirement-to-resume relationship."""

    evidence: ResumeEvidence
    exact_or_alias_match: bool = False
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    cosine_rank: int | None = Field(default=None, ge=1)
    cosine_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    judge_support: Literal["supported", "partial", "unsupported"] | None = None
    judge_reason: str | None = None
    safe_keywords: list[str] = Field(default_factory=list)


class RequirementEvidenceMatch(BaseModel):
    """Support classification for one parsed job requirement."""

    requirement_id: UUID
    requirement_text: str
    requirement_kind: Literal[
        "skill",
        "experience",
        "education",
        "certification",
        "responsibility",
        "other",
    ] = "other"
    importance: Literal["critical", "important", "supporting"]
    support: Literal["supported", "partial", "unsupported"]
    matches: list[EvidenceMatch] = Field(default_factory=list)
    decision_source: Literal["exact", "embedding", "llm", "none"] = "none"
    adjudication_reason: str | None = None


class EvidenceJudgment(BaseModel):
    """One bounded semantic judgment with source evidence citations."""

    requirement_id: UUID
    support: Literal["supported", "partial", "unsupported"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str
    relationship: Literal[
        "equivalent",
        "parent_generalization",
        "transferable",
        "adjacent",
        "none",
    ] = "none"
    safe_keywords: list[str] = Field(default_factory=list)


class FitRubricAxis(BaseModel):
    axis: Literal[
        "functional_alignment",
        "transferable_experience",
        "skill_coverage",
        "domain_alignment",
        "seniority_scope",
        "evidence_strength",
    ]
    score: int = Field(ge=0, le=5)
    reason: str


class CandidateFitRubric(BaseModel):
    axes: list[FitRubricAxis] = Field(default_factory=list)
    summary: str = ""


class EvidenceJudgmentBatch(BaseModel):
    judgments: list[EvidenceJudgment] = Field(default_factory=list)
    fit_rubric: CandidateFitRubric | None = None


class KeywordAssignment(BaseModel):
    """A supported job keyword assigned to an output section."""

    keyword: str
    section: str
    requirement_id: UUID
    evidence_ids: list[str] = Field(default_factory=list)
    importance: Literal["critical", "important", "supporting"]


class CoveragePlan(BaseModel):
    """Explainable, section-aware keyword coverage derived for one run."""

    requirement_matches: list[RequirementEvidenceMatch] = Field(
        default_factory=list
    )
    assignments: list[KeywordAssignment] = Field(default_factory=list)
    fit_rubric: CandidateFitRubric | None = None

    @property
    def unsupported_requirements(self) -> list[RequirementEvidenceMatch]:
        return [
            match
            for match in self.requirement_matches
            if match.support == "unsupported"
        ]

    def keywords_for(self, section: str) -> list[str]:
        return list(
            dict.fromkeys(
                assignment.keyword
                for assignment in self.assignments
                if assignment.section == section
            )
        )
