"""Deterministic resume-to-job match scoring from evidence coverage."""

from typing import Literal

from pydantic import BaseModel, Field

from app.features.keyword_evidence.schema import CoveragePlan

IMPORTANCE_WEIGHTS = {
    "critical": 3.0,
    "important": 2.0,
    "supporting": 1.0,
}
SUPPORT_CREDIT = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
}


class MatchScoreBreakdown(BaseModel):
    kind: str
    score: int = Field(ge=0, le=100)
    supported: int = Field(ge=0)
    partial: int = Field(ge=0)
    unsupported: int = Field(ge=0)


class ResumeMatchScore(BaseModel):
    stage: Literal["initial", "final"]
    score: int = Field(ge=0, le=100)
    supported: int = Field(ge=0)
    partial: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    total_requirements: int = Field(ge=0)
    breakdown: list[MatchScoreBreakdown] = Field(default_factory=list)
    largest_gaps: list[str] = Field(default_factory=list)
    evidence_coverage_score: int = Field(ge=0, le=100)
    holistic_fit_score: int | None = Field(default=None, ge=0, le=100)
    holistic_summary: str | None = None
    rubric_observations: list[str] = Field(default_factory=list)


def score_coverage_plan(
    plan: CoveragePlan,
    *,
    stage: Literal["initial", "final"],
) -> ResumeMatchScore:
    """Return a stable 0–100 weighted evidence-coverage score."""
    matches = plan.requirement_matches
    evidence_score = _weighted_score(matches)
    rubric_score = _rubric_score(plan)
    score = (
        evidence_score
        if rubric_score is None
        else round(0.35 * evidence_score + 0.65 * rubric_score)
    )
    breakdown = []
    for kind in sorted({match.requirement_kind for match in matches}):
        grouped = [
            match for match in matches if match.requirement_kind == kind
        ]
        breakdown.append(
            MatchScoreBreakdown(
                kind=kind,
                score=_weighted_score(grouped),
                **_counts(grouped),
            )
        )
    gaps = sorted(
        (
            match
            for match in matches
            if match.support != "supported"
        ),
        key=lambda match: (
            -IMPORTANCE_WEIGHTS[match.importance],
            match.support != "unsupported",
        ),
    )
    return ResumeMatchScore(
        stage=stage,
        score=score,
        evidence_coverage_score=evidence_score,
        holistic_fit_score=rubric_score,
        holistic_summary=(
            plan.fit_rubric.summary if plan.fit_rubric else None
        ),
        rubric_observations=[
            f"{axis.axis.replace('_', ' ')}: {axis.score}/5 — {axis.reason}"
            for axis in (plan.fit_rubric.axes if plan.fit_rubric else [])
        ],
        total_requirements=len(matches),
        breakdown=breakdown,
        largest_gaps=[
            f"{match.requirement_text} ({match.support})"
            for match in gaps[:5]
        ],
        **_counts(matches),
    )


def _weighted_score(matches) -> int:
    if not matches:
        return 0
    available = sum(
        IMPORTANCE_WEIGHTS[match.importance]
        for match in matches
    )
    earned = sum(
        IMPORTANCE_WEIGHTS[match.importance]
        * SUPPORT_CREDIT[match.support]
        for match in matches
    )
    return round(100 * earned / available)


def _counts(matches) -> dict[str, int]:
    return {
        support: sum(match.support == support for match in matches)
        for support in ("supported", "partial", "unsupported")
    }


def _rubric_score(plan: CoveragePlan) -> int | None:
    rubric = plan.fit_rubric
    if rubric is None or not rubric.axes:
        return None
    weights = {
        "functional_alignment": 0.30,
        "transferable_experience": 0.20,
        "skill_coverage": 0.20,
        "domain_alignment": 0.10,
        "seniority_scope": 0.10,
        "evidence_strength": 0.10,
    }
    axes = {axis.axis: axis.score for axis in rubric.axes}
    present_weight = sum(weights[name] for name in axes)
    if not present_weight:
        return None
    weighted = sum(
        axes[name] / 5 * weights[name]
        for name in axes
    )
    return round(100 * weighted / present_weight)
