from uuid import uuid4

from app.features.keyword_evidence.schema import (
    CandidateFitRubric,
    CoveragePlan,
    FitRubricAxis,
    RequirementEvidenceMatch,
)
from app.features.keyword_evidence.scoring import score_coverage_plan


def match(text, importance, support, kind="skill"):
    return RequirementEvidenceMatch(
        requirement_id=uuid4(),
        requirement_text=text,
        requirement_kind=kind,
        importance=importance,
        support=support,
    )


def test_match_score_weights_importance_and_partial_credit() -> None:
    plan = CoveragePlan(
        requirement_matches=[
            match("Critical supported", "critical", "supported"),
            match("Important partial", "important", "partial"),
            match("Supporting missing", "supporting", "unsupported"),
        ]
    )

    score = score_coverage_plan(plan, stage="initial")

    assert score.score == 67
    assert score.supported == 1
    assert score.partial == 1
    assert score.unsupported == 1
    assert score.largest_gaps[0] == "Important partial (partial)"


def test_match_score_includes_requirement_kind_breakdown() -> None:
    plan = CoveragePlan(
        requirement_matches=[
            match("Python", "critical", "supported", "skill"),
            match("Three years", "critical", "unsupported", "experience"),
        ]
    )

    score = score_coverage_plan(plan, stage="final")

    assert score.score == 50
    assert {item.kind: item.score for item in score.breakdown} == {
        "experience": 0,
        "skill": 100,
    }


def test_holistic_rubric_prevents_keyword_only_zero_for_adjacent_fit() -> None:
    plan = CoveragePlan(
        requirement_matches=[
            match(
                "Digital product management",
                "critical",
                "unsupported",
                "experience",
            )
        ],
        fit_rubric=CandidateFitRubric(
            summary="Strong transferable product work with a domain gap.",
            axes=[
                FitRubricAxis(
                    axis="functional_alignment",
                    score=4,
                    reason="Owned a roadmap and backlog.",
                ),
                FitRubricAxis(
                    axis="transferable_experience",
                    score=4,
                    reason="Product strategy transfers directly.",
                ),
                FitRubricAxis(
                    axis="skill_coverage",
                    score=3,
                    reason="Jira and Figma support the workflow.",
                ),
                FitRubricAxis(
                    axis="domain_alignment",
                    score=1,
                    reason="No banking domain evidence.",
                ),
                FitRubricAxis(
                    axis="seniority_scope",
                    score=3,
                    reason="Demonstrated ownership.",
                ),
                FitRubricAxis(
                    axis="evidence_strength",
                    score=4,
                    reason="Concrete work bullets.",
                ),
            ],
        ),
    )

    score = score_coverage_plan(plan, stage="initial")

    assert score.evidence_coverage_score == 0
    assert score.holistic_fit_score == 68
    assert score.score == 44
