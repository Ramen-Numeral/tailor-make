import numpy as np

from app.features.ai_detection.schema import (
    AIDetectionResult,
    RubricAxisResult,
)
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.features.keyword_evidence.planner import (
    apply_coverage_plan,
    build_coverage_plan,
    extract_resume_evidence,
)
from app.features.keyword_evidence.schema import (
    EvidenceJudgment,
    EvidenceJudgmentBatch,
)
from app.features.validator.decision import evaluate_attempt
from app.features.validator.policy import policy_for_section
from app.features.validator.validator import validate_section_with_trace
from app.resume_schema.resume_schema import (
    Candidate,
    Constraints,
    Resume,
    SkillCategoryItem,
    SkillsSection,
    WorkExperienceItem,
    WorkExperienceSection,
)
from config.resume.candidate_profile import build_resume


class SemanticEmbedder:
    """Small deterministic stand-in for SentenceTransformer."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if "event-driven" in lowered or "kafka" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "angular" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors)


class EvidenceJudge:
    def __init__(self, response: EvidenceJudgmentBatch):
        self.response = response
        self.calls = 0

    def invoke_structured(self, **kwargs):
        self.calls += 1
        return self.response


def axis(name: str, score: int) -> RubricAxisResult:
    return RubricAxisResult(
        axis=name,
        label=name.replace("_", " ").title(),
        definition=f"Definition for {name}",
        score=score,
        interpretation=f"Level {score}",
    )


def test_resume_evidence_preserves_atomic_provenance() -> None:
    evidence = extract_resume_evidence(build_resume())
    postgres = [
        item
        for item in evidence
        if item.text == "PostgreSQL" and item.section == "skills"
    ]

    assert postgres
    assert postgres[0].field == "skills"
    assert str(postgres[0].item_id) in postgres[0].evidence_id


def test_coverage_plan_requires_only_exact_supported_job_terms() -> None:
    job = JobListing(
        requirements=[
            Requirement(
                text="PostgreSQL",
                kind="skill",
                importance="critical",
            ),
            Requirement(
                text="Angular",
                kind="skill",
                importance="critical",
            ),
            Requirement(
                text="event-driven architecture",
                kind="skill",
                importance="important",
            ),
        ]
    )

    plan = build_coverage_plan(
        job,
        build_resume(),
        embedder=SemanticEmbedder(),
    )
    matches = {
        match.requirement_text: match
        for match in plan.requirement_matches
    }

    assert matches["PostgreSQL"].support == "supported"
    assert matches["Angular"].support == "unsupported"
    assert matches["event-driven architecture"].support == "partial"
    assert "PostgreSQL" in plan.keywords_for("skills")
    assert "PostgreSQL" in plan.keywords_for("summary")
    assert "Angular" not in {
        assignment.keyword
        for assignment in plan.assignments
    }
    assert "event-driven architecture" not in {
        assignment.keyword
        for assignment in plan.assignments
    }


def test_llm_judge_recognizes_transferable_product_evidence() -> None:
    requirement = Requirement(
        text="project management",
        kind="skill",
        importance="critical",
    )
    resume = Resume(
        candidate=Candidate(name="Candidate"),
        work_experience=WorkExperienceSection(
            items=[
                WorkExperienceItem(
                    title="Product Strategist",
                    company="Medical education SaaS",
                    start_date="2024",
                    bullets=[
                        "Organized the roadmap, ticket backlog, and "
                        "prototyped UI features in Figma."
                    ],
                )
            ]
        ),
    )
    evidence = extract_resume_evidence(resume)
    cited = next(
        item for item in evidence if "roadmap" in item.text
    )
    judge = EvidenceJudge(
        EvidenceJudgmentBatch(
            judgments=[
                EvidenceJudgment(
                    requirement_id=requirement.id,
                    support="supported",
                    evidence_ids=[cited.evidence_id],
                    relationship="transferable",
                    safe_keywords=["project management"],
                    reason=(
                        "Roadmap and backlog ownership directly demonstrate "
                        "project management."
                    ),
                )
            ]
        )
    )

    plan = build_coverage_plan(
        JobListing(requirements=[requirement]),
        resume,
        adjudicator=judge,
    )

    match = plan.requirement_matches[0]
    assert match.support == "supported"
    assert match.decision_source == "llm"
    assert match.matches[0].evidence.evidence_id == cited.evidence_id
    assert match.matches[0].judge_support == "supported"
    assert "project management" in plan.keywords_for("work_experience")
    assert judge.calls == 1


def test_llm_judge_cannot_support_with_unknown_evidence() -> None:
    requirement = Requirement(text="SEO", kind="skill")
    resume = build_resume()
    judge = EvidenceJudge(
        EvidenceJudgmentBatch(
            judgments=[
                EvidenceJudgment(
                    requirement_id=requirement.id,
                    support="supported",
                    evidence_ids=["invented:evidence:id"],
                    reason="Unsupported citation.",
                )
            ]
        )
    )

    plan = build_coverage_plan(
        JobListing(requirements=[requirement]),
        resume,
        adjudicator=judge,
    )

    assert plan.requirement_matches[0].support == "unsupported"
    assert plan.keywords_for("skills") == []


def test_llm_judge_allows_child_to_parent_keyword_generalization() -> None:
    requirement = Requirement(text="SQL", kind="skill")
    resume = Resume(
        candidate=Candidate(name="Candidate"),
        skills=SkillsSection(
            items=[
                SkillCategoryItem(
                    name="Databases",
                    skills=["PostgreSQL"],
                )
            ]
        ),
    )
    evidence = extract_resume_evidence(resume)
    postgres = next(item for item in evidence if item.text == "PostgreSQL")
    judge = EvidenceJudge(
        EvidenceJudgmentBatch(
            judgments=[
                EvidenceJudgment(
                    requirement_id=requirement.id,
                    support="supported",
                    relationship="parent_generalization",
                    evidence_ids=[postgres.evidence_id],
                    safe_keywords=["SQL"],
                    reason="PostgreSQL is a SQL relational database.",
                )
            ]
        )
    )

    plan = build_coverage_plan(
        JobListing(requirements=[requirement]),
        resume,
        adjudicator=judge,
    )

    assert plan.keywords_for("skills") == ["SQL"]


def test_llm_judge_does_not_allow_parent_to_child_keyword_claim() -> None:
    requirement = Requirement(text="PostgreSQL", kind="skill")
    resume = Resume(
        candidate=Candidate(name="Candidate"),
        skills=SkillsSection(
            items=[SkillCategoryItem(name="Languages", skills=["SQL"])]
        ),
    )
    evidence = extract_resume_evidence(resume)
    sql = next(item for item in evidence if item.text == "SQL")
    judge = EvidenceJudge(
        EvidenceJudgmentBatch(
            judgments=[
                EvidenceJudgment(
                    requirement_id=requirement.id,
                    support="partial",
                    relationship="adjacent",
                    evidence_ids=[sql.evidence_id],
                    safe_keywords=[],
                    reason="SQL knowledge does not prove PostgreSQL experience.",
                )
            ]
        )
    )

    plan = build_coverage_plan(
        JobListing(requirements=[requirement]),
        resume,
        adjudicator=judge,
    )

    assert plan.keywords_for("skills") == []


def test_coverage_plan_replaces_static_keyword_constraints() -> None:
    resume = build_resume()
    mutable = resume.model_dump()
    from app.resume_schema.resume_schema import MutableResume

    selected = MutableResume.model_validate(mutable)
    plan = build_coverage_plan(
        JobListing(
            requirements=[
                Requirement(text="Python", kind="skill"),
            ]
        ),
        selected,
    )

    planned = apply_coverage_plan(selected, plan)

    assert planned.summary.constraints.required_keywords == ["Python"]
    assert planned.skills.constraints.required_keywords == ["Python"]
    assert "machine learning" not in planned.summary.constraints.required_keywords


def test_work_policy_does_not_block_on_stylistic_variation() -> None:
    detection = AIDetectionResult(
        ai_probability=0.2,
        threshold=0.5,
        rubric_axes=[
            axis("specificity", 4),
            axis("idea_compression", 4),
            axis("substantive_value", 4),
            axis("stylistic_variation", 2),
        ],
    )
    policy = policy_for_section(
        "work_experience",
        ai_likeness_retry_threshold=0.7,
    )

    decision = evaluate_attempt(detection, [], policy)

    assert decision.outcome == "accept"
    assert decision.failed_rubric_count == 0


def test_skills_policy_skips_prose_detector() -> None:
    class UnexpectedDetector:
        def classify(self, text):
            raise AssertionError("skills should not run prose scoring")

    section = SkillsSection(
        constraints=Constraints(
            min_items=1,
            max_items=2,
            min_skills_per_category=1,
        ),
        items=[
            SkillCategoryItem(
                name="Languages",
                skills=["Python"],
            )
        ],
    )

    result = validate_section_with_trace(
        JobListing(title="Engineer"),
        section,
        UnexpectedDetector(),
        section_name="skills",
    )

    assert result.status == "accepted"
    assert result.attempts[0].detection.components == []
    assert result.attempts[0].detection.prediction == "not-evaluated"
    assert result.attempts[0].decision.failed_rubric_count == 0
