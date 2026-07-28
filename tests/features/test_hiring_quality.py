from app.features.agent.hiring_quality import (
    evaluate_bullets,
    evaluate_recruiter_quality,
)
from app.features.agent.positioning import build_positioning_brief
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.features.keyword_evidence.planner import build_coverage_plan
from app.infrastructure.cache import clear_stage_cache
from config.resume.candidate_profile import build_resume as example_resume


def test_positioning_brief_separates_supported_evidence_and_gaps() -> None:
    listing = JobListing(
        title="Backend Engineer",
        requirements=[
            Requirement(text="Python", kind="skill"),
            Requirement(text="Angular", kind="skill"),
        ],
    )
    plan = build_coverage_plan(listing, example_resume())

    brief = build_positioning_brief(listing, plan)

    assert brief.target_identity == "Backend Engineer"
    assert "Python" in brief.primary_evidence
    assert "Angular" in brief.gaps
    assert brief.section_plan


def test_bullet_quality_rewards_action_context_and_outcome() -> None:
    resume = example_resume()

    results = evaluate_bullets(resume)

    assert results
    assert all(0 <= result.score <= 100 for result in results)
    assert any("supported outcome" in result.passed_dimensions for result in results)
    assert any("specific implementation" in result.passed_dimensions for result in results)


def test_recruiter_evaluation_is_cached_and_schema_bounded() -> None:
    clear_stage_cache()
    resume = example_resume()
    brief = build_positioning_brief(
        JobListing(
            title="Backend Engineer",
            requirements=[Requirement(text="Python", kind="skill")],
        ),
        build_coverage_plan(
            JobListing(
                title="Backend Engineer",
                requirements=[Requirement(text="Python", kind="skill")],
            ),
            resume,
        ),
    )

    class Critic:
        calls = 0

        def invoke_structured(self, *, schema, **kwargs):
            self.calls += 1
            return schema(
                axes=[
                    {"axis": axis, "score": 4, "reason": "Clear evidence."}
                    for axis in (
                        "target_clarity",
                        "evidence_visibility",
                        "coherence",
                        "seniority_credibility",
                        "specificity",
                        "scanability",
                    )
                ],
                strengths=["Python delivery"],
                summary="Strong positioning.",
                ready=True,
            )

    critic = Critic()
    first = evaluate_recruiter_quality(resume, brief, critic)
    second = evaluate_recruiter_quality(resume, brief, critic)

    assert first.ready is True
    assert second == first
    assert critic.calls == 1
