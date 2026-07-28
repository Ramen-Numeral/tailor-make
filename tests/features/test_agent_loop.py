from app.features.agent import orchestrator
from app.features.agent.page_fit import fit_resume_to_page_limit
from app.features.agent.schema import GlobalPageCheck
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.renderer.renderer import count_pdf_pages, render_html
from app.features.validator.factual import evaluate_factual_integrity
from app.features.validator.schema import ResumeValidationResult
from app.resume_schema.resume_schema import (
    Constraints,
    MutableResume,
    ProjectSection,
    SkillCategoryItem,
    SkillsSection,
    WorkExperienceItem,
    WorkExperienceSection,
)
from config.resume.candidate_profile import build_resume


def test_factual_integrity_blocks_new_numbers_and_locked_field_changes() -> None:
    original_item = WorkExperienceItem(
        title="Engineer",
        company="Example",
        start_date="2020",
        bullets=["Improved throughput."],
    )
    source = WorkExperienceSection(items=[original_item])
    candidate = WorkExperienceSection(
        items=[
            original_item.model_copy(
                update={
                    "company": "Different Company",
                    "bullets": ["Improved throughput by 50%."],
                },
                deep=True,
            )
        ]
    )

    checks = evaluate_factual_integrity(source, candidate)

    assert any(
        check.constraint == "factual_locked_field" and not check.passed
        for check in checks
    )
    assert any(
        check.constraint == "factual_numbers" and not check.passed
        for check in checks
    )


def test_factual_integrity_blocks_new_structured_skills() -> None:
    original_item = SkillCategoryItem(
        name="Languages",
        skills=["Python"],
    )
    source = SkillsSection(items=[original_item])
    candidate = SkillsSection(
        items=[
            original_item.model_copy(
                update={"skills": ["Python", "Rust"]},
                deep=True,
            )
        ]
    )

    checks = evaluate_factual_integrity(source, candidate)

    assert any(
        check.constraint == "factual_structured_terms"
        and not check.passed
        and "rust" in check.observed
        for check in checks
    )


def test_candidate_notes_authorize_only_literal_supplemental_skill() -> None:
    original_item = SkillCategoryItem(name="Languages", skills=["Python"])
    source = SkillsSection(items=[original_item])
    candidate = SkillsSection(
        items=[
            original_item.model_copy(
                update={"skills": ["Python", "PostgreSQL", "Rust"]},
                deep=True,
            )
        ]
    )

    checks = evaluate_factual_integrity(
        source,
        candidate,
        supplemental_evidence=(
            "Used PostgreSQL to build reporting queries at Beacon Digital."
        ),
    )

    structured = next(
        check for check in checks
        if check.constraint == "factual_structured_terms"
    )
    assert structured.passed is False
    assert "rust" in structured.observed
    assert "postgresql" not in structured.observed


def test_page_fit_removes_tail_ranked_items_until_counter_fits() -> None:
    resume = MutableResume.model_validate(build_resume().model_dump())
    resume.projects = ProjectSection(
        constraints=Constraints(min_items=1),
        items=resume.projects.items[:3],
    )

    def counter(candidate):
        return 2 if len(candidate.projects.items) > 1 else 1

    fitted, result = fit_resume_to_page_limit(
        resume,
        maximum_pages=1,
        max_trims=5,
        page_counter=counter,
    )

    assert result.passed is True
    assert result.initial_pages == 2
    assert result.final_pages == 1
    assert len(fitted.projects.items) == 1
    assert [action.removed for action in result.trim_actions] == [
        resume.projects.items[2].name,
        resume.projects.items[1].name,
    ]


def test_page_fit_preserves_item_containing_required_keyword() -> None:
    resume = MutableResume.model_validate(build_resume().model_dump())
    items = resume.projects.items[:3]
    protected_keyword = items[2].technologies[0]
    resume.projects = ProjectSection(
        constraints=Constraints(
            min_items=1,
            required_keywords=[protected_keyword],
        ),
        items=items,
    )

    fitted, result = fit_resume_to_page_limit(
        resume,
        maximum_pages=1,
        max_trims=2,
        page_counter=lambda candidate: (
            2 if len(candidate.projects.items) > 2 else 1
        ),
    )

    remaining_ids = {item.id for item in fitted.projects.items}
    assert result.passed is True
    assert items[2].id in remaining_ids
    assert items[1].id not in remaining_ids


def test_page_fit_never_removes_a_work_history_entry() -> None:
    resume = MutableResume.model_validate(build_resume().model_dump())
    original_ids = [item.id for item in resume.work_experience.items]
    resume.projects = None
    resume.research = None

    fitted, result = fit_resume_to_page_limit(
        resume,
        maximum_pages=1,
        max_trims=50,
        page_counter=lambda candidate: (
            2
            if len(candidate.work_experience.items) == len(original_ids)
            else 1
        ),
    )

    assert [item.id for item in fitted.work_experience.items] == original_ids
    assert not any(
        action.section == "work_experience" and action.field == "item"
        for action in result.trim_actions
    )
    assert result.passed is False


def test_pdf_page_counter_uses_physical_render() -> None:
    pages = count_pdf_pages(render_html(build_resume()))

    assert pages >= 1


def test_renderer_supports_sections_with_empty_constraints() -> None:
    resume = build_resume()
    resume = resume.model_copy(
        update={
            "work_experience": resume.work_experience.model_copy(
                update={"constraints": Constraints()},
                deep=True,
            ),
            "projects": resume.projects.model_copy(
                update={"constraints": Constraints()},
                deep=True,
            ),
            "education": resume.education.model_copy(
                update={"constraints": Constraints()},
                deep=True,
            ),
        },
        deep=True,
    )

    html = render_html(resume)

    assert resume.candidate.name in html


def test_agent_orchestrator_retains_state_and_pipeline_events(
    monkeypatch,
) -> None:
    source = build_resume()
    mutable = MutableResume.model_validate(source.model_dump())
    monkeypatch.setattr(
        orchestrator,
        "parse_listing",
        lambda text, **kwargs: JobListing(title="Engineer"),
    )
    monkeypatch.setattr(
        orchestrator,
        "match_resume",
        lambda job, resume: mutable,
    )
    monkeypatch.setattr(orchestrator, "_load_embedder", lambda: None)
    coverage_inputs = []

    def coverage(job, resume, **kwargs):
        coverage_inputs.append(resume)
        return CoveragePlan()

    monkeypatch.setattr(
        orchestrator,
        "build_coverage_plan",
        coverage,
    )
    monkeypatch.setattr(
        orchestrator,
        "apply_coverage_plan",
        lambda resume, plan: resume,
    )
    monkeypatch.setattr(
        orchestrator,
        "global_resume_rewrite",
        lambda job, resume, special_instructions: resume,
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_resume_with_trace",
        lambda *args, **kwargs: ResumeValidationResult(
            resume=mutable,
            status="completed",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "fit_resume_to_page_limit",
        lambda resume, **kwargs: (
            resume,
            GlobalPageCheck(
                maximum_pages=1,
                initial_pages=1,
                final_pages=1,
                passed=True,
            ),
        ),
    )
    streamed = []

    result = orchestrator.tailor_resume_agent(
        source,
        "Engineer",
        trace_callback=streamed.append,
    )

    assert result.state.status == "completed"
    assert result.state.source_resume == source
    assert result.page_check.passed is True
    assert streamed[0].event_type == "run_started"
    assert streamed[-1].event_type == "run_completed"
    score_events = [
        event
        for event in streamed
        if event.event_type == "match_score_completed"
    ]
    assert [event.title for event in score_events] == [
        "Initial profile match score",
        "Final resume match score",
    ]
    assert result.state.initial_match_score.stage == "initial"
    assert result.state.final_match_score.stage == "final"
    assert len(coverage_inputs) == 2
    assert coverage_inputs[0] is source
    assert coverage_inputs[1] is result.resume


def test_summary_option_creates_or_omits_only_working_summary() -> None:
    source = build_resume()
    source_without_summary = source.model_copy(
        update={
            "summary": source.summary.model_copy(
                update={"items": []},
                deep=True,
            )
        },
        deep=True,
    )

    included = orchestrator._prepare_summary(
        source_without_summary,
        True,
    )
    excluded = orchestrator._prepare_summary(source, False)

    assert len(included.summary.items) == 1
    assert included.summary.items[0].content is None
    assert excluded.summary.items == []
    assert source.summary.items
    assert source_without_summary.summary.items == []


def test_supplemental_evidence_is_labeled_in_rewrite_instructions() -> None:
    instructions = orchestrator._rewrite_instructions(
        "Keep the tone direct.",
        "Used PostgreSQL for weekly reporting.",
    )

    assert "Keep the tone direct." in instructions
    assert "Candidate-supplied supplemental evidence" in instructions
    assert "Used PostgreSQL" in instructions
