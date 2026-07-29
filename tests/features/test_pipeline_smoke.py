from types import SimpleNamespace
from typing import get_args

import pytest
from pydantic import ValidationError

from app.features.content_extractor import content_extractor
from app.features.job_listing_parser import parse_listing as parser_module
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.features.renderer.renderer import render_html
from app.features import pipeline as pipeline_module
from app.features.writer import writer
from app.resume_schema.resume_schema import (
    Constraints,
    SkillCategoryItem,
    SkillsSection,
    WorkExperienceItem,
    WorkExperienceSection,
)
from config.resume.candidate_profile import build_resume


class StructuredClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke_structured(self, *, schema, **kwargs):
        self.calls += 1
        response = next(self.responses)
        return response(schema) if callable(response) else response


def candidate_selections(resume):
    def build(schema):
        values = {}
        for name in schema.model_fields:
            section = getattr(resume, name)
            maximum = section.constraints.max_items
            count = min(maximum or len(section.items), len(section.items))
            values[name] = [str(item.id) for item in section.items[:count]]
        return schema(**values)

    return build


def test_content_selection_uses_minimum_to_maximum_range(
    monkeypatch,
) -> None:
    source = build_resume()
    source.skills.constraints.min_items = 2
    source.skills.constraints.max_items = 5
    source.skills.constraints.max_skill_categories = 4

    def select_two_skills(schema):
        skill_schema = schema.model_json_schema()["properties"]["skills"]
        assert skill_schema["minItems"] == 2
        assert skill_schema["maxItems"] == 4

        values = {}
        for name in schema.model_fields:
            section = getattr(source, name)
            count = 2 if name == "skills" else min(1, len(section.items))
            values[name] = [
                str(item.id)
                for item in section.items[:count]
            ]
        return schema(**values)

    client = StructuredClient([select_two_skills])
    monkeypatch.setattr(
        content_extractor,
        "get_llm_clients",
        lambda: SimpleNamespace(content_matcher=client),
    )

    selected = content_extractor.match_resume(JobListing(), source)

    assert len(selected.skills.items) == 2


def test_content_selection_always_preserves_jobs_and_education(
    monkeypatch,
) -> None:
    source = build_resume()

    def select(schema):
        assert "work_experience" not in schema.model_fields
        assert "education" not in schema.model_fields
        return candidate_selections(source)(schema)

    client = StructuredClient([select])
    monkeypatch.setattr(
        content_extractor,
        "get_llm_clients",
        lambda: SimpleNamespace(content_matcher=client),
    )

    selected = content_extractor.match_resume(JobListing(), source)

    assert selected.work_experience.items == source.work_experience.items
    assert selected.education.items == source.education.items


def global_rewrites(schema):
    counts = {
        "summary": 1,
        "skills": 4,
        "work_experience": 2,
        "education": 1,
        "projects": 3,
    }
    values = {}
    for name, field in schema.model_fields.items():
        item_type = get_args(field.annotation)[0]
        values[name] = [item_type() for _ in range(counts[name])]
    return schema(**values)


def test_non_ai_feature_pipeline_wiring(monkeypatch) -> None:
    listing = JobListing(
        title="Senior Software Engineer",
        requirements=[Requirement(text="Python", kind="skill")],
    )
    source = build_resume()
    clients = SimpleNamespace(
        job_parser=StructuredClient([listing]),
        content_matcher=StructuredClient([candidate_selections(source)]),
        resume_writer=StructuredClient([global_rewrites]),
    )
    monkeypatch.setattr(parser_module, "get_llm_clients", lambda: clients)
    monkeypatch.setattr(content_extractor, "get_llm_clients", lambda: clients)
    monkeypatch.setattr(writer, "get_llm_clients", lambda: clients)

    parsed = parser_module.parse_listing("Senior Software Engineer")
    selected = content_extractor.match_resume(parsed, source)
    final = writer.global_resume_rewrite(parsed, selected)
    html = render_html(final)

    assert parsed.title == "Senior Software Engineer"
    assert clients.job_parser.calls == 1
    assert clients.content_matcher.calls == 1
    assert clients.resume_writer.calls == 1
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert source.candidate.name in html
    assert "Professional Summary" in html
    assert final.projects is not None
    assert len(final.projects.items) == 3
    assert "OpenTrail" in html
    assert "LogLens" in html


def test_pipeline_derives_supported_keywords_for_each_run(
    monkeypatch,
) -> None:
    listing = JobListing(
        title="Backend Engineer",
        requirements=[
            Requirement(
                text="Python",
                kind="skill",
                importance="critical",
            ),
            Requirement(
                text="Angular",
                kind="skill",
                importance="critical",
            ),
        ],
    )
    source = build_resume()
    clients = SimpleNamespace(
        job_parser=StructuredClient([listing, listing]),
        content_matcher=StructuredClient([candidate_selections(source)]),
        resume_writer=StructuredClient([global_rewrites]),
    )
    monkeypatch.setattr(parser_module, "get_llm_clients", lambda: clients)
    monkeypatch.setattr(content_extractor, "get_llm_clients", lambda: clients)
    monkeypatch.setattr(writer, "get_llm_clients", lambda: clients)
    monkeypatch.setattr(
        pipeline_module,
        "get_keyword_embedder",
        lambda: None,
    )
    plans = []

    result = pipeline_module.tailor_resume(
        source,
        "Backend Engineer",
        coverage_hook=plans.append,
    )

    assert plans[0].keywords_for("summary") == ["Python"]
    assert "Python" in result.skills.constraints.required_keywords
    assert "Angular" not in result.skills.constraints.required_keywords
    assert (
        plans[0].unsupported_requirements[0].requirement_text
        == "Angular"
    )


def test_writer_enforces_max_bullets() -> None:
    section = WorkExperienceSection(
        constraints=Constraints(max_bullets_per_item=2),
        items=[
            WorkExperienceItem(
                title="Engineer",
                company="Example",
                start_date="2020",
                bullets=["one", "two", "three"],
            )
        ],
    )

    limited = writer.apply_structural_limits(section)

    assert limited.items[0].bullets == ["one", "two"]


def test_writer_keeps_each_skill_in_only_its_original_category() -> None:
    product = SkillCategoryItem(name="Product", skills=["Roadmapping"])
    analytics = SkillCategoryItem(name="Analytics", skills=["SQL"])
    tools = SkillCategoryItem(name="Tools", skills=["Jira"])
    reference = SkillsSection(items=[product, analytics, tools])
    rewritten = reference.model_copy(
        update={
            "items": [
                product.model_copy(
                    update={"skills": ["Roadmapping", "SQL", "Jira"]}
                ),
                analytics.model_copy(update={"skills": ["SQL", "sql"]}),
                tools.model_copy(update={"skills": ["Jira", "SQL"]}),
            ]
        },
        deep=True,
    )

    limited = writer.apply_structural_limits(
        rewritten,
        reference=reference,
    )

    assert limited.items[0].skills == ["Roadmapping"]
    assert limited.items[1].skills == ["SQL"]
    assert limited.items[2].skills == ["Jira"]


def test_writer_item_limits_do_not_remove_jobs_or_degrees() -> None:
    resume = build_resume()
    work = resume.work_experience.model_copy(
        update={
            "constraints": resume.work_experience.constraints.model_copy(
                update={"max_items": 1}
            )
        }
    )
    education = resume.education.model_copy(
        update={
            "constraints": resume.education.constraints.model_copy(
                update={"max_items": 1}
            )
        }
    )

    assert len(writer.apply_structural_limits(work).items) == len(work.items)
    assert len(writer.apply_structural_limits(education).items) == len(
        education.items
    )


def test_writer_applies_writable_fields_to_original_item() -> None:
    section = WorkExperienceSection(
        items=[
            WorkExperienceItem(
                title="Engineer",
                company="Example",
                start_date="2020",
                bullets=["original"],
            )
        ],
    )
    rewrite = WorkExperienceItem.WritableForm(bullets=["rewritten"])

    rewritten = writer._apply_rewrites(
        section,
        [rewrite],
        context="work_experience",
    )

    assert rewritten.items[0].title == "Engineer"
    assert rewritten.items[0].bullets == ["rewritten"]


def test_renderer_enforces_max_bullets() -> None:
    resume = build_resume()
    html = render_html(resume)

    first_job = html.index("Software Engineer II")
    second_job = html.index("Software Engineer", first_job + 1)
    assert html[first_job:second_job].count("<li>") == 2


def test_global_rewrite_schema_requires_one_result_per_project() -> None:
    projects = build_resume().projects
    assert projects is not None
    selected = projects.model_copy(
        update={"items": projects.items[:3]},
        deep=True,
    )
    schema = writer._make_global_response_schema({"projects": selected})
    item_type = get_args(schema.model_fields["projects"].annotation)[0]

    with pytest.raises(ValidationError):
        schema(projects=[item_type(), item_type()])


def test_summary_alternatives_are_selected_by_deterministic_constraints() -> None:
    section = build_resume().summary.model_copy(deep=True)
    section.constraints.required_keywords = ["Python"]
    section.constraints.forbidden_phrases = ["results-driven"]
    rewrites = [
        writer.SummaryRewriteForm(
            content="Results-driven engineer.",
            alternatives=[
                "Engineer with Python experience.",
                "Engineer with general software experience.",
            ],
        )
    ]

    selected = writer._select_summary_candidates(section, rewrites)

    assert selected[0].content == "Engineer with Python experience."
