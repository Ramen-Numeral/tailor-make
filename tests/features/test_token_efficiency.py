from types import SimpleNamespace
from typing import get_args

import pytest

from app.features.content_extractor.prompt import make_resume_prompt
from app.features.job_listing_parser import parse_listing as parser_module
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.features.keyword_evidence.adjudicator import judge_ambiguous_evidence
from app.features.keyword_evidence.schema import ResumeEvidence
from app.features.writer import writer as writer_module
from app.infrastructure.cache import clear_stage_cache
from app.resume_schema.resume_schema import WorkExperienceItem, WorkExperienceSection
from config.resume.candidate_profile import build_resume


@pytest.fixture(autouse=True)
def isolated_stage_cache():
    clear_stage_cache()
    yield
    clear_stage_cache()


class StructuredClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.prompts = []

    def invoke_structured(self, *, prompt, schema, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response(schema) if callable(self.response) else self.response


def test_deterministic_first_parser_uses_one_call_and_then_cache(
    monkeypatch,
) -> None:
    text = """
Senior Product Manager
Qualifications:
- Proven digital product management experience
- Strong understanding of SEO and web analytics
- Excellent cross-functional project management
- Ability to translate data into actionable strategy
""" + ("Digital product organization. " * 30)
    client = StructuredClient(JobListing(title="Senior Product Manager"))
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    first = parser_module.parse_listing(text, minimum_attempts=1)
    second = parser_module.parse_listing(text, minimum_attempts=1)

    assert len(first.requirements) == 4
    assert second == first
    assert client.calls == 1


def test_content_selection_prompt_uses_compact_local_indexes() -> None:
    resume = build_resume()
    prompt = make_resume_prompt(
        job_listing=JobListing(title="Engineer"),
        sections={"projects": (resume.projects, 1, 2)},
    )

    assert '"id":"0"' in prompt
    assert str(resume.projects.items[0].id) not in prompt


def test_evidence_judge_maps_compact_indexes_back_to_provenance() -> None:
    requirement = Requirement(text="Project management", kind="skill")
    source_item = build_resume().work_experience.items[0]
    evidence = [
        ResumeEvidence(
            evidence_id=f"work_experience:{source_item.id}:bullets:0",
            section="work_experience",
            item_id=source_item.id,
            field="bullets",
            text="Owned a roadmap and coordinated delivery.",
        )
    ]

    def response(schema):
        return schema(
            judgments=[
                {
                    "requirement": 0,
                    "support": "supported",
                    "evidence": [0],
                    "reason": "Roadmap ownership supports project management.",
                    "relationship": "transferable",
                    "safe_keywords": ["project management"],
                }
            ]
        )

    client = StructuredClient(response)
    result = judge_ambiguous_evidence([requirement], evidence, client)

    assert result.judgments[0].requirement_id == requirement.id
    assert result.judgments[0].evidence_ids == [evidence[0].evidence_id]
    assert '"id":0' in client.prompts[0]
    assert str(requirement.id) not in client.prompts[0]


def test_targeted_section_rewrite_sends_only_failed_item_and_caches(
    monkeypatch,
) -> None:
    section = WorkExperienceSection(
        items=[
            WorkExperienceItem(
                title="Engineer",
                company="One",
                start_date="2020",
                bullets=["Keep this."],
            ),
            WorkExperienceItem(
                title="Engineer",
                company="Two",
                start_date="2021",
                bullets=["Rewrite this."],
            ),
        ]
    )

    def response(schema):
        item_type = get_args(schema.model_fields["items"].annotation)[0]
        return schema(items=[item_type(bullets=["Targeted rewrite."])])

    client = StructuredClient(response)
    monkeypatch.setattr(
        writer_module,
        "get_llm_clients",
        lambda: SimpleNamespace(resume_writer=client),
    )

    first = writer_module.write_section(
        JobListing(title="Engineer"),
        section,
        target_item_indices=[1],
    )
    second = writer_module.write_section(
        JobListing(title="Engineer"),
        section,
        target_item_indices=[1],
    )

    assert first.items[0].bullets == ["Keep this."]
    assert first.items[1].bullets == ["Targeted rewrite."]
    assert second == first
    assert client.calls == 1
