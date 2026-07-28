from types import SimpleNamespace

import pytest

from app.features.job_listing_parser import parse_listing as parser_module
from app.features.job_listing_parser.listing_schema import (
    JobListing,
    Requirement,
)
from app.infrastructure.llm.errors import LLMInvalidOutputError


class ParserClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke_structured(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_job_listing_schema_defaults() -> None:
    listing = JobListing()

    assert listing.title is None
    assert listing.requirements == []


def test_requirement_normalizes_provider_name_variant() -> None:
    requirement = Requirement.model_validate(
        {
            "name": "project management",
            "kind": "skill",
            "source_text": "Excellent project management skills",
        }
    )

    assert requirement.text == "project management"
    assert requirement.source_text == "Excellent project management skills"


def test_requirement_uses_source_evidence_when_text_is_missing() -> None:
    requirement = Requirement.model_validate(
        {
            "kind": "experience",
            "source_text": "Proven experience in digital product management.",
            "min_years": 3,
        }
    )

    assert requirement.text == (
        "Proven experience in digital product management."
    )


def test_requirement_discards_malformed_provider_uuid() -> None:
    requirement = Requirement.model_validate(
        {
            "id": "e1a2b3c4-5678-90ab-cdef-1234567890ag",
            "text": "Stakeholder management",
            "kind": "skill",
        }
    )

    assert requirement.text == "Stakeholder management"
    assert str(requirement.id) != "e1a2b3c4-5678-90ab-cdef-1234567890ag"


def test_job_listing_removes_response_time_metadata() -> None:
    listing = JobListing(
        requirements=[
            Requirement(
                text="Typically responds to applications within 2 days",
                source_text="Typically responds to applications within 2 days",
                kind="other",
            ),
            Requirement(text="Project management", kind="skill"),
        ]
    )

    assert [item.text for item in listing.requirements] == [
        "Project management"
    ]


def test_job_listing_normalizes_camel_case_provider_metadata() -> None:
    listing = JobListing.model_validate(
        {
            "jobTitle": "Senior Product Manager - Digital Channels",
            "company": "F.N.B. Corp.",
            "location": {
                "street": "626 Washington Place",
                "city": "Pittsburgh",
                "state": "PA",
                "zip": "15219",
            },
            "employmentType": "Full-time",
            "commute": "Over an hour",
        }
    )

    assert listing.title == "Senior Product Manager - Digital Channels"
    assert listing.location == "Pittsburgh, PA 15219"
    assert listing.employment_type == "Full-time"
    assert listing.requirements == []


def test_parser_recovers_bullets_after_provider_retry_exhaustion(
    monkeypatch,
) -> None:
    listing_text = """
Senior Product Manager

Qualifications:
- Proven experience in digital product management
- Strong understanding of SEO and web analytics
- Excellent cross-functional project management skills
- Ability to translate data insights into actionable strategies

Benefits:
- Medical coverage
""" + ("Product organization background. " * 30)
    client = ParserClient(
        [
            JobListing(title="Senior Product Manager"),
            LLMInvalidOutputError("metadata-only provider output"),
        ]
    )
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    parsed = parser_module.parse_listing(listing_text)

    assert parsed.title == "Senior Product Manager"
    assert len(parsed.requirements) == 4
    assert all("Medical coverage" != item.text for item in parsed.requirements)


def test_attached_groq_requirement_shape_is_schema_valid() -> None:
    listing = JobListing.model_validate(
        {
            "location": "Pittsburgh, PA 15219",
            "requirements": [
                {
                    "kind": "education",
                    "importance": "critical",
                    "required": True,
                    "source_text": "BA or BS",
                    "level": "Bachelor's",
                },
                {
                    "kind": "skill",
                    "importance": "critical",
                    "required": True,
                    "source_text": "MS Excel - Intermediate Level",
                    "name": "MS Excel",
                },
            ],
        }
    )

    assert [item.text for item in listing.requirements] == [
        "BA or BS",
        "MS Excel",
    ]


def test_job_listing_normalizes_top_level_requirement_array() -> None:
    listing = JobListing.model_validate(
        [
            {
                "text": "Full-time",
                "kind": "other",
                "importance": "critical",
                "required": True,
                "source_text": "Job type",
            },
            {
                "text": "Over an hour from 1425 S 19th St",
                "kind": "other",
                "importance": "critical",
                "required": True,
                "source_text": "Estimated commute",
            },
            {
                "text": "Proven experience in digital product management",
                "kind": "experience",
                "importance": "critical",
                "required": True,
                "source_text": (
                    "Proven experience in digital product management, "
                    "preferably with a focus on web platforms."
                ),
            },
        ]
    )

    assert [requirement.text for requirement in listing.requirements] == [
        "Proven experience in digital product management"
    ]


def test_parse_listing_returns_complete_first_response(monkeypatch) -> None:
    expected = JobListing(
        title="Senior Engineer",
        requirements=[Requirement(text="Python", kind="skill")],
    )
    client = ParserClient([expected])
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    assert parser_module.parse_listing("Senior Engineer") == expected
    assert client.calls == 1


def test_parse_listing_merges_partial_retries(monkeypatch) -> None:
    client = ParserClient(
        [
            JobListing(title="Engineer"),
            JobListing(
                requirements=[Requirement(text="Python", kind="skill")]
            ),
        ]
    )
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    result = parser_module.parse_listing("Engineer", max_attempts=2)

    assert result.title == "Engineer"
    assert result.requirements[0].text == "Python"
    assert client.calls == 2


def test_parse_listing_unions_variable_requirement_counts(monkeypatch) -> None:
    def listing(count):
        return JobListing(
            title="Product Manager",
            requirements=[
                Requirement(
                    text=f"Requirement {index}",
                    kind="skill",
                )
                for index in range(count)
            ],
        )

    client = ParserClient([listing(3), listing(11), listing(14)])
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    result = parser_module.parse_listing(
        "Product Manager",
        max_attempts=3,
        minimum_attempts=2,
    )

    assert len(result.requirements) == 14
    assert client.calls == 3


def test_parse_listing_stops_when_second_pass_adds_nothing(
    monkeypatch,
) -> None:
    first = JobListing(
        title="Product Manager",
        requirements=[Requirement(text="Roadmaps", kind="skill")],
    )
    duplicate = JobListing(
        title="Product Manager",
        requirements=[Requirement(text="roadmaps", kind="skill")],
    )
    client = ParserClient([first, duplicate])
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    result = parser_module.parse_listing(
        "Product Manager",
        max_attempts=3,
        minimum_attempts=2,
    )

    assert len(result.requirements) == 1
    assert client.calls == 2


def test_long_listing_recovers_explicit_requirement_bullets(
    monkeypatch,
) -> None:
    posting = """Project Manager
About the company
""" + ("We build useful products. " * 30) + """
Qualifications:
- Five years of project management experience
- Proficiency with Jira and Excel
- Bachelor's degree in engineering
- PMP certification preferred
Benefits:
- Health insurance
- Typically responds to applications within 2 days
"""
    client = ParserClient([
        JobListing(title="Project Manager"),
        JobListing(title="Project Manager"),
    ])
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(job_parser=client),
    )

    result = parser_module.parse_listing(
        posting,
        max_attempts=2,
        minimum_attempts=2,
    )

    assert len(result.requirements) == 4
    assert all("responds" not in item.text for item in result.requirements)
    assert result.requirements[-1].required is False


@pytest.mark.parametrize("text, attempts", [("", 1), ("Engineer", 0)])
def test_parse_listing_rejects_invalid_inputs(text, attempts) -> None:
    with pytest.raises(ValueError):
        parser_module.parse_listing(text, max_attempts=attempts)
