"""Compact prompts for candidate-content selection."""

import json

from app.features.job_listing_parser.listing_schema import JobListing
from app.resume_schema.resume_schema import Section


def make_resume_prompt(
    *,
    job_listing: JobListing,
    sections: dict[str, tuple[Section, int, int]],
) -> str:
    """Build one compact ranking prompt covering all candidate sections."""
    payload = {
        "job": _job_payload(job_listing),
        "sections": {
            name: {
                "minimum_count": minimum,
                "maximum_count": maximum,
                "items": [
                    _item_payload(item, index)
                    for index, item in enumerate(section.items)
                ],
            }
            for name, (section, minimum, maximum) in sections.items()
        },
    }
    return _selection_instruction(payload)


def _selection_instruction(payload: dict) -> str:
    return """Select existing resume items most relevant to the job. This is
selection only: never rewrite content or invent IDs. IDs are compact zero-based
indexes local to each section. For each section return between `minimum_count`
and `maximum_count` unique ID strings, best first.
Include an item only when it adds relevant evidence. Favor direct required
evidence, then relevant duties/skills and complementary coverage. Return only
a schema-valid JSON object.

DATA:
""" + _json(payload)


def _job_payload(job: JobListing) -> dict:
    return {
        key: value
        for key, value in {
            "title": job.title,
            "seniority": job.seniority,
            "requirements": [
                {
                    "text": requirement.text,
                    "kind": requirement.kind,
                    "required": requirement.required,
                }
                for requirement in job.requirements
            ],
        }.items()
        if value not in (None, [], "")
    }


def _item_payload(item, index: int) -> dict:
    return {
        "id": str(index),
        "content": item.model_dump(
            mode="json",
            exclude={"id"},
            exclude_none=True,
        ),
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
