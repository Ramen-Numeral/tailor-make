"""Compact prompts for factual resume rewriting."""

import json

from app.features.job_listing_parser.listing_schema import JobListing
from app.resume_schema.resume_schema import Section


def make_writer_prompt(
    *,
    job_listing: JobListing,
    section: Section,
    special_instructions: str | None = None,
) -> str:
    """Build the optional one-section rewrite prompt."""
    return _prompt(
        job_listing,
        {
            "section": {
                "constraints": _constraints(section),
                "items": _items(section),
            }
        },
        special_instructions,
    )


def make_global_writer_prompt(
    *,
    job_listing: JobListing,
    sections: dict[str, Section],
    special_instructions: str | None = None,
) -> str:
    """Build one compact whole-resume rewrite prompt."""
    payload = {
        name: {
            "constraints": _constraints(section),
            "items": _items(section),
        }
        for name, section in sections.items()
    }
    return _prompt(job_listing, payload, special_instructions)


def _prompt(
    job: JobListing,
    sections: dict,
    special_instructions: str | None,
) -> str:
    data = {
        "job": {
            "title": job.title,
            "requirements": [
                {
                    "text": requirement.text,
                    "kind": requirement.kind,
                    "required": requirement.required,
                }
                for requirement in job.requirements
            ],
        },
        "sections": sections,
    }
    instruction = special_instructions or "Use concise, natural ATS wording."
    return f"""Create an evidence-grounded factual draft for all supplied
resume items in one pass.
Return one output item per input item in identical section/list order. Modify
only fields allowed by the response schema; all omitted fields are locked.
Preserve every fact, employer, title, date, institution, technology, metric,
responsibility, and outcome. Never invent or strengthen claims. Prefer
supported required qualifications, follow constraints, vary phrasing, and
return only a schema-valid JSON object. Within the skills section, include each
skill keyword in at most one category and preserve its original category when
it already exists. Nested accomplishment fields are
When the response schema offers `alternatives` for a professional summary,
return the primary version plus up to two materially different candidates:
one conservative and one more assertive but fully supported.
Additional instruction: {instruction}

DATA:
{_json(data)}"""


def _items(section: Section) -> list[dict]:
    return [
        item.model_dump(
            mode="json",
            exclude={"id"},
            exclude_none=True,
        )
        for item in section.items
    ]


def _constraints(section: Section) -> dict:
    keep = {
        "max_words",
        "max_sentences",
        "max_bullets_per_item",
        "max_words_per_bullet",
        "max_skills_per_category",
        "max_technologies",
        "required_keywords",
        "forbidden_phrases",
        "require_metrics",
    }
    return section.constraints.model_dump(
        include=keep,
        exclude_none=True,
        exclude_defaults=True,
    )


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
