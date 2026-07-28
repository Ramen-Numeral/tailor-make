
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conlist,
    create_model,
    model_validator,
)

from app.bootstrap import get_llm_clients
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.content_extractor.prompt import make_resume_prompt
from app.resume_schema.resume_schema import (
    MutableResume,
    RESUME_SECTION_FIELDS,
    Resume,
    Section,
    SectionItem,
)

CANONICAL_HISTORY_SECTIONS = {"education", "work_experience"}


class RankedSelection(BaseModel):
    """Item IDs ordered from most relevant to least relevant."""

    ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids(self) -> "RankedSelection":
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Ranked item IDs must be unique")

        return self


def resolve_ranked_items(
    *,
    section: Section,
    selection: RankedSelection,
    minimum_count: int | None = None,
    maximum_count: int | None = None,
) -> list[SectionItem]:
    """Resolve ranked IDs to the original, untouched section items."""
    if minimum_count is not None and len(selection.ids) < minimum_count:
        raise ValueError(
            f"Matcher returned {len(selection.ids)} choices; "
            f"minimum is {minimum_count}"
        )
    if maximum_count is not None and len(selection.ids) > maximum_count:
        raise ValueError(
            f"Matcher returned {len(selection.ids)} choices; "
            f"maximum is {maximum_count}"
        )

    try:
        return [
            section.get_item(item_id)
            for item_id in selection.ids
        ]
    except KeyError as error:
        raise ValueError(
            f"Matcher returned an unknown item ID: {error}"
        ) from error


def match_resume(
    job_listing: JobListing,
    resume: Resume,
) -> MutableResume:
    """Select all resume sections in one model call."""
    source_sections = {
        name: section
        for name in RESUME_SECTION_FIELDS
        if (section := getattr(resume, name, None)) is not None
        and section.items
        and section.constraints.max_items != 0
        and name not in CANONICAL_HISTORY_SECTIONS
    }
    sections = {
        name: (section, *_selection_bounds(section))
        for name, section in source_sections.items()
    }

    if any(
        minimum < 0 or maximum < 0
        for _, minimum, maximum in sections.values()
    ):
        raise ValueError("section item constraints cannot be negative")
    if not sections:
        return MutableResume.model_validate(resume.model_dump())

    matcher = get_llm_clients().content_matcher
    cache_key = content_key(
        "content_selection",
        object_identity(matcher),
        job_listing,
        resume,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return MutableResume.model_validate(cached)

    response_schema = create_model(
        "CandidateContentSelection",
        __config__=ConfigDict(extra="forbid"),
        **{
            name: (
                conlist(
                    str,
                    min_length=minimum,
                    max_length=maximum,
                ),
                ...,
            )
            for name, (_, minimum, maximum) in sections.items()
        },
    )
    response = matcher.invoke_structured(
        prompt=make_resume_prompt(job_listing=job_listing, sections=sections),
        schema=response_schema,
        temperature=0,
        max_tokens=800,
        structured_strict=True,
        trace_context="candidate_content_selection",
    )

    selected = {
        name: section.model_copy(
            update={
                "items": resolve_ranked_items(
                    section=section,
                    selection=RankedSelection(
                        ids=_expand_compact_ids(
                            getattr(response, name),
                            section,
                        )
                    ),
                    minimum_count=minimum,
                    maximum_count=maximum,
                )
            },
            deep=True,
        )
        for name, (section, minimum, maximum) in sections.items()
    }

    values = {
        "candidate": resume.candidate.model_copy(deep=True),
        **{
            name: selected.get(name, getattr(resume, name, None))
            for name in RESUME_SECTION_FIELDS
        },
    }
    result = MutableResume(**values)
    set_cached(cache_key, result.model_dump(mode="json"))
    return result


def _selection_bounds(section: Section) -> tuple[int, int]:
    limits = (
        section.constraints.max_items,
        section.constraints.max_skill_categories,
        len(section.items),
    )
    maximum = min(limit for limit in limits if limit is not None)
    configured_minimum = section.constraints.min_items
    minimum = (
        min(1, maximum)
        if configured_minimum is None
        else min(configured_minimum, maximum)
    )
    return minimum, maximum


def _expand_compact_ids(
    values: list[str | UUID],
    section: Section,
) -> list[UUID]:
    """Map prompt-local numeric indexes back to canonical item UUIDs."""
    expanded: list[UUID] = []
    for value in values:
        rendered = str(value)
        if rendered.isdigit():
            index = int(rendered)
            if not 0 <= index < len(section.items):
                raise ValueError(
                    f"Matcher returned an unknown item index: {index}"
                )
            expanded.append(section.items[index].id)
        else:
            expanded.append(UUID(rendered))
    return expanded
