"""Resume rewriting with one global model call for the MVP pipeline."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, conlist, create_model

from app.bootstrap import get_llm_clients
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.writer.prompts import make_global_writer_prompt, make_writer_prompt
from app.resume_schema.resume_schema import (
    MutableResume,
    RESUME_SECTION_FIELDS,
    Resume,
    Section,
    SkillsSection,
    EducationSection,
    WorkExperienceSection,
    WRITABLE_SECTION_FIELDS,
)


class SummaryRewriteForm(BaseModel):
    content: str | None = None
    alternatives: list[str] = Field(default_factory=list, max_length=2)


def global_resume_rewrite(
    job_listing: JobListing,
    resume: MutableResume,
    special_instructions: str | None = None,
) -> MutableResume:
    """Rewrite all writable resume sections in one structured model call."""
    sections = {
        name: section
        for name in WRITABLE_SECTION_FIELDS
        if (section := getattr(resume, name, None)) is not None
        and section.items
    }
    if not sections:
        return apply_resume_limits(resume)

    writer = get_llm_clients().resume_writer
    cache_key = content_key(
        "global_resume_rewrite",
        object_identity(writer),
        job_listing,
        resume,
        special_instructions,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return MutableResume.model_validate(cached)

    response_schema = _make_global_response_schema(sections)
    response = writer.invoke_structured(
        prompt=make_global_writer_prompt(
            job_listing=job_listing,
            sections=sections,
            special_instructions=special_instructions,
        ),
        schema=response_schema,
        temperature=0,
        max_tokens=1800,
        trace_context="global_resume_rewrite",
    )

    rewritten_sections = {
        name: _apply_rewrites(
            section,
            (
                _select_summary_candidates(
                    section,
                    getattr(response, name),
                )
                if name == "summary"
                else getattr(response, name)
            ),
            context=name,
        )
        for name, section in sections.items()
    }

    result = resume.model_copy(update=rewritten_sections, deep=True)
    rewritten = MutableResume.model_validate(result.model_dump())
    set_cached(cache_key, rewritten.model_dump(mode="json"))
    return rewritten


def write_section(
    job_listing: JobListing,
    section: Section,
    special_instructions: str | None = None,
    target_item_indices: list[int] | None = None,
) -> Section:
    """Rewrite one section for the optional AI-detection retry hook."""
    if not section.items:
        return section

    indexes = (
        list(range(len(section.items)))
        if target_item_indices is None
        else list(dict.fromkeys(target_item_indices))
    )
    if not indexes:
        return section
    if any(index < 0 or index >= len(section.items) for index in indexes):
        raise ValueError("target_item_indices contains an unknown item")
    target_section = section.model_copy(
        update={"items": [section.items[index] for index in indexes]},
        deep=True,
    )

    writer = get_llm_clients().resume_writer
    cache_key = content_key(
        "section_rewrite",
        object_identity(writer),
        job_listing,
        target_section,
        special_instructions,
        indexes,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return _merge_targeted_items(
            section,
            type(section).model_validate(cached),
            indexes,
        )

    item_type = type(target_section.items[0])
    if any(type(item) is not item_type for item in target_section.items):
        raise TypeError(f"{section.heading!r} contains mixed item types")

    response_schema = create_model(
        f"{item_type.__name__}RewriteResponse",
        __config__=ConfigDict(extra="forbid"),
        items=(list[item_type.WritableForm], ...),
    )
    response = writer.invoke_structured(
        prompt=make_writer_prompt(
            job_listing=job_listing,
            section=target_section,
            special_instructions=special_instructions,
        ),
        schema=response_schema,
        temperature=0,
        trace_context=f"section_rewrite section={section.heading}",
    )
    rewritten = _apply_rewrites(
        target_section,
        response.items,
        context=section.heading,
    )
    set_cached(cache_key, rewritten.model_dump(mode="json"))
    return _merge_targeted_items(section, rewritten, indexes)


def _merge_targeted_items(
    original: Section,
    rewritten: Section,
    indexes: list[int],
) -> Section:
    items = list(original.items)
    for index, item in zip(indexes, rewritten.items, strict=True):
        items[index] = item
    return original.model_copy(update={"items": items}, deep=True)


def apply_resume_limits(
    resume: MutableResume,
    reference: Resume | None = None,
) -> MutableResume:
    """Apply deterministic section constraints to a resume copy."""
    updates = {
        name: apply_structural_limits(
            section,
            reference=(
                getattr(reference, name, None)
                if reference is not None
                else None
            ),
        )
        for name in RESUME_SECTION_FIELDS
        if (section := getattr(resume, name, None)) is not None
    }
    result = resume.model_copy(update=updates, deep=True)
    return MutableResume.model_validate(result.model_dump())


def apply_structural_limits(
    section: Section,
    *,
    reference: Section | None = None,
) -> Section:
    """Apply configured list-size maximums without modifying prose."""
    constraints = section.constraints
    item_limits = [
        limit
        for limit in (
            constraints.max_items,
            constraints.max_skill_categories,
        )
        if limit is not None
    ]
    max_items = min(item_limits) if item_limits else None
    preserve_all_items = isinstance(
        section,
        (EducationSection, WorkExperienceSection),
    )
    items = (
        section.items
        if max_items is None or preserve_all_items
        else section.items[:max_items]
    )
    if isinstance(section, SkillsSection):
        items = _deduplicate_skill_categories(
            items,
            (
                reference.items
                if isinstance(reference, SkillsSection)
                else []
            ),
        )
    limited_items = []

    for item in items:
        updates = {}
        for field_name, limit in (
            ("bullets", constraints.max_bullets_per_item),
            ("skills", constraints.max_skills_per_category),
            ("coursework", constraints.max_courses),
            ("technologies", constraints.max_technologies),
        ):
            value = getattr(item, field_name, None)
            if value is not None and limit is not None:
                updates[field_name] = value[:limit]

        limited_items.append(
            item.model_copy(update=updates, deep=True) if updates else item
        )

    return section.model_copy(update={"items": limited_items}, deep=True)


def _deduplicate_skill_categories(items, reference_items) -> list:
    """Keep each normalized skill in one category, preferring its source owner."""
    current_owners: dict[str, list] = {}
    for item in items:
        for skill in item.skills:
            key = _skill_key(skill)
            if key and item.id not in current_owners.setdefault(key, []):
                current_owners[key].append(item.id)

    reference_owners = {}
    for item in reference_items:
        for skill in item.skills:
            reference_owners.setdefault(_skill_key(skill), item.id)

    selected_owners = {
        key: (
            reference_owners[key]
            if reference_owners.get(key) in owners
            else owners[0]
        )
        for key, owners in current_owners.items()
    }
    deduplicated = []
    for item in items:
        retained = []
        seen: set[str] = set()
        for skill in item.skills:
            key = _skill_key(skill)
            if (
                not key
                or key in seen
                or selected_owners.get(key) != item.id
            ):
                continue
            seen.add(key)
            retained.append(skill.strip())
        deduplicated.append(
            item.model_copy(update={"skills": retained}, deep=True)
        )
    return deduplicated


def _skill_key(skill: str) -> str:
    return " ".join(skill.casefold().split())


def _make_global_response_schema(
    sections: dict[str, Section],
) -> type[BaseModel]:
    """Build a compact top-level response schema with one list per section."""
    fields: dict[str, Any] = {}
    for name, section in sections.items():
        item_type = type(section.items[0])
        if any(type(item) is not item_type for item in section.items):
            raise TypeError(f"{section.heading!r} contains mixed item types")
        item_count = len(section.items)
        writable_form = (
            SummaryRewriteForm
            if name == "summary"
            else item_type.WritableForm
        )
        fields[name] = (
            conlist(
                writable_form,
                min_length=item_count,
                max_length=item_count,
            ),
            ...,
        )

    return create_model(
        "GlobalResumeRewrite",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _select_summary_candidates(
    section: Section,
    rewrites: list[SummaryRewriteForm],
) -> list[BaseModel]:
    selected = []
    constraints = section.constraints
    for rewrite in rewrites:
        candidates = [
            value.strip()
            for value in [rewrite.content, *rewrite.alternatives]
            if value and value.strip()
        ]
        content = (
            max(
                candidates,
                key=lambda value: _summary_candidate_score(
                    value,
                    constraints,
                ),
            )
            if candidates
            else rewrite.content
        )
        selected.append(
            type(section.items[0]).WritableForm(content=content)
        )
    return selected


def _summary_candidate_score(text: str, constraints) -> tuple:
    lowered = text.casefold()
    words = len(text.split())
    required = sum(
        keyword.casefold() in lowered
        for keyword in constraints.required_keywords
    )
    forbidden = sum(
        phrase.casefold() in lowered
        for phrase in constraints.forbidden_phrases
    )
    within_max = (
        constraints.max_words is None
        or words <= constraints.max_words
    )
    within_min = (
        constraints.min_words is None
        or words >= constraints.min_words
    )
    concrete = int(any(character.isdigit() for character in text))
    return (
        -forbidden,
        required,
        int(within_max and within_min),
        concrete,
        -abs(words - 50),
    )


def _apply_rewrites(
    section: Section,
    rewrites: list[BaseModel],
    *,
    context: str,
) -> Section:
    """Merge writable responses into original items and enforce limits."""
    if len(rewrites) != len(section.items):
        raise ValueError(
            f"Expected {len(section.items)} rewrites for {context!r}; "
            f"received {len(rewrites)}"
        )

    items = [
        original.model_copy(
            update=rewrite.model_dump(
                include=set(type(original).WritableForm.model_fields),
                exclude_unset=True,
            ),
            deep=True,
        )
        for original, rewrite in zip(section.items, rewrites, strict=True)
    ]
    return apply_structural_limits(
        section.model_copy(update={"items": items}, deep=True),
        reference=section,
    )
