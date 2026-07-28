"""Deterministically trim least-relevant content to a rendered page limit."""

from collections.abc import Callable

from app.features.agent.schema import GlobalPageCheck, PageTrimAction
from app.features.renderer.renderer import count_pdf_pages, render_html
from app.features.validator.constraints import evaluate_constraints
from app.resume_schema.resume_schema import MutableResume, Section

PageCounter = Callable[[MutableResume], int]


def rendered_page_count(resume: MutableResume) -> int:
    return count_pdf_pages(render_html(resume))


def fit_resume_to_page_limit(
    resume: MutableResume,
    *,
    maximum_pages: int,
    max_trims: int,
    page_counter: PageCounter = rendered_page_count,
) -> tuple[MutableResume, GlobalPageCheck]:
    """Trim tail-ranked content until the physical render fits."""
    current = resume.model_copy(deep=True)
    initial_pages = page_counter(current)
    pages = initial_pages
    actions: list[PageTrimAction] = []

    while pages > maximum_pages and len(actions) < max_trims:
        trimmed = _trim_once(current)
        if trimmed is None:
            break
        current, action = trimmed
        actions.append(action)
        pages = page_counter(current)

    return current, GlobalPageCheck(
        maximum_pages=maximum_pages,
        initial_pages=initial_pages,
        final_pages=pages,
        passed=pages <= maximum_pages,
        trim_actions=actions,
    )


def _trim_once(
    resume: MutableResume,
) -> tuple[MutableResume, PageTrimAction] | None:
    for section_name in ("research", "projects"):
        section = getattr(resume, section_name, None)
        if section is None:
            continue
        minimum = section.constraints.min_items or 0
        if len(section.items) <= minimum:
            continue
        for item_index in range(len(section.items) - 1, -1, -1):
            removed = section.items[item_index]
            candidate = section.model_copy(
                update={
                    "items": [
                        *section.items[:item_index],
                        *section.items[item_index + 1:],
                    ]
                },
                deep=True,
            )
            if _safe(candidate):
                return _updated(
                    resume,
                    section_name,
                    candidate,
                    PageTrimAction(
                        section=section_name,
                        item_id=removed.id,
                        field="item",
                        removed=_item_label(removed),
                        reason="Removed the lowest-ranked safe selected item.",
                    ),
                )

    for section_name in ("projects", "work_experience", "research"):
        section = getattr(resume, section_name, None)
        if section is None:
            continue
        for item_index in range(len(section.items) - 1, -1, -1):
            item = section.items[item_index]
            bullets = getattr(item, "bullets", None)
            minimum = section.constraints.min_bullets_per_item or 0
            if bullets is None or len(bullets) <= minimum:
                continue
            removed = bullets[-1]
            updated_item = item.model_copy(
                update={"bullets": bullets[:-1]},
                deep=True,
            )
            items = list(section.items)
            items[item_index] = updated_item
            candidate = section.model_copy(
                update={"items": items},
                deep=True,
            )
            if _safe(candidate):
                return _updated(
                    resume,
                    section_name,
                    candidate,
                    PageTrimAction(
                        section=section_name,
                        item_id=item.id,
                        field="bullets",
                        removed=removed,
                        reason="Removed the lowest-ranked trailing bullet.",
                    ),
                )

    skills = resume.skills
    for item_index in range(len(skills.items) - 1, -1, -1):
        item = skills.items[item_index]
        minimum = skills.constraints.min_skills_per_category or 0
        if len(item.skills) <= minimum:
            continue
        for skill_index in range(len(item.skills) - 1, -1, -1):
            removed = item.skills[skill_index]
            if removed.casefold() in {
                keyword.casefold()
                for keyword in skills.constraints.required_keywords
            }:
                continue
            updated_skills = [
                *item.skills[:skill_index],
                *item.skills[skill_index + 1:],
            ]
            updated_item = item.model_copy(
                update={"skills": updated_skills},
                deep=True,
            )
            items = list(skills.items)
            items[item_index] = updated_item
            candidate = skills.model_copy(
                update={"items": items},
                deep=True,
            )
            if _safe(candidate):
                return _updated(
                    resume,
                    "skills",
                    candidate,
                    PageTrimAction(
                        section="skills",
                        item_id=item.id,
                        field="skills",
                        removed=removed,
                        reason="Removed the lowest-ranked non-required skill.",
                    ),
                )
    return None


def _safe(section: Section) -> bool:
    blocking = {
        "min_items",
        "min_bullets_per_item",
        "min_skills_per_item",
        "required_keyword",
    }
    return all(
        check.passed
        for check in evaluate_constraints(section)
        if check.constraint in blocking
    )


def _updated(
    resume: MutableResume,
    section_name: str,
    section: Section,
    action: PageTrimAction,
) -> tuple[MutableResume, PageTrimAction]:
    return (
        resume.model_copy(
            update={section_name: section},
            deep=True,
        ),
        action,
    )


def _item_label(item) -> str:
    for field in ("name", "title", "institution"):
        value = getattr(item, field, None)
        if value:
            return str(value)
    return str(item.id)
