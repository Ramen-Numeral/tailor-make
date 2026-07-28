"""Deterministic observations for configured resume constraints."""

import re
from collections.abc import Iterable
from typing import Any

from app.features.ai_detection.schema import ConstraintCheck
from app.resume_schema.resume_schema import (
    EducationSection,
    Section,
    WorkExperienceSection,
)

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_METRIC_RE = re.compile(r"(?:\d[\d,.]*%?|\$\s?\d)")


def evaluate_constraints(section: Section) -> list[ConstraintCheck]:
    """Return checklist observations without making an overall decision."""
    constraints = section.constraints
    checks: list[ConstraintCheck] = []
    text = _section_text(section)
    words = _word_count(text)
    sentences = len(_SENTENCE_RE.findall(text))
    item_count = len(section.items)

    if isinstance(section, (EducationSection, WorkExperienceSection)):
        if constraints.max_items is not None:
            checks.append(
                ConstraintCheck(
                    constraint="max_items",
                    label="Canonical history preservation",
                    expected="All source entries preserved",
                    observed=f"{item_count} entries",
                    passed=True,
                    severity="advisory",
                )
            )
    else:
        _bounded_check(
            checks,
            "max_items",
            "Maximum items",
            item_count,
            constraints.max_items,
            "<=",
        )
    _bounded_check(
        checks,
        "min_items",
        "Minimum items",
        item_count,
        constraints.min_items,
        ">=",
    )
    _bounded_check(
        checks,
        "max_words",
        "Maximum words",
        words,
        constraints.max_words,
        "<=",
    )
    _bounded_check(
        checks,
        "min_words",
        "Minimum words",
        words,
        constraints.min_words,
        ">=",
    )
    _bounded_check(
        checks,
        "max_sentences",
        "Maximum sentences",
        sentences,
        constraints.max_sentences,
        "<=",
    )
    _bounded_check(
        checks,
        "max_skill_categories",
        "Maximum skill categories",
        item_count,
        constraints.max_skill_categories,
        "<=",
    )

    for index, item in enumerate(section.items):
        bullets = getattr(item, "bullets", None)
        if bullets is not None:
            _bounded_check(
                checks,
                "max_bullets_per_item",
                "Maximum bullets",
                len(bullets),
                constraints.max_bullets_per_item,
                "<=",
                index,
            )
            _bounded_check(
                checks,
                "min_bullets_per_item",
                "Minimum bullets",
                len(bullets),
                constraints.min_bullets_per_item,
                ">=",
                index,
            )
            if constraints.max_words_per_bullet is not None:
                for bullet_index, bullet in enumerate(bullets):
                    count = _word_count(bullet)
                    limit = constraints.max_words_per_bullet
                    checks.append(
                        ConstraintCheck(
                            constraint="max_words_per_bullet",
                            label=f"Maximum words in bullet {bullet_index + 1}",
                            expected=f"At most {limit}",
                            observed=f"{count} words",
                            passed=count <= limit,
                            item_index=index,
                        )
                    )

        for field_name, label, maximum, minimum in (
            (
                "skills",
                "Skills per category",
                constraints.max_skills_per_category,
                constraints.min_skills_per_category,
            ),
            ("coursework", "Courses", constraints.max_courses, None),
            (
                "technologies",
                "Technologies",
                constraints.max_technologies,
                None,
            ),
        ):
            values = getattr(item, field_name, None)
            if values is None:
                continue
            _bounded_check(
                checks,
                f"max_{field_name}_per_item",
                f"Maximum {label.lower()}",
                len(values),
                maximum,
                "<=",
                index,
            )
            _bounded_check(
                checks,
                f"min_{field_name}_per_item",
                f"Minimum {label.lower()}",
                len(values),
                minimum,
                ">=",
                index,
            )

        if constraints.require_metrics:
            item_bullets = bullets or []
            metric_count = sum(
                bool(_METRIC_RE.search(bullet))
                for bullet in item_bullets
            )
            checks.append(
                ConstraintCheck(
                    constraint="require_metrics",
                    label="Measurable impact",
                    expected="At least one bullet with a number or percentage",
                    observed=f"{metric_count} metric-bearing bullets",
                    passed=metric_count > 0,
                    item_index=index,
                )
            )

    lowered = text.casefold()
    for keyword in constraints.required_keywords:
        checks.append(
            ConstraintCheck(
                constraint="required_keyword",
                label=f'Required keyword: "{keyword}"',
                expected="Present",
                observed="Present" if keyword.casefold() in lowered else "Not found",
                passed=keyword.casefold() in lowered,
            )
        )
    for phrase in constraints.forbidden_phrases:
        found = phrase.casefold() in lowered
        checks.append(
            ConstraintCheck(
                constraint="forbidden_phrase",
                label=f'Forbidden phrase: "{phrase}"',
                expected="Absent",
                observed="Found" if found else "Not found",
                passed=not found,
            )
        )

    _visibility_checks(section, checks)
    return checks


def _bounded_check(
    checks: list[ConstraintCheck],
    constraint: str,
    label: str,
    observed: int,
    expected: int | None,
    operator: str,
    item_index: int | None = None,
) -> None:
    if expected is None:
        return
    passed = observed <= expected if operator == "<=" else observed >= expected
    checks.append(
        ConstraintCheck(
            constraint=constraint,
            label=label,
            expected=(
                f"At most {expected}" if operator == "<=" else f"At least {expected}"
            ),
            observed=str(observed),
            passed=passed,
            item_index=item_index,
        )
    )


def _visibility_checks(
    section: Section,
    checks: list[ConstraintCheck],
) -> None:
    for field_name, configured, label in (
        ("gpa", section.constraints.show_gpa, "GPA visibility"),
        (
            "coursework",
            section.constraints.show_coursework,
            "Coursework visibility",
        ),
    ):
        if configured is None:
            continue
        has_content = any(
            bool(getattr(item, field_name, None))
            for item in section.items
        )
        visible = configured and has_content
        checks.append(
            ConstraintCheck(
                constraint=f"show_{field_name}",
                label=label,
                expected="Shown" if configured else "Hidden",
                observed="Shown" if visible else "Hidden",
                passed=visible is configured,
            )
        )


def _section_text(section: Section) -> str:
    return " ".join(_string_values(section.model_dump(include={"items"})))


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))
