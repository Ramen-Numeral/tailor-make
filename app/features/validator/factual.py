"""Deterministic factual-integrity checks for rewritten sections."""

import re
from typing import Any

from app.features.ai_detection.schema import ConstraintCheck
from app.resume_schema.resume_schema import Section

_NUMBER_RE = re.compile(
    r"(?<!\w)(?:\$\s*)?\d[\d,.]*(?:%|\+|[KMBkmb])?(?!\w)"
)


def evaluate_factual_integrity(
    source: Section,
    candidate: Section,
    *,
    supplemental_evidence: str | None = None,
) -> list[ConstraintCheck]:
    """Detect new metrics and mutations to fields that were locked."""
    checks: list[ConstraintCheck] = []
    source_by_id = {item.id: item for item in source.items}
    source_terms = _structured_terms(source)
    candidate_terms = _structured_terms(candidate)
    supplemental = (supplemental_evidence or "").casefold()
    introduced_terms = sorted(
        term
        for term in candidate_terms - source_terms
        if term not in supplemental
    )
    checks.append(
        ConstraintCheck(
            constraint="factual_structured_terms",
            label="Skills and technologies",
            expected="No structured terms absent from the source section",
            observed=(
                f"Introduced: {', '.join(introduced_terms)}"
                if introduced_terms
                else "No new structured terms"
            ),
            passed=not introduced_terms,
        )
    )

    for index, item in enumerate(candidate.items):
        original = source_by_id.get(item.id)
        if original is None:
            checks.append(
                ConstraintCheck(
                    constraint="factual_item_provenance",
                    label="Source item provenance",
                    expected="Every item must exist in the selected source",
                    observed=f"Unknown item ID {item.id}",
                    passed=False,
                    item_index=index,
                )
            )
            continue

        writable = set(type(original).WritableForm.model_fields)
        for field_name in type(original).model_fields:
            if field_name == "id" or field_name in writable:
                continue
            before = getattr(original, field_name, None)
            after = getattr(item, field_name, None)
            checks.append(
                ConstraintCheck(
                    constraint="factual_locked_field",
                    label=f"Locked field: {field_name}",
                    expected=_display(before),
                    observed=_display(after),
                    passed=before == after,
                    item_index=index,
                )
            )

        source_numbers = set(_NUMBER_RE.findall(_item_text(original)))
        candidate_numbers = set(_NUMBER_RE.findall(_item_text(item)))
        supplemental_numbers = set(_NUMBER_RE.findall(supplemental_evidence or ""))
        introduced = sorted(
            candidate_numbers - source_numbers - supplemental_numbers
        )
        checks.append(
            ConstraintCheck(
                constraint="factual_numbers",
                label="Metrics and numbers",
                expected="No numbers absent from the source item",
                observed=(
                    f"Introduced: {', '.join(introduced)}"
                    if introduced
                    else "No new numbers"
                ),
                passed=not introduced,
                item_index=index,
            )
        )

    return checks


def _item_text(item) -> str:
    return " ".join(_strings(item.model_dump(exclude={"id"})))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)


def _display(value: Any) -> str:
    if value in (None, "", []):
        return "Empty"
    return str(value)


def _structured_terms(section: Section) -> set[str]:
    return {
        str(term).strip().casefold()
        for item in section.items
        for field in ("skills", "technologies", "coursework")
        for term in (getattr(item, field, None) or [])
        if str(term).strip()
    }
