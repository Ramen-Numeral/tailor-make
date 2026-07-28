"""UUID-aware, word-level diffs across resume pipeline stages."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from app.features.resume_diff.schema import DiffToken, DiffValue, FieldDiff
from app.resume_schema.resume_schema import RESUME_SECTION_FIELDS, Resume

if TYPE_CHECKING:
    from app.features.agent.schema import PageTrimAction

_TOKEN_RE = re.compile(r"\s+|[^\s]+")


def build_resume_diffs(
    source: Resume,
    selected: Resume,
    final: Resume,
    *,
    page_trim_actions: list[PageTrimAction] | None = None,
) -> list[FieldDiff]:
    """Explain selection, rewriting, and page trimming as separate changes."""
    trim_actions = page_trim_actions or []
    trim_lookup = {
        (action.section, action.item_id, action.field): action
        for action in trim_actions
    }
    diffs: list[FieldDiff] = []

    for section_name in RESUME_SECTION_FIELDS:
        source_section = getattr(source, section_name, None)
        selected_section = getattr(selected, section_name, None)
        final_section = getattr(final, section_name, None)
        if source_section is None:
            continue

        selected_by_id = {
            item.id: item
            for item in (selected_section.items if selected_section else [])
        }
        final_by_id = {
            item.id: item
            for item in (final_section.items if final_section else [])
        }
        source_ids = {item.id for item in source_section.items}

        for original_item in source_section.items:
            item_id = original_item.id
            item_label = _item_label(original_item)
            final_item = final_by_id.get(item_id)
            if final_item is None:
                if not _has_item_content(original_item):
                    continue
                if item_id not in selected_by_id:
                    change_type = "not_selected"
                    reason = (
                        "Excluded during job-relevance selection before rewriting."
                    )
                elif (
                    action := trim_lookup.get(
                        (section_name, item_id, "item")
                    )
                ) is not None:
                    change_type = "trimmed_for_page_limit"
                    reason = action.reason
                else:
                    change_type = "removed"
                    reason = "Removed after selection."
                original_text = _item_text(original_item)
                diffs.append(
                    FieldDiff(
                        section=section_name,
                        item_id=item_id,
                        item_label=item_label,
                        field="__item__",
                        change_type=change_type,
                        original=original_text,
                        reason=reason,
                        tokens=inline_diff(original_text, ""),
                    )
                )
                continue

            for field_name in type(original_item).model_fields:
                if field_name == "id":
                    continue
                before = getattr(original_item, field_name, None)
                after = getattr(final_item, field_name, None)
                if before == after:
                    continue
                action = trim_lookup.get(
                    (section_name, item_id, field_name)
                )
                change_type = (
                    "trimmed_for_page_limit"
                    if action is not None
                    else "rewritten"
                )
                reason = (
                    action.reason
                    if action is not None
                    else (
                        "Changed during job tailoring and retained by the "
                        "section evaluation."
                    )
                )
                before_value = _diff_value(before)
                after_value = _diff_value(after)
                diffs.append(
                    FieldDiff(
                        section=section_name,
                        item_id=item_id,
                        item_label=item_label,
                        field=field_name,
                        change_type=change_type,
                        original=before_value,
                        final=after_value,
                        reason=reason,
                        tokens=inline_diff(
                            _value_text(before_value),
                            _value_text(after_value),
                        ),
                    )
                )

        for final_item in final_by_id.values():
            if final_item.id in source_ids:
                continue
            if not _has_item_content(final_item):
                continue
            final_text = _item_text(final_item)
            diffs.append(
                FieldDiff(
                    section=section_name,
                    item_id=final_item.id,
                    item_label=_item_label(final_item),
                    field="__item__",
                    change_type="added",
                    final=final_text,
                    reason="The final resume contains an item absent from the source.",
                    tokens=inline_diff("", final_text),
                )
            )

    return diffs


def inline_diff(original: str, final: str) -> list[DiffToken]:
    """Return whitespace-preserving word groups for inline rendering."""
    left = _TOKEN_RE.findall(original)
    right = _TOKEN_RE.findall(final)
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    tokens: list[DiffToken] = []

    for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if operation == "equal":
            _append(tokens, "equal", "".join(left[left_start:left_end]))
        elif operation == "delete":
            _append(tokens, "remove", "".join(left[left_start:left_end]))
        elif operation == "insert":
            _append(tokens, "add", "".join(right[right_start:right_end]))
        else:
            _append(tokens, "remove", "".join(left[left_start:left_end]))
            _append(tokens, "add", "".join(right[right_start:right_end]))
    return tokens


def _append(
    tokens: list[DiffToken],
    operation: str,
    text: str,
) -> None:
    if not text:
        return
    if tokens and tokens[-1].operation == operation:
        tokens[-1] = tokens[-1].model_copy(
            update={"text": tokens[-1].text + text}
        )
    else:
        tokens.append(DiffToken(operation=operation, text=text))


def _diff_value(value) -> DiffValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def _value_text(value: DiffValue) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(value)
    return value


def _item_text(item) -> str:
    return json.dumps(
        _item_payload(item),
        ensure_ascii=False,
        sort_keys=True,
    )


def _item_payload(item) -> dict:
    return item.model_dump(
        mode="json",
        exclude={"id"},
        exclude_none=True,
    )


def _has_item_content(item) -> bool:
    def meaningful(value) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(meaningful(child) for child in value)
        if isinstance(value, dict):
            return any(meaningful(child) for child in value.values())
        return value is not None

    return meaningful(_item_payload(item))


def _item_label(item) -> str:
    for field in ("name", "title", "institution", "content"):
        value = getattr(item, field, None)
        if value:
            return str(value)
    return str(item.id)
