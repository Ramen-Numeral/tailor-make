"""Typed frontend contracts for resume changes."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

DiffValue = str | list[str] | None


class DiffToken(BaseModel):
    """One renderable token group in an inline diff."""

    operation: Literal["equal", "add", "remove"]
    text: str


class FieldDiff(BaseModel):
    """One explainable field or item-level resume change."""

    section: str
    item_id: UUID
    item_label: str
    field: str
    change_type: Literal[
        "rewritten",
        "not_selected",
        "trimmed_for_page_limit",
        "removed",
        "added",
    ]
    original: DiffValue = None
    final: DiffValue = None
    tokens: list[DiffToken] = Field(default_factory=list)
    reason: str
