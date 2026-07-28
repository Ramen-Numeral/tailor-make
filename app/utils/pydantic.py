from __future__ import annotations

from collections.abc import Collection, Iterable
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

from app.utils.text import deduplicate_list

ModelT = TypeVar("ModelT", bound=BaseModel)


def is_empty_value(
    value: Any,
    *,
    treat_unknown_enum_as_empty: bool = False,
) -> bool:
    """Return whether a value should be treated as missing."""

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, Collection) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return len(value) == 0

    if (
        treat_unknown_enum_as_empty
        and isinstance(value, Enum)
        and value.value == "unknown"
    ):
        return True

    if isinstance(value, BaseModel):
        return all(
            is_empty_value(
                field_value,
                treat_unknown_enum_as_empty=treat_unknown_enum_as_empty,
            )
            for field_value in value.__dict__.values()
        )

    return False


def find_missing_fields(
    model: BaseModel,
    required_fields: Iterable[str] | None = None,
    *,
    treat_unknown_enum_as_empty: bool = False,
) -> list[str]:
    """Return field names whose values are empty.

    When required_fields is omitted, every field on the model is checked.
    """

    field_names = (
        tuple(required_fields)
        if required_fields is not None
        else tuple(type(model).model_fields)
    )

    missing: list[str] = []

    for field_name in field_names:
        if field_name not in type(model).model_fields:
            raise ValueError(
                f"{field_name!r} is not a field on "
                f"{type(model).__name__}"
            )

        value = getattr(model, field_name)

        if is_empty_value(
            value,
            treat_unknown_enum_as_empty=treat_unknown_enum_as_empty,
        ):
            missing.append(field_name)

    return missing


def merge_pydantic_models(
    current: ModelT,
    new: ModelT,
) -> ModelT:
    """Merge two models of the same type.

    Rules:
    - Empty values are filled from the new model.
    - Nested Pydantic models are merged recursively.
    - Lists are combined and deduplicated.
    - Existing populated scalar values are preserved.
    """

    if type(current) is not type(new):
        raise TypeError(
            "Models must have the same type, got "
            f"{type(current).__name__} and {type(new).__name__}"
        )

    merged: dict[str, Any] = {}

    for field_name in type(current).model_fields:
        current_value = getattr(current, field_name)
        new_value = getattr(new, field_name)

        merged[field_name] = merge_values(
            current_value,
            new_value,
        )

    return type(current).model_validate(merged)


def merge_values(
    current: Any,
    new: Any,
) -> Any:
    """Merge two values using conservative defaults."""

    if is_empty_value(current):
        return new

    if is_empty_value(new):
        return current

    if isinstance(current, BaseModel) and isinstance(new, BaseModel):
        return merge_pydantic_models(current, new)

    if isinstance(current, list) and isinstance(new, list):
        return deduplicate_list([*current, *new])

    return current

