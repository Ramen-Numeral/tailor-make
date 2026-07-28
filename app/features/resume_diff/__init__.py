"""Frontend-ready resume diff generation."""

from app.features.resume_diff.differ import build_resume_diffs, inline_diff
from app.features.resume_diff.schema import DiffToken, FieldDiff

__all__ = [
    "DiffToken",
    "FieldDiff",
    "build_resume_diffs",
    "inline_diff",
]
