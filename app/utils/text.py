"""Text processing utilities.

Generic functions for text manipulation, parsing, and analysis.
Extracted from app/data/feature_util.py for reuse across the codebase.
"""
from __future__ import annotations

import re
from typing import Any

# Regex patterns
SENT_END_RE = re.compile(r"[.?!]+")
PUNCT_RE = re.compile(r"[^\w\s]")


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text.

    Args:
        text: Input text.

    Returns:
        Text with punctuation removed.
    """
    return PUNCT_RE.sub("", text)


def get_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentences (stripped of whitespace).
    """
    return [s.strip() for s in SENT_END_RE.split(text) if s.strip()]


def get_words(text: str, lowercase: bool = True) -> list[str]:
    """Split text into words.

    Removes punctuation and optionally lowercases.

    Args:
        text: Input text.
        lowercase: Whether to lowercase words (default: True).

    Returns:
        List of words.
    """
    text = remove_punctuation(text)
    text = text.lower() if lowercase else text
    return text.split()


def word_lengths(words: list[str]) -> list[int]:
    """Get character length of each word.

    Args:
        words: List of words.

    Returns:
        List of character counts.
    """
    return [len(word) for word in words]


def safe_divide(numerator: float, denominator: float, precision: int = 2) -> float:
    """Safely divide, returning 0 if denominator is 0.

    Args:
        numerator: Numerator.
        denominator: Denominator.
        precision: Decimal places to round to (default: 2).

    Returns:
        Quotient rounded to precision, or 0 if denominator is 0.
    """
    return round(numerator / denominator, precision) if denominator else 0.0


def average(values: list[int | float]) -> float:
    """Calculate average of numeric list.

    Args:
        values: List of numbers.

    Returns:
        Average (0 if empty).
    """
    return safe_divide(sum(values), len(values)) if values else 0.0


def population_variance(values: list[int | float], precision: int = 2) -> float:
    """Calculate population variance.

    Args:
        values: List of numbers.
        precision: Decimal places to round to (default: 2).

    Returns:
        Population variance.
    """
    if not values:
        return 0.0

    mean = sum(values) / len(values)
    return round(sum((x - mean) ** 2 for x in values) / len(values), precision)


def deduplicate_list(items: list[Any]) -> list[Any]:
    """Deduplicate a list while preserving order."""

    result: list[Any] = []

    for item in items:
        if item not in result:
            result.append(item)

    return result
