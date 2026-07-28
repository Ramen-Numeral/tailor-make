"""Feature augmentation for text datasets.

Extracts linguistic features from text for use in ML models.
Reuses text processing utilities from app/utils/text.py.
"""

from collections import Counter
from math import sqrt, log2

from app.utils.text import (
    remove_punctuation,
    word_lengths,
    safe_divide,
    population_variance as utils_population_variance,
)
from config.train_data import TrainDataConfig


data_cfg = TrainDataConfig()
PRECISION = data_cfg.augmentation_precision



def safe_div(n: float, d: float) -> float:
    """Wrapper for safe_divide with augment-specific precision."""
    return safe_divide(n, d, precision=PRECISION)


def population_variance(xs: list[int]) -> float:
    """Wrapper for population_variance with augment-specific precision."""
    return utils_population_variance(xs, precision=PRECISION)


def population_std(xs: list[int]) -> float:
    return round(sqrt(population_variance(xs)), PRECISION)


def lengths(items: list[str]) -> list[int]:
    """Get word count of each item (sentence, etc.)."""
    return [len(x.split()) for x in items]


def avg_len(xs: list[int]) -> float:
    """Calculate average with augment-specific precision."""
    return safe_div(sum(xs), len(xs)) if xs else 0.0


def coefficient_of_variation(xs: list[int]) -> float:
    return safe_div(population_std(xs), avg_len(xs)) if avg_len(xs) else 0.0


def word_counts(words: list[str]) -> Counter:
    return Counter(words)


def unique_word_ratio(counts: Counter) -> float:
    return safe_div(len(counts), sum(counts.values()))


def words_that_appear_once(counts: Counter) -> list[str]:
    return [word for word, count in counts.items() if count == 1]


def hapax_ratio(counts: Counter) -> float:
    return safe_div(len(words_that_appear_once(counts)), len(counts))


def repeat_word_ratio(counts: Counter) -> float:
    return safe_div(sum(count > 1 for count in counts.values()), len(counts))


def avg_word_repetition(counts: Counter) -> float:
    return safe_div(sum(counts.values()), len(counts))


def count_occurrences(items: list[str], token: str) -> int:
    return sum(item.count(token) for item in items)


def avg_token_per_sentence(sentences: list[str], token: str) -> float:
    return safe_div(count_occurrences(sentences, token), len(sentences))


def avg_comma_per_sent(sentences: list[str]) -> float:
    return avg_token_per_sentence(sentences, ",")


def avg_semicolon_per_sent(sentences: list[str]) -> float:
    return avg_token_per_sentence(sentences, ";")


def avg_em_dash_per_sent(sentences: list[str]) -> float:
    return avg_token_per_sentence(sentences, "--")


def avg_parenthetical_per_sent(sentences: list[str]) -> float:
    return avg_token_per_sentence(sentences, "(")


def avg_quotation_per_sent(sentences: list[str]) -> float:
    return avg_token_per_sentence(sentences, '"')


def punct_per_word(text: str) -> float:
    """Calculate punctuation marks per word."""
    cleaned = remove_punctuation(text)
    return safe_div(len(text) - len(cleaned), len(cleaned.split()))


def max_word_length(words: list[str]) -> int:
    return max((len(word) for word in words), default=0)


def newline_count(text: str) -> int:
    return text.count("\n")


def avg_adverbs_per_sent(words: list[str], sentences: list[str]) -> float:
    return safe_div(sum(word.endswith("ly") for word in words), len(sentences))


def entropy(xs: list[int]) -> float:
    counts = Counter(xs)
    total = sum(counts.values())

    return round(
        -sum((count / total) * log2(count / total) for count in counts.values()),
        PRECISION,
    ) if total else 0.0


def word_length_entropy(words: list[str]) -> float:
    return entropy(word_lengths(words))


def sentence_length_entropy(sentences: list[str]) -> float:
    return entropy(lengths(sentences))

