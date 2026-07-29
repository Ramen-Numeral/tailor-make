import polars as pl

from app.utils.text import get_sentences, get_words
from config.train_data import TrainDataConfig
from ml_pipelines.data.feature_util import (
    avg_adverbs_per_sent,
    avg_comma_per_sent,
    avg_em_dash_per_sent,
    avg_len,
    avg_parenthetical_per_sent,
    avg_quotation_per_sent,
    avg_semicolon_per_sent,
    avg_word_repetition,
    coefficient_of_variation,
    hapax_ratio,
    lengths,
    max_word_length,
    newline_count,
    population_std,
    population_variance,
    punct_per_word,
    repeat_word_ratio,
    sentence_length_entropy,
    unique_word_ratio,
    word_counts,
    word_length_entropy,
    word_lengths,
)


data_cfg = TrainDataConfig()


FEATURE_DESCRIPTIONS = {
    "num_sentences": "Total number of sentences in the text.",
    "num_words": "Total number of words in the text.",
    "num_chars": "Total number of characters, including spaces and punctuation.",
    "newline_count": "Number of newline characters in the text.",

    "avg_sent_len": "Average number of words per sentence.",
    "sent_len_var": "Variance in sentence lengths, measured in words.",
    "sent_len_std": "Standard deviation of sentence lengths, measured in words.",
    "sent_len_cv": (
        "Sentence-length variability relative to the average sentence length."
    ),
    "max_sent_len": "Number of words in the longest sentence.",

    "avg_word_len": "Average number of characters per word.",
    "word_len_var": "Variance in word lengths, measured in characters.",
    "word_len_std": "Standard deviation of word lengths, measured in characters.",
    "max_word_len": "Number of characters in the longest word.",

    "unique_word_ratio": (
        "Proportion of all words that are distinct vocabulary terms."
    ),
    "hapax_ratio": (
        "Proportion of distinct vocabulary terms that appear exactly once."
    ),
    "repeat_word_ratio": (
        "Proportion of distinct vocabulary terms that appear more than once."
    ),
    "avg_word_repetition": (
        "Average number of occurrences per distinct vocabulary term."
    ),

    "comma_per_sent": "Average number of commas per sentence.",
    "semicolon_per_sent": "Average number of semicolons per sentence.",
    "em_dash_per_sent": (
        "Average number of double-hyphen em-dash markers per sentence."
    ),
    "parenthetical_per_sent": (
        "Average number of opening parentheses per sentence."
    ),
    "quotation_per_sent": (
        "Average number of double-quotation marks per sentence."
    ),
    "punct_per_word": "Number of punctuation characters per word.",

    "avg_adverbs_per_sent": (
        "Average number of words ending in 'ly' per sentence, used as an "
        "approximation of adverb frequency."
    ),
    "word_len_entropy": (
        "Diversity and unpredictability of word lengths throughout the text."
    ),
    "sent_len_entropy": (
        "Diversity and unpredictability of sentence lengths throughout the text."
    ),
}


def extract_features(text: str) -> dict[str, float | int]:
    sentences = get_sentences(text)
    words = get_words(text)
    counts = word_counts(words)

    sent_lens = lengths(sentences)
    word_lens = word_lengths(words)

    return {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_chars": len(text),
        "newline_count": newline_count(text),

        "avg_sent_len": avg_len(sent_lens),
        "sent_len_var": population_variance(sent_lens),
        "sent_len_std": population_std(sent_lens),
        "sent_len_cv": coefficient_of_variation(sent_lens),
        "max_sent_len": max(sent_lens, default=0),

        "avg_word_len": avg_len(word_lens),
        "word_len_var": population_variance(word_lens),
        "word_len_std": population_std(word_lens),
        "max_word_len": max_word_length(words),

        "unique_word_ratio": unique_word_ratio(counts),
        "hapax_ratio": hapax_ratio(counts),
        "repeat_word_ratio": repeat_word_ratio(counts),
        "avg_word_repetition": avg_word_repetition(counts),

        "comma_per_sent": avg_comma_per_sent(sentences),
        "semicolon_per_sent": avg_semicolon_per_sent(sentences),
        "em_dash_per_sent": avg_em_dash_per_sent(sentences),
        "parenthetical_per_sent": avg_parenthetical_per_sent(sentences),
        "quotation_per_sent": avg_quotation_per_sent(sentences),
        "punct_per_word": punct_per_word(text),

        "avg_adverbs_per_sent": avg_adverbs_per_sent(words, sentences),
        "word_len_entropy": word_length_entropy(words),
        "sent_len_entropy": sentence_length_entropy(sentences),
    }



def augment_data(
    df: pl.DataFrame | pl.Series,
    text_col: str,
    label_col: str | None = None,
) -> pl.DataFrame:
    if isinstance(df, pl.Series):
        df = df.to_frame()

    required = {text_col}
    if label_col is not None:
        required.add(label_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rename_map = {text_col: data_cfg.text_column}
    if label_col is not None:
        rename_map[label_col] = data_cfg.target_column

    df = df.rename({
        old: new
        for old, new in rename_map.items()
        if old != new
    })

    feats = pl.DataFrame([
        extract_features(text or "")
        for text in df[data_cfg.text_column].cast(pl.String)
    ])

    return pl.concat(
        [df.drop([c for c in feats.columns if c in df.columns]), feats],
        how="horizontal_extend",
    )
