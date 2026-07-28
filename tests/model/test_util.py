import polars as pl

from ml_pipelines.models.util import extract_text, split_Xy


def test_split_Xy_removes_target_and_optional_columns():
    df = pl.DataFrame({"label": [0, 1], "text": ["a", "b"], "feature": [1.0, 2.0]})

    X, y = split_Xy(df)

    assert X.columns == ["text", "feature"]
    assert y.to_list() == [0, 1]


def test_split_Xy_drops_extra_columns():
    df = pl.DataFrame({"label": [0, 1], "text": ["a", "b"], "junk": ["x", "y"]})

    X, y = split_Xy(df, drop_cols=["junk"])

    assert X.columns == ["text"]
    assert y.to_list() == [0, 1]


def test_split_Xy_ignores_missing_optional_drop_columns():
    df = pl.DataFrame({
        "text": ["a", "b"],
        "label": [0, 1],
        "feature": [1.0, 2.0],
    })

    X, y = split_Xy(
        df,
        drop_cols=["text", "abstract", "title", "newline_count"],
    )

    assert X.columns == ["feature"]
    assert y.to_list() == [0, 1]


def test_extract_text_fills_nulls_with_empty_string():
    df = pl.DataFrame({"abstract": ["hello", None]})

    texts = extract_text(df, "abstract")

    assert texts == ["hello", ""]


def test_extract_text_casts_non_string_column_to_utf8():
    df = pl.DataFrame({"abstract": [1, 2, 3]})

    texts = extract_text(df, "abstract")

    assert texts == ["1", "2", "3"]
