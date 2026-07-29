"""Training utility functions for data manipulation."""

import polars as pl
from config.train_data import TrainDataConfig


data_cfg = TrainDataConfig()


def split_Xy(
    df: pl.DataFrame,
    target_label: str = data_cfg.target_column,
    drop_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.Series]:
    """Split dataframe into features and target."""
    if target_label not in df.columns:
        raise ValueError(f"Missing target column: {target_label!r}")

    optional_columns = [
        column for column in (drop_cols or []) if column in df.columns
    ]
    X = df.drop([target_label, *optional_columns])
    y = df[target_label]
    return X, y


def extract_text(
    df: pl.DataFrame,
    text_column: str = data_cfg.text_column,
) -> list[str]:
    """Extract text column as list of strings."""
    return (
        df
        .get_column(text_column)
        .cast(pl.Utf8)
        .fill_null("")
        .to_list()
    )
