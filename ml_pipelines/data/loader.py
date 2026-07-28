from pathlib import Path

import numpy as np
import polars as pl
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)

from config.settings import get_settings
from config.train_data import TrainDataConfig
from ml_pipelines.data.data_source_cfg import DatasetSource


data_cfg = TrainDataConfig()
CANONICAL_FEATURES = Features({
    data_cfg.text_column: Value("string"),
    data_cfg.target_column: Value("int64"),
})


def _normalize(
    dataset: Dataset | DatasetDict,
    text_column: str,
    label_column: str,
) -> Dataset:
    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets(list(dataset.values()))

    normalized = dataset.select_columns(
        [text_column, label_column]
    ).rename_columns({
        text_column: data_cfg.text_column,
        label_column: data_cfg.target_column,
    })
    return normalized.cast(CANONICAL_FEATURES)


def load_ds(
    address: str,
    text_column: str,
    label_column: str,
) -> Dataset:
    """Load every split and return canonical ``text`` and ``label`` columns."""
    return _normalize(load_dataset(address), text_column, label_column)


def load_batch(sources: list[DatasetSource]) -> Dataset:
    """Load multiple sources into one canonical ``text``/``label`` dataset."""
    if not sources:
        raise ValueError("At least one dataset source is required")

    datasets = [
        load_ds(source.address, source.text_column, source.label_column)
        for source in sources
    ]
    return concatenate_datasets(datasets)


def split_ds(ds: Dataset) -> dict[str, Dataset]:
    expected = [data_cfg.text_column, data_cfg.target_column]
    if ds.column_names != expected:
        raise ValueError(f"Expected columns {expected}; received {ds.column_names}")

    splits = ds.train_test_split(
        train_size=data_cfg.train_pct,
        test_size=data_cfg.validation_pct + data_cfg.test_pct,
        shuffle=True,
    )
    holdout = splits["test"].train_test_split(
        train_size=data_cfg.test_pct / (data_cfg.test_pct + data_cfg.validation_pct),
        shuffle=True,
    )
    return {
        "train": splits["train"],
        "test": holdout["train"],
        "val": holdout["test"],
    }


def print_ds_stats(ds: dict[str, Dataset]) -> None:
    label = data_cfg.target_column
    for name, split in ds.items():
        print(f"\nKey: {name}, Length: {len(split)}, Columns: {split.column_names}")
        values = np.asarray(split[label], dtype=float)
        print(f"{label} stats:")
        print(f"mean: {values.mean():.4f}")
        print(f"std:  {values.std(ddof=1):.4f}")
        print(f"min:  {values.min():.4f}")
        print(f"max:  {values.max():.4f}")


def load_dataframe(
    file_name: str | Path,
    text_column: str = data_cfg.text_column,
    label_column: str = data_cfg.target_column,
) -> pl.DataFrame:
    """Load a CSV, preserving features while canonicalizing text and label."""
    return (
        pl.read_csv(get_settings().io.data_dir / file_name)
        .rename({
            text_column: data_cfg.text_column,
            label_column: data_cfg.target_column,
        })
        .cast({
            data_cfg.text_column: pl.String,
            data_cfg.target_column: pl.Int64,
        })
    )
