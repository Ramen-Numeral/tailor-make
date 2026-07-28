import polars as pl
import pytest
from datasets import Dataset, DatasetDict, Features, Value

from ml_pipelines.data import loader, data_source_cfg


def test_load_ds_combines_splits_and_normalizes_columns(monkeypatch) -> None:
    source = DatasetDict({
        "train": Dataset.from_dict({"abstract": ["one"], "target": [0]}),
        "test": Dataset.from_dict({"abstract": ["two"], "target": [1]}),
    })
    monkeypatch.setattr(loader, "load_dataset", lambda address: source)

    loaded = loader.load_ds("example/source", "abstract", "target")

    assert loaded.column_names == ["text", "label"]
    assert loaded.to_dict() == {"text": ["one", "two"], "label": [0, 1]}


def test_load_batch_uses_each_sources_column_names(monkeypatch) -> None:
    datasets = {
        "first": Dataset.from_dict({"body": ["one"], "class": [0]}),
        "second": Dataset.from_dict({"content": ["two"], "is_ai": [1]}),
    }
    monkeypatch.setattr(loader, "load_dataset", datasets.get)
    sources = [
        data_source_cfg.DatasetSource("first", "body", "class"),
        data_source_cfg.DatasetSource("second", "content", "is_ai"),
    ]

    loaded = loader.load_batch(sources)

    assert loaded.column_names == ["text", "label"]
    assert loaded.to_dict() == {"text": ["one", "two"], "label": [0, 1]}


def test_load_batch_normalizes_incompatible_label_types(monkeypatch) -> None:
    datasets = {
        "int8": Dataset.from_dict(
            {"body": ["one"], "class": [0]},
            features=Features({
                "body": Value("string"),
                "class": Value("int8"),
            }),
        ),
        "int64": Dataset.from_dict(
            {"content": ["two"], "is_ai": [1]},
            features=Features({
                "content": Value("string"),
                "is_ai": Value("int64"),
            }),
        ),
    }
    monkeypatch.setattr(loader, "load_dataset", datasets.get)

    loaded = loader.load_batch([
        data_source_cfg.DatasetSource("int8", "body", "class"),
        data_source_cfg.DatasetSource("int64", "content", "is_ai"),
    ])

    assert loaded.features == Features({
        "text": Value("string"),
        "label": Value("int64"),
    })
    assert loaded.to_dict() == {"text": ["one", "two"], "label": [0, 1]}


def test_load_batch_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="At least one"):
        loader.load_batch([])


def test_split_ds_preserves_schema_and_ratios() -> None:
    source = Dataset.from_dict({
        "text": [f"text {index}" for index in range(100)],
        "label": [index % 2 for index in range(100)],
    })

    splits = loader.split_ds(source)

    assert {name: len(split) for name, split in splits.items()} == {
        "train": 80,
        "test": 10,
        "val": 10,
    }
    assert all(split.column_names == ["text", "label"] for split in splits.values())


def test_load_dataframe_normalizes_columns_and_keeps_features(tmp_path) -> None:
    path = tmp_path / "source.csv"
    pl.DataFrame({"body": ["one"], "target": [0], "feature": [2.5]}).write_csv(path)

    loaded = loader.load_dataframe(path, "body", "target")

    assert loaded.columns == ["text", "label", "feature"]
