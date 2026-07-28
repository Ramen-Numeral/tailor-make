import polars as pl

from config.train_data import TrainDataConfig
from ml_pipelines.data.features import (
    FEATURE_DESCRIPTIONS,
    augment_data,
    extract_features,
)

data_config = TrainDataConfig()


def test_feature_descriptions_cover_every_extracted_feature() -> None:
    assert set(extract_features("A short sentence.")) == set(
        FEATURE_DESCRIPTIONS
    )


def test_augment_data_normalizes_columns_and_adds_features() -> None:
    source = pl.DataFrame({"essay": ["A short sentence."], "generated": [1]})

    result = augment_data(source, "essay", "generated")

    assert data_config.text_column in result
    assert data_config.target_column in result
    assert set(FEATURE_DESCRIPTIONS) <= set(result.columns)
    assert result[data_config.target_column].to_list() == [1]
