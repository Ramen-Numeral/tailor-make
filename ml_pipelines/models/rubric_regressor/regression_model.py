from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings
from config.train_data import TrainDataConfig
from ml_pipelines.models.util import split_Xy


data_cfg = TrainDataConfig()

def fit_regression_head(
    df: pl.DataFrame,
) -> tuple[Pipeline, list[str]]:
    target = data_cfg.target_column
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target!r}")

    X, y = split_Xy(df, drop_cols=[data_cfg.text_column])

    if X.is_empty():
        raise ValueError("No complete rows remain after dropping nulls.")

    feature_columns = X.columns

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])
    model.fit(
        X.to_numpy().astype(np.float64),
        y.to_numpy().astype(np.float64),
    )
    model.feature_cols_ = list(feature_columns)
    return model, feature_columns

def save_regression_head(
    model: Any,
    feature_cols: Sequence[str],
) -> Path:
    path = get_settings().io.rubric_regressor_path
    feature_cols = list(feature_cols)
    model.feature_cols_ = feature_cols

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"ai_detection": model, "feature_cols": feature_cols},
        path,
    )

    return path
