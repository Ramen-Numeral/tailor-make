"""CatBoost model loading and inference."""

from pathlib import Path
from typing import Any

import polars as pl
from catboost import CatBoostClassifier

from app.features.ai_detection.schema import ComponentScore
from config.settings import get_settings
from config.train_data import TrainDataConfig
from ml_pipelines.data.features import FEATURE_DESCRIPTIONS, augment_data
from ml_pipelines.models.catboost.cb_model import make_pool

data_cfg = TrainDataConfig()
def predict_catboost(
    model: CatBoostClassifier,
    text: str,
) -> tuple[ComponentScore, dict[str, Any] | None]:
    """Return CatBoost's AI probability and explanation payload."""
    frame = pl.DataFrame({data_cfg.text_column: [text]})
    features = augment_data(frame, data_cfg.text_column).drop(
        data_cfg.text_column
    )
    pool = make_pool(features)
    probability = float(model.predict_proba(pool)[0, 1])
    shap_row = model.get_feature_importance(
        pool,
        type="ShapValues",
    )[0]
    ranked = sorted(
        zip(features.columns, features.row(0), shap_row[:-1], strict=True),
        key=lambda item: abs(float(item[2])),
        reverse=True,
    )
    top_features = [
        {
            "feature": feature,
            "label": feature.replace("_", " ").title(),
            "description": FEATURE_DESCRIPTIONS[feature],
            "observed_value": float(observed),
            "shap_value": float(shap_value),
            "direction": (
                "machine_like" if shap_value >= 0 else "human_like"
            ),
            "importance_rank": rank,
        }
        for rank, (feature, observed, shap_value) in enumerate(ranked, start=1)
    ]
    return ComponentScore(
        model_name="catboost",
        ai_probability=probability,
    ), {
        "base_value": float(shap_row[-1]),
        "top_features": top_features,
    }


def load_cb_model(
    file_name: str | None = None,
    path: str | Path | None = None,
) -> CatBoostClassifier:
    io = get_settings().io
    model_path = Path(path or io.model_dir) / (
        file_name or io.catboost_model_filename
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model
