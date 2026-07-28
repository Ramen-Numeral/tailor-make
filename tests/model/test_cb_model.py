import numpy as np
from catboost import CatBoostClassifier

from app.infrastructure.ai_detection.catboost_component import (
    load_cb_model,
    predict_catboost,
)
from ml_pipelines.data.features import extract_features
from ml_pipelines.models.catboost.cb_model import save_model


def test_catboost_model_round_trip(tmp_path) -> None:
    model = CatBoostClassifier(
        iterations=2,
        depth=2,
        verbose=False,
        train_dir=str(tmp_path / "training"),
    )
    model.fit([[0, 0], [1, 1]], [0, 1])

    save_model(model, "model.cbm", path=tmp_path)
    loaded = load_cb_model("model.cbm", path=tmp_path)

    assert isinstance(loaded, CatBoostClassifier)
    assert loaded.predict_proba([[1, 1]]).shape == (1, 2)


def test_catboost_prediction_returns_ranked_local_feature_evidence() -> None:
    feature_count = len(extract_features("A short sentence."))

    class ExplainableModel:
        def predict_proba(self, pool):
            return np.array([[0.25, 0.75]])

        def get_feature_importance(self, pool, *, type):
            assert type == "ShapValues"
            contributions = np.arange(feature_count, dtype=float)
            return np.array([[*contributions, -0.5]])

    component, details = predict_catboost(
        ExplainableModel(),
        "A short sentence.",
    )

    assert component.ai_probability == 0.75
    assert details is not None
    assert details["base_value"] == -0.5
    assert len(details["top_features"]) == feature_count
    assert details["top_features"][0]["importance_rank"] == 1
    assert details["top_features"][0]["direction"] == "machine_like"
    assert details["top_features"][0]["description"]
