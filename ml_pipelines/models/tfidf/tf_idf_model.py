from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config.settings import get_settings
from ml_pipelines.models.config import TfidfConfig


tfidf_config = TfidfConfig()

def fit_tfidf_svm(
    tr_X: list[str],
    tr_y: list[int],
) -> Pipeline:
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=tfidf_config.ngram_range,
                max_features=tfidf_config.max_features,
                min_df=tfidf_config.min_df,
                max_df=tfidf_config.max_df,
            ),
        ),
        (
            "svm",
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
    ])

    model.fit(tr_X, tr_y)
    return model


def save_tfidf_model(
    model: Pipeline,
    path: str | Path | None = None,
) -> Path:
    model_path = Path(path or get_settings().io.tfidf_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
