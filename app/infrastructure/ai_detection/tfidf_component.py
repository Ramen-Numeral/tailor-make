"""TF-IDF/SVM model loading and inference."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline

from app.features.ai_detection.schema import ComponentScore, TermContribution
from config.settings import get_settings

def predict_tfidf(
    model: Pipeline,
    text: str,
) -> ComponentScore:
    """Return calibrated probability and exact linear-margin contributions."""
    probability = float(model.predict_proba([text])[0, 1])
    contributions: list[TermContribution] = []
    base_value: float | None = None
    note: str | None = None
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["svm"]
        vector = vectorizer.transform([text])
        estimators = [
            getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", None))
            for calibrated in classifier.calibrated_classifiers_
        ]
        estimators = [estimator for estimator in estimators if estimator is not None]
        coefficients = np.mean(
            [np.asarray(estimator.coef_)[0] for estimator in estimators],
            axis=0,
        )
        base_value = float(np.mean([
            np.asarray(estimator.intercept_).ravel()[0]
            for estimator in estimators
        ]))
        names = vectorizer.get_feature_names_out()
        for index, value in zip(vector.indices, vector.data, strict=True):
            term = str(names[index])
            if len(term.split()) < 2:
                continue
            coefficient = float(coefficients[index])
            contribution = float(value * coefficient)
            contributions.append(
                TermContribution(
                    term=term,
                    tfidf_value=float(value),
                    coefficient=coefficient,
                    contribution=contribution,
                    direction=(
                        "machine_like" if contribution >= 0 else "human_like"
                    ),
                )
            )
        contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        note = (
            "Phrase-level contributions are shown; noisy unigram terms are "
            "suppressed. Probability calibration is nonlinear."
        )
    except Exception as error:
        note = f"Linear term explanation unavailable: {error}"

    return ComponentScore(
        model_name="tfidf_svm",
        ai_probability=probability,
        base_value=base_value,
        term_contributions=contributions,
        explanation_note=note,
    )


def load_tfidf_model(path: str | Path | None = None) -> Pipeline:
    model_path = Path(path or get_settings().io.tfidf_model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"TF-IDF SVM model file not found: {model_path}")

    return joblib.load(model_path)
