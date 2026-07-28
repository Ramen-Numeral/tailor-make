import pytest
from sklearn.pipeline import Pipeline

from app.infrastructure.ai_detection.tfidf_component import (
    load_tfidf_model,
    predict_tfidf,
)
from ml_pipelines.models.tfidf.tf_idf_model import (
    fit_tfidf_svm,
    save_tfidf_model,
)
from ml_pipelines.models.config import TfidfConfig

HUMAN_TEXTS = [
    "forgot my keys again",
    "my dog ate lunch",
    "coffee tastes bad",
] * 3
AI_TEXTS = [
    "Furthermore this comprehensive framework improves evaluation",
    "In conclusion the methodology provides significant insights",
    "Moreover the analysis demonstrates substantial improvement",
] * 3


@pytest.fixture(scope="module")
def fitted_model() -> Pipeline:
    return fit_tfidf_svm(
        HUMAN_TEXTS + AI_TEXTS,
        [0] * len(HUMAN_TEXTS) + [1] * len(AI_TEXTS),
    )


def test_tfidf_prediction_is_ai_probability(fitted_model) -> None:
    text = "Furthermore this framework provides comprehensive analysis"
    result = predict_tfidf(fitted_model, text)

    assert result.ai_probability == pytest.approx(
        fitted_model.predict_proba([text])[0, 1]
    )
    assert result.term_contributions
    assert all(" " in term.term for term in result.term_contributions)
    assert not any(
        term.term == "demonstrates"
        for term in result.term_contributions
    )


def test_tfidf_training_uses_phrase_level_ngrams() -> None:
    assert TfidfConfig().ngram_range == (2, 4)


def test_tfidf_model_round_trip(tmp_path, fitted_model) -> None:
    path = tmp_path / "tfidf.joblib"

    save_tfidf_model(fitted_model, path)
    loaded = load_tfidf_model(path)

    assert loaded.predict_proba(["forgot lunch"]) == pytest.approx(
        fitted_model.predict_proba(["forgot lunch"])
    )
