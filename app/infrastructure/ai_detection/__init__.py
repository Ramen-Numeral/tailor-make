"""Component wrappers for individual classification models.

Each component returns a ComponentScore with the model's probability and metadata.
"""

from app.infrastructure.ai_detection.catboost_component import (
    load_cb_model,
    predict_catboost,
)
from app.infrastructure.ai_detection.distilbert_component import (
    load_dbert_model,
    predict_distilbert,
)
from app.infrastructure.ai_detection.tfidf_component import (
    load_tfidf_model,
    predict_tfidf,
)
from app.infrastructure.ai_detection.rubric_component import (
    load_rubric_regressor,
    predict_rubric,
)

__all__ = [
    "predict_catboost",
    "predict_distilbert",
    "predict_tfidf",
    "predict_rubric",
    "load_cb_model",
    "load_dbert_model",
    "load_tfidf_model",
    "load_rubric_regressor",
]
