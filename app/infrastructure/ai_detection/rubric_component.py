"""Rubric-regression model loading and inference."""

from typing import Any

import joblib
import numpy as np

from app.features.ai_detection.schema import ComponentScore
from app.features.ai_detection.config import ai_detection_config
from config.settings import get_settings
from ml_pipelines.models.rubric_regressor.rubric import AXES_DEFINITIONS
from ml_pipelines.models.rubric_regressor.score import score_text


def predict_rubric(
    text: str,
    regression_model: Any | None = None,
) -> tuple[ComponentScore, dict[str, Any] | None]:
    """Return the regression head's AI probability and rubric scores."""
    scores = score_text(
        text,
        max_attempts=ai_detection_config.rubric_max_attempts,
    )
    if not scores:
        raise ValueError("Rubric scoring returned no scores")

    details: dict[str, Any] = {"rubric_scores": scores}
    if regression_model is None:
        probability = sum((5 - score) / 4 for score in scores.values()) / len(
            scores
        )
        details.update(
            ai_probability=probability,
            note="Using rubric average because no regression model was loaded",
            axis_contributions={
                axis: (3.0 - score) / (4.0 * len(scores))
                for axis, score in scores.items()
            },
        )
    else:
        try:
            features = np.array(
                [[scores[axis] for axis in AXES_DEFINITIONS]],
                dtype=np.float64,
            )
            classes = np.asarray(regression_model.classes_)
            ai_indexes = np.flatnonzero(classes == 1)
            if not len(ai_indexes):
                raise ValueError("Regression model has no AI-written class")
            probability = float(
                regression_model.predict_proba(features)[0, ai_indexes[0]]
            )
            details.update(
                ai_probability=probability,
                prediction=(
                    "AI-written" if probability >= 0.5 else "human-written"
                ),
            )
            try:
                coefficients = np.asarray(regression_model.coef_)
                if coefficients.shape[0] == 1:
                    class_coefficients = coefficients[0]
                else:
                    class_coefficients = coefficients[ai_indexes[0]]
                centered = features[0] - 3.0
                details.update(
                    axis_contributions={
                        axis: float(centered[index] * class_coefficients[index])
                        for index, axis in enumerate(AXES_DEFINITIONS)
                    },
                    contribution_note=(
                        "Axis contributions exactly decompose the logistic "
                        "regression log-odds around a neutral score of 3."
                    ),
                )
            except Exception as explanation_error:
                try:
                    import shap

                    background = np.full(
                        (1, len(AXES_DEFINITIONS)),
                        3.0,
                        dtype=np.float64,
                    )
                    explanation = shap.Explainer(
                        regression_model.predict_proba,
                        background,
                    )(features)
                    values = np.asarray(explanation.values)
                    class_values = (
                        values[0, :, ai_indexes[0]]
                        if values.ndim == 3
                        else values[0]
                    )
                    details.update(
                        axis_contributions={
                            axis: float(class_values[index])
                            for index, axis in enumerate(AXES_DEFINITIONS)
                        },
                        contribution_note=(
                            "Axis contributions are local SHAP values against "
                            "a neutral 3/5 rubric baseline."
                        ),
                    )
                except Exception as shap_error:
                    details["contribution_error"] = (
                        f"Linear explanation: {explanation_error}; "
                        f"SHAP explanation: {shap_error}"
                    )
        except Exception as error:
            details["error"] = str(error)
            return ComponentScore(
                model_name="rubric_regressor",
                ai_probability=0.5,
                error=str(error),
            ), details

    return ComponentScore(
        model_name="rubric_regressor",
        ai_probability=probability,
        base_value=0.5 if regression_model is None else None,
        explanation_note=details.get(
            "contribution_note",
            details.get("note") or details.get("contribution_error"),
        ),
    ), details


def load_rubric_regressor() -> tuple[Any, list[str]]:
    path = get_settings().io.rubric_regressor_path

    if not path.exists():
        raise FileNotFoundError(f"Regression head not found: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError("Saved regression artifact must be a dictionary.")

    required = {"ai_detection", "feature_cols"}
    if missing := required - artifact.keys():
        raise ValueError(
            f"Saved artifact is missing: {', '.join(sorted(missing))}"
        )

    return artifact["ai_detection"], list(artifact["feature_cols"])
