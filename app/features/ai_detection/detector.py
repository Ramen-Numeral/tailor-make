
"""Small orchestration layer for the AI-detection ensemble."""

from typing import Any, Callable

from app.features.ai_detection.config import (
    AIDetectionConfig,
    ai_detection_config,
)
from app.features.ai_detection.schema import (
    AIDetectionResult,
    ComponentScore,
    EnsembleComponentContribution,
    EnsembleExplanation,
    FeatureEvidence,
    RubricAxisResult,
)
from app.infrastructure.ai_detection import (
    predict_catboost,
    predict_distilbert,
    predict_rubric,
    predict_tfidf,
)
from ml_pipelines.models.rubric_regressor.rubric import (
    AXES_DEFINITIONS,
    LEVEL_DEFINITIONS,
)


class AIDetector:
    """Owns loaded models and exposes one text-classification method."""

    def __init__(
        self,
        *,
        catboost_model: Any,
        distilbert_model: Any,
        distilbert_tokenizer: Any,
        tfidf_svm_model: Any,
        rubric_regression_model: Any | None = None,
        config: AIDetectionConfig = ai_detection_config,
    ) -> None:
        self.catboost_model = catboost_model
        self.distilbert_model = distilbert_model
        self.distilbert_tokenizer = distilbert_tokenizer
        self.tfidf_svm_model = tfidf_svm_model
        self.rubric_regression_model = rubric_regression_model
        self.config = config

    def classify(self, text: str) -> AIDetectionResult:
        if not text.strip():
            raise ValueError("text cannot be empty")

        components: list[ComponentScore] = []
        shap_details: dict[str, Any] | None = None
        rubric_details: dict[str, Any] | None = None

        component, shap_details = self._run_component(
            "catboost",
            lambda: predict_catboost(self.catboost_model, text),
        )
        components.append(component)

        components.append(
            self._run_component(
                "distilbert",
                lambda: predict_distilbert(
                    self.distilbert_model,
                    self.distilbert_tokenizer,
                    text,
                ),
            )[0]
        )

        components.append(
            self._run_component(
                "tfidf_svm",
                lambda: predict_tfidf(self.tfidf_svm_model, text),
            )[0]
        )

        if self.config.run_rubric:
            component, rubric_details = self._run_component(
                "rubric_regressor",
                lambda: predict_rubric(
                    text,
                    self.rubric_regression_model,
                ),
            )
            components.append(component)

        probability, ensemble_explanation = _combine(
            components,
            config=self.config,
        )
        feature_evidence = _feature_evidence(shap_details)
        rubric_axes = _rubric_axes(rubric_details)

        feedback = (
            _build_feedback(
                feature_evidence=feature_evidence,
                rubric_axes=rubric_axes,
            )
            if self.config.include_feedback
            else []
        )

        return AIDetectionResult(
            ai_probability=probability,
            threshold=self.config.threshold,
            components=components,
            feature_evidence=feature_evidence,
            rubric_axes=rubric_axes,
            ensemble_explanation=ensemble_explanation,
            feedback=feedback,
        )

    __call__ = classify

    def _run_component(
        self,
        model_name: str,
        prediction: Callable[[], Any],
    ) -> tuple[ComponentScore, dict[str, Any] | None]:
        try:
            result = prediction()
            return result if isinstance(result, tuple) else (result, None)
        except Exception as error:
            if not self.config.soft_fail:
                raise

            return ComponentScore(
                model_name=model_name,
                ai_probability=0.5,
                error=str(error),
            ), None


def _combine(
    components: list[ComponentScore],
    *,
    config: AIDetectionConfig,
) -> tuple[float, EnsembleExplanation]:
    available = [
        component
        for component in components
        if component.available
    ]

    if not available:
        return 0.5, EnsembleExplanation(
            method=config.ensemble_method,
            weighted_sum=0.0,
            weight_total=0.0,
            combined_probability=0.5,
            minimum_probability=0.5,
            maximum_probability=0.5,
            spread=0.0,
            standard_deviation=0.0,
            agreement="high",
        )

    weights = [
        1.0 if config.ensemble_method == "average"
        else config.weights.get(component.model_name, 1.0)
        for component in available
    ]
    weighted_sum = sum(
        component.ai_probability * weight
        for component, weight in zip(available, weights, strict=True)
    )
    total_weight = sum(weights)
    probability = weighted_sum / total_weight if total_weight > 0 else 0.5
    values = [component.ai_probability for component in available]
    # Very low-weight legacy/diagnostic components remain visible and still
    # contribute to the ensemble, but do not dominate the agreement label.
    agreement_values = [
        component.ai_probability
        for component, weight in zip(available, weights, strict=True)
        if weight >= 0.5
    ] or values
    mean = sum(agreement_values) / len(agreement_values)
    spread = max(agreement_values) - min(agreement_values)
    deviation = (
        sum((value - mean) ** 2 for value in agreement_values)
        / len(agreement_values)
    ) ** 0.5
    explanation = EnsembleExplanation(
        method=config.ensemble_method,
        components=[
            EnsembleComponentContribution(
                model_name=component.model_name,
                probability=component.ai_probability,
                weight=weight,
                weighted_value=component.ai_probability * weight,
                normalized_contribution=(
                    component.ai_probability * weight / total_weight
                    if total_weight > 0 else 0.0
                ),
            )
            for component, weight in zip(available, weights, strict=True)
        ],
        weighted_sum=weighted_sum,
        weight_total=total_weight,
        combined_probability=probability,
        minimum_probability=min(agreement_values),
        maximum_probability=max(agreement_values),
        spread=spread,
        standard_deviation=deviation,
        agreement="high" if spread <= 0.15 else (
            "moderate" if spread <= 0.30 else "low"
        ),
    )
    return probability, explanation


def _feature_evidence(
    details: dict[str, Any] | None,
) -> list[FeatureEvidence]:
    if not details:
        return []

    return [
        FeatureEvidence.model_validate(feature)
        for feature in details.get("top_features", [])
    ]


def _rubric_axes(
    details: dict[str, Any] | None,
) -> list[RubricAxisResult]:
    if not details:
        return []

    scores = details.get("rubric_scores", {})
    contributions = details.get("axis_contributions", {})
    return [
        RubricAxisResult(
            axis=axis,
            label=axis.replace("_", " ").title(),
            definition=AXES_DEFINITIONS[axis],
            score=int(score),
            interpretation=LEVEL_DEFINITIONS.get(
                int(score),
                "Needs improvement",
            ),
            contribution=contributions.get(axis),
            direction=(
                "machine_like"
                if contributions.get(axis, 0) >= 0
                else "human_like"
            ) if axis in contributions else None,
        )
        for axis, score in scores.items()
        if axis in AXES_DEFINITIONS
    ]


def _build_feedback(
    *,
    feature_evidence: list[FeatureEvidence],
    rubric_axes: list[RubricAxisResult],
) -> list[str]:
    feedback = [
        f"{axis.label}: {axis.interpretation}"
        for axis in sorted(rubric_axes, key=lambda item: item.score)[:3]
        if axis.score <= 3
    ]

    feedback.extend(
        feature.description
        for feature in feature_evidence
        if feature.direction == "machine_like"
    )

    if not feedback:
        feedback.append(
            "Use more natural sentence variation and less repetitive phrasing."
        )

    return feedback
