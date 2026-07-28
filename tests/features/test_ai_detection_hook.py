from app.features.ai_detection.detector import (
    _combine,
    _build_feedback,
    _feature_evidence,
    _rubric_axes,
)
from app.features.ai_detection.schema import AIDetectionResult
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.validator import validator
from app.features.validator.decision import (
    _component_guidance,
    _feature_guidance,
    attempt_rank,
)
from app.features.ai_detection.schema import (
    ComponentScore,
    EvaluationDecision,
    FeatureEvidence,
)
from app.features.validator.constraints import evaluate_constraints
from app.resume_schema.resume_schema import (
    Candidate,
    Constraints,
    MutableResume,
    ProfessionalSummaryItem,
    SummarySection,
    WorkExperienceItem,
    WorkExperienceSection,
)
from config.runtime import RuntimeConfig
from app.features.ai_detection.config import AIDetectionConfig
from config.resume.candidate_profile import build_resume


class Detector:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.calls = 0

    def classify(self, text: str) -> AIDetectionResult:
        self.calls += 1
        return AIDetectionResult(
            ai_probability=next(self.probabilities),
            threshold=0.7,
            feedback=["Vary sentence structure."],
        )


def summary(text: str) -> SummarySection:
    return SummarySection(
        constraints=Constraints(),
        items=[ProfessionalSummaryItem(content=text)],
    )


def test_ai_detection_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("AI_DETECTION_ENABLED", raising=False)

    assert RuntimeConfig(_env_file=None).ai_detection_enabled is True


def test_ai_detection_accepts_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("AI_DETECTION_ENABLED", "false")

    assert RuntimeConfig(_env_file=None).ai_detection_enabled is False


def test_passing_section_does_not_trigger_rewrite(monkeypatch) -> None:
    detector = Detector([0.2])

    def unexpected_rewrite(**kwargs):
        raise AssertionError("passing content must not be rewritten")

    monkeypatch.setattr(validator, "write_section", unexpected_rewrite)

    result = validator.validate_section(
        JobListing(title="Engineer"),
        summary("Original"),
        detector,
    )

    assert result.items[0].content == "Original"
    assert detector.calls == 1


def test_failing_section_uses_feedback_and_returns_passing_retry(
    monkeypatch,
) -> None:
    detector = Detector([0.9, 0.4])
    calls = []

    def rewrite(**kwargs):
        calls.append(kwargs)
        return summary("More natural")

    monkeypatch.setattr(validator, "write_section", rewrite)

    result = validator.validate_section(
        JobListing(title="Engineer"),
        summary("Original"),
        detector,
    )

    assert result.items[0].content == "More natural"
    assert "Vary sentence structure." in calls[0]["special_instructions"]
    assert detector.calls == 2


def test_detector_initialization_failure_preserves_resume(
    monkeypatch,
) -> None:
    source = build_resume()
    monkeypatch.setattr(
        "app.bootstrap.get_ai_detector",
        lambda: (_ for _ in ()).throw(RuntimeError("missing model")),
    )

    result = validator.validate_resume(
        JobListing(title="Engineer"),
        source,
    )

    assert result is source


def test_detector_preserves_structured_feature_and_rubric_evidence() -> None:
    features = _feature_evidence(
        {
            "top_features": [
                {
                    "feature": "sent_len_cv",
                    "label": "Sentence Length Variation",
                    "description": "Relative sentence-length variation.",
                    "observed_value": 0.12,
                    "shap_value": 0.8,
                    "direction": "machine_like",
                    "importance_rank": 1,
                }
            ]
        }
    )
    axes = _rubric_axes(
        {"rubric_scores": {"specificity": 2, "idea_compression": 4}}
    )

    feedback = _build_feedback(
        feature_evidence=features,
        rubric_axes=axes,
    )

    assert features[0].feature == "sent_len_cv"
    assert features[0].shap_value == 0.8
    assert axes[0].label == "Specificity"
    assert axes[0].score == 2
    assert "Specificity" in feedback[0]
    assert features[0].description in feedback


def test_constraint_evaluation_returns_observed_checklist() -> None:
    section = WorkExperienceSection(
        constraints=Constraints(
            max_items=1,
            max_bullets_per_item=2,
            max_words_per_bullet=4,
            require_metrics=True,
            required_keywords=["Python"],
            forbidden_phrases=["responsible for"],
        ),
        items=[
            WorkExperienceItem(
                title="Engineer",
                company="Example",
                start_date="2020",
                bullets=[
                    "Improved Python throughput by 30%.",
                    "Responsible for platform delivery.",
                ],
            )
        ],
    )

    checks = evaluate_constraints(section)
    by_constraint = {}
    for check in checks:
        by_constraint.setdefault(check.constraint, []).append(check)

    assert by_constraint["max_items"][0].passed is True
    assert by_constraint["max_bullets_per_item"][0].observed == "2"
    assert any(
        check.passed is False
        for check in by_constraint["max_words_per_bullet"]
    )
    assert by_constraint["require_metrics"][0].passed is True
    assert by_constraint["required_keyword"][0].passed is True
    assert by_constraint["forbidden_phrase"][0].passed is False


def test_traced_section_streams_and_retains_accepted_attempt(
    monkeypatch,
) -> None:
    detector = Detector([0.9, 0.3])
    streamed = []
    monkeypatch.setattr(
        validator,
        "write_section",
        lambda **kwargs: summary("More natural"),
    )

    result = validator.validate_section_with_trace(
        JobListing(title="Engineer"),
        summary("Original"),
        detector,
        trace_callback=streamed.append,
    )

    assert result.final_section.items[0].content == "More natural"
    assert len(result.attempts) == 2
    assert result.attempts[1].selected is True
    assert [event.event_type for event in result.events] == [
        "evaluation_completed",
        "rewrite_started",
        "evaluation_completed",
        "attempt_accepted",
    ]
    assert streamed == result.events
    assert result.events[-1].evaluation is not None
    assert result.events[-1].evaluation.decision.passed_checks


def test_traced_section_records_deterministic_best_attempt(
    monkeypatch,
) -> None:
    detector = Detector([0.9, 0.8, 0.7])
    rewrites = iter([summary("Attempt one"), summary("Attempt two")])
    monkeypatch.setattr(
        validator,
        "write_section",
        lambda **kwargs: next(rewrites),
    )

    result = validator.validate_section_with_trace(
        JobListing(title="Engineer"),
        summary("Original"),
        detector,
        max_attempts=2,
        max_ai_probability=0.6,
    )

    assert result.status == "best_attempt_selected"
    assert result.final_section.items[0].content == "Attempt two"
    assert [attempt.selected for attempt in result.attempts] == [
        False,
        False,
        True,
    ]
    assert result.events[-1].event_type == "best_attempt_selected"
    assert result.events[-1].attempt == 2


def test_best_attempt_prefers_passing_ai_threshold_before_rubric() -> None:
    passing_ai = EvaluationDecision(
        outcome="retry",
        failed_rubric_count=2,
        ai_likeness_failed=False,
    )
    failing_ai = EvaluationDecision(
        outcome="retry",
        failed_rubric_count=0,
        ai_likeness_failed=True,
    )

    assert attempt_rank(
        passing_ai,
        AIDetectionResult(ai_probability=0.5, threshold=0.7),
    ) < attempt_rank(
        failing_ai,
        AIDetectionResult(ai_probability=0.9, threshold=0.7),
    )


def test_rewrite_cycle_builds_on_previous_attempt(monkeypatch) -> None:
    detector = Detector([0.9, 0.85, 0.3])
    inputs = []

    def rewrite(**kwargs):
        content = kwargs["section"].items[0].content
        inputs.append(content)
        return summary(f"{content} revised")

    monkeypatch.setattr(validator, "write_section", rewrite)

    result = validator.validate_section_with_trace(
        JobListing(title="Engineer"),
        summary("Original"),
        detector,
        max_attempts=2,
        max_ai_probability=0.7,
    )

    assert inputs == ["Original", "Original revised"]
    assert result.final_section.items[0].content == (
        "Original revised revised"
    )


def test_feature_guidance_uses_top_five_machine_like_features() -> None:
    features = [
        FeatureEvidence(
            feature=f"feature_{index}",
            label=f"Feature {index}",
            description="Diagnostic feature.",
            observed_value=float(index),
            shap_value=0.1 * index,
            direction="machine_like",
            importance_rank=index,
        )
        for index in range(1, 8)
    ]

    guidance = _feature_guidance(features)

    assert len(guidance) == 5
    assert any("Feature 7" in item for item in guidance)
    assert not any("Feature 1" in item for item in guidance)


def test_low_weight_tfidf_does_not_dominate_agreement_label() -> None:
    components = [
        ComponentScore(model_name="catboost", ai_probability=0.20),
        ComponentScore(model_name="distilbert", ai_probability=0.25),
        ComponentScore(model_name="tfidf_svm", ai_probability=0.95),
        ComponentScore(model_name="rubric_regressor", ai_probability=0.22),
    ]
    config = AIDetectionConfig(
        weights={
            "catboost": 1.0,
            "distilbert": 0.8,
            "tfidf_svm": 0.35,
            "rubric_regressor": 0.9,
        }
    )

    probability, explanation = _combine(components, config=config)

    assert probability > 0.22
    assert len(explanation.components) == 4
    assert explanation.agreement == "high"
    assert explanation.maximum_probability == 0.25


def test_traced_resume_returns_section_and_workflow_history(
    monkeypatch,
) -> None:
    resume = MutableResume(
        candidate=Candidate(name="Example"),
        summary=summary("Original"),
    )
    detector = Detector([0.2])
    monkeypatch.setattr("app.bootstrap.get_ai_detector", lambda: detector)

    result = validator.validate_resume_with_trace(
        JobListing(title="Engineer"),
        resume,
    )

    assert len(result.sections) == 1
    assert result.sections[0].section_name == "summary"
    assert result.events[-1].event_type == "workflow_completed"
    assert result.events[-1].decision == "accept_with_warnings"
