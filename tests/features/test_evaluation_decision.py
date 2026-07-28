from app.features.ai_detection.schema import (
    AIDetectionResult,
    ComponentScore,
    ConstraintCheck,
    EvaluationPolicy,
    FeatureEvidence,
    RubricAxisResult,
)
from app.features.validator.decision import attempt_rank, evaluate_attempt


def rubric_axes(**scores: int) -> list[RubricAxisResult]:
    return [
        RubricAxisResult(
            axis=axis,
            label=axis.replace("_", " ").title(),
            definition=f"Definition for {axis}",
            score=score,
            interpretation=f"Level {score}",
        )
        for axis, score in scores.items()
    ]


def passing_detection(**updates) -> AIDetectionResult:
    values = {
        "ai_probability": 0.2,
        "threshold": 0.5,
        "rubric_axes": rubric_axes(
            specificity=4,
            stylistic_variation=4,
            idea_compression=4,
            semantic_novelty=4,
        ),
    }
    values.update(updates)
    return AIDetectionResult(**values)


def test_all_blocking_rules_pass() -> None:
    checks = [
        ConstraintCheck(
            constraint="max_items",
            label="Maximum items",
            expected="At most 3",
            observed="2",
            passed=True,
        )
    ]

    decision = evaluate_attempt(passing_detection(), checks)

    assert decision.outcome == "accept"
    assert not decision.hard_failures
    assert any("Maximum items" in check for check in decision.passed_checks)


def test_advisory_failure_accepts_with_warning() -> None:
    checks = [
        ConstraintCheck(
            constraint="require_metrics",
            label="Measurable impact",
            expected="At least one",
            observed="None",
            passed=False,
            severity="advisory",
        )
    ]

    decision = evaluate_attempt(passing_detection(), checks)

    assert decision.outcome == "accept_with_warnings"
    assert decision.warnings
    assert not decision.hard_failures


def test_required_constraint_failure_retries_with_guidance() -> None:
    checks = [
        ConstraintCheck(
            constraint="max_words_per_bullet",
            label="Maximum bullet words",
            expected="At most 20",
            observed="27",
            passed=False,
        )
    ]

    decision = evaluate_attempt(passing_detection(), checks)

    assert decision.outcome == "retry"
    assert decision.failed_constraint_count == 1
    assert "Shorten" in decision.retry_instructions[0]


def test_required_rubric_axis_failure_retries() -> None:
    detection = passing_detection(
        rubric_axes=rubric_axes(
            specificity=2,
            stylistic_variation=4,
            idea_compression=4,
            semantic_novelty=4,
        )
    )

    decision = evaluate_attempt(detection, [])

    assert decision.outcome == "retry"
    assert decision.failed_rubric_count == 1
    assert any("concrete" in item for item in decision.retry_instructions)


def test_unavailable_component_is_non_blocking_warning() -> None:
    detection = passing_detection(
        components=[
            ComponentScore(
                model_name="catboost",
                ai_probability=0.5,
                error="model missing",
            )
        ]
    )

    decision = evaluate_attempt(detection, [])

    assert decision.outcome == "accept_with_warnings"
    assert "model missing" in decision.warnings[0]


def test_ai_likeness_failure_uses_machine_like_feature_guidance() -> None:
    detection = passing_detection(
        ai_probability=0.8,
        feature_evidence=[
            FeatureEvidence(
                feature="sent_len_cv",
                label="Sentence Length Variation",
                description="Relative sentence-length variation.",
                observed_value=0.1,
                shap_value=0.9,
                direction="machine_like",
                importance_rank=1,
            )
        ],
    )

    decision = evaluate_attempt(detection, [])

    assert decision.outcome == "retry"
    assert any(
        "sentence and bullet lengths" in item
        for item in decision.retry_instructions
    )


def test_attempt_rank_prioritizes_required_constraints_over_probability() -> None:
    low_probability = passing_detection(ai_probability=0.1)
    high_probability = passing_detection(ai_probability=0.6)
    constraint_failure = evaluate_attempt(
        low_probability,
        [
            ConstraintCheck(
                constraint="required_keyword",
                label="Required keyword",
                expected="Present",
                observed="Missing",
                passed=False,
            )
        ],
    )
    no_constraint_failure = evaluate_attempt(
        high_probability,
        [],
        EvaluationPolicy(ai_likeness_retry_threshold=0.5),
    )

    assert attempt_rank(
        no_constraint_failure,
        high_probability,
    ) < attempt_rank(
        constraint_failure,
        low_probability,
    )


def test_identical_evidence_produces_identical_decision() -> None:
    detection = passing_detection()

    first = evaluate_attempt(detection, [])
    second = evaluate_attempt(detection, [])

    assert first == second
