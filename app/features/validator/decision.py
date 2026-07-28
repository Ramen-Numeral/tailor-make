"""Pure deterministic decisions over structured evaluation evidence."""

from app.features.ai_detection.schema import (
    AIDetectionResult,
    ConstraintCheck,
    EvaluationDecision,
    EvaluationPolicy,
    FeatureEvidence,
)


def evaluate_attempt(
    detection: AIDetectionResult,
    constraint_checks: list[ConstraintCheck],
    policy: EvaluationPolicy | None = None,
) -> EvaluationDecision:
    """Decide whether to accept or retry without invoking another model."""
    policy = policy or EvaluationPolicy()
    hard_failures: list[str] = []
    warnings: list[str] = []
    passed_checks: list[str] = []
    retry_instructions: list[str] = []
    failed_constraint_count = 0
    failed_rubric_count = 0

    for check in constraint_checks:
        description = (
            f"{check.label}: expected {check.expected}; "
            f"observed {check.observed}."
        )
        if check.passed:
            passed_checks.append(description)
        elif check.severity == "required" and policy.required_constraints_must_pass:
            hard_failures.append(description)
            failed_constraint_count += 1
            retry_instructions.append(_constraint_guidance(check))
        else:
            warnings.append(description)

    failed_advisories = sum(
        not check.passed and check.severity == "advisory"
        for check in constraint_checks
    )
    if (
        policy.maximum_failed_advisories is not None
        and failed_advisories > policy.maximum_failed_advisories
    ):
        message = (
            f"{failed_advisories} advisory constraints failed; policy permits "
            f"at most {policy.maximum_failed_advisories}."
        )
        hard_failures.append(message)
        failed_constraint_count += 1
        retry_instructions.append("Address the failed advisory constraints.")

    axes = {axis.axis: axis for axis in detection.rubric_axes}
    available_scores = [axis.score for axis in detection.rubric_axes]
    for axis_name, minimum in policy.minimum_rubric_scores.items():
        axis = axes.get(axis_name)
        label = axis_name.replace("_", " ").title()
        if axis is None:
            warnings.append(f"{label} was not scored; it did not block acceptance.")
            continue
        if axis.score < minimum:
            hard_failures.append(
                f"{axis.label}: required at least {minimum}/5; "
                f"observed {axis.score}/5."
            )
            failed_rubric_count += 1
            retry_instructions.append(_rubric_guidance(axis_name))
        else:
            passed_checks.append(
                f"{axis.label}: {axis.score}/5 met the {minimum}/5 minimum."
            )

    for component in detection.components:
        if not component.available:
            warnings.append(
                f"{component.model_name} was unavailable: {component.error}"
            )

    ai_likeness_failed = (
        policy.ai_likeness_blocks
        and detection.ai_probability >= policy.ai_likeness_retry_threshold
    )
    if ai_likeness_failed:
        hard_failures.append(
            "AI-likeness exceeded the retry threshold: "
            f"{detection.ai_probability:.3f} observed versus "
            f"{policy.ai_likeness_retry_threshold:.3f} maximum."
        )
        retry_instructions.extend(
            _feature_guidance(detection.feature_evidence)
        )
        retry_instructions.extend(_component_guidance(detection))
    elif policy.ai_likeness_blocks:
        passed_checks.append(
            "AI-likeness was below the retry threshold: "
            f"{detection.ai_probability:.3f} observed versus "
            f"{policy.ai_likeness_retry_threshold:.3f} maximum."
        )
    else:
        passed_checks.append(
            detection.skipped_reason
            or "AI-likeness is diagnostic-only for this section."
        )

    retry_instructions = _deduplicate(retry_instructions)
    warnings = _deduplicate(warnings)
    hard_failures = _deduplicate(hard_failures)

    if hard_failures:
        outcome = "retry"
        reasons = [
            f"Retry required because {len(hard_failures)} blocking rule(s) failed.",
            *hard_failures,
        ]
    elif warnings:
        outcome = "accept_with_warnings"
        reasons = [
            "Accepted because all blocking rules passed.",
            f"{len(warnings)} non-blocking warning(s) remain.",
        ]
    else:
        outcome = "accept"
        reasons = ["Accepted because all configured blocking rules passed."]

    return EvaluationDecision(
        outcome=outcome,
        hard_failures=hard_failures,
        warnings=warnings,
        passed_checks=passed_checks,
        retry_instructions=retry_instructions,
        reasons=reasons,
        failed_constraint_count=failed_constraint_count,
        failed_rubric_count=failed_rubric_count,
        minimum_rubric_score=min(available_scores) if available_scores else None,
        average_rubric_score=(
            sum(available_scores) / len(available_scores)
            if available_scores
            else None
        ),
        ai_likeness_failed=ai_likeness_failed,
    )


def attempt_rank(
    decision: EvaluationDecision,
    detection: AIDetectionResult,
) -> tuple[int, int, int, int, float, float]:
    """Return a stable lower-is-better rank for unsuccessful attempts."""
    minimum = decision.minimum_rubric_score or 0
    average = decision.average_rubric_score or 0.0
    return (
        decision.failed_constraint_count,
        int(decision.ai_likeness_failed),
        decision.failed_rubric_count,
        -minimum,
        -average,
        detection.ai_probability,
    )


def _constraint_guidance(check: ConstraintCheck) -> str:
    guidance = {
        "max_items": "Reduce the section to the configured item limit.",
        "min_items": "Add supported content to meet the minimum item count.",
        "max_words": "Shorten the section without removing supported facts.",
        "min_words": "Add relevant supported detail.",
        "max_sentences": "Combine or remove redundant sentences.",
        "max_bullets_per_item": "Reduce the number of bullets.",
        "min_bullets_per_item": "Add a supported bullet.",
        "max_words_per_bullet": (
            "Shorten the affected bullet without removing supported facts."
        ),
        "required_keyword": "Include the required keyword where factually supported.",
        "forbidden_phrase": "Remove or replace the forbidden phrase.",
        "require_metrics": (
            "Use an existing supported metric; do not invent a number."
        ),
        "factual_numbers": (
            "Remove introduced numbers and use only metrics from the source item."
        ),
        "factual_locked_field": "Restore the locked field to its source value.",
        "factual_item_provenance": "Remove items not present in the selected source.",
        "factual_structured_terms": (
            "Remove skills or technologies not present in the source section."
        ),
    }
    return guidance.get(
        check.constraint,
        f"Revise the section to satisfy: {check.label}.",
    )


def _rubric_guidance(axis: str) -> str:
    guidance = {
        "specificity": (
            "Replace generic wording with concrete, fact-supported responsibilities."
        ),
        "stylistic_variation": (
            "Vary sentence openings and clause structure while preserving meaning."
        ),
        "idea_compression": (
            "Remove unnecessary wording while retaining the substantive claim."
        ),
        "semantic_novelty": (
            "Ensure each sentence or bullet contributes distinct information."
        ),
        "global_coherence": "Make the section develop one clear professional theme.",
        "claim_development": (
            "Support each claim with concrete context already present in the source."
        ),
        "voice_and_stance": "Use direct, consistent, and purposeful phrasing.",
        "contextual_judgment": (
            "Preserve relevant context, qualifications, and constraints."
        ),
        "formality_fit": "Use concise professional language appropriate for a resume.",
        "substantive_value": "Prioritize meaningful responsibilities and outcomes.",
    }
    return guidance.get(axis, f"Improve the {axis.replace('_', ' ')} score.")


def _feature_guidance(features: list[FeatureEvidence]) -> list[str]:
    guidance: list[str] = []
    for feature in sorted(
        (
            evidence
            for evidence in features
            if evidence.direction == "machine_like"
        ),
        key=lambda evidence: evidence.shap_value,
        reverse=True,
    )[:5]:
        name = feature.feature
        if "sent_len" in name or name == "sent_len_entropy":
            instruction = "Vary sentence and bullet lengths more naturally."
        elif "repeat" in name or name in {"unique_word_ratio", "hapax_ratio"}:
            instruction = "Reduce repetitive wording and use more specific vocabulary."
        elif "punct" in name or name.endswith("_per_sent"):
            instruction = "Vary punctuation and sentence structure naturally."
        elif name == "avg_adverbs_per_sent":
            instruction = "Replace unnecessary adverbs with precise verbs."
        else:
            instruction = (
                f"Review {feature.label.lower()}, which strongly influenced "
                "the writing-pattern assessment."
            )
        guidance.append(
            f"{instruction} SHAP signal: {feature.label}; "
            f"observed {feature.observed_value:.3f}; "
            f"contribution {feature.shap_value:+.3f}."
        )
    return guidance


def _component_guidance(detection: AIDetectionResult) -> list[str]:
    guidance: list[str] = []
    for component in detection.components:
        if component.model_name == "tfidf_svm":
            terms = [
                term for term in component.term_contributions
                if term.direction == "machine_like"
            ][:5]
            if terms:
                guidance.append(
                    "Reduce formulaic use of these machine-like TF-IDF terms "
                    "where meaning can be preserved: "
                    + ", ".join(term.term for term in terms)
                    + "."
                )
    return guidance


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
