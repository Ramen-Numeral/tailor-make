"""Section-aware deterministic evaluation policies."""

from app.features.ai_detection.schema import EvaluationPolicy


def policy_for_section(
    section_name: str,
    *,
    ai_likeness_retry_threshold: float,
) -> EvaluationPolicy:
    """Return rubric axes appropriate for the section's writing form."""
    normalized = section_name.casefold().replace(" ", "_")
    minimums = {
        "summary": {
            "specificity": 3,
            "idea_compression": 3,
            "formality_fit": 3,
            "substantive_value": 3,
        },
        "professional_summary": {
            "specificity": 3,
            "idea_compression": 3,
            "formality_fit": 3,
            "substantive_value": 3,
        },
        "work_experience": {
            "specificity": 3,
            "idea_compression": 3,
            "substantive_value": 3,
        },
        "projects": {
            "claim_development": 3,
            "specificity": 3,
            "idea_compression": 3,
            "substantive_value": 3,
        },
        "research": {
            "global_coherence": 3,
            "claim_development": 3,
            "specificity": 3,
            "contextual_judgment": 3,
        },
    }
    if normalized == "skills":
        return EvaluationPolicy(
            minimum_rubric_scores={},
            evaluate_writing=False,
            ai_likeness_blocks=False,
            ai_likeness_retry_threshold=ai_likeness_retry_threshold,
        )
    return EvaluationPolicy(
        minimum_rubric_scores=minimums.get(
            normalized,
            EvaluationPolicy().minimum_rubric_scores,
        ),
        ai_likeness_retry_threshold=ai_likeness_retry_threshold,
    )
