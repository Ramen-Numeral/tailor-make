"""Runtime configuration for AI-detection inference."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class AIDetectionConfig:
    threshold: float = 0.5
    ensemble_method: Literal["average", "weighted"] = "weighted"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "catboost": 1.0,
            "distilbert": 0.8,
            # Keep the legacy 1–2 gram artifact diagnostic until it is
            # retrained with the phrase-level TfidfConfig.
            "tfidf_svm": 0.35,
            "rubric_regressor": 0.9,
        }
    )
    run_rubric: bool = True
    rubric_max_attempts: int = 3
    rewrite_threshold: float = 0.7
    rewrite_attempts: int = 3
    include_feedback: bool = True
    soft_fail: bool = True
    integrated_gradients_steps: int = 16

    def __post_init__(self) -> None:
        for name in ("threshold", "rewrite_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("component weights cannot be negative")

        for name in ("rubric_max_attempts", "rewrite_attempts"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

        if (
            self.ensemble_method == "weighted"
            and not any(self.weights.values())
        ):
            raise ValueError(
                "weighted ensemble requires at least one positive weight"
            )
        if self.integrated_gradients_steps < 2:
            raise ValueError("integrated_gradients_steps must be at least 2")


ai_detection_config = AIDetectionConfig()
