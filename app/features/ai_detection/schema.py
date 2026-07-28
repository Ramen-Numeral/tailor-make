"""Typed, explainable outputs for AI-detection inference."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

RubricMinimum = Annotated[int, Field(ge=1, le=5)]


class FeatureEvidence(BaseModel):
    """One local CatBoost feature contribution for the supplied text."""

    feature: str
    label: str
    description: str
    observed_value: float
    shap_value: float
    direction: Literal["machine_like", "human_like"]
    importance_rank: int = Field(ge=1)


class RubricAxisResult(BaseModel):
    """Human-readable evidence for one rubric-scoring axis."""

    axis: str
    label: str
    definition: str
    score: int = Field(ge=1, le=5)
    interpretation: str
    contribution: float | None = None
    direction: Literal["machine_like", "human_like"] | None = None


class TokenAttribution(BaseModel):
    token: str
    attribution: float
    direction: Literal["machine_like", "human_like"]


class TermContribution(BaseModel):
    term: str
    tfidf_value: float
    coefficient: float
    contribution: float
    direction: Literal["machine_like", "human_like"]


class EnsembleComponentContribution(BaseModel):
    model_name: str
    probability: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    weighted_value: float
    normalized_contribution: float


class EnsembleExplanation(BaseModel):
    method: Literal["average", "weighted"]
    components: list[EnsembleComponentContribution] = Field(default_factory=list)
    weighted_sum: float
    weight_total: float
    combined_probability: float = Field(ge=0.0, le=1.0)
    minimum_probability: float = Field(ge=0.0, le=1.0)
    maximum_probability: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    standard_deviation: float = Field(ge=0.0)
    agreement: Literal["high", "moderate", "low"]


class ComponentDelta(BaseModel):
    model_name: str
    before: float = Field(ge=0.0, le=1.0)
    after: float = Field(ge=0.0, le=1.0)
    delta: float


class CounterfactualComparison(BaseModel):
    before_probability: float = Field(ge=0.0, le=1.0)
    after_probability: float = Field(ge=0.0, le=1.0)
    delta: float
    components: list[ComponentDelta] = Field(default_factory=list)


class ConstraintCheck(BaseModel):
    """Observed result for one configured resume constraint."""

    constraint: str
    label: str
    expected: str
    observed: str
    passed: bool
    severity: Literal["required", "advisory"] = "required"
    item_index: int | None = Field(default=None, ge=0)


class EvaluationPolicy(BaseModel):
    """Deterministic acceptance policy for one rewrite attempt."""

    minimum_rubric_scores: dict[str, RubricMinimum] = Field(
        default_factory=lambda: {
            "specificity": 3,
            "stylistic_variation": 3,
            "idea_compression": 3,
            "semantic_novelty": 3,
        }
    )
    required_constraints_must_pass: bool = True
    maximum_failed_advisories: int | None = Field(default=None, ge=0)
    evaluate_writing: bool = True
    ai_likeness_blocks: bool = True
    ai_likeness_retry_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )


class EvaluationDecision(BaseModel):
    """Auditable decision derived only from structured evidence and policy."""

    outcome: Literal["accept", "retry", "accept_with_warnings"]
    hard_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    retry_instructions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    failed_constraint_count: int = Field(default=0, ge=0)
    failed_rubric_count: int = Field(default=0, ge=0)
    minimum_rubric_score: int | None = Field(default=None, ge=1, le=5)
    average_rubric_score: float | None = Field(default=None, ge=1.0, le=5.0)
    ai_likeness_failed: bool = False


class ComponentScore(BaseModel):
    """Score returned by one detection component."""

    model_name: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    error: str | None = None
    base_value: float | None = None
    explanation_note: str | None = None
    token_attributions: list[TokenAttribution] = Field(default_factory=list)
    term_contributions: list[TermContribution] = Field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.error is None


class AIDetectionResult(BaseModel):
    """Combined AI-detection result used by the rewrite validator."""

    ai_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    components: list[ComponentScore] = Field(default_factory=list)
    feature_evidence: list[FeatureEvidence] = Field(default_factory=list)
    rubric_axes: list[RubricAxisResult] = Field(default_factory=list)
    ensemble_explanation: EnsembleExplanation | None = None
    feedback: list[str] = Field(default_factory=list)
    scoring_status: Literal["completed", "skipped"] = "completed"
    skipped_reason: str | None = None

    @property
    def prediction(
        self,
    ) -> Literal["AI-written", "human-written", "not-evaluated"]:
        if self.scoring_status == "skipped":
            return "not-evaluated"
        return (
            "AI-written"
            if self.ai_probability >= self.threshold
            else "human-written"
        )

    @property
    def passed(self) -> bool:
        return (
            self.scoring_status == "skipped"
            or self.ai_probability < self.threshold
        )
