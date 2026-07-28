"""Bounded LLM adjudication for semantically ambiguous evidence."""

import json
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from app.features.job_listing_parser.listing_schema import Requirement
from app.features.keyword_evidence.schema import (
    CandidateFitRubric,
    EvidenceJudgment,
    EvidenceJudgmentBatch,
    ResumeEvidence,
)
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)

T = TypeVar("T", bound=BaseModel)


class CompactEvidenceJudgment(BaseModel):
    requirement: int = Field(ge=0)
    support: Literal["supported", "partial", "unsupported"]
    evidence: list[int] = Field(default_factory=list)
    reason: str
    relationship: Literal[
        "equivalent",
        "parent_generalization",
        "transferable",
        "adjacent",
        "none",
    ] = "none"
    safe_keywords: list[str] = Field(default_factory=list)


class CompactEvidenceJudgmentBatch(BaseModel):
    judgments: list[CompactEvidenceJudgment] = Field(default_factory=list)
    fit_rubric: CandidateFitRubric | None = None


class StructuredJudge(Protocol):
    def invoke_structured(
        self,
        prompt: str,
        schema: type[T],
        **kwargs,
    ) -> T: ...


def judge_ambiguous_evidence(
    requirements: list[Requirement],
    evidence: list[ResumeEvidence],
    judge: StructuredJudge,
) -> EvidenceJudgmentBatch:
    """Classify semantic support in one auditable, citation-only call."""
    if not requirements:
        return EvidenceJudgmentBatch()
    cache_key = content_key(
        "evidence_judgment",
        object_identity(judge),
        requirements,
        evidence,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return EvidenceJudgmentBatch.model_validate(cached)
    prompt = _prompt(requirements, evidence)
    raw_result = judge.invoke_structured(
        prompt=prompt,
        schema=CompactEvidenceJudgmentBatch,
        temperature=0,
        max_tokens=2400,
        trace_context="semantic_evidence_adjudication",
    )
    result = (
        raw_result
        if isinstance(raw_result, EvidenceJudgmentBatch)
        else _expand_judgments(raw_result, requirements, evidence)
    )
    set_cached(cache_key, result.model_dump(mode="json"))
    return result


def _prompt(
    requirements: list[Requirement],
    evidence: list[ResumeEvidence],
) -> str:
    payload = {
        "requirements": [
            {
                "id": index,
                "text": requirement.text,
                "kind": requirement.kind,
                "required": requirement.required,
            }
            for index, requirement in enumerate(requirements)
        ],
        "evidence": [
            {
                "id": index,
                "section": item.section,
                "field": item.field,
                "text": item.text,
            }
            for index, item in enumerate(evidence)
        ],
    }
    return """Return one schema-valid JSON object that judges whether existing
resume evidence supports each job
requirement. This is semantic evidence matching, not keyword matching.

Definitions:
- supported: cited evidence directly demonstrates the capability or a clear
  transferable equivalent (for example roadmap/backlog ownership supports
  project management).
- partial: cited evidence is relevant but an important domain, scope, or skill
  element is missing.
- unsupported: no cited evidence establishes the requirement.

Relationship labels:
- equivalent: same capability under different wording.
- parent_generalization: evidence is a narrower child technology/capability
  that truthfully entails the broader requirement (PostgreSQL -> SQL).
- transferable: demonstrated work maps to the function in another context
  (roadmap/backlog ownership -> project management).
- adjacent: relevant foundation, but not enough to claim the requirement.
- none: no defensible relationship.

Rules:
- Return exactly one judgment per numeric requirement index.
- Cite only numeric evidence indexes supplied below.
- Never infer an unstated tool, credential, domain, or responsibility.
- supported and partial require at least one evidence index.
- unsupported must use an empty evidence list.
- Explain the concrete relationship in one short sentence.
- safe_keywords may contain only job-listing language that the cited evidence
  makes truthful. Generalization may go child-to-parent (PostgreSQL -> SQL),
  never parent-to-child (SQL does not prove PostgreSQL).
- For partial/unsupported judgments, safe_keywords must be empty.

Also score the candidate holistically on all six fit_rubric axes from 0 to 5.
Functional and transferable alignment must receive credit even when exact job
vocabulary or industry domain is absent. A zero means no relevant evidence at
all, such as cashier experience for a physician role. Cite concrete reasoning;
do not reward credentials or experience that are not present.

DATA:
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _expand_judgments(
    result: CompactEvidenceJudgmentBatch,
    requirements: list[Requirement],
    evidence: list[ResumeEvidence],
) -> EvidenceJudgmentBatch:
    judgments = []
    for judgment in result.judgments:
        if judgment.requirement >= len(requirements):
            continue
        evidence_ids = [
            evidence[index].evidence_id
            for index in judgment.evidence
            if 0 <= index < len(evidence)
        ]
        judgments.append(
            EvidenceJudgment(
                requirement_id=requirements[judgment.requirement].id,
                support=judgment.support,
                evidence_ids=evidence_ids,
                reason=judgment.reason,
                relationship=judgment.relationship,
                safe_keywords=judgment.safe_keywords,
            )
        )
    return EvidenceJudgmentBatch(
        judgments=judgments,
        fit_rubric=result.fit_rubric,
    )
