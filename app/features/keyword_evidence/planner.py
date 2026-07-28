"""Build a conservative, explainable job-keyword coverage plan."""

import logging
import re
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from app.features.job_listing_parser.listing_schema import (
    JobListing,
    Requirement,
)
from app.features.keyword_evidence.schema import (
    CoveragePlan,
    EvidenceMatch,
    KeywordAssignment,
    RequirementEvidenceMatch,
    ResumeEvidence,
)
from app.features.keyword_evidence.adjudicator import (
    StructuredJudge,
    judge_ambiguous_evidence,
)
from app.infrastructure.keyword_models import (
    build_cosine_index,
    index_bm25,
    query_bm25_index,
    query_cosine_index,
)
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.resume_schema.resume_schema import (
    MutableResume,
    RESUME_SECTION_FIELDS,
    Resume,
)

_NON_EVIDENCE_FIELDS = {
    "id",
    "start_date",
    "end_date",
    "graduation_date",
    "location",
    "url",
}
_KEYWORD_FIELDS = {"skills", "technologies", "coursework"}
_ALIASES = {
    "amazon web services": "aws",
    "apis": "api",
    "asp net core": "asp.net core",
    "dotnet": ".net",
    "event based": "event driven",
    "k8s": "kubernetes",
    "node js": "node.js",
    "postgres": "postgresql",
    "restful api": "rest api",
    "soa": "service oriented architecture",
}
_DIRECTIONAL_GENERALIZATIONS = {
    "postgresql": {"sql", "relational database", "relational databases"},
    "mysql": {"sql", "relational database", "relational databases"},
    "mariadb": {"sql", "relational database", "relational databases"},
    "sql server": {"sql", "relational database", "relational databases"},
}
logger = logging.getLogger(__name__)


class SentenceEmbedder(Protocol):
    """Minimal SentenceTransformer-compatible encoding interface."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def build_coverage_plan(
    job_listing: JobListing,
    resume: Resume,
    *,
    embedder: SentenceEmbedder | None = None,
    top_k: int = 5,
    semantic_threshold: float = 0.58,
    adjudicator: StructuredJudge | None = None,
) -> CoveragePlan:
    """Match parsed requirements to atomic source evidence."""
    evidence = extract_resume_evidence(resume)
    if not job_listing.requirements or not evidence:
        return CoveragePlan()

    corpus = [item.text for item in evidence]
    bm25_index = index_bm25(corpus)
    cosine_index = None
    if embedder is not None:
        try:
            vector_key = content_key(
                "resume_embeddings",
                object_identity(embedder),
                corpus,
            )
            cached_vectors = get_cached(vector_key)
            vectors = (
                np.asarray(cached_vectors)
                if cached_vectors is not None
                else _encode(embedder, corpus)
            )
            if cached_vectors is None:
                set_cached(vector_key, vectors.tolist())
            cosine_index = build_cosine_index(vectors, evidence)
        except Exception as error:
            logger.warning(
                "keyword_embedding_failed using_exact_and_bm25=true error=%s",
                error,
            )
            embedder = None

    requirement_matches = [
        _match_requirement(
            requirement,
            evidence,
            bm25_index=bm25_index,
            cosine_index=cosine_index,
            embedder=embedder,
            top_k=top_k,
            semantic_threshold=semantic_threshold,
        )
        for requirement in job_listing.requirements
    ]
    fit_rubric = None
    if adjudicator is not None:
        requirement_matches, fit_rubric = _adjudicate_matches(
            job_listing.requirements,
            evidence,
            requirement_matches,
            adjudicator,
        )
    assignments = _build_assignments(
        job_listing.requirements,
        requirement_matches,
    )
    return CoveragePlan(
        requirement_matches=requirement_matches,
        assignments=assignments,
        fit_rubric=fit_rubric,
    )


def extract_resume_evidence(resume: Resume) -> list[ResumeEvidence]:
    """Preserve section, item, and field provenance for atomic text values."""
    evidence: list[ResumeEvidence] = []
    for section_name in RESUME_SECTION_FIELDS:
        section = getattr(resume, section_name, None)
        if section is None:
            continue
        for item in section.items:
            for field_name in type(item).model_fields:
                if field_name in _NON_EVIDENCE_FIELDS:
                    continue
                value = getattr(item, field_name, None)
                for value_index, text in enumerate(_text_values(value)):
                    evidence.append(
                        ResumeEvidence(
                            evidence_id=(
                                f"{section_name}:{item.id}:"
                                f"{field_name}:{value_index}"
                            ),
                            section=section_name,
                            item_id=item.id,
                            field=field_name,
                            text=text,
                        )
                    )
    return evidence


def apply_coverage_plan(
    resume: MutableResume,
    plan: CoveragePlan,
) -> MutableResume:
    """Replace static keyword constraints with supported per-run keywords."""
    updates = {}
    for section_name in RESUME_SECTION_FIELDS:
        section = getattr(resume, section_name, None)
        if section is None:
            continue
        constraints = section.constraints.model_copy(
            update={
                "required_keywords": plan.keywords_for(section_name),
            },
            deep=True,
        )
        updates[section_name] = section.model_copy(
            update={"constraints": constraints},
            deep=True,
        )
    return resume.model_copy(update=updates, deep=True)


def _match_requirement(
    requirement: Requirement,
    evidence: list[ResumeEvidence],
    *,
    bm25_index,
    cosine_index,
    embedder: SentenceEmbedder | None,
    top_k: int,
    semantic_threshold: float,
) -> RequirementEvidenceMatch:
    exact_indexes = {
        index
        for index, item in enumerate(evidence)
        if _is_exact_or_alias_match(requirement.text, item.text)
    }
    matches: dict[int, EvidenceMatch] = {
        index: EvidenceMatch(
            evidence=evidence[index],
            exact_or_alias_match=True,
        )
        for index in exact_indexes
    }

    for rank, (index, _, score) in enumerate(
        query_bm25_index(bm25_index, requirement.text, k=top_k),
        start=1,
    ):
        if score <= 0:
            continue
        current = matches.get(index) or EvidenceMatch(
            evidence=evidence[index]
        )
        matches[index] = current.model_copy(
            update={"bm25_rank": rank, "bm25_score": score}
        )

    if cosine_index is not None and embedder is not None:
        query_key = content_key(
            "requirement_embedding",
            object_identity(embedder),
            requirement.text,
        )
        cached_query = get_cached(query_key)
        query_vector = (
            np.asarray(cached_query)
            if cached_query is not None
            else _encode(embedder, [requirement.text])[0]
        )
        if cached_query is None:
            set_cached(query_key, query_vector.tolist())
        for result in query_cosine_index(
            cosine_index,
            query_vector,
            k=top_k,
        ):
            current = matches.get(result.index) or EvidenceMatch(
                evidence=evidence[result.index]
            )
            matches[result.index] = current.model_copy(
                update={
                    "cosine_rank": result.rank,
                    "cosine_score": result.score,
                }
            )

    ranked = sorted(
        matches.values(),
        key=lambda match: (
            not match.exact_or_alias_match,
            match.cosine_rank or top_k + 1,
            match.bm25_rank or top_k + 1,
        ),
    )
    support = "unsupported"
    if any(match.exact_or_alias_match for match in ranked):
        support = "supported"
    elif any(
        (match.cosine_score or -1.0) >= semantic_threshold
        for match in ranked
    ):
        support = "partial"

    exact_matches = [
        match
        for match in ranked
        if match.exact_or_alias_match
    ]
    retrieved_matches = [
        match
        for match in ranked
        if not match.exact_or_alias_match
    ]
    retained_matches = [
        *exact_matches,
        *retrieved_matches[:max(0, top_k - len(exact_matches))],
    ]
    return RequirementEvidenceMatch(
        requirement_id=requirement.id,
        requirement_text=requirement.text,
        requirement_kind=requirement.kind,
        importance=requirement.importance,
        support=support,
        matches=retained_matches,
        decision_source=(
            "exact"
            if support == "supported"
            else "embedding"
            if support == "partial"
            else "none"
        ),
    )


def _adjudicate_matches(
    requirements: list[Requirement],
    evidence: list[ResumeEvidence],
    matches: list[RequirementEvidenceMatch],
    adjudicator: StructuredJudge,
) -> tuple[list[RequirementEvidenceMatch], Any]:
    """Let an LLM judge only non-exact matches and verify every citation."""
    ambiguous = [
        requirement
        for requirement, match in zip(requirements, matches)
        if match.decision_source != "exact"
    ]
    if not ambiguous:
        return matches, None
    candidate_ids = {
        candidate.evidence.evidence_id
        for match in matches
        if match.decision_source != "exact"
        for candidate in match.matches
    }
    candidate_evidence = [
        item for item in evidence if item.evidence_id in candidate_ids
    ]
    # Add a small cross-section sample so the holistic fit rubric still sees
    # broader experience without receiving the entire canonical resume.
    represented_sections = {item.section for item in candidate_evidence}
    for item in evidence:
        if len(candidate_evidence) >= 30:
            break
        if (
            item.evidence_id not in candidate_ids
            and (
                item.section not in represented_sections
                or item.field in {"title", "bullets", "content", "skills"}
            )
        ):
            candidate_evidence.append(item)
            represented_sections.add(item.section)
    try:
        batch = judge_ambiguous_evidence(
            ambiguous,
            candidate_evidence,
            adjudicator,
        )
    except Exception as error:
        logger.warning(
            "semantic_evidence_judge_failed preserving_retrieval=true error=%s",
            error,
        )
        return matches, None

    evidence_by_id = {item.evidence_id: item for item in evidence}
    judgments = {
        judgment.requirement_id: judgment
        for judgment in batch.judgments
    }
    requirements_by_id = {
        requirement.id: requirement
        for requirement in requirements
    }
    updated: list[RequirementEvidenceMatch] = []
    for match in matches:
        if match.decision_source == "exact":
            updated.append(match)
            continue
        judgment = judgments.get(match.requirement_id)
        if judgment is None:
            updated.append(match)
            continue
        cited_ids = list(dict.fromkeys(judgment.evidence_ids))
        valid_ids = [
            evidence_id
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id
        ]
        support = judgment.support
        if support in {"supported", "partial"} and not valid_ids:
            logger.warning(
                "semantic_evidence_judge_missing_valid_citation "
                "requirement_id=%s",
                match.requirement_id,
            )
            updated.append(match)
            continue
        if support == "unsupported":
            valid_ids = []
        requirement = requirements_by_id[match.requirement_id]
        safe_keywords = [
            keyword.strip()
            for keyword in judgment.safe_keywords
            if (
                keyword.strip()
                and _normalize(keyword) in _normalize(requirement.text)
                and support == "supported"
                and _safe_keyword_entailment(
                    keyword,
                    [
                        evidence_by_id[evidence_id]
                        for evidence_id in valid_ids
                    ],
                    judgment.relationship,
                )
            )
        ]

        existing = {
            candidate.evidence.evidence_id: candidate
            for candidate in match.matches
        }
        adjudicated_matches = []
        for evidence_id in valid_ids:
            candidate = existing.get(evidence_id) or EvidenceMatch(
                evidence=evidence_by_id[evidence_id]
            )
            adjudicated_matches.append(
                candidate.model_copy(
                    update={
                        "judge_support": support,
                        "judge_reason": judgment.reason,
                        "safe_keywords": safe_keywords,
                    }
                )
            )
        remaining = [
            candidate
            for candidate in match.matches
            if candidate.evidence.evidence_id not in valid_ids
        ]
        updated.append(
            match.model_copy(
                update={
                    "support": support,
                    "matches": [*adjudicated_matches, *remaining],
                    "decision_source": "llm",
                    "adjudication_reason": judgment.reason,
                }
            )
        )
    return updated, batch.fit_rubric


def _safe_keyword_entailment(
    keyword: str,
    evidence: list[ResumeEvidence],
    relationship: str,
) -> bool:
    target = _normalize(keyword)
    evidence_terms = {_normalize(item.text) for item in evidence}
    if target in evidence_terms:
        return True
    if relationship == "parent_generalization":
        return any(
            target in _DIRECTIONAL_GENERALIZATIONS.get(term, set())
            for term in evidence_terms
        )
    return relationship in {"equivalent", "transferable"}


def _build_assignments(
    requirements: list[Requirement],
    matches: list[RequirementEvidenceMatch],
) -> list[KeywordAssignment]:
    requirements_by_id = {
        requirement.id: requirement
        for requirement in requirements
    }
    assignments: list[KeywordAssignment] = []
    summary_count = 0

    importance_order = {"critical": 0, "important": 1, "supporting": 2}
    for match in sorted(
        matches,
        key=lambda item: importance_order[item.importance],
    ):
        if match.support != "supported":
            continue
        requirement = requirements_by_id[match.requirement_id]
        exact_matches = [
            candidate
            for candidate in match.matches
            if (
                candidate.exact_or_alias_match
                or candidate.judge_support == "supported"
            )
        ]
        keywords_by_section: dict[str, list[EvidenceMatch]] = {}
        for candidate in exact_matches:
            keyword = _keyword_for(requirement, candidate)
            if keyword is None:
                continue
            key = f"{candidate.evidence.section}\0{keyword.casefold()}"
            keywords_by_section.setdefault(key, []).append(candidate)

        for grouped in keywords_by_section.values():
            first = grouped[0]
            keyword = _keyword_for(requirement, first)
            if keyword is None:
                continue
            assignments.append(
                KeywordAssignment(
                    keyword=keyword,
                    section=first.evidence.section,
                    requirement_id=requirement.id,
                    evidence_ids=[
                        candidate.evidence.evidence_id
                        for candidate in grouped
                    ],
                    importance=requirement.importance,
                )
            )

        if (
            requirement.kind == "skill"
            and requirement.importance in {"critical", "important"}
            and summary_count < 3
        ):
            keyword_candidates = [
                _keyword_for(requirement, candidate)
                for candidate in exact_matches
            ]
            keyword = next(
                (
                    candidate
                    for candidate in keyword_candidates
                    if candidate is not None
                ),
                None,
            )
            if keyword is not None:
                assignments.append(
                    KeywordAssignment(
                        keyword=keyword,
                        section="summary",
                        requirement_id=requirement.id,
                        evidence_ids=[
                            candidate.evidence.evidence_id
                            for candidate in exact_matches
                        ],
                        importance=requirement.importance,
                    )
                )
                summary_count += 1

    return _deduplicate_assignments(assignments)


def _keyword_for(
    requirement: Requirement,
    match: EvidenceMatch,
) -> str | None:
    evidence = match.evidence
    if match.safe_keywords:
        return match.safe_keywords[0]
    if match.judge_support is not None:
        return None
    if requirement.kind == "skill" and len(requirement.text.split()) <= 5:
        return requirement.text
    if evidence.field in _KEYWORD_FIELDS:
        return evidence.text
    return None


def _is_exact_or_alias_match(requirement: str, evidence: str) -> bool:
    left = f" {_normalize(requirement)} "
    right = f" {_normalize(evidence)} "
    stripped_left = left.strip()
    stripped_right = right.strip()
    if min(len(stripped_left), len(stripped_right)) < 2:
        return False
    return left in right or right in left


def _normalize(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9+#.]+", " ", text.casefold()).strip()
    for alias, canonical in sorted(
        _ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        normalized = re.sub(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            canonical,
            normalized,
        )
    return " ".join(normalized.split())


def _encode(
    embedder: SentenceEmbedder,
    texts: list[str],
) -> np.ndarray:
    return np.asarray(
        embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float64,
    )


def _text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, list):
        for nested in value:
            yield from _text_values(nested)


def _deduplicate_assignments(
    assignments: list[KeywordAssignment],
) -> list[KeywordAssignment]:
    unique = {}
    for assignment in assignments:
        key = (assignment.section, assignment.keyword.casefold())
        existing = unique.get(key)
        if existing is None:
            unique[key] = assignment
        else:
            unique[key] = existing.model_copy(
                update={
                    "evidence_ids": list(
                        dict.fromkeys(
                            [
                                *existing.evidence_ids,
                                *assignment.evidence_ids,
                            ]
                        )
                    )
                }
            )
    return list(unique.values())
