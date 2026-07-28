"""Bullet-level and final recruiter-quality evaluation."""

import json
import re

from app.features.agent.schema import (
    BulletQualityResult,
    PositioningBrief,
    RecruiterAxis,
    RecruiterEvaluation,
)
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.resume_schema.resume_schema import RESUME_SECTION_FIELDS, Resume

_ACTION = re.compile(
    r"^(?:built|created|designed|developed|delivered|drove|improved|increased|"
    r"launched|led|managed|optimized|organized|reduced|scaled|shipped|used)\b",
    re.I,
)
_OUTCOME = re.compile(
    r"(?:\d|%|\$|\bresult(?:ed)?\b|\bimprov|\breduc|\bincreas|\benabl|\bcut\b)",
    re.I,
)
_IMPLEMENTATION = re.compile(r"\b(?:using|with|through|by|via|in)\b", re.I)


def evaluate_bullets(resume: Resume) -> list[BulletQualityResult]:
    results = []
    for section_name in ("work_experience", "projects", "research"):
        section = getattr(resume, section_name, None)
        if section is None:
            continue
        for item_index, item in enumerate(section.items):
            for bullet_index, bullet in enumerate(getattr(item, "bullets", [])):
                dimensions = {
                    "action clarity": bool(_ACTION.search(bullet)),
                    "specific implementation": bool(_IMPLEMENTATION.search(bullet)),
                    "scope or context": len(bullet.split()) >= 9,
                    "supported outcome": bool(_OUTCOME.search(bullet)),
                    "conciseness": len(bullet.split()) <= 30,
                    "credibility": not bool(
                        re.search(r"\b(?:world-class|revolutionary|best-in-class)\b", bullet, re.I)
                    ),
                }
                passed = [name for name, value in dimensions.items() if value]
                results.append(
                    BulletQualityResult(
                        section=section_name,
                        item_index=item_index,
                        bullet_index=bullet_index,
                        text=bullet,
                        score=round(100 * len(passed) / len(dimensions)),
                        passed_dimensions=passed,
                        improvement_dimensions=[
                            name for name, value in dimensions.items() if not value
                        ],
                    )
                )
    return results


def evaluate_recruiter_quality(
    resume: Resume,
    brief: PositioningBrief,
    critic=None,
) -> RecruiterEvaluation:
    if critic is None:
        return _fallback_recruiter_evaluation(resume, brief)
    key = content_key(
        "recruiter_quality",
        object_identity(critic),
        resume,
        brief,
    )
    cached = get_cached(key)
    if cached is not None:
        return RecruiterEvaluation.model_validate(cached)
    payload = {
        "target": brief.target_identity,
        "supported_positioning": brief.model_dump(mode="json"),
        "resume": resume.model_dump(
            mode="json",
            exclude={
                section: {"items": {"__all__": {"id"}}}
                for section in RESUME_SECTION_FIELDS
            },
        ),
    }
    prompt = """Evaluate this final resume as a recruiter scanning it for the
target role. Score every required axis. Judge whether the target is immediately
clear, the strongest evidence is visible, the story is coherent, seniority is
credible, claims are specific, and the document is easy to scan. Do not reward
unsupported claims or missing requirements. Give concise, actionable reasons.
Return only schema-valid JSON.

DATA:
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    try:
        result = critic.invoke_structured(
            prompt=prompt,
            schema=RecruiterEvaluation,
            temperature=0,
            max_tokens=900,
            trace_context="final_recruiter_evaluation",
        )
    except Exception:
        return _fallback_recruiter_evaluation(resume, brief)
    set_cached(key, result.model_dump(mode="json"))
    return result


def _fallback_recruiter_evaluation(
    resume: Resume,
    brief: PositioningBrief,
) -> RecruiterEvaluation:
    bullets = evaluate_bullets(resume)
    average = (
        sum(item.score for item in bullets) / len(bullets)
        if bullets else 0
    )
    clarity = 4 if resume.candidate.target_title else 2
    evidence = 4 if average >= 70 else 3 if average >= 50 else 2
    axes = [
        RecruiterAxis(axis="target_clarity", score=clarity, reason="Based on the resume headline."),
        RecruiterAxis(axis="evidence_visibility", score=evidence, reason="Based on bullet action, context, and outcome coverage."),
        RecruiterAxis(axis="coherence", score=3, reason="Sections preserve one canonical candidate history."),
        RecruiterAxis(axis="seniority_credibility", score=3, reason="No unsupported seniority inference was added."),
        RecruiterAxis(axis="specificity", score=evidence, reason="Based on deterministic bullet specificity checks."),
        RecruiterAxis(axis="scanability", score=4, reason="The rendered resume uses a concise single-column structure."),
    ]
    gaps = list(brief.gaps[:3])
    return RecruiterEvaluation(
        axes=axes,
        strengths=brief.primary_evidence[:3],
        gaps=gaps,
        recommendations=[
            "Add source-backed scope or outcomes to weak bullets."
        ] if average < 70 else [],
        summary="Deterministic fallback evaluation; recruiter judge unavailable.",
        ready=average >= 60 and clarity >= 3,
    )
