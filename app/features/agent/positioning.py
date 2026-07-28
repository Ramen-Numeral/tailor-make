"""Deterministic job-positioning strategy from evidence coverage."""

from app.features.agent.schema import PositioningBrief, RequirementPosition
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.job_listing_parser.listing_schema import JobListing


def build_positioning_brief(
    listing: JobListing,
    plan: CoveragePlan,
) -> PositioningBrief:
    positions = []
    section_plan: dict[str, list[str]] = {}
    primary = []
    transferable = []
    gaps = []
    for match in plan.requirement_matches:
        sections = list(
            dict.fromkeys(item.evidence.section for item in match.matches[:3])
        )
        strength = (
            "strong" if match.support == "supported" and match.matches
            else "moderate" if match.support == "partial"
            else "weak" if match.matches
            else "none"
        )
        positions.append(
            RequirementPosition(
                requirement=match.requirement_text,
                support=match.support,
                evidence_strength=strength,
                destination_sections=sections,
            )
        )
        if match.support == "supported":
            primary.append(match.requirement_text)
            for section in sections:
                section_plan.setdefault(section, []).append(match.requirement_text)
            if match.decision_source == "llm":
                transferable.append(match.requirement_text)
        elif match.support == "unsupported":
            gaps.append(match.requirement_text)
    return PositioningBrief(
        target_identity=listing.title or "Target role",
        primary_evidence=primary[:6],
        transferable_narrative=transferable[:4],
        gaps=gaps[:6],
        writing_priorities=[
            "Lead with the strongest supported evidence.",
            "Use exact job language only when source evidence makes it truthful.",
            "Emphasize scope, implementation, and outcomes before generic traits.",
            "Do not imply unsupported tools, domains, credentials, or seniority.",
        ],
        requirement_positions=positions,
        section_plan={
            section: list(dict.fromkeys(requirements))[:5]
            for section, requirements in section_plan.items()
        },
    )


def brief_instruction(brief: PositioningBrief) -> str:
    section_plan = "; ".join(
        f"{section}: {', '.join(requirements)}"
        for section, requirements in brief.section_plan.items()
    )
    return (
        f"Positioning target: {brief.target_identity}. "
        f"Lead evidence: {', '.join(brief.primary_evidence) or 'none'}. "
        f"Transferable framing: {', '.join(brief.transferable_narrative) or 'none'}. "
        f"Unsupported gaps to avoid claiming: {', '.join(brief.gaps) or 'none'}. "
        f"Section emphasis: {section_plan or 'use strongest evidence first'}."
    )
