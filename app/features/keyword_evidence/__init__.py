"""Supported job-keyword planning with source-resume provenance."""

from app.features.keyword_evidence.planner import (
    apply_coverage_plan,
    build_coverage_plan,
    extract_resume_evidence,
)
from app.features.keyword_evidence.schema import (
    CoveragePlan,
    EvidenceMatch,
    KeywordAssignment,
    RequirementEvidenceMatch,
    ResumeEvidence,
)

__all__ = [
    "CoveragePlan",
    "EvidenceMatch",
    "KeywordAssignment",
    "RequirementEvidenceMatch",
    "ResumeEvidence",
    "apply_coverage_plan",
    "build_coverage_plan",
    "extract_resume_evidence",
]
