"""Convert PDF resume uploads into the canonical Resume model."""

from pathlib import Path
from typing import BinaryIO

from app.bootstrap import get_llm_clients
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.features.resume_parser.pdf import extract_pdf_text
from app.features.resume_parser.prompts import get_resume_extraction_prompt
from app.features.resume_parser.schema import ExtractedResume, ResumeParseResult
from app.resume_schema.resume_schema import (
    Candidate,
    EducationItem,
    EducationSection,
    ProfessionalSummaryItem,
    ProjectItem,
    ProjectSection,
    ResearchItem,
    ResearchSection,
    Resume,
    SkillCategoryItem,
    SkillsSection,
    SummarySection,
    WorkExperienceItem,
    WorkExperienceSection,
)
from config.resume.section_constraints import (
    EDUCATION_CONSTRAINTS,
    PROJECT_CONSTRAINTS,
    RESEARCH_CONSTRAINTS,
    SKILLS_CONSTRAINTS,
    SUMMARY_CONSTRAINTS,
    WORK_EXPERIENCE_CONSTRAINTS,
)


class ResumeParseError(ValueError):
    """Raised when extracted resume data cannot form a canonical resume."""


def _clean_required(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise ResumeParseError(f"Resume is missing required field: {field}")
    return value


def build_resume(extracted: ExtractedResume) -> Resume:
    """Apply application section constraints to factual extracted content."""
    candidate_data = extracted.candidate.model_dump()
    candidate_data["name"] = _clean_required(
        extracted.candidate.name, "candidate.name"
    )

    summary_items = (
        [ProfessionalSummaryItem(content=extracted.summary.strip())]
        if extracted.summary and extracted.summary.strip()
        else []
    )
    projects = [
        ProjectItem(**item.model_dump())
        for item in extracted.projects
    ]
    research = [
        ResearchItem(**item.model_dump())
        for item in extracted.research
    ]

    return Resume(
        candidate=Candidate(**candidate_data),
        summary=SummarySection(
            items=summary_items,
            constraints=SUMMARY_CONSTRAINTS.model_copy(deep=True),
        ),
        skills=SkillsSection(
            items=[
                SkillCategoryItem(**item.model_dump())
                for item in extracted.skills
            ],
            constraints=SKILLS_CONSTRAINTS.model_copy(deep=True),
        ),
        work_experience=WorkExperienceSection(
            items=[
                WorkExperienceItem(**item.model_dump())
                for item in extracted.work_experience
            ],
            constraints=WORK_EXPERIENCE_CONSTRAINTS.model_copy(deep=True),
        ),
        education=EducationSection(
            items=[
                EducationItem(**item.model_dump())
                for item in extracted.education
            ],
            constraints=EDUCATION_CONSTRAINTS.model_copy(deep=True),
        ),
        projects=ProjectSection(
            items=projects,
            constraints=PROJECT_CONSTRAINTS.model_copy(deep=True),
        ) if projects else None,
        research=ResearchSection(
            items=research,
            constraints=RESEARCH_CONSTRAINTS.model_copy(deep=True),
        ) if research else None,
    )


def parse_resume_text(resume_text: str) -> Resume:
    """Parse already-extracted resume text into a canonical Resume."""
    if not resume_text.strip():
        raise ResumeParseError("resume_text cannot be empty")
    parser = get_llm_clients().resume_parser
    cache_key = content_key(
        "resume_parse",
        object_identity(parser),
        resume_text,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return Resume.model_validate(cached)
    extracted = parser.invoke_structured(
        prompt=get_resume_extraction_prompt(resume_text),
        schema=ExtractedResume,
        temperature=0,
        trace_context="resume_parser",
    )
    resume = build_resume(extracted)
    set_cached(cache_key, resume.model_dump(mode="json"))
    return resume


def parse_resume_pdf(
    source: bytes | bytearray | BinaryIO | str | Path,
    *,
    filename: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    max_pages: int = 10,
) -> ResumeParseResult:
    """Parse uploaded PDF bytes/file into a Resume and provenance record."""
    extraction = extract_pdf_text(
        source,
        max_bytes=max_bytes,
        max_pages=max_pages,
    )
    return ResumeParseResult(
        resume=parse_resume_text(extraction.text),
        source_text=extraction.text,
        page_count=extraction.page_count,
        filename=filename,
        warnings=extraction.warnings,
    )
