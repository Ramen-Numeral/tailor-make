"""Transport schemas for factual resume extraction."""

from pydantic import BaseModel, Field


class ExtractedCandidate(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    target_title: str | None = None
    github: str | None = None
    linkedin: str | None = None
    website: str | None = None


class ExtractedSkillCategory(BaseModel):
    name: str
    skills: list[str] = Field(default_factory=list)


class ExtractedWorkExperience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str | None = None
    location: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ExtractedEducation(BaseModel):
    degree: str
    institution: str
    graduation_date: str | None = None
    location: str | None = None
    gpa: str | None = None
    coursework: list[str] = Field(default_factory=list)
    honors: list[str] = Field(default_factory=list)


class ExtractedProject(BaseModel):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    url: str | None = None


class ExtractedResearch(ExtractedProject):
    pass


class ExtractedResume(BaseModel):
    candidate: ExtractedCandidate
    summary: str | None = None
    skills: list[ExtractedSkillCategory] = Field(default_factory=list)
    work_experience: list[ExtractedWorkExperience] = Field(default_factory=list)
    education: list[ExtractedEducation] = Field(default_factory=list)
    projects: list[ExtractedProject] = Field(default_factory=list)
    research: list[ExtractedResearch] = Field(default_factory=list)


class PDFTextExtraction(BaseModel):
    text: str
    page_count: int
    page_texts: list[str]
    warnings: list[str] = Field(default_factory=list)


class ResumeParseResult(BaseModel):
    """Canonical resume plus source provenance useful to the UI."""

    resume: "Resume"
    source_text: str
    page_count: int
    filename: str | None = None
    warnings: list[str] = Field(default_factory=list)


from app.resume_schema.resume_schema import Resume  # noqa: E402

ResumeParseResult.model_rebuild()
