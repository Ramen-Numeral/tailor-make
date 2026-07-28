from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

RESUME_SECTION_FIELDS = (
    "summary",
    "skills",
    "work_experience",
    "education",
    "projects",
    "research",
)
WRITABLE_SECTION_FIELDS = tuple(
    field for field in RESUME_SECTION_FIELDS if field != "education"
)


class Constraints(BaseModel):
    max_items: int | None = None
    min_items: int | None = None

    max_words: int | None = None
    min_words: int | None = None
    max_sentences: int | None = None

    max_bullets_per_item: int | None = None
    min_bullets_per_item: int | None = None
    max_words_per_bullet: int | None = None

    max_skill_categories: int | None = None
    max_skills_per_category: int | None = None
    min_skills_per_category: int | None = None
    max_courses: int | None = None
    max_technologies: int | None = None

    show_gpa: bool | None = None
    show_coursework: bool | None = None

    require_metrics: bool | None = None
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)

    style: str | None = None

    def active(self) -> dict:
        """Return only explicitly meaningful constraints."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        )


class Candidate(BaseModel):
    """Candidate identity and contact fields used by the resume features."""

    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    target_title: str | None = None
    github: str | None = None
    linkedin: str | None = None
    website: str | None = None
class SectionItem(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        frozen=True,
    )


class ProfessionalSummaryItem(SectionItem):
    content: str | None = None

    class WritableForm(BaseModel):
        content: str | None = None


class EducationItem(SectionItem):
    degree: str
    institution: str
    graduation_date: str | None = None
    location: str | None = None
    gpa: str | None = None
    coursework: list[str] = Field(default_factory=list)
    honors: list[str] = Field(default_factory=list)

    class WritableForm(BaseModel):
        coursework_indices: list[int] = Field(default_factory=list)


class SkillCategoryItem(SectionItem):
    name: str
    skills: list[str] = Field(default_factory=list)

    class WritableForm(BaseModel):
        skills: list[str] = Field(default_factory=list)


class WorkExperienceItem(SectionItem):
    title: str
    company: str
    start_date: str
    end_date: str | None = None
    location: str | None = None
    bullets: list[str] = Field(default_factory=list)

    class WritableForm(BaseModel):
        bullets: list[str] = Field(default_factory=list)

class ProjectItem(SectionItem):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    url: str | None = None

    class WritableForm(BaseModel):
        description: str | None = None
        bullets: list[str] = Field(default_factory=list)

class ResearchItem(SectionItem):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    url: str | None = None

    class WritableForm(BaseModel):
        description: str | None = None
        bullets: list[str] = Field(default_factory=list)

class Section(BaseModel):
    heading: str
    items: list[SectionItem] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)

    def get_item(self, item_id: UUID) -> SectionItem:
        for item in self.items:
            if item.id == item_id:
                return item

        raise KeyError(
            f"Item {item_id} not found in section {self.heading!r}"
        )


class SummarySection(Section):
    heading: str = "Professional Summary"
    items: list[ProfessionalSummaryItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=1
        )
    )


class SkillsSection(Section):
    heading: str = "Skills"
    items: list[SkillCategoryItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=4
        )
    )


class WorkExperienceSection(Section):
    heading: str = "Work Experience"
    items: list[WorkExperienceItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=5
        )
    )


class ProjectSection(Section):
    heading: str = "Projects"
    items: list[ProjectItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=3
        )
    )


class ResearchSection(Section):
    heading: str = "Research"
    items: list[ResearchItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=2
        )
    )


class EducationSection(Section):
    heading: str = "Education"
    items: list[EducationItem] = Field(default_factory=list)
    constraints: Constraints = Field(
        default_factory=lambda: Constraints(
            max_items=1
        )
    )


class Resume(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate: Candidate

    summary: SummarySection = Field(default_factory=SummarySection)
    skills: SkillsSection = Field(default_factory=SkillsSection)
    work_experience: WorkExperienceSection = Field(
        default_factory=WorkExperienceSection
    )
    education: EducationSection = Field(
        default_factory=EducationSection
    )

    projects: ProjectSection | None = None
    research: ResearchSection | None = None

class MutableResume(Resume):
    model_config = ConfigDict(frozen=False)
