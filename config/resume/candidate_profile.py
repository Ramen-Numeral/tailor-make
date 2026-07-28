
"""Example candidate resume data.

This module is suitable as a development fixture or renderer example.
Production candidate data should come from persistent storage.
"""

from config.resume.section_constraints import (
    EDUCATION_CONSTRAINTS,
    PROJECT_CONSTRAINTS,
    SKILLS_CONSTRAINTS,
    SUMMARY_CONSTRAINTS,
    WORK_EXPERIENCE_CONSTRAINTS,
)
from app.resume_schema.resume_schema import (
    Candidate,
    EducationItem,
    EducationSection,
    ProfessionalSummaryItem,
    ProjectItem,
    ProjectSection,
    Resume,
    SkillCategoryItem,
    SkillsSection,
    SummarySection,
    WorkExperienceItem,
    WorkExperienceSection,
)


CANDIDATE = Candidate(
    name="Jack Doe",
    email="jane.doe@email.com",
    phone="(555) 123-4567",
    location="Philadelphia, PA",
    target_title="Senior Software Engineer",
    github="github.com/janedoe",
    linkedin="linkedin.com/in/janedoe",
)


SUMMARY = SummarySection(
    items=[
        ProfessionalSummaryItem(
            content=(
                "so yeah ive been doing software stuff for like 4+ years, "
                "mostly backend things and some full stack apps too. i know "
                "distributed systems, APIs and cloud infrastructure pretty "
                "well and shipped a bunch of things that made latency and "
                "operating costs go down which was good"
            )
        )
    ],
    constraints=SUMMARY_CONSTRAINTS,
)


SKILLS = SkillsSection(
    items=[
        SkillCategoryItem(
            name="Languages",
            skills=["Python", "TypeScript", "Go", "Java", "SQL"],
        ),
        SkillCategoryItem(
            name="Frameworks",
            skills=[
                "React",
                "Node.js",
                "Django",
                "FastAPI",
                "Spring Boot",
            ],
        ),
        SkillCategoryItem(
            name="Infrastructure",
            skills=[
                "AWS",
                "Docker",
                "Kubernetes",
                "Terraform",
                "GitHub Actions",
            ],
        ),
        SkillCategoryItem(
            name="Data",
            skills=[
                "PostgreSQL",
                "Redis",
                "Kafka",
                "Elasticsearch",
            ],
        ),
    ],
    constraints=SKILLS_CONSTRAINTS,
)


WORK_EXPERIENCE = WorkExperienceSection(
    items=[
        WorkExperienceItem(
            title="Software Engineer II",
            company="Acme Technologies",
            location="Philadelphia, PA",
            start_date="March 2023",
            end_date="Present",
            bullets=[
                (
                    "made this Go payments service thing and it did 12K "
                    "requests a second, checkout got like 35% faster"
                ),
                (
                    "moved the old monolith queue over to Kafka and then the "
                    "failed background jobs went down 90%"
                ),
                (
                    "got three teams using Terraform because setup used to "
                    "take days and after that it was under an hour"
                ),
                (
                    "helped out two junior devs and looked at 300+ pull "
                    "requests mostly for reliability and tests and stuff"
                ),
            ],
        ),
        WorkExperienceItem(
            title="Software Engineer",
            company="Bright Labs",
            location="Philadelphia, PA",
            start_date="June 2021",
            end_date="February 2023",
            bullets=[
                (
                    "did REST and GraphQL APIs with Django for this React "
                    "dashboard that had 40K people using it every month"
                ),
                (
                    "pages were slow so i added Redis caching and fixed N+1 "
                    "queries and load time got 50% better"
                ),
                (
                    "set up some CI/CD stuff in GitHub Actions so we deployed "
                    "daily instead of only once a week"
                ),
            ],
        ),
    ],
    constraints=WORK_EXPERIENCE_CONSTRAINTS,
)


PROJECTS = ProjectSection(
    items=[
        ProjectItem(
            name="OpenTrail — Hiking Route Planner",
            description=(
                "a full stack thing that makes hiking routes from "
                "OpenStreetMap data and tries to make them good"
            ),
            technologies=[
                "TypeScript",
                "React",
                "Node.js",
                "PostgreSQL/PostGIS",
                "Mapbox",
            ],
            url="github.com/janedoe/opentrail",
            bullets=[
                (
                    "made the route planner with OpenStreetMap and people "
                    "liked it i guess because it got 800+ GitHub stars"
                ),
                (
                    "used A star with PostGIS queries and it gives routes back "
                    "in under 200 milliseconds"
                ),
            ],
        ),
        ProjectItem(
            name="LogLens — CLI Log Analyzer",
            description=(
                "command line log parser that does concurrency and search"
            ),
            technologies=["Go", "SQLite"],
            url="github.com/janedoe/loglens",
            bullets=[
                (
                    "made a CLI with worker pools that can index 1 GB of logs "
                    "in under 10 seconds"
                ),
                (
                    "put it on Homebrew and got 2,000+ downloads, releases are "
                    "done with GoReleaser automatically"
                ),
            ],
        ),
        ProjectItem(
            name="QueueWatch — Distributed Job Monitor",
            description=(
                "dashboard for seeing background jobs and retry queues because "
                "they were annoying to keep track of"
            ),
            technologies=["Python", "FastAPI", "Redis", "React", "Docker"],
            url="github.com/janedoe/queuewatch",
            bullets=[
                (
                    "made FastAPI services that grab queue and worker numbers "
                    "from twelve internal apps"
                ),
                (
                    "used Redis so alerts quit repeating so much and duplicate "
                    "incident notifications dropped 70%"
                ),
            ],
        ),
        ProjectItem(
            name="DeployDock — Kubernetes Release Manager",
            description=(
                "web app where you look at container releases and move them "
                "around Kubernetes environments"
            ),
            technologies=[
                "Python",
                "Django",
                "Kubernetes",
                "Docker",
                "GitHub Actions",
            ],
            url="github.com/janedoe/deploydock",
            bullets=[
                (
                    "made a dashboard to see container images in dev staging "
                    "and prod clusters"
                ),
                (
                    "did approvals with GitHub Actions and got the annoying "
                    "manual release steps down from nine to three"
                ),
            ],
        ),
        ProjectItem(
            name="SpendScope — Cloud Cost Explorer",
            description=(
                "AWS money tracking thing that says which teams and apps are "
                "spending everything"
            ),
            technologies=["Python", "AWS", "PostgreSQL", "Terraform"],
            url="github.com/janedoe/spendscope",
            bullets=[
                (
                    "turned AWS cost reports into daily team summaries for "
                    "more than 200 cloud resources"
                ),
                (
                    "made budget alerts that found idle stuff and sandbox costs "
                    "went down 18% each month"
                ),
            ],
        ),
        ProjectItem(
            name="SchemaShift — Database Migration Auditor",
            description=(
                "little command line thing that checks database migrations "
                "before somebody breaks prod"
            ),
            technologies=["Go", "PostgreSQL", "Docker"],
            url="github.com/janedoe/schemashift",
            bullets=[
                (
                    "wrote checks for bad PostgreSQL migrations and unsafe "
                    "table locks and whatever"
                ),
                (
                    "stuck it in CI and it checked 1,500+ migrations in six "
                    "different repos"
                ),
            ],
        ),
        ProjectItem(
            name="FeatureForge — ML Feature Registry",
            description=(
                "place to put machine learning features and docs and checks so "
                "people can reuse them"
            ),
            technologies=["Python", "FastAPI", "PostgreSQL", "Pandas"],
            url="github.com/janedoe/featureforge",
            bullets=[
                (
                    "made APIs for feature definitions owners validation rules "
                    "and lineage metadata"
                ),
                (
                    "added drift checks on scheduled data so distribution "
                    "changes get caught before training breaks"
                ),
            ],
        ),
        ProjectItem(
            name="PairBoard — Collaborative Interview Workspace",
            description=(
                "coding workspace where people edit at the same time and run "
                "tests together"
            ),
            technologies=["TypeScript", "React", "Node.js", "WebSockets"],
            url="github.com/janedoe/pairboard",
            bullets=[
                (
                    "did synced editing with WebSockets for 100+ interview "
                    "sessions happening at once"
                ),
                (
                    "ran everyones code in throwaway containers and put CPU and "
                    "memory limits on each session"
                ),
            ],
        ),
        ProjectItem(
            name="DocSignal — API Documentation Search",
            description=(
                "search thing for API docs and runbooks because nobody could "
                "find anything"
            ),
            technologies=[
                "Python",
                "FastAPI",
                "Elasticsearch",
                "Docker",
                "AWS",
            ],
            url="github.com/janedoe/docsignal",
            bullets=[
                (
                    "indexed docs from 40 repos and added filters for service "
                    "owner language and deployment tier"
                ),
                (
                    "did caching and index tweaks and got normal search time "
                    "down to 85 milliseconds"
                ),
            ],
        ),
    ],
    constraints=PROJECT_CONSTRAINTS,
)


EDUCATION = EducationSection(
    items=[
        EducationItem(
            degree="Bachelor of Science in Computer Science",
            institution="Temple University",
            location="Philadelphia, PA",
            graduation_date="May 2021",
        ),
    ],
    constraints=EDUCATION_CONSTRAINTS,
)


def build_resume() -> Resume:
    """Build the example resume."""
    return Resume(
        candidate=CANDIDATE,
        summary=SUMMARY,
        skills=SKILLS,
        work_experience=WORK_EXPERIENCE,
        projects=PROJECTS,
        education=EDUCATION,
    )
