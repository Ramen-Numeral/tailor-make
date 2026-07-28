"""Default constraints for generated resume sections."""

from app.resume_schema.resume_schema import Constraints


GLOBAL_STYLE = (
    "Use concise, ATS-friendly language. Prefer concrete nouns and active "
    "verbs. Do not invent facts, credentials, technologies, responsibilities, "
    "or metrics. Preserve the meaning of verified candidate facts."
)

GLOBAL_FORBIDDEN_KEYWORDS = ["team-player", "stakeholder"]


SUMMARY_CONSTRAINTS = Constraints(
    min_words=40,
    max_words=60,
    max_sentences=3,
    forbidden_phrases=[
        "hard-working",
        "team player",
        "results-driven",
        *GLOBAL_FORBIDDEN_KEYWORDS
    ],
    style=(
        f"{GLOBAL_STYLE} "
        "Write one professional summary paragraph tailored to the target role. "
        "Emphasize relevant technical experience and measurable impact."
    ),
)


WORK_EXPERIENCE_CONSTRAINTS = Constraints(
    min_items=1,
    max_items=4,
    min_bullets_per_item=2,
    max_bullets_per_item=2,
    max_words_per_bullet=28,
    require_metrics=False,
    forbidden_phrases=[
        "responsible for",
        "worked on",
        "helped with",
        "participated in",
        *GLOBAL_FORBIDDEN_KEYWORDS
    ],
    style=(
        f"{GLOBAL_STYLE} "
        "Write achievement-oriented bullets. Begin each bullet with a strong "
        "action verb, describe the technical work performed, and include the "
        "outcome when supported by verified facts."
    ),
)


SKILLS_CONSTRAINTS = Constraints(
    min_items=2,
    max_items=5,
    min_skills_per_category=2,
    max_skills_per_category=8,
    max_skill_categories=4,
    forbidden_phrases=[
        "expert",
        "guru",
        "ninja",
        "rockstar",
        *GLOBAL_FORBIDDEN_KEYWORDS
    ],
    style=(
        f"{GLOBAL_STYLE} "
        "Select only verified skills relevant to the target job and group them "
        "into clear, conventional categories."
    ),
)


EDUCATION_CONSTRAINTS = Constraints(
    min_items=1,
    max_items=2,
    max_courses=5,
    show_coursework=True,
    show_gpa=True,
    style=(
        f"{GLOBAL_STYLE} "
        "Present education entries clearly and factually. Include coursework, "
        "GPA, and honors only when available, verified, and relevant."
    ),
)


PROJECT_CONSTRAINTS = Constraints(
    min_items=0,
    max_items=3,
    min_bullets_per_item=1,
    max_bullets_per_item=3,
    max_words_per_bullet=26,
    max_technologies=6,
    require_metrics=False,
    forbidden_phrases=[
        "simple",
        "basic",
        "toy project",
        "just",
        *GLOBAL_FORBIDDEN_KEYWORDS
    ],
    style=(
        f"{GLOBAL_STYLE} "
        "Select the projects most relevant to the target job. Describe the "
        "problem, implementation, technical stack, and measurable outcome when "
        "supported by verified facts."
    ),
)


RESEARCH_CONSTRAINTS = Constraints(
    min_items=0,
    max_items=2,
    min_bullets_per_item=1,
    max_bullets_per_item=3,
    max_words_per_bullet=28,
    max_technologies=6,
    require_metrics=False,
    forbidden_phrases=[
        "groundbreaking",
        "revolutionary",
        "novel",
        *GLOBAL_FORBIDDEN_KEYWORDS
    ],
    style=(
        f"{GLOBAL_STYLE} "
        "Select research relevant to the target role. Describe the problem, "
        "methodology, implementation, and results without overstating novelty "
        "or impact."
    ),
)
