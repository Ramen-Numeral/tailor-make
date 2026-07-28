"""Integration tests spanning schema, renderer, PDF, and extraction boundaries."""

from io import BytesIO

from pypdf import PdfReader

from app.features.keyword_evidence.planner import build_coverage_plan
from app.features.keyword_evidence.scoring import score_coverage_plan
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.features.renderer.renderer import render_html, render_pdf_bytes
from app.features.resume_diff.differ import build_resume_diffs
from app.features.resume_parser.pdf import extract_pdf_text
from app.resume_schema.resume_schema import (
    Candidate,
    Constraints,
    EducationItem,
    EducationSection,
    Resume,
)
from config.resume.candidate_profile import build_resume


def test_resume_json_render_pdf_extract_round_trip() -> None:
    """Exercise real Pydantic, Jinja, WeasyPrint, and PyPDF together."""
    source = build_resume()
    serialized = source.model_dump_json()
    restored = Resume.model_validate_json(serialized)

    html = render_html(restored)
    pdf = render_pdf_bytes(html)
    reader = PdfReader(BytesIO(pdf))
    extraction = extract_pdf_text(pdf, min_text_characters=20)

    assert pdf.startswith(b"%PDF-")
    assert len(reader.pages) == extraction.page_count >= 1
    assert source.candidate.name in extraction.text
    assert "work experience" in extraction.text.casefold()
    assert "Acme Technologies" in extraction.text


def test_real_keyword_plan_score_and_diff_share_source_ids() -> None:
    source = build_resume()
    listing = JobListing(
        title="Backend Engineer",
        requirements=[
            Requirement(
                text="Python",
                kind="skill",
                importance="critical",
            ),
            Requirement(
                text="PostgreSQL",
                kind="skill",
                importance="important",
            ),
            Requirement(
                text="Angular",
                kind="skill",
                importance="supporting",
            ),
        ],
    )

    plan = build_coverage_plan(listing, source)
    score = score_coverage_plan(plan, stage="initial")
    first_job = source.work_experience.items[0]
    final = source.model_copy(
        update={
            "work_experience": source.work_experience.model_copy(
                update={
                    "items": [
                        first_job.model_copy(
                            update={"bullets": ["Improved Python service throughput."]}
                        ),
                        *source.work_experience.items[1:],
                    ]
                }
            )
        },
        deep=True,
    )
    diffs = build_resume_diffs(source, source, final)

    assert score.total_requirements == 3
    assert score.supported >= 2
    assert score.unsupported >= 1
    assert 1 <= score.score < 100
    assert any(diff.item_id == first_job.id for diff in diffs)
    assert final.work_experience.items[0].id == first_job.id


def test_renderer_escapes_candidate_supplied_markup_in_every_output() -> None:
    source = build_resume()
    unsafe = source.model_copy(
        update={
            "candidate": source.candidate.model_copy(
                update={"name": '<script>alert("x")</script>'}
            )
        },
        deep=True,
    )

    html = render_html(unsafe)
    pdf = render_pdf_bytes(html)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert pdf.startswith(b"%PDF-")


def test_populated_education_details_show_unless_explicitly_disabled() -> None:
    education = EducationSection(
        constraints=Constraints(max_courses=2),
        items=[
            EducationItem(
                degree="MS Computer Science",
                institution="Example University",
                gpa="3.8",
                coursework=["Machine Learning", "Optimization", "Databases"],
            )
        ],
    )
    resume = Resume(
        candidate=Candidate(name="Candidate"),
        education=education,
    )

    visible = render_html(resume)
    hidden = render_html(
        resume.model_copy(
            update={
                "education": education.model_copy(
                    update={
                        "constraints": Constraints(
                            show_gpa=False,
                            show_coursework=False,
                        )
                    }
                )
            }
        )
    )

    assert "GPA: 3.8" in visible
    assert "Machine Learning, Optimization" in visible
    assert "Databases" not in visible
    assert "GPA: 3.8" not in hidden
    assert "Machine Learning" not in hidden
