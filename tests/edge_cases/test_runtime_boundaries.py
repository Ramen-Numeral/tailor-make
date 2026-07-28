"""Edge cases and installed-dependency compatibility contracts."""

from importlib import import_module
from io import StringIO

import pytest
from pydantic import ValidationError

from app.features.job_listing_parser.listing_schema import JobListing
from app.features.renderer.renderer import render_html, render_pdf_bytes
from app.features.resume_parser.pdf import ResumePDFError, extract_pdf_text
from app.resume_schema.resume_schema import Candidate, Resume
from config.resume.candidate_profile import build_resume


@pytest.mark.parametrize(
    ("distribution_import", "public_symbol"),
    [
        ("fastapi", "FastAPI"),
        ("pydantic", "BaseModel"),
        ("jinja2", "Environment"),
        ("pypdf", "PdfReader"),
        ("bm25s", "BM25"),
        ("sklearn", "__version__"),
        ("numpy", "ndarray"),
        ("polars", "DataFrame"),
        ("groq", "Groq"),
    ],
)
def test_core_runtime_dependencies_import_with_expected_api(
    distribution_import: str,
    public_symbol: str,
) -> None:
    module = import_module(distribution_import)

    assert hasattr(module, public_symbol)


def test_weasyprint_native_dependency_works_through_supported_boundary() -> None:
    pdf = render_pdf_bytes(render_html(build_resume()))

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1_000


@pytest.mark.parametrize(
    "payload",
    [
        None,
        17,
        "requirements",
        {"requirements": "not-a-list"},
    ],
)
def test_job_listing_rejects_invalid_root_shapes(payload) -> None:
    with pytest.raises(ValidationError):
        JobListing.model_validate(payload)


def test_minimal_resume_with_absent_optional_sections_renders() -> None:
    resume = Resume(candidate=Candidate(name="Minimal Candidate"))

    html = render_html(resume)

    assert "Minimal Candidate" in html
    assert "Work Experience" not in html


def test_pdf_reader_rejects_text_stream_and_truncated_pdf() -> None:
    with pytest.raises(ResumePDFError, match="binary mode"):
        extract_pdf_text(StringIO("%PDF-1.7 text"), min_text_characters=1)

    with pytest.raises(ResumePDFError, match="could not be read"):
        extract_pdf_text(b"%PDF-1.7\ntruncated", min_text_characters=1)


def test_pdf_limits_must_be_positive_before_reading_input() -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        extract_pdf_text(b"not even a pdf", max_pages=0)


def test_resume_schema_rejects_missing_candidate_name() -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate({"candidate": {}})
