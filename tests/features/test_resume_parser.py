from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from app.features.resume_parser import parser as parser_module
from app.features.resume_parser.pdf import ResumePDFError, extract_pdf_text
from app.features.resume_parser.schema import (
    ExtractedCandidate,
    ExtractedEducation,
    ExtractedResume,
    ExtractedSkillCategory,
    ExtractedWorkExperience,
)
from config.resume.section_constraints import WORK_EXPERIENCE_CONSTRAINTS


class ParserClient:
    def __init__(self, response):
        self.response = response
        self.call = None

    def invoke_structured(self, **kwargs):
        self.call = kwargs
        return self.response


def _blank_pdf() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def _extracted_resume() -> ExtractedResume:
    return ExtractedResume(
        candidate=ExtractedCandidate(
            name="Ada Lovelace",
            email="ada@example.com",
        ),
        summary="Engineer who builds reliable analytical systems.",
        skills=[
            ExtractedSkillCategory(name="Languages", skills=["Python", "SQL"])
        ],
        work_experience=[
            ExtractedWorkExperience(
                title="Engineer",
                company="Analytical Engines",
                start_date="2023",
                bullets=["Built a data processing engine."],
            )
        ],
        education=[
            ExtractedEducation(
                degree="B.S. Mathematics",
                institution="Example University",
            )
        ],
    )


def test_extract_pdf_rejects_non_pdf() -> None:
    with pytest.raises(ResumePDFError, match="not a valid PDF"):
        extract_pdf_text(b"not a PDF")


def test_extract_pdf_explains_scanned_or_blank_files() -> None:
    with pytest.raises(ResumePDFError, match="OCR"):
        extract_pdf_text(_blank_pdf())


def test_parse_resume_text_builds_canonical_resume(monkeypatch) -> None:
    client = ParserClient(_extracted_resume())
    monkeypatch.setattr(
        parser_module,
        "get_llm_clients",
        lambda: SimpleNamespace(resume_parser=client),
    )

    resume = parser_module.parse_resume_text("Ada Lovelace\nEngineer")

    assert resume.candidate.name == "Ada Lovelace"
    assert resume.skills.items[0].skills == ["Python", "SQL"]
    assert resume.work_experience.items[0].company == "Analytical Engines"
    assert resume.work_experience.constraints == WORK_EXPERIENCE_CONSTRAINTS
    assert client.call["schema"] is ExtractedResume
    assert client.call["temperature"] == 0


def test_parse_resume_pdf_preserves_provenance(monkeypatch) -> None:
    extraction = SimpleNamespace(
        text="Ada Lovelace\nEngineer",
        page_count=2,
        warnings=["Page 2 contained no extractable text."],
    )
    monkeypatch.setattr(
        parser_module, "extract_pdf_text", lambda *args, **kwargs: extraction
    )
    monkeypatch.setattr(
        parser_module,
        "parse_resume_text",
        lambda text: parser_module.build_resume(_extracted_resume()),
    )

    result = parser_module.parse_resume_pdf(
        b"%PDF-fake",
        filename="ada.pdf",
    )

    assert result.resume.candidate.name == "Ada Lovelace"
    assert result.source_text == extraction.text
    assert result.page_count == 2
    assert result.filename == "ada.pdf"
    assert result.warnings == extraction.warnings
