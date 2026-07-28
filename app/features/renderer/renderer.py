"""Render Resume models to ATS-friendly HTML and PDF."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    select_autoescape,
)

from app.resume_schema.resume_schema import Resume
from config.settings import get_settings


LAYOUT_PROFILES = {
    "normal": {
        "font_size": "10.5pt",
        "line_height": "1.35",
        "section_margin": "12px",
    },
    "compact": {
        "font_size": "10pt",
        "line_height": "1.25",
        "section_margin": "9px",
    },
    "dense": {
        "font_size": "9.5pt",
        "line_height": "1.18",
        "section_margin": "7px",
    },
}

SECTION_TEMPLATES = (
    ("summary", "sections/summary.html"),
    ("skills", "sections/skills.html"),
    ("work_experience", "sections/work_experience.html"),
    ("projects", "sections/projects.html"),
    ("research", "sections/research.html"),
    ("education", "sections/education.html"),
)


def create_environment(
    template_dir: str | Path | None = None,
) -> Environment:
    """Create the shared Jinja environment."""
    template_dir = template_dir or get_settings().io.template_dir
    return Environment(
        loader=FileSystemLoader(str(Path(template_dir))),
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_contact_line(resume: Resume) -> str:
    """Build one plain-text contact line."""
    candidate = resume.candidate

    parts = (
        candidate.location,
        candidate.phone,
        candidate.email,
        candidate.github,
        candidate.linkedin,
        candidate.website,
    )

    return " • ".join(str(part) for part in parts if part)


def build_render_sections(resume: Resume) -> list[dict[str, Any]]:
    """Return populated sections in render order."""
    sections: list[dict[str, Any]] = []

    for field_name, template in SECTION_TEMPLATES:
        section = getattr(resume, field_name, None)

        if section is None or not section.items:
            continue

        sections.append(
            {
                "template": template,
                "section": section.model_dump(
                    mode="json",
                ),
            }
        )

    return sections


def render_html(
    resume: Resume,
    *,
    template_dir: str | Path | None = None,
    template_file: str | None = None,
    layout_profile: str = "normal",
) -> str:
    """Render a Resume object to HTML."""
    try:
        layout = LAYOUT_PROFILES[layout_profile]
    except KeyError as error:
        options = ", ".join(LAYOUT_PROFILES)
        raise ValueError(
            f"Unknown layout profile {layout_profile!r}. "
            f"Expected one of: {options}"
        ) from error

    io = get_settings().io
    template_dir = template_dir or io.template_dir
    template_file = template_file or io.resume_template_filename
    template = create_environment(template_dir).get_template(template_file)

    return template.render(
        resume=resume,
        name=resume.candidate.name,
        target_title=resume.candidate.target_title,
        contact=build_contact_line(resume),
        sections=build_render_sections(resume),
        layout=layout,
        layout_profile=layout_profile,
    )


def save_html(
    html: str,
    output_path: str | Path,
) -> Path:
    """Write rendered HTML to disk."""
    path = Path(output_path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path.resolve()


def save_pdf(
    html: str,
    output_path: str | Path,
    *,
    base_url: str | Path | None = None,
) -> Path:
    """Render an HTML string to PDF with WeasyPrint."""
    if sys.platform == "darwin":
        homebrew_lib = get_settings().io.homebrew_library_dir
        if homebrew_lib.is_dir():
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
            paths = [str(homebrew_lib), *(existing.split(":") if existing else [])]
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                dict.fromkeys(paths)
            )

    try:
        from weasyprint import HTML
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PDF generation requires WeasyPrint and its native libraries. "
            "On macOS install them with: brew install pango"
        ) from error

    path = Path(output_path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    base_url = base_url or get_settings().io.template_dir
    HTML(
        string=html,
        base_url=str(Path(base_url).resolve()),
    ).write_pdf(str(path))

    return path.resolve()

def render_pdf_bytes(
    html: str,
    *,
    base_url: str | Path | None = None,
) -> bytes:
    """Render an HTML string to in-memory PDF bytes."""
    if sys.platform == "darwin":
        homebrew_lib = get_settings().io.homebrew_library_dir
        if homebrew_lib.is_dir():
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
            paths = [str(homebrew_lib), *(existing.split(":") if existing else [])]
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                dict.fromkeys(paths)
            )
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PDF generation requires WeasyPrint and its native libraries."
        ) from error
    base_url = base_url or get_settings().io.template_dir
    return HTML(
        string=html,
        base_url=str(Path(base_url).resolve()),
    ).write_pdf()


def count_pdf_pages(
    html: str,
    *,
    base_url: str | Path | None = None,
) -> int:
    """Render HTML in memory and return its physical PDF page count."""
    if sys.platform == "darwin":
        homebrew_lib = get_settings().io.homebrew_library_dir
        if homebrew_lib.is_dir():
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
            paths = [str(homebrew_lib), *(existing.split(":") if existing else [])]
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                dict.fromkeys(paths)
            )

    try:
        from weasyprint import HTML
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PDF page counting requires WeasyPrint and its native libraries."
        ) from error

    base_url = base_url or get_settings().io.template_dir
    document = HTML(
        string=html,
        base_url=str(Path(base_url).resolve()),
    ).render()
    return len(document.pages)
