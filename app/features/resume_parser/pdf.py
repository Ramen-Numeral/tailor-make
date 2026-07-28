"""Bounded local text extraction for uploaded PDF resumes."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.features.resume_parser.schema import PDFTextExtraction

DEFAULT_MAX_PDF_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES = 10
DEFAULT_MIN_TEXT_CHARACTERS = 40


class ResumePDFError(ValueError):
    """Raised when an uploaded PDF cannot be safely converted to text."""


def _read_upload(
    source: bytes | bytearray | BinaryIO | str | Path,
    max_bytes: int,
) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise ResumePDFError(
                f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
            )
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read(max_bytes + 1)
        if not isinstance(data, bytes):
            raise ResumePDFError("Uploaded PDF must be opened in binary mode.")
    else:
        raise TypeError("source must be PDF bytes, a binary upload, or a path")

    if len(data) > max_bytes:
        raise ResumePDFError(
            f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    if not data.lstrip().startswith(b"%PDF-"):
        raise ResumePDFError("The uploaded file is not a valid PDF.")
    return data


def extract_pdf_text(
    source: bytes | bytearray | BinaryIO | str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    min_text_characters: int = DEFAULT_MIN_TEXT_CHARACTERS,
) -> PDFTextExtraction:
    """Extract text while bounding file size and page count."""
    if max_bytes < 1 or max_pages < 1 or min_text_characters < 1:
        raise ValueError("PDF extraction limits must be positive")

    data = _read_upload(source, max_bytes)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumePDFError(
                "Password-protected PDFs are not supported. Upload an unlocked copy."
            )
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise ResumePDFError(
                f"PDF has {page_count} pages; the limit is {max_pages}."
            )

        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except ResumePDFError:
        raise
    except (PdfReadError, ValueError, TypeError, OSError) as exc:
        raise ResumePDFError("The uploaded PDF could not be read.") from exc

    warnings = [
        f"Page {number} contained no extractable text."
        for number, text in enumerate(page_texts, start=1)
        if not text
    ]
    text = "\n\n".join(
        f"--- Page {number} ---\n{page_text}"
        for number, page_text in enumerate(page_texts, start=1)
        if page_text
    )
    if len("".join(page_texts).strip()) < min_text_characters:
        raise ResumePDFError(
            "The PDF contains too little extractable text. "
            "It may be scanned; upload a text-based PDF or run OCR first."
        )
    return PDFTextExtraction(
        text=text,
        page_count=page_count,
        page_texts=page_texts,
        warnings=warnings,
    )
