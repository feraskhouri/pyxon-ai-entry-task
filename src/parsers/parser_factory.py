"""Unified parser factory that routes by file extension."""

from pathlib import Path
from typing import Any

from .pdf_parser import parse_pdf
from .docx_parser import parse_docx
from .txt_parser import parse_txt

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a document based on its file extension.
    Returns raw text and optional structure hints (headings, sections).

    Args:
        file_path: Path to the document file.

    Returns:
        Dict with text, structure (if available), and metadata.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".txt":
        return parse_txt(path)

    raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")
