"""PDF document parser using PyMuPDF for robust Arabic and diacritics support."""

import io
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def _ocr_page(page: "fitz.Page") -> str:
    """Extract text from a page image using OCR (for scanned PDFs). Requires Tesseract installed."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def parse_pdf(file_path: str | Path, use_ocr_fallback: bool = True, max_ocr_pages: int = 50) -> dict[str, Any]:
    """
    Extract text and structure from a PDF file.
    Uses PyMuPDF for better Arabic/diacritics support than PyPDF2.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Dict with keys: text (full text), pages (list of page texts), metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    pages_text: list[str] = []
    full_text_parts: list[str] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text", sort=True)
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks if b[4].strip()) if blocks else ""
            if not text.strip() and use_ocr_fallback and page_num < max_ocr_pages:
                text = _ocr_page(page)
            pages_text.append(text)
            full_text_parts.append(text)
        full_text = "\n\n".join(full_text_parts)
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "format": "pdf",
            "page_count": len(doc),
        }

        return {
            "text": full_text,
            "pages": pages_text,
            "metadata": metadata,
        }
    finally:
        doc.close()
