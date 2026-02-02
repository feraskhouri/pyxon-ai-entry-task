"""DOCX document parser with structure extraction (paragraphs, headings)."""

from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.style import WD_STYLE_TYPE


def parse_docx(file_path: str | Path) -> dict[str, Any]:
    """
    Extract text and structure from a DOCX file.
    Preserves paragraphs and heading structure for intelligent chunking.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Dict with keys: text, structure (list of blocks with type/level), metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    doc = Document(file_path)
    structure: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else ""
        is_heading = "heading" in style_name.lower() or para.style and para.style.type == WD_STYLE_TYPE.PARAGRAPH

        level = 0
        if "heading" in style_name.lower():
            try:
                level = int(style_name.split()[-1]) if style_name.split()[-1].isdigit() else 1
            except (ValueError, IndexError):
                level = 1

        block = {"type": "heading" if "heading" in style_name.lower() else "paragraph", "level": level, "text": text}
        structure.append(block)
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "format": "docx",
        "block_count": len(structure),
    }

    return {
        "text": full_text,
        "structure": structure,
        "metadata": metadata,
    }
