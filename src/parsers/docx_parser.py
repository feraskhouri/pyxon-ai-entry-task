"""DOCX document parser with structure extraction (paragraphs, headings, tables)."""

from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def _iter_block_items(doc: Document) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document order (so middle/end content in tables is included)."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _table_to_text(table: Table) -> str:
    """Extract all text from a table, cell by cell, row by row."""
    parts: list[str] = []
    for row in table.rows:
        row_parts: list[str] = []
        for cell in row.cells:
            cell_text = " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
            if cell_text:
                row_parts.append(cell_text)
        if row_parts:
            parts.append(" ".join(row_parts))
    return "\n\n".join(parts) if parts else ""


def parse_docx(file_path: str | Path) -> dict[str, Any]:
    """
    Extract text and structure from a DOCX file.
    Preserves paragraphs, headings, and table content in document order.

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

    for block_item in _iter_block_items(doc):
        if isinstance(block_item, Paragraph):
            text = block_item.text.strip()
            if not text:
                continue
            style_name = block_item.style.name if block_item.style else ""
            level = 0
            if "heading" in style_name.lower():
                try:
                    level = int(style_name.split()[-1]) if style_name.split()[-1].isdigit() else 1
                except (ValueError, IndexError):
                    level = 1
            block = {
                "type": "heading" if "heading" in style_name.lower() else "paragraph",
                "level": level,
                "text": text,
            }
            structure.append(block)
            full_text_parts.append(text)
        elif isinstance(block_item, Table):
            text = _table_to_text(block_item)
            if not text:
                continue
            block = {"type": "paragraph", "level": 0, "text": text}
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
