"""Content analyzer for intelligent chunking strategy selection."""

from typing import Any, Literal

ChunkingStrategy = Literal["fixed", "dynamic"]


def analyze_document(parsed: dict[str, Any]) -> ChunkingStrategy:
    """
    Analyze document content and determine the best chunking strategy.

    - Fixed: For short, uniform documents (reports, forms, repetitive content)
    - Dynamic: For long documents with clear structure (chapters, headers, mixed content)

    Heuristics:
    - Many headings/sections + variable structure -> dynamic
    - Short + uniform paragraph sizes -> fixed
    - Long document -> dynamic
    """
    text = parsed.get("text", "")
    structure = parsed.get("structure", [])
    pages = parsed.get("pages", [])

    # Use structure if available (DOCX, TXT with paragraphs)
    if structure:
        headings = [b for b in structure if b.get("type") == "heading"]
        paragraphs = [b for b in structure if b.get("type") == "paragraph"]

        heading_count = len(headings)
        para_lengths = [len(p.get("text", "")) for p in paragraphs]
    else:
        # Fallback for PDF: split by double newline
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        heading_count = sum(1 for b in blocks if _looks_like_heading(b))
        para_lengths = [len(b) for b in blocks]

    doc_length = len(text)
    para_count = len(para_lengths)
    avg_para_len = sum(para_lengths) / para_count if para_count else 0
    variance = (
        sum((x - avg_para_len) ** 2 for x in para_lengths) / para_count
        if para_count
        else 0
    )
    std_dev = variance ** 0.5

    # Dynamic when: has clear sections, long doc, or high variance in block sizes
    if heading_count >= 2:
        return "dynamic"
    if doc_length > 5000:  # ~1-2 pages
        return "dynamic"
    if para_count > 0 and std_dev > avg_para_len * 0.5:  # High variance
        return "dynamic"

    return "fixed"


def _looks_like_heading(line: str) -> bool:
    """Heuristic: short line, possibly numbered, or all caps."""
    if len(line) > 80:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) <= 6 and stripped[0].isdigit():
        return True
    if stripped.isupper() and len(stripped) < 50:
        return True
    return False
