"""TXT document parser with UTF-8 encoding for Arabic support."""

from pathlib import Path
from typing import Any


def parse_txt(file_path: str | Path) -> dict[str, Any]:
    """
    Read TXT file with UTF-8 encoding (critical for Arabic and diacritics).

    Args:
        file_path: Path to the TXT file.

    Returns:
        Dict with keys: text, structure (paragraphs), metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    structure = [{"type": "paragraph", "level": 0, "text": p} for p in paragraphs]

    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "format": "txt",
        "paragraph_count": len(paragraphs),
    }

    return {
        "text": content,
        "structure": structure,
        "metadata": metadata,
    }
