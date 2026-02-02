"""Intelligent chunking: fixed and dynamic strategies."""

from typing import Any

from .analyzer import ChunkingStrategy


def chunk_document(
    parsed: dict[str, Any],
    strategy: ChunkingStrategy,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[dict[str, Any]]:
    """
    Split document into chunks based on the selected strategy.

    Args:
        parsed: Output from parse_document (text, structure, metadata)
        strategy: "fixed" or "dynamic"
        chunk_size: Max chars for fixed strategy; target for dynamic
        chunk_overlap: Overlap between consecutive chunks (fixed only)

    Returns:
        List of chunks, each with text, chunk_id, metadata
    """
    if strategy == "fixed":
        return _chunk_fixed(parsed, chunk_size, chunk_overlap)
    return _chunk_dynamic(parsed, chunk_size)


def _chunk_fixed(
    parsed: dict[str, Any],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Split by character count with overlap."""
    text = parsed.get("text", "")
    metadata_base = parsed.get("metadata", {})
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if not chunk_text.strip():
            start = end - overlap
            continue

        chunks.append({
            "text": chunk_text,
            "chunk_id": chunk_idx,
            "metadata": {
                **metadata_base,
                "strategy": "fixed",
                "chunk_index": chunk_idx,
            },
        })
        chunk_idx += 1
        start = end - overlap

    return chunks


def _chunk_dynamic(parsed: dict[str, Any], target_size: int) -> list[dict[str, Any]]:
    """Split on semantic boundaries (paragraphs, sections, headings)."""
    structure = parsed.get("structure", [])
    metadata_base = parsed.get("metadata", {})

    if structure:
        return _chunk_from_structure(structure, metadata_base, target_size)

    # Fallback: split by paragraphs (double newline)
    text = parsed.get("text", "")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    structure = [{"type": "paragraph", "level": 0, "text": b} for b in blocks]
    return _chunk_from_structure(structure, metadata_base, target_size)


def _chunk_from_structure(
    structure: list[dict],
    metadata_base: dict,
    target_size: int,
) -> list[dict[str, Any]]:
    """Chunk using structure, respecting heading boundaries."""
    chunks: list[dict[str, Any]] = []
    current_chunk: list[str] = []
    current_size = 0
    chunk_idx = 0

    for block in structure:
        block_text = block.get("text", "")
        if not block_text:
            continue

        block_len = len(block_text) + 2  # +2 for \n\n

        is_heading = block.get("type") == "heading"

        if is_heading and current_chunk and current_size + block_len > target_size:
            # Flush current chunk before starting new section
            if current_chunk:
                chunks.append(_make_chunk(current_chunk, chunk_idx, metadata_base))
                chunk_idx += 1
            current_chunk = [block_text]
            current_size = block_len
        elif current_size + block_len > target_size and current_chunk:
            chunks.append(_make_chunk(current_chunk, chunk_idx, metadata_base))
            chunk_idx += 1
            current_chunk = [block_text]
            current_size = block_len
        else:
            current_chunk.append(block_text)
            current_size += block_len

    if current_chunk:
        chunks.append(_make_chunk(current_chunk, chunk_idx, metadata_base))

    return chunks


def _make_chunk(
    parts: list[str],
    chunk_idx: int,
    metadata_base: dict,
) -> dict[str, Any]:
    text = "\n\n".join(parts)
    return {
        "text": text,
        "chunk_id": chunk_idx,
        "metadata": {
            **metadata_base,
            "strategy": "dynamic",
            "chunk_index": chunk_idx,
        },
    }
