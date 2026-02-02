"""Build co-occurrence graph from chunks."""

from typing import Any

from .entity_extractor import extract_entities, detect_language


def build_cooccurrence_edges(
    chunks: list[dict[str, Any]],
    doc_id: str | None = None,
) -> tuple[list[tuple[str, str, float]], dict[str, list[str]]]:
    """
    Build edges from entity co-occurrence in same chunk.
    Returns (edges, entity_to_chunk_ids).
    """
    edges: dict[tuple[str, str], float] = {}
    entity_chunks: dict[str, list[str]] = {}

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        chunk_idx = chunk.get("metadata", {}).get("chunk_index", i)
        chunk_id_str = f"{doc_id}_{chunk_idx}" if doc_id else str(chunk_idx)
        if not text:
            continue
        lang = detect_language(text)
        entities = extract_entities(text, lang)
        entities = [e for e in entities if len(e) > 1]
        for e in entities:
            entity_chunks.setdefault(e, []).append(chunk_id_str)
        for i_a, a in enumerate(entities):
            for b in entities[i_a + 1:]:
                if a != b:
                    key = tuple(sorted([a, b]))
                    edges[key] = edges.get(key, 0) + 1.0

    return ([(a, b, w) for (a, b), w in edges.items()], entity_chunks)
