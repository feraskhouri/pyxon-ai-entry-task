"""Unified retrieval engine: vector, graph, RAPTOR, hybrid."""

from typing import Any

from ..graphrag.entity_extractor import extract_entities, detect_language


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}

    for lst in result_lists:
        for rank, item in enumerate(lst, 1):
            doc_id = item.get("id") or item.get("text", "")[:100]
            key = str(doc_id)
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
            if key not in seen:
                seen[key] = item

    sorted_keys = sorted(scores.keys(), key=lambda x: -scores[x])
    return [seen[k] for k in sorted_keys]


def _deduplicate_results(results: list[dict], key: str = "id") -> list[dict]:
    """Deduplicate by key, preserving order."""
    seen = set()
    out = []
    for r in results:
        k = r.get(key) or r.get("text", "")[:80]
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


class RetrievalEngine:
    """Unified retrieval with vector, graph, RAPTOR, hybrid modes."""

    def __init__(self, vector_db, sql_db):
        self.vector_db = vector_db
        self.sql_db = sql_db

    def _route_mode(self, query: str) -> str:
        """Simple heuristic routing."""
        q = query.lower()
        if any(w in q for w in ("related", "connection", "between", "relationship", "connect")):
            return "graph"
        if any(w in q for w in ("summarize", "overview", "main points", "summary")):
            return "raptor"
        return "vector"

    def retrieve(
        self,
        query: str,
        mode: str | None = None,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks.

        mode: "vector" | "graph" | "raptor" | "hybrid" | None (auto-route)
        doc_id: optional; limit retrieval to this document only.
        """
        mode = mode or self._route_mode(query)

        if mode == "vector":
            return self.vector_db.search(query, top_k=top_k, doc_id=doc_id)

        if mode == "graph":
            return self._retrieve_graph(query, top_k, doc_id=doc_id)

        if mode == "raptor":
            return self._retrieve_raptor(query, top_k, doc_id=doc_id)

        if mode == "hybrid":
            return self._retrieve_hybrid(query, top_k, doc_id=doc_id)

        return self.vector_db.search(query, top_k=top_k, doc_id=doc_id)

    def _retrieve_graph(
        self, query: str, top_k: int, doc_id: str | None = None
    ) -> list[dict]:
        """Vector search -> extract entities from query + results -> expand via graph -> fuse."""
        fetch_k = max(10, top_k * 2)
        initial = self.vector_db.search(query, top_k=fetch_k, doc_id=doc_id)
        if not initial:
            return []

        all_text = " ".join(r.get("text", "") for r in initial)
        lang = detect_language(all_text)
        entities_from_results = extract_entities(all_text, lang)
        entities_from_results = [e for e in entities_from_results if len(e) > 2][:15]

        query_lang = detect_language(query)
        entities_from_query = extract_entities(query, query_lang)
        entities_from_query = [e for e in entities_from_query if len(e) > 2][:10]

        entities = list(dict.fromkeys(entities_from_query + entities_from_results))[:15]

        related = []
        for e in entities[:8]:
            related.extend(
                t[0] for t in self.sql_db.get_related_entities(e, top_k=4, doc_id=doc_id)
            )
        related = list(dict.fromkeys(related))[:15]

        expanded = []
        for ent in related[:5]:
            chunk_ids = self.sql_db.get_chunk_ids_for_entities(
                [ent], limit=6, doc_id=doc_id
            )
            if chunk_ids:
                chunks = self.vector_db.get_by_ids(chunk_ids)
                expanded.extend(chunks)

        merged = _reciprocal_rank_fusion([initial, expanded], k=60)
        return _deduplicate_results(merged)[:top_k]

    def _retrieve_raptor(
        self, query: str, top_k: int, doc_id: str | None = None
    ) -> list[dict]:
        """Multi-level RAPTOR search (level 1 summaries + level 0 chunks). doc_id filters when set."""
        fetch_k = max(8, top_k * 2)
        from_levels = []
        for level in [1, 0]:
            r = self.vector_db.search_raptor(
                query, top_k=fetch_k, level=level, doc_id=doc_id
            )
            if r:
                from_levels.append(r)
        if not from_levels:
            return self.vector_db.search(query, top_k=top_k, doc_id=doc_id)
        merged = _reciprocal_rank_fusion(from_levels, k=60)
        return _deduplicate_results(merged)[:top_k]

    def _retrieve_hybrid(
        self, query: str, top_k: int, doc_id: str | None = None
    ) -> list[dict]:
        """Vector + graph fusion with more candidates for better RRF."""
        fetch_k = max(10, top_k * 2)
        vector_res = self.vector_db.search(query, top_k=fetch_k, doc_id=doc_id)
        graph_res = self._retrieve_graph(query, fetch_k, doc_id=doc_id)
        merged = _reciprocal_rank_fusion([vector_res, graph_res], k=60)
        return _deduplicate_results(merged)[:top_k]
