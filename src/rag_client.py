"""RAG integration interface for retrieval-augmented generation systems."""

from typing import Any

from .storage import VectorDB, SQLDB
from .retrieval import RetrievalEngine


class RAGClient:
    """
    Client for RAG systems: semantic retrieval (Vector DB) and structured queries (SQL DB).
    Supports vector, graph, RAPTOR, and hybrid retrieval modes.
    """

    def __init__(self, vector_db: VectorDB, sql_db: SQLDB):
        self.vector_db = vector_db
        self.sql_db = sql_db
        self.retrieval_engine = RetrievalEngine(vector_db, sql_db)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search over document chunks. Uses retrieval engine when mode is set.

        mode: None (vector only) | "vector" | "graph" | "raptor" | "hybrid"
        doc_id: optional; limit search to this document only.
        """
        if mode and mode != "vector":
            return self.retrieval_engine.retrieve(
                query, mode=mode, top_k=top_k, doc_id=doc_id
            )
        return self.vector_db.search(query, top_k=top_k, doc_id=doc_id)

    def retrieve(
        self,
        query: str,
        mode: str | None = None,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve with optional mode (auto-routes if None). Optionally limit to doc_id."""
        return self.retrieval_engine.retrieve(
            query, mode=mode, top_k=top_k, doc_id=doc_id
        )

    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        mode: str | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Retrieve chunks and format as context string for LLM prompt."""
        results = self.retrieve(
            query, mode=mode, top_k=top_k, doc_id=doc_id
        )
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['text']}")
        return "\n\n---\n\n".join(parts) if parts else ""

    def list_documents(self) -> list[dict]:
        """List all ingested documents (from SQL DB)."""
        return self.sql_db.list_documents()

    def get_document(self, doc_id: str) -> dict | None:
        """Get document and chunks by ID."""
        return self.sql_db.get_document(doc_id)

    def answer(
        self,
        query: str,
        use_llm: bool = False,
        api_key: str | None = None,
        mode: str | None = None,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve context and optionally generate answer via OpenAI.

        Returns:
            {answer, sources, context}
        """
        results = self.retrieve(
            query, mode=mode, top_k=top_k, doc_id=doc_id
        )
        context = "\n\n---\n\n".join(r.get("text", "") for r in results)
        sources = [{"id": r.get("id"), "text": (r.get("text", ""))[:200]} for r in results]

        if use_llm and api_key:
            from .llm.openai_client import generate_answer
            answer = generate_answer(query, context, api_key=api_key)
        else:
            answer = "Enable LLM and provide API key to generate answers."

        return {"answer": answer, "sources": sources, "context": context}
