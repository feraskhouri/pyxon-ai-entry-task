"""Main document processing orchestrator."""

from pathlib import Path
from typing import Any
import uuid

from .parsers import parse_document
from .analyzer import analyze_document
from .chunker import chunk_document
from .storage import VectorDB, SQLDB


def _process_parsed(
    processor: "DocumentProcessor",
    parsed: dict,
    filename: str,
    format_type: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> dict[str, Any]:
    """Shared logic for processing a parsed document dict."""
    from .graphrag.graph_builder import build_cooccurrence_edges
    from .raptor.raptor import build_raptor_tree_with_model

    strategy = analyze_document(parsed)
    chunks = chunk_document(parsed, strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if not chunks:
        raise ValueError(f"No text extracted from {filename}. File may be empty, corrupted, or image-only (OCR failed).")
    
    for c in chunks:
        c.setdefault("metadata", {})["source_type"] = parsed.get("metadata", {}).get("source_type", "local")

    doc_id = str(uuid.uuid4())
    processor.vector_db.add_chunks(chunks, doc_id=doc_id)

    if len(chunks) >= 2:
        raptor_nodes = build_raptor_tree_with_model(
            chunks,
            processor.vector_db.model,
            doc_id=doc_id,
        )
        processor.vector_db.add_raptor_nodes(raptor_nodes)

    processor.sql_db.add_document(
        filename=filename,
        format_type=format_type,
        strategy=strategy,
        chunks=chunks,
        doc_id=doc_id,
    )

    edges, entity_chunks = build_cooccurrence_edges(chunks, doc_id=doc_id)
    if edges:
        processor.sql_db.add_graph_edges(doc_id, edges)
    if entity_chunks:
        processor.sql_db.add_entity_chunks(doc_id, entity_chunks)

    return {
        "doc_id": doc_id,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "filename": filename,
        "metadata": parsed.get("metadata", {}),
        "graph_edges": len(edges),
    }


class DocumentProcessor:
    """Orchestrates parsing, analysis, chunking, and storage."""

    def __init__(
        self,
        vector_db: VectorDB | None = None,
        sql_db: SQLDB | None = None,
        persist_vector_path: str | Path | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        embedding_model: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            kwargs = {"persist_directory": persist_vector_path}
            if embedding_model:
                kwargs["embedding_model"] = embedding_model
            self.vector_db = VectorDB(**kwargs)
        self.sql_db = sql_db or SQLDB()

    def process(self, file_path: str | Path) -> dict[str, Any]:
        """
        Process a local document: parse, analyze, chunk, embed, store.

        Returns:
            Dict with doc_id, strategy, chunk_count, metadata.
        """
        path = Path(file_path)
        parsed = parse_document(path)
        parsed.setdefault("metadata", {})["source_type"] = "local"
        return _process_parsed(
            self,
            parsed,
            filename=path.name,
            format_type=parsed.get("metadata", {}).get("format", "unknown"),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
