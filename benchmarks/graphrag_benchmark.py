"""GraphRAG benchmark: entity extraction coverage, graph density."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.storage import VectorDB, SQLDB


def run_graphrag_benchmark(doc_path: str | Path) -> dict:
    """Measure entity extraction and graph construction."""
    doc_path = Path(doc_path)
    vector_db = VectorDB(persist_directory=None)
    sql_db = SQLDB(db_path=":memory:")
    processor = DocumentProcessor(vector_db=vector_db, sql_db=sql_db)
    result = processor.process(doc_path)
    doc_id = result["doc_id"]
    edges_count = result.get("graph_edges", 0)
    chunk_count = result["chunk_count"]

    return {
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "graph_edges": edges_count,
        "edges_per_chunk": edges_count / chunk_count if chunk_count else 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graphrag_benchmark.py <path_to_document>")
        sys.exit(1)
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)
    metrics = run_graphrag_benchmark(doc_path)
    print("GraphRAG Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
