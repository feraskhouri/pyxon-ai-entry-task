"""RAPTOR benchmark: retrieval at different levels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.rag_client import RAGClient
from src.storage import VectorDB, SQLDB


def run_raptor_benchmark(doc_path: str | Path) -> dict:
    """Test RAPTOR multi-level retrieval."""
    doc_path = Path(doc_path)
    vector_db = VectorDB(persist_directory=None)
    sql_db = SQLDB(db_path=":memory:")
    processor = DocumentProcessor(vector_db=vector_db, sql_db=sql_db)
    result = processor.process(doc_path)
    rag = RAGClient(vector_db, sql_db)

    query = "machine learning"
    vector_res = rag.vector_db.search(query, top_k=3)
    raptor_res = rag.vector_db.search_raptor(query, top_k=3)

    return {
        "chunk_count": result["chunk_count"],
        "vector_results": len(vector_res),
        "raptor_results": len(raptor_res),
        "raptor_works": len(raptor_res) > 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python raptor_benchmark.py <path_to_document>")
        sys.exit(1)
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)
    metrics = run_raptor_benchmark(doc_path)
    print("RAPTOR Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
