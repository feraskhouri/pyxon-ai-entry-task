"""Performance benchmark: parse time, chunk time, embed time, memory."""

import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.storage import VectorDB, SQLDB


def run_performance_benchmark(doc_path: str | Path) -> dict:
    """Measure timing and memory for full pipeline."""
    doc_path = Path(doc_path)

    tracemalloc.start()
    t0 = time.perf_counter()

    processor = DocumentProcessor(
        vector_db=VectorDB(persist_directory=None),
        sql_db=SQLDB(db_path=":memory:"),
    )
    result = processor.process(doc_path)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "total_time_seconds": round(elapsed, 3),
        "peak_memory_mb": round(peak / 1024 / 1024, 2),
        "chunk_count": result["chunk_count"],
        "strategy": result["strategy"],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python performance_benchmark.py <path_to_document>")
        sys.exit(1)
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)
    metrics = run_performance_benchmark(doc_path)
    print("Performance Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
