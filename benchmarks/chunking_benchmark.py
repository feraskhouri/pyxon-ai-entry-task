"""Chunking quality benchmark: coherence, size distribution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import parse_document
from src.analyzer import analyze_document
from src.chunker import chunk_document


def run_chunking_benchmark(doc_path: str | Path) -> dict:
    """Evaluate chunk coherence and size distribution."""
    doc_path = Path(doc_path)
    parsed = parse_document(doc_path)
    strategy = analyze_document(parsed)
    chunks = chunk_document(parsed, strategy)

    lengths = [len(c["text"]) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    variance = sum((x - avg_len) ** 2 for x in lengths) / len(lengths) if lengths else 0
    std_dev = variance ** 0.5

    return {
        "strategy": strategy,
        "chunk_count": len(chunks),
        "avg_chunk_length": avg_len,
        "std_dev_length": std_dev,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunking_benchmark.py <path_to_document>")
        sys.exit(1)
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)
    metrics = run_chunking_benchmark(doc_path)
    print("Chunking Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
