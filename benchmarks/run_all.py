"""Run all benchmarks and print summary."""

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval_benchmark import run_retrieval_benchmark
from chunking_benchmark import run_chunking_benchmark
from arabic_benchmark import run_arabic_benchmark
from performance_benchmark import run_performance_benchmark
from graphrag_benchmark import run_graphrag_benchmark
from raptor_benchmark import run_raptor_benchmark


def main(doc_path: str | None = None):
    if not doc_path:
        print("Usage: python run_all.py <path_to_document>")
        print("Example: python run_all.py ../myfile.pdf")
        return {}
    test_file = Path(doc_path)
    if not test_file.exists():
        print(f"Document not found: {doc_path}")
        return {}
    print(f"Running benchmarks on: {test_file}\n")

    results = {}

    print("=== Retrieval Benchmark ===")
    try:
        results["retrieval"] = run_retrieval_benchmark(test_file)
        for k, v in results["retrieval"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== Chunking Benchmark ===")
    try:
        results["chunking"] = run_chunking_benchmark(test_file)
        for k, v in results["chunking"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== Performance Benchmark ===")
    try:
        results["performance"] = run_performance_benchmark(test_file)
        for k, v in results["performance"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== GraphRAG Benchmark ===")
    try:
        results["graphrag"] = run_graphrag_benchmark(test_file)
        for k, v in results["graphrag"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== RAPTOR Benchmark ===")
    try:
        results["raptor"] = run_raptor_benchmark(test_file)
        for k, v in results["raptor"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <path_to_document>")
        sys.exit(1)
    main(sys.argv[1])
