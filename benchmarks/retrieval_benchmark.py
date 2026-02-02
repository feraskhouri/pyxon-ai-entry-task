"""Retrieval accuracy benchmark: precision@k, recall@k, MRR."""

import os
import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from src.document_processor import DocumentProcessor
from src.rag_client import RAGClient
from src.storage import VectorDB, SQLDB


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "may",
    "might", "can", "this", "that", "these", "those", "it", "its", "from",
    "by", "as", "if", "when", "where", "which", "who", "what", "how", "why"
}


def extract_keywords(text: str, n: int = 5) -> str:
    """Extract top n keywords from text, excluding stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not filtered:
        words_any = re.findall(r'\b\w{3,}\b', text.lower())
        return " ".join(words_any[:n]) if words_any else text[:50]
    counter = Counter(filtered)
    top_words = [w for w, _ in counter.most_common(n * 2)]
    unique = []
    for w in top_words:
        if w not in unique:
            unique.append(w)
        if len(unique) >= n:
            break
    return " ".join(unique) if unique else " ".join(filtered[:n])


def run_retrieval_benchmark(
    doc_path: str | Path,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    embedding_model: str | None = None,
    top_k: int = 10,
) -> dict:
    """Run retrieval benchmark on any document. Uses self-retrieval: query from chunk text, expect that chunk in top-k."""
    doc_path = Path(doc_path)
    vector_db = VectorDB(persist_directory=None, embedding_model=embedding_model) if embedding_model else VectorDB(persist_directory=None)
    sql_db = SQLDB(db_path=":memory:")
    processor = DocumentProcessor(
        vector_db=vector_db,
        sql_db=sql_db,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    result = processor.process(doc_path)
    doc_id = result["doc_id"]
    chunk_count = result["chunk_count"]

    if chunk_count == 0:
        return {
            "doc_id": doc_id,
            "chunk_count": 0,
            "strategy": result["strategy"],
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "note": "No chunks extracted (empty or image-only document)",
        }

    rag = RAGClient(vector_db, sql_db)

    doc = sql_db.get_document(doc_id)
    chunks = doc.get("chunks", []) if doc else []
    pairs = []
    for c in chunks:
        text = c.get("text_preview", "").strip()
        if len(text) < 30:
            continue
        query = extract_keywords(text, n=5)
        if len(query) < 10:
            continue
        pairs.append((query, c.get("id", "")))
    pairs = pairs[:10]
    if not pairs and chunks:
        for c in chunks[:5]:
            q = (c.get("text_preview", "") or "")[:80].strip()
            if q:
                pairs.append((q, c.get("id", "")))

    hits = []
    rr_values = []
    search_k = max(top_k, 5)
    for query, expected_id in pairs:
        results = rag.search(query, top_k=search_k)
        retrieved_ids = [r.get("id") for r in results]
        hit = expected_id in retrieved_ids
        hits.append(1 if hit else 0)
        rank = next((i + 1 for i, rid in enumerate(retrieved_ids) if rid == expected_id), None)
        rr_values.append(1.0 / rank if rank else 0.0)

    n = len(pairs)
    precision_at_k = sum(hits) / n if n else 0.0
    recall_at_k = precision_at_k
    mrr = sum(rr_values) / n if n else 0.0

    return {
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "strategy": result["strategy"],
        "queries_tested": n,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
    }


if __name__ == "__main__":
    import argparse
    from src.parsers import parse_document

    parser = argparse.ArgumentParser()
    parser.add_argument("document", nargs="?", default=None, help="Path to document")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chars per chunk")
    parser.add_argument("--target-chunks", type=int, default=None, help="Target chunk count (estimates chunk_size)")
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=None, help="Embedding model (e.g. BAAI/bge-m3)")
    args = parser.parse_args()
    if not args.document:
        print("Usage: python retrieval_benchmark.py <path_to_document> [options]")
        sys.exit(1)
    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)

    chunk_size = args.chunk_size
    if args.target_chunks:
        parsed = parse_document(doc_path)
        total_chars = len(parsed.get("text", ""))
        chunk_size = max(100, total_chars // args.target_chunks)
        print(f"Target {args.target_chunks} chunks: estimated chunk_size={chunk_size} (doc={total_chars} chars)\n")

    chunk_size = chunk_size or 1024
    print(f"Testing: {doc_path} (chunk_size={chunk_size}, top_k={args.top_k})\n")
    metrics = run_retrieval_benchmark(
        doc_path,
        chunk_size=chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.model,
        top_k=args.top_k,
    )
    print("Retrieval Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
