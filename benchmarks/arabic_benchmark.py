"""Arabic language and diacritics benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.rag_client import RAGClient
from src.storage import VectorDB, SQLDB

# Arabic diacritics (harakat) to verify preservation
# Each entry includes both the literal character and its Unicode code point for clarity.
ARABIC_DIACRITICS = [
    "\u064E",  #  Fatha
    "\u064F",  #  Damma
    "\u0650",  #  Kasra
    "\u0652",  #  Sukun
    "\u064B",  #  Tanwin Fath
    "\u064C",  #  Tanwin Damm
    "\u064D",  #  Tanwin Kasr
    "\u0651",  #  Shadda
]



def run_arabic_benchmark(doc_path: str | Path) -> dict:
    """Verify Arabic support: encoding, diacritics preservation, retrieval."""
    doc_path = Path(doc_path)
    with open(doc_path, encoding="utf-8") as f:
        raw_text = f.read()

    # Check source has diacritics
    diacritics_in_source = sum(1 for c in raw_text if c in ARABIC_DIACRITICS)

    vector_db = VectorDB(persist_directory=None)
    sql_db = SQLDB(db_path=":memory:")
    processor = DocumentProcessor(vector_db=vector_db, sql_db=sql_db)
    result = processor.process(doc_path)

    rag = RAGClient(vector_db, sql_db)
    results = rag.search("التشكيل الحركات", top_k=3)
    retrieved = " ".join(r["text"] for r in results)

    diacritics_in_retrieved = sum(1 for c in retrieved if c in ARABIC_DIACRITICS)
    encoding_ok = "التشكيل" in retrieved or "الحركات" in retrieved or "العربية" in retrieved

    return {
        "diacritics_in_source": diacritics_in_source,
        "diacritics_in_retrieved": diacritics_in_retrieved,
        "encoding_ok": encoding_ok,
        "retrieval_works": len(results) > 0,
        "chunk_count": result["chunk_count"],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python arabic_benchmark.py <path_to_document>")
        sys.exit(1)
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        sys.exit(1)
    metrics = run_arabic_benchmark(doc_path)
    print("Arabic Benchmark Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if metrics["encoding_ok"] and metrics["diacritics_in_retrieved"] > 0:
        print("\n  PASS: Arabic and diacritics supported correctly.")
    else:
        print("\n  Check: Ensure UTF-8 and diacritics preservation.")
