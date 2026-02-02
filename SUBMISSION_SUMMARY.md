# Pyxon AI Entry Task — Submission Summary

**My implementation: AI-Powered Document Parser with Advanced RAG**

---

## Overview

I built an end-to-end document processing and retrieval pipeline for PDF, DOCX, and TXT files. It includes intelligent chunking, dual storage (vector + SQL), GraphRAG, RAPTOR, and full Arabic support with diacritics. It is designed for integration with RAG applications and includes a benchmark suite and interactive demo.

---

## Features

### Core (Required)

| Feature                                        | Status |
| ---------------------------------------------- | ------ |
| PDF, DOCX, TXT parsing                         | ✅     |
| Content analysis & chunking strategy selection | ✅     |
| Fixed and dynamic chunking                     | ✅     |
| Vector DB (ChromaDB)                           | ✅     |
| SQL DB (SQLite)                                | ✅     |
| Arabic language support                        | ✅     |
| Arabic diacritics support                      | ✅     |
| Benchmark suite                                | ✅     |
| RAG integration ready                          | ✅     |

### Advanced (Bonus)

| Feature                                           | Status |
| ------------------------------------------------- | ------ |
| GraphRAG (entity graph, expansion retrieval)      | ✅     |
| RAPTOR (hierarchical retrieval)                   | ✅     |
| 4 retrieval modes (vector, graph, raptor, hybrid) | ✅     |
| OpenAI LLM answer generation                      | ✅     |
| OCR fallback for scanned PDFs                     | ✅     |
| Arabic stopwords filtering ([linuxscout/arabicstopwords](https://github.com/linuxscout/arabicstopwords)) | ✅ |

---

## Architecture

```
Document (PDF/DOCX/TXT)
        │
        ▼
   Unified Parser ──────────────────────────────────────┐
        │                                               │
        ▼                                               │
   Content Analyzer → Chunking Strategy (fixed/dynamic)  │
        │                                               │
        ▼                                               │
   Chunker ─────────────────────────────────────────────┤
        │                                               │
        ├──► Vector DB (ChromaDB) ◄── RAPTOR nodes      │
        │                                               │
        └──► SQL DB (SQLite) ◄── GraphRAG edges         │
        │                                               │
        ▼                                               │
   User Query ──► Retrieval Engine ──► Vector / Graph / RAPTOR / Hybrid
        │                                               │
        ▼                                               │
   Retrieved Chunks ──► (Optional) OpenAI LLM ──► Answer │
```

---

## Technologies

| Component         | Choice                                                        |
| ----------------- | ------------------------------------------------------------- |
| Parsing           | PyMuPDF, python-docx, pytesseract (OCR)                       |
| NLP / Embeddings  | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Entity Extraction | spaCy (EN), [Arabic-Stopwords](https://github.com/linuxscout/arabicstopwords) (13k+ forms), regex (AR) |
| Vector DB         | ChromaDB                                                      |
| SQL DB            | SQLite, SQLAlchemy                                            |
| Clustering        | scikit-learn (RAPTOR)                                         |
| LLM               | OpenAI gpt-4o-mini (optional)                                 |
| Demo              | Streamlit                                                     |

---

## How to Run

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Optional:** Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) for scanned PDF OCR.

### Demo

```bash
streamlit run demo/app.py
```

- Upload PDF, DOCX, or TXT
- Search with vector, graph, raptor, or hybrid mode
- Generate answers with OpenAI (shared demo key; no sign-up required)
- Try `sample_docs/arabic_with_harakat.txt` for Arabic
- **Note:** The demo OpenAI API key will be revoked after 14 days.

### Benchmarks

```bash
python benchmarks/run_all.py
python benchmarks/run_all.py path/to/document.pdf
```

### Programmatic Use

```python
from src.document_processor import DocumentProcessor
from src.rag_client import RAGClient
from src.storage import VectorDB, SQLDB

processor = DocumentProcessor()
result = processor.process("document.pdf")

rag = RAGClient(processor.vector_db, processor.sql_db)
chunks = rag.retrieve("your query", mode="hybrid", top_k=5)
answer = rag.answer("question?", use_llm=True, api_key="sk-...")
```

---

## Retrieval Modes

| Mode             | Best for                                    |
| ---------------- | ------------------------------------------- |
| **vector** | General semantic search                     |
| **graph**  | Relationship, entity, or connection queries |
| **raptor** | Overview, summarization, main points        |
| **hybrid** | Combined vector + graph retrieval           |

---

## Benchmark Highlights

### English Article (~19 chunks)

| Metric          | Value  |
| --------------- | ------ |
| Precision@k     | 1.0    |
| Recall@k        | 1.0    |
| MRR             | 0.73   |
| GraphRAG edges  | 1,684  |
| Processing time | ~11 s  |
| Peak memory     | ~76 MB |

### Arabic Document (1,500 words, harakat)

| Metric         | Value        |
| -------------- | ------------ |
| Precision@k    | 0.75         |
| Recall@k       | 0.75         |
| MRR            | 0.38         |
| GraphRAG edges | 152          |
| Diacritics     | Preserved ✓ |

---

## Design Decisions

| Decision                            | Rationale                                    |
| ----------------------------------- | -------------------------------------------- |
| Heuristic chunking (no LLM)         | Fast, deterministic, offline                 |
| spaCy + regex entities (no LLM NER) | Lower cost, faster batch processing          |
| K-means RAPTOR (no LLM summaries)   | No API cost, deterministic                   |
| Keyword-based benchmark             | Document-agnostic, reproducible              |
| ChromaDB + SQLite                   | Minimal setup, good for demo and development |

---

## References

- **Arabic Stopwords**: [linuxscout/arabicstopwords](https://github.com/linuxscout/arabicstopwords) — Classified Arabic stop word list with 13,000+ inflected forms, used for GraphRAG entity filtering.

---

## Limitations

- DOCX supported; legacy .doc not supported
- Tesseract required for OCR on scanned PDFs
- Large files (>100MB) may cause memory pressure
- Benchmark uses self-retrieval; real user queries may differ
- GraphRAG filters rare entities (≥2 occurrences)

---

## Demo API Key

I provide a shared OpenAI API key so reviewers can test **Generate Answer** without signing up. **The key will be revoked after 14 days** for security.

---

## Contact

- **Email:** khouriferas74@gmail.com
- **Phone:** 00962772462582
- **Demo:** [https://pyxon-ai-entry-task-feras-khouri.streamlit.app/](https://pyxon-ai-entry-task-feras-khouri.streamlit.app/)

---

*Pyxon AI Junior Engineer Entry Task — February 2025*
