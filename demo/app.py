"""Streamlit demo for the AI document parser."""

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.document_processor import DocumentProcessor
from src.rag_client import RAGClient
from src.storage import VectorDB, SQLDB


def init_session():
    """Use session-scoped storage so each browser session gets a fresh DB (fixes old files on Streamlit deploy)."""
    if "processor" not in st.session_state:
        if "_storage_id" not in st.session_state:
            st.session_state._storage_id = str(uuid.uuid4())
        data_dir = Path(tempfile.gettempdir()) / f"pyxon_docs_{st.session_state._storage_id}"
        data_dir.mkdir(parents=True, exist_ok=True)
        vector_path = data_dir / "chroma"
        vector_path.mkdir(exist_ok=True)
        db_path = data_dir / "documents.db"
        vector_db = VectorDB(persist_directory=str(vector_path))
        sql_db = SQLDB(db_path=str(db_path))
        st.session_state.vector_db = vector_db
        st.session_state.sql_db = sql_db
        st.session_state.processor = DocumentProcessor(vector_db=vector_db, sql_db=sql_db)
        st.session_state.rag = RAGClient(vector_db, sql_db)
    if "current_doc_id" not in st.session_state:
        st.session_state.current_doc_id = None


def main():
    st.set_page_config(page_title="Pyxon AI Document Parser", page_icon="📄", layout="wide")
    st.title("AI Document Parser")
    st.caption("PDF, DOCX, DOC, TXT | GraphRAG | RAPTOR | Vector + SQL | Arabic + Harakat support")

    init_session()
    processor = st.session_state.processor
    rag = st.session_state.rag
    selected_doc_id = None

    with st.sidebar:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "") or ""
        except Exception:
            api_key = ""
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["vector", "graph", "raptor", "hybrid"],
            help="Choose how to retrieve: meaning (vector), relations (graph), overview (raptor), or combined (hybrid).",
        )
        with st.expander("What each mode does"):
            st.markdown("""
            **Vector:** Retrieves the same relevant text even when you ask in different wording — it matches *meaning*, not just keywords.

            **Graph:** Extracts entities and relationships (e.g. people, organizations, “X reports to Y”) and returns supporting quotes from the document as evidence.

            **Raptor:** Builds a faithful hierarchical summary of the relevant section(s), preserving key named entities (people, organizations, places) instead of dropping them.

            **Hybrid:** Combines vector + graph into one coherent answer: a direct response, quoted excerpts, and extracted relationships, with no contradictions between the answer, quotes, and relations.
            """)
        docs = rag.list_documents()
        doc_options = ["All documents"] + [f"{d['filename']} ({d['id'][:8]}…)" for d in docs]
        doc_id_map = {f"{d['filename']} ({d['id'][:8]}…)": d["id"] for d in docs}
        default_idx = 0
        if st.session_state.current_doc_id and docs:
            for i, d in enumerate(docs):
                if d["id"] == st.session_state.current_doc_id:
                    default_idx = i + 1  # +1 because "All documents" is index 0
                    break
        selected_label = st.selectbox(
            "Search in document",
            doc_options,
            index=min(default_idx, len(doc_options) - 1) if doc_options else 0,
            help="Choose which document to search. 'All documents' uses everything in this session.",
        )
        selected_doc_id = None if selected_label == "All documents" else (doc_id_map.get(selected_label) or None)
        if selected_doc_id is not None:
            selected_doc_id = str(selected_doc_id).strip() or None

    tab1, tab3, tab4, tab5 = st.tabs(["Upload", "Search", "Generate Answer", "Documents"])

    with tab1:
        st.subheader("Upload Document")
        uploaded = st.file_uploader("Choose PDF, DOCX, DOC, or TXT", type=["pdf", "docx", "doc", "txt"])
        if uploaded:
            with st.spinner("Processing..."):
                tmp = Path("temp_upload") / uploaded.name
                tmp.parent.mkdir(exist_ok=True)
                tmp.write_bytes(uploaded.getvalue())
                try:
                    result = processor.process(tmp)
                    tmp.unlink(missing_ok=True)
                    st.session_state.current_doc_id = result["doc_id"]
                    extra = f" | Graph edges: {result.get('graph_edges', 0)}" if result.get("graph_edges") else ""
                    st.success(f"Processed: {result['filename']} | Strategy: {result['strategy']} | Chunks: {result['chunk_count']}{extra}")
                except ValueError as e:
                    st.error(f"Processing failed: {e}")
                    tmp.unlink(missing_ok=True)
                except FileNotFoundError as e:
                    st.error(f"File error: {e}")
                    tmp.unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"Unexpected error: {type(e).__name__}: {e}")
                    tmp.unlink(missing_ok=True)

    with tab3:
        st.subheader("Semantic Search")
        st.caption(
            "Uses the retrieval mode selected in the sidebar. See **What each mode does** in the sidebar for details."
        )
        query = st.text_input("Enter query (English or Arabic)", key="search_query")
        if query:
            try:
                with st.spinner(f"Searching with **{retrieval_mode}** mode..."):
                    results = rag.retrieve(
                        query, mode=retrieval_mode, top_k=8, doc_id=selected_doc_id
                    )
            except Exception as e:
                st.error(f"Search error: {type(e).__name__}: {e}")
                results = []
            if results:
                for i, r in enumerate(results, 1):
                    dist = r.get("distance")
                    dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
                    with st.expander(f"Result {i} (distance: {dist_str})"):
                        text = r.get("text", "")
                        st.markdown(f'<p dir="auto" lang="ar">{text}</p>' if _is_arabic(text) else text, unsafe_allow_html=True)
            else:
                st.info("No results. Upload a document first, or try a different query or mode.")

    with tab4:
        st.subheader("Generate Answer (OpenAI)")
        if api_key:
            st.caption("_Uses demo API key. Key will be revoked after 14 days._")
        q = st.text_input("Question", key="answer_query")
        if q:
            if api_key and api_key.strip():
                with st.spinner("Generating answer..."):
                    try:
                        out = rag.answer(
                            q,
                            use_llm=True,
                            api_key=api_key.strip(),
                            mode=retrieval_mode,
                            doc_id=selected_doc_id,
                        )
                        if "Error:" in out["answer"]:
                            st.error(out["answer"])
                        else:
                            st.write(out["answer"])
                        with st.expander("Sources"):
                            for s in out.get("sources", [])[:5]:
                                st.caption(s.get("text", "")[:150] + "...")
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
            else:
                st.info("OpenAI API key not configured. For local runs, add OPENAI_API_KEY to .streamlit/secrets.toml")

    with tab5:
        st.subheader("Stored Documents")
        docs = rag.list_documents()
        if docs:
            for d in docs:
                st.write(f"- **{d['filename']}** | {d['format']} | {d['strategy']} | ID: `{d['id']}`")
        else:
            st.info("No documents yet. Upload or crawl in the first tabs.")

def _is_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


if __name__ == "__main__":
    main()
