"""Vector database storage using ChromaDB for semantic search."""

from pathlib import Path
from typing import Any
import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDB:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.persist_directory = str(persist_directory) if persist_directory else None
        self.embedding_model_name = embedding_model
        self._model: SentenceTransformer | None = None
        if persist_directory:
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection_name = "documents"
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._raptor_collection = self._client.get_or_create_collection(
            name="raptor",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        doc_id: str | None = None,
    ) -> list[str]:
        """Add chunks to the vector DB. Returns list of chunk IDs."""
        if not chunks:
            return []

        doc_id = doc_id or str(uuid.uuid4())
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts).tolist()

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            ids.append(chunk_id)
            meta = chunk.get("metadata", {}).copy()
            meta["doc_id"] = doc_id
            meta["chunk_index"] = i
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    meta[k] = str(v)
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    def search(self, query: str, top_k: int = 5, doc_id: str | None = None) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {text, metadata, distance}. Optionally filter by doc_id."""
        query_embedding = self.model.encode([query]).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        count = self._collection.count() or 1
        if doc_id:
            try:
                existing = self._collection.get(where={"doc_id": doc_id})
                count = len(existing["ids"]) if existing and existing["ids"] else 0
            except Exception:
                count = 0
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            where=where,
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, cid in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results.get("distances") else None
            output.append({
                "id": cid,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })
        return output

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch chunks by their IDs."""
        if not ids:
            return []
        try:
            result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        except Exception:
            return []
        if not result or not result["ids"]:
            return []
        output = []
        for i, cid in enumerate(result["ids"]):
            doc = result["documents"][i] if result.get("documents") else ""
            meta = result["metadatas"][i] if result.get("metadatas") else {}
            output.append({"id": cid, "text": doc, "metadata": meta})
        return output

    def add_raptor_nodes(self, nodes: list[dict[str, Any]]) -> list[str]:
        """Add RAPTOR tree nodes to the raptor collection."""
        if not nodes:
            return []
        texts = [n["text"] for n in nodes]
        embeddings = self.model.encode(texts).tolist()
        ids = []
        metadatas = []
        for n in nodes:
            cid = n.get("chunk_id", str(len(ids)))
            ids.append(cid)
            meta = n.get("metadata", {}).copy()
            clean = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (list, dict)):
                    clean[k] = str(v)
                elif isinstance(v, (int, float)) and k == "raptor_level":
                    clean[k] = str(v)
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            metadatas.append(clean)
        self._raptor_collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return ids

    def search_raptor(self, query: str, top_k: int = 5, level: int | None = None) -> list[dict[str, Any]]:
        """Search RAPTOR collection. Optionally filter by level."""
        coll = self._raptor_collection
        count = coll.count() or 0
        if count == 0:
            return []
        where = {"raptor_level": str(level)} if level is not None else None
        query_emb = self.model.encode([query]).tolist()
        results = coll.query(
            query_embeddings=query_emb,
            n_results=min(top_k, count),
            where=where,
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        output = []
        for i, cid in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results.get("distances") else None
            output.append({"id": cid, "text": doc, "metadata": meta, "distance": dist})
        return output

    def delete_raptor_by_doc_id(self, doc_id: str) -> None:
        """Remove RAPTOR nodes for a document."""
        try:
            existing = self._raptor_collection.get(where={"doc_id": doc_id})
            if existing and existing["ids"]:
                self._raptor_collection.delete(ids=existing["ids"])
        except Exception:
            pass

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks and RAPTOR nodes belonging to a document."""
        existing = self._collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        self.delete_raptor_by_doc_id(doc_id)
