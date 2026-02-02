"""SQL database storage using SQLite for structured metadata and relational queries."""

from pathlib import Path
from datetime import datetime
from typing import Any
import json
import uuid

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(36), nullable=False)
    source = Column(String(255), nullable=False)
    target = Column(String(255), nullable=False)
    weight = Column(Float, default=1.0)


class EntityChunk(Base):
    __tablename__ = "entity_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity = Column(String(255), nullable=False)
    chunk_id = Column(String(64), nullable=False)
    doc_id = Column(String(36), nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    format = Column(String(20), nullable=False)
    strategy = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(64), primary_key=True)
    doc_id = Column(String(36), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text_preview = Column(Text)
    meta_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class SQLDB:
    """SQLite-backed relational store for documents and chunks."""

    def __init__(self, db_path: str | Path = "documents.db"):
        self.db_path = Path(db_path) if db_path != ":memory:" else None
        url = "sqlite:///:memory:" if db_path == ":memory:" else f"sqlite:///{db_path}"
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_document(
        self,
        filename: str,
        format_type: str,
        strategy: str,
        chunks: list[dict[str, Any]],
        doc_id: str | None = None,
    ) -> str:
        """Insert document and its chunks. Returns doc_id."""
        doc_id = doc_id or str(uuid.uuid4())
        with self.Session() as session:
            doc = Document(
                id=doc_id,
                filename=filename,
                format=format_type,
                strategy=strategy,
            )
            session.add(doc)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                text_preview = chunk.get("text", "")[:500]
                meta = chunk.get("metadata", {})
                meta_json = json.dumps(meta, default=str)

                c = Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=i,
                    text_preview=text_preview,
                    meta_json=meta_json,
                )
                session.add(c)

            session.commit()
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        """Get document by ID."""
        with self.Session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                return None
            chunks = session.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index).all()
            return {
                "id": doc.id,
                "filename": doc.filename,
                "format": doc.format,
                "strategy": doc.strategy,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "chunks": [
                    {
                        "id": c.id,
                        "chunk_index": c.chunk_index,
                        "text_preview": c.text_preview,
                        "metadata": json.loads(c.meta_json) if c.meta_json else {},
                    }
                    for c in chunks
                ],
            }

    def list_documents(self) -> list[dict]:
        """List all documents."""
        with self.Session() as session:
            docs = session.query(Document).order_by(Document.created_at.desc()).all()
            return [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "format": d.format,
                    "strategy": d.strategy,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in docs
            ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete document, chunks, and graph edges."""
        with self.Session() as session:
            session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
            session.query(GraphEdge).filter(GraphEdge.doc_id == doc_id).delete()
            session.query(EntityChunk).filter(EntityChunk.doc_id == doc_id).delete()
            deleted = session.query(Document).filter(Document.id == doc_id).delete()
            session.commit()
            return deleted > 0

    def add_graph_edges(self, doc_id: str, edges: list[tuple[str, str, float]]) -> None:
        """Insert graph edges from co-occurrence."""
        with self.Session() as session:
            for source, target, weight in edges:
                e = GraphEdge(doc_id=doc_id, source=source, target=target, weight=weight)
                session.add(e)
            session.commit()

    def add_entity_chunks(self, doc_id: str, entity_chunks: dict[str, list[str]]) -> None:
        """Insert entity->chunk_id mappings."""
        with self.Session() as session:
            for entity, chunk_ids in entity_chunks.items():
                for cid in chunk_ids:
                    ec = EntityChunk(entity=entity, chunk_id=cid, doc_id=doc_id)
                    session.add(ec)
            session.commit()

    def get_chunk_ids_for_entities(
        self, entities: list[str], limit: int = 20, doc_id: str | None = None
    ) -> list[str]:
        """Get chunk IDs that contain any of the given entities. Optionally filter by doc_id."""
        if not entities:
            return []
        with self.Session() as session:
            q = session.query(EntityChunk.chunk_id).filter(EntityChunk.entity.in_(entities))
            if doc_id:
                q = q.filter(EntityChunk.doc_id == doc_id)
            rows = q.limit(limit).all()
            return list(dict.fromkeys(r.chunk_id for r in rows))

    def get_related_entities(
        self, entity: str, top_k: int = 10, doc_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Get entities related to the given entity by graph edges. Optionally filter by doc_id."""
        with self.Session() as session:
            q_source = (
                session.query(GraphEdge.target, GraphEdge.weight)
                .filter(GraphEdge.source == entity)
            )
            if doc_id:
                q_source = q_source.filter(GraphEdge.doc_id == doc_id)
            rows = q_source.order_by(GraphEdge.weight.desc()).limit(top_k).all()
            result = [(r.target, r.weight) for r in rows]
            q_target = (
                session.query(GraphEdge.source, GraphEdge.weight)
                .filter(GraphEdge.target == entity)
            )
            if doc_id:
                q_target = q_target.filter(GraphEdge.doc_id == doc_id)
            rows2 = q_target.order_by(GraphEdge.weight.desc()).limit(top_k).all()
            seen = {entity}
            for r in rows2:
                if r.source not in seen:
                    result.append((r.source, r.weight))
                    seen.add(r.source)
            result.sort(key=lambda x: -x[1])
            return result[:top_k]
