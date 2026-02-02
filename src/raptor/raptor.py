"""RAPTOR: Hierarchical tree from clustering on embeddings (no LLM summarization)."""

from typing import Any

import numpy as np


def build_raptor_tree(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    doc_id: str,
    n_clusters: int = 4,
) -> list[dict[str, Any]]:
    """
    Build 2-level RAPTOR tree from chunks.
    Level 0: leaf chunks. Level 1: cluster representatives.

    Returns list of nodes to store in vector DB, each with text, metadata.raptor_level.
    """
    from sklearn.cluster import KMeans

    if not chunks or not embeddings:
        return []

    emb = np.array(embeddings, dtype=np.float32)
    n = len(chunks)
    nodes: list[dict[str, Any]] = []

    for i, (chunk, _) in enumerate(zip(chunks, embeddings)):
        nodes.append({
            "text": chunk.get("text", ""),
            "chunk_id": f"{doc_id}_{i}",
            "metadata": {
                "doc_id": doc_id,
                "raptor_level": 0,
                "chunk_index": i,
            },
        })

    if n <= n_clusters:
        return nodes

    kmeans = KMeans(n_clusters=min(n_clusters, n), random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb)
    centroids = kmeans.cluster_centers_

    for c in range(min(n_clusters, n)):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        centroid = centroids[c]
        cluster_embeddings = emb[indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        nearest_idx = indices[np.argmin(distances)]
        rep_chunk = chunks[nearest_idx]
        summary_id = f"{doc_id}_L1_{c}"
        nodes.append({
            "text": rep_chunk.get("text", ""),
            "chunk_id": summary_id,
            "metadata": {
                "doc_id": doc_id,
                "raptor_level": 1,
                "represents": str([f"{doc_id}_{i}" for i in indices]),
            },
        })

    return nodes


def build_raptor_tree_with_model(
    chunks: list[dict[str, Any]],
    embedding_model,
    doc_id: str,
    n_clusters: int = 4,
) -> list[dict[str, Any]]:
    """Build RAPTOR tree using an embedding model."""
    texts = [c.get("text", "") for c in chunks]
    embeddings = embedding_model.encode(texts).tolist()
    return build_raptor_tree(chunks, embeddings, doc_id, n_clusters)
