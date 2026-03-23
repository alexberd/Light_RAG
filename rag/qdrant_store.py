from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.config import Settings
from rag.embeddings import EMBED_DIM


def get_client(settings: Settings) -> QdrantClient:
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create collection (1024-dim cosine) and keyword index on doc_id for filters."""
    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{doc_id}:{chunk_index}"))


def delete_doc_chunks(
    client: QdrantClient, collection_name: str, doc_id: str
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ],
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    points = [
        PointStruct(
            id=_point_id(doc_id, i),
            vector=vectors[i],
            payload={
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunks[i],
            },
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    doc_id: Optional[str] = None,
) -> list[tuple[str, float]]:
    """
    Returns (content, score). Cosine similarity for COSINE metric (higher is better).
    """
    flt: Optional[Filter] = None
    if doc_id is not None:
        flt = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ],
        )

    # qdrant-client 1.17+: use query_points (client.search was removed)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    out: list[tuple[str, float]] = []
    for h in response.points or []:
        payload = h.payload or {}
        content = str(payload.get("content", ""))
        out.append((content, float(h.score)))
    return out
