from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rag.chunking import chunk_text
from rag.config import Settings, get_settings
from rag.embeddings import get_embedder
from rag.qdrant_store import (
    delete_doc_chunks,
    ensure_collection,
    get_client,
    search,
    upsert_chunks,
)
from rag.tokenizers import get_chunk_tokenizer


@dataclass(frozen=True)
class RetrievalHit:
    content: str
    score: float


def ingest(
    doc_id: str,
    text: str,
    *,
    settings: Optional[Settings] = None,
    replace_existing: bool = True,
) -> int:
    """
    Chunk `text`, embed each chunk, upsert into Qdrant. If `replace_existing`,
    deletes prior points for `doc_id` first.
    """
    s = settings or get_settings()
    tokenizer = get_chunk_tokenizer(s.embedding_provider, s.local_embedding_model)
    chunks = chunk_text(
        text, tokenizer, chunk_size=s.chunk_size, overlap=s.chunk_overlap
    )
    if not chunks:
        return 0

    embedder = get_embedder(s.embedding_provider, s)
    vectors = [embedder.embed(c) for c in chunks]

    client = get_client(s)
    ensure_collection(client, s.qdrant_collection)
    if replace_existing:
        delete_doc_chunks(client, s.qdrant_collection, doc_id)
    upsert_chunks(client, s.qdrant_collection, doc_id, chunks, vectors)
    return len(chunks)


def retrieve(
    query: str,
    *,
    settings: Optional[Settings] = None,
    limit: Optional[int] = None,
    doc_id: Optional[str] = None,
) -> list[RetrievalHit]:
    """
    Embed `query`, cosine ANN search in Qdrant. Optional `doc_id` filters to one document.
    """
    s = settings or get_settings()
    lim = limit if limit is not None else s.retrieve_limit
    embedder = get_embedder(s.embedding_provider, s)
    q_vec = embedder.embed(query)

    client = get_client(s)
    rows = search(client, s.qdrant_collection, q_vec, lim, doc_id=doc_id)
    return [RetrievalHit(content=content, score=score) for content, score in rows]


def retrieve_dicts(
    query: str,
    *,
    settings: Optional[Settings] = None,
    limit: Optional[int] = None,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """Same as `retrieve`, using dicts like the spec examples."""
    hits = retrieve(query, settings=settings, limit=limit, doc_id=doc_id)
    return [{"content": h.content, "score": h.score} for h in hits]
