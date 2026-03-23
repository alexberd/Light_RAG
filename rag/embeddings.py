from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from rag.config import Settings

EMBED_DIM = 1024


def l2_normalize(vec: list[float]) -> list[float]:
    """Cosine similarity in Qdrant assumes unit vectors; keep outputs normalized."""
    a = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        raise ValueError("Zero-norm embedding cannot be normalized")
    return (a / n).tolist()


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str, *, for_query: bool = False) -> list[float]:
        ...


# BAAI/bge-m3 dense retrieval prompts (https://huggingface.co/BAAI/bge-m3)
_BGE_M3_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_M3_DOC_PREFIX = "Represent this document for retrieval: "


def _flatten_embedding_vector(v: object) -> list[float]:
    """Sentence-transformers may return (dim,) or (1, dim); normalize to a flat list."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D embedding, got shape {arr.shape}")
    return arr.tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str, *, for_query: bool = False) -> list[float]:
        r = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=EMBED_DIM,
        )
        vec = list(r.data[0].embedding)
        if len(vec) != EMBED_DIM:
            raise ValueError(f"Expected {EMBED_DIM} dimensions, got {len(vec)}")
        return l2_normalize(vec)


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._use_bge_m3_prompts = "bge-m3" in model_name.lower()
        self._model = SentenceTransformer(model_name)

    def _prepare_text(self, text: str, *, for_query: bool) -> str:
        if self._use_bge_m3_prompts:
            return (
                _BGE_M3_QUERY_PREFIX + text
                if for_query
                else _BGE_M3_DOC_PREFIX + text
            )
        return text

    def embed(self, text: str, *, for_query: bool = False) -> list[float]:
        to_encode = self._prepare_text(text, for_query=for_query)
        v = self._model.encode(
            to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = _flatten_embedding_vector(v)
        if len(vec) != EMBED_DIM:
            raise ValueError(
                f"Expected {EMBED_DIM} dimensions, got {len(vec)} "
                f"(model={self._model_name!r})"
            )
        return l2_normalize(vec)


def get_embedder(provider: str, settings: "Settings") -> Embedder:
    s = settings
    if provider == "openai":
        if not s.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        return OpenAIEmbedder(s.openai_api_key, s.openai_embedding_model)
    if provider == "local":
        return LocalEmbedder(s.local_embedding_model)
    raise ValueError(f"Unknown embedding provider: {provider}")
