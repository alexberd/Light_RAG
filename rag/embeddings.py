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
    def embed(self, text: str) -> list[float]:
        ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
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

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        v = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = v.tolist() if hasattr(v, "tolist") else list(v)  # type: ignore[arg-type]
        if len(vec) != EMBED_DIM:
            raise ValueError(f"Expected {EMBED_DIM} dimensions, got {len(vec)}")
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
