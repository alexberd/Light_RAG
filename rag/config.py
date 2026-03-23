import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

EmbeddingProvider = Literal["openai", "local"]


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: str | None
    embedding_provider: EmbeddingProvider
    openai_api_key: str | None
    openai_embedding_model: str
    local_embedding_model: str
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500)), #512,
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50)), #50,
    retrieve_limit: int = int(os.getenv("RETRIEVE_LIMIT", 10)), #3


def get_settings() -> Settings:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    if not qdrant_url:
        raise ValueError("QDRANT_URL cannot be empty")

    collection = os.getenv("QDRANT_COLLECTION", "chunks").strip()
    if not collection:
        raise ValueError("QDRANT_COLLECTION cannot be empty")

    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None

    provider_raw = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider_raw not in ("openai", "local"):
        raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'local'")
    provider: EmbeddingProvider = provider_raw  # type: ignore[assignment]

    return Settings(
        qdrant_url=qdrant_url,
        qdrant_collection=collection,
        qdrant_api_key=api_key,
        embedding_provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        ),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        retrieve_limit=int(os.getenv("RETRIEVE_LIMIT", "3")),
    )
