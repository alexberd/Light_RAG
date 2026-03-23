from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken


class ChunkTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        pass


class TiktokenTokenizer(ChunkTokenizer):
    """Aligns chunk windows with OpenAI embedding models (cl100k_base)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._enc.decode(token_ids)


class HuggingFaceTokenizer(ChunkTokenizer):
    """Aligns chunk windows with a HuggingFace tokenizer (e.g. BGE)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def encode(self, text: str) -> list[int]:
        # No special tokens — raw text windows for chunking
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)


def get_chunk_tokenizer(provider: str, local_model: str) -> ChunkTokenizer:
    if provider == "openai":
        return TiktokenTokenizer("cl100k_base")
    if provider == "local":
        return HuggingFaceTokenizer(local_model)
    raise ValueError(f"Unknown provider: {provider}")
