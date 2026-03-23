from __future__ import annotations

import re

from rag.tokenizers import ChunkTokenizer

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s+|$)")


def _sentence_end_char_positions(text: str) -> list[int]:
    """Character offsets in `text` immediately after sentence-ending punctuation."""
    if not text.strip():
        return []
    ends: list[int] = []
    for m in _SENTENCE_END.finditer(text):
        ends.append(m.end())
    if not ends:
        ends = [len(text)]
    return sorted(set(ends))


def _token_boundaries_for_sentence_ends(
    text: str, sentence_ends_char: list[int], tokenizer: ChunkTokenizer
) -> list[int]:
    """Map each sentence end to a token index in the full-text encoding."""
    boundaries: list[int] = []
    for c in sentence_ends_char:
        n = len(tokenizer.encode(text[:c]))
        if n > 0:
            boundaries.append(n)
    return sorted(set(boundaries))


def chunk_text(
    text: str,
    tokenizer: ChunkTokenizer,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Sentence-aware chunking: prefer ends at sentence boundaries; if no boundary
    falls inside (start, start + chunk_size], cut at start + chunk_size (hard-cut),
    including when a single sentence is longer than the limit.

    Overlap is applied in token space between consecutive windows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    sentence_ends = _sentence_end_char_positions(text)
    boundaries = _token_boundaries_for_sentence_ends(text, sentence_ends, tokenizer)

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        max_end = min(start + chunk_size, len(tokens))
        candidates = [b for b in boundaries if start < b <= max_end]
        end = max(candidates) if candidates else max_end

        if end <= start:
            end = min(start + chunk_size, len(tokens))

        chunks.append(tokenizer.decode(tokens[start:end]))

        if end >= len(tokens):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks
