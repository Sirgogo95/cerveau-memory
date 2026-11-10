"""Embedding through the cache (PIPELINE.md step 1): the same text under the same
model is never embedded twice."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .hashing import text_hash
from .ports import Embedder, EmbeddingCacheStore
from .types import Embedding


def embed_cached(
    texts: Sequence[str], embedder: Embedder, cache: EmbeddingCacheStore
) -> list[Embedding]:
    """One Embedding per text, in order, each carrying its text_hash. Cache misses
    are embedded in one batch (duplicates once) and written back to the cache."""
    keys = [text_hash(t) for t in texts]
    found = cache.get_cached(list(dict.fromkeys(keys)), embedder.model)

    missing: dict[str, str] = {}  # key -> text, first occurrence of each
    for key, text in zip(keys, texts):
        if key not in found:
            missing.setdefault(key, text)
    if missing:
        fresh = embedder.embed(list(missing.values()))
        new = {
            key: replace(embedding, text_hash=key)
            for key, embedding in zip(missing, fresh)
        }
        cache.put_cached(new)
        found = {**found, **new}

    return [found[key] for key in keys]
