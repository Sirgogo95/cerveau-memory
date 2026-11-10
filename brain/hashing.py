"""The one definition of a text's hash, shared by the embedding cache (key is
`(model, text_hash)`) and the embedding tables (staleness checks)."""

from __future__ import annotations

import hashlib


def text_hash(text: str) -> str:
    """sha256 of the text with surrounding whitespace removed. The model is not part
    of the hash; it is stored and looked up alongside it."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def is_stale(stored_hash: str, current_text: str) -> bool:
    """True when a vector was embedded from a different text than the current one."""
    return stored_hash != text_hash(current_text)
