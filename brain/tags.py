"""Topic tags: cleaning the names a model wrote, and turning them into tags, the same
way for conversations (brain/consolidation.py) and shared files
(brain/attachments.py)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .embedding import embed_cached
from .ports import Embedder, MemoryStore
from .text import normalize_key
from .types import Tag

MAX_TAGS = 5
MAX_TAG_LENGTH = 40


def clean_tags(raw: Sequence[object]) -> tuple[str, ...]:
    """Lowercase, trimmed, deduplicated, at most MAX_TAGS."""
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = " ".join(str(item).strip().strip(".,;:#").lower().split())
        if tag and len(tag) <= MAX_TAG_LENGTH and normalize_key(tag) not in seen:
            seen.add(normalize_key(tag))
            tags.append(tag)
    return tuple(tags[:MAX_TAGS])



@dataclass
class ResolvedTags:
    tags: list[Tag] = field(default_factory=list)              # distinct, in order
    merged: list[tuple[str, str]] = field(default_factory=list)  # (new name, existing it became)
    skipped: list[str] = field(default_factory=list)            # forgotten, not relearned


def resolve_tags(
    names: Sequence[str], store: MemoryStore, embedder: Embedder, tag_same_score: float | None
) -> ResolvedTags:
    """The tags these names stand for: an existing tag that means the same (at
    `tag_same_score` or above) is reused, so "coding" and "programming" don't split
    one subject; a name close to a forgotten tag is dropped, so forgetting holds;
    otherwise the tag is created with its vector. Shared by consolidation and
    attachments."""
    resolved = ResolvedTags()
    vectors = embed_cached(list(names), embedder, store) if names else []
    for name, vector in zip(names, vectors):
        tag = None
        if tag_same_score is not None:
            [closest] = store.similar_tags(vector, 1) or [(None, 0.0)]
            existing, similarity = closest
            if existing is not None and similarity >= tag_same_score and normalize_key(existing.name) != normalize_key(name):
                if existing.status == "Z":
                    resolved.skipped.append(name)
                    continue
                tag = existing
                resolved.merged.append((name, existing.name))
        if tag is None:
            tag = store.upsert_tag(name, vector)
        if tag is None:
            resolved.skipped.append(name)
            continue
        if tag.id not in {t.id for t in resolved.tags}:
            resolved.tags.append(tag)
    return resolved
