"""Episode boundaries (PIPELINE.md steps 2 and 3).

A silence longer than the gap closes an episode (ensure_episode). A change of subject
can too (drift, below): two messages in a row unlike the conversation's recent ones.

Closing an episode is announced through an `on_close` callback with its id. The
engine doesn't know what happens next; the app passes a function that queues
consolidation, tests pass a list's append.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime

from .config import EngineConfig
from .embedding import embed_cached
from .ports import Embedder, EpisodeStore, MemoryStore
from .types import Embedding, Episode, Message, Role

OnClose = Callable[[int], None]


def close_stale_episode(
    store: EpisodeStore, now: datetime, config: EngineConfig, on_close: OnClose | None = None
) -> Episode | None:
    """Close the open episode if it has been quiet for longer than the gap. Returns
    the episode that is still open, or None. The closed episode ends at its last
    message, not at `now`."""
    current = store.current_episode()
    if current is None:
        return None
    last = current.last_message_at or current.started_at
    if now - last <= config.episode_gap:
        return current
    store.close_episode(current.id, last)
    if on_close is not None:
        on_close(current.id)
    return None


def ensure_episode(
    store: EpisodeStore, now: datetime, config: EngineConfig, on_close: OnClose | None = None
) -> Episode:
    """Return the episode a message arriving at `now` belongs to: the open one if it
    is recent enough, otherwise a new one (closing the stale one first)."""
    return close_stale_episode(store, now, config, on_close) or store.open_episode(now)


# -- Drift (PIPELINE.md step 3) -------------------------------------------------------


def drift_similarity(query: Sequence[float], recent: Sequence[Sequence[float]]) -> float:
    """How close a message is to the conversation's recent subject: cosine with the
    average of the recent messages' vectors."""
    centroid = [sum(values) / len(recent) for values in zip(*recent)]
    norm = math.sqrt(sum(c * c for c in centroid)) * math.sqrt(sum(q * q for q in query))
    return sum(c * q for c, q in zip(centroid, query)) / norm if norm else 0.0


def follow_subject(
    text: str,
    message: Message,
    query: Embedding,
    episode: Episode,
    *,
    store: MemoryStore,
    embedder: Embedder,
    config: EngineConfig,
    on_close: OnClose | None = None,
) -> Episode:
    """The episode `message` (already stored in `episode`) belongs to, after the
    drift check: the same one, or, on the second message in a row unlike the
    conversation's recent ones, a new episode starting at the first of them, which
    moves into it with everything said since (its reply included)."""
    if config.drift_threshold == 0:
        return episode
    users = [m for m in store.get_messages(episode.id) if m.role is Role.USER and m.id != message.id]
    pending = episode.drift_pending_id
    before = [m for m in users if pending is None or m.id < pending]
    similarity = None
    if len(before) >= config.drift_min_messages:
        recent = embed_cached([m.content for m in before[-config.drift_window:]], embedder, store)
        similarity = drift_similarity(query.vector, [v.vector for v in recent])
    decision, strikes = drift_step(text, similarity, episode.drift_strikes, config)
    if decision == "strike":
        store.save_drift(episode.id, strikes, pending if pending is not None else message.id)
    elif decision == "same" and (episode.drift_strikes or pending is not None):
        store.save_drift(episode.id, 0, None)
    if decision != "split":
        return episode
    start = pending if pending is not None else message.id
    everything = store.get_messages(episode.id)
    moving = [m for m in everything if m.id >= start]
    staying = [m for m in everything if m.id < start]
    store.close_episode(episode.id, staying[-1].created_at if staying else episode.started_at)
    store.save_drift(episode.id, 0, None)
    if on_close is not None:
        on_close(episode.id)
    new = store.open_episode(moving[0].created_at)
    store.move_messages([m.id for m in moving], new.id)
    [opened] = store.get_episodes([new.id])
    return opened


def carried_over(store: MemoryStore, episode: Episode, config: EngineConfig) -> list[Message]:
    """The previous episode's last messages, to lead this one's prompt for its first
    turns, when it was split off by a change of subject (it started less than a gap
    after the previous one ended): carryover_messages on the first turn, one fewer
    each turn after. A split that was wrong then still has its thread in view."""
    if config.carryover_messages == 0:
        return []
    # A split brings drift_strikes of the user's messages into the new episode, so
    # its first turn already holds that many.
    turns = sum(m.role is Role.USER for m in store.get_messages(episode.id))
    keep = config.carryover_messages - max(turns - config.drift_strikes, 0)
    if keep <= 0:
        return []
    [previous] = store.recent_episodes(1, exclude_ids=[episode.id]) or [None]
    if previous is None or previous.ended_at is None or episode.started_at - previous.ended_at >= config.episode_gap:
        return []
    return store.get_messages(previous.id, last=keep)


def drift_step(text: str, similarity: float | None, strikes: int, config: EngineConfig) -> tuple[str, int]:
    """One message through the two-strike rule: ("same", 0), ("strike", n), or
    ("split", 0). A message too short to say much, or an episode too young to have a
    subject (`similarity` None), changes nothing."""
    if similarity is None or len(text.split()) < config.drift_min_words:
        return "skip", strikes
    if similarity >= config.drift_threshold:
        return "same", 0
    if strikes + 1 >= config.drift_strikes:
        return "split", 0
    return "strike", strikes + 1
