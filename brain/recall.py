"""What a turn recalls before the prompt is built (PIPELINE.md steps 4 to 7).

One function, shared by the turn and the multi-turn eval, so the eval measures
exactly what a reply would see: the names and topics in the message, the period it
mentions, the recent conversations, and the memories retrieved with all of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo

from .activation import update_activation
from .attachments import reading_notes
from .config import EngineConfig, all_sources
from .embedding import embed_cached
from .entity_links import without_names
from .keywords import keyword_terms
from .hashing import text_hash
from .names import named_entities
from .periods import Period, find_period
from .ports import Embedder, MemoryStore, Reranker
from .retrieval import (
    RecentEpisode,
    ScoredCandidate,
    cosine,
    matched_tags,
    recent_context,
    retrieve,
    worth_recalling,
)
from .types import Candidate, CarriedMemory, Embedding, MemoryKind, Role


@dataclass(frozen=True, slots=True)
class Recall:
    memories: list[ScoredCandidate]
    recent_episodes: list[RecentEpisode] = field(default_factory=list)
    period: Period | None = None
    # Lines for the memory block about files the new message shared that can't be
    # quoted yet (still being read, unreadable).
    notes: list[str] = field(default_factory=list)


def previous_messages(store: MemoryStore, episode_id: int, count: int) -> list[str]:
    """The user's messages before the newest one in this episode (the one being
    answered, already stored), oldest first, at most `count`."""
    if count == 0:
        return []
    recent = store.get_messages(episode_id, last=4 * (count + 1))
    users = [m.content for m in recent if m.role is Role.USER][:-1]
    return users[-count:]


def conversation_query(query: Embedding, previous: Sequence[Embedding], weight: float) -> Embedding:
    """The search vector for the newest message (`query`) with the user's `previous`
    messages (oldest first) added, the k-th most recent weighted `weight`^k."""
    if weight == 0 or not previous:
        return query
    combined = list(query.vector)
    for k, vector in enumerate(reversed(previous), start=1):
        combined = [c + weight ** k * v for c, v in zip(combined, vector.vector)]
    norm = math.sqrt(sum(c * c for c in combined)) or 1.0
    return Embedding([c / norm for c in combined], query.model,
                     text_hash("\n".join([*(v.text_hash for v in previous), query.text_hash])))


def carried_candidates(store: MemoryStore, config: EngineConfig, episode_id: int) -> list[ScoredCandidate]:
    """What earlier turns of this episode recalled, re-read from the store (so a
    memory forgotten since never comes back) and faded by one turn."""
    saved = store.carried_memories(episode_id)
    scores = {(m.kind, m.id): m.score * config.carry_decay for m in saved}
    found: list[Candidate] = []
    ids = {kind: [m.id for m in saved if m.kind is kind] for kind in MemoryKind}
    for e in store.get_episodes(ids[MemoryKind.EPISODE]):
        if e.summary_short:
            found.append(Candidate(MemoryKind.EPISODE, e.id, e.summary_short, episode_id=e.id, at=e.started_at))
    for message_id in ids[MemoryKind.MESSAGE]:
        m = store.get_message(message_id)
        if m is not None:
            found.append(Candidate(MemoryKind.MESSAGE, m.id, m.content, episode_id=m.episode_id,
                                   at=m.created_at, role=m.role))
    for f in store.get_facts(ids[MemoryKind.FACT]):
        if f.status == "A" and f.superseded_by is None:
            found.append(Candidate(MemoryKind.FACT, f.id, f.statement, episode_id=f.episode_id, at=f.learned_at))
    return [ScoredCandidate(c, scores[(c.kind, c.id)]) for c in found if worth_recalling(c, config)]


def recall(
    text: str,
    query: Embedding,
    store: MemoryStore,
    config: EngineConfig,
    *,
    embedder: Embedder,
    episode_id: int,
    now: datetime,
    tz: tzinfo | None = None,
    reranker: Reranker | None = None,
) -> Recall:
    """The memory for a message `text` (embedded as `query`) sent in episode
    `episode_id`, whose own messages are left out (they're in the prompt already).
    Searches by meaning with the conversation's recent subject (conversation_query);
    names and periods come from the message alone. With `carry_decay`, what earlier
    turns recalled competes too, fading each turn, while the conversation stays on
    its subject; what this turn recalls is saved for the next one."""
    carrying = config.carry_decay > 0
    activating = any(s.w_act > 0 for s in all_sources(config))
    wanted = max(config.context_messages if config.context_weight else 0, 1 if carrying or activating else 0)
    previous = embed_cached(previous_messages(store, episode_id, wanted), embedder, store) if wanted else []
    search_query = conversation_query(
        query, previous[-config.context_messages:] if config.context_messages else [], config.context_weight
    )
    named = named_entities(text, store, config.fuzzy_names)
    period = find_period(text, now, tz)               # "in August", "last week"
    recent_episodes = recent_context(store, config, current_episode_id=episode_id)
    if period is not None:
        # Only the recent conversations from that period: the others would be
        # described as if they were from it.
        recent_episodes = [r for r in recent_episodes if period.contains(r.episode.started_at)]
    tags = matched_tags(search_query, store, config)
    # Still on the subject of the message before? Decides whether what earlier turns
    # recalled (carry-over) and what was on your mind (activation) come along.
    on_subject = not previous or cosine(query.vector, previous[-1].vector) >= config.subject_similarity
    activation = None
    if activating:
        # Step 5: what the message names and is about comes to mind, the rest fades.
        matched = {(MemoryKind.ENTITY, e.id): 1.0 for e in named}
        matched.update({(MemoryKind.TAG, t.id): similarity for t, similarity in tags})
        # A message unlike the one before has moved on: what was on your mind then
        # doesn't carry over (the new subject pushes it out). Time and turns fade
        # it otherwise.
        state = store.load_activation() if on_subject else {}
        activation = update_activation(state, matched, store, now, config)
        store.save_activation({node: (value, now) for node, value in activation.items()})

    # Entity blocks rank an entity's facts partly by the message without its name
    # (brain/retrieval.py with_entity_blocks).
    entity_query = None
    block = config.entity_block
    if block.enabled and block.without_name_weight > 0 and named:
        bare = without_names(text, named[:block.max_entity_blocks])
        if bare:
            [entity_query] = embed_cached([bare], embedder, store)

    def search(carried: Sequence[ScoredCandidate] = ()) -> list[ScoredCandidate]:
        return retrieve(
            search_query, store, config, current_episode_id=episode_id,
            shown_episode_ids={r.episode.id for r in recent_episodes}, entities=named,
            tags=tags, carried=carried, activation=activation, terms=keyword_terms(text),
            period=period, reranker=reranker, text=text, entity_query=entity_query,
        )

    memories = search()
    # What earlier turns recalled competes only while the conversation is still on
    # that subject: the new message finds nothing strong of its own (a clear subject
    # means it moved on) and it's close to the message before. Not with a period:
    # that asks about then, not about what was on the table.
    own_best = max((m.score for m in memories if m.candidate.kind is not MemoryKind.ENTITY), default=0.0)
    if carrying and period is None and previous and on_subject and own_best < config.carry_gate:
        memories = search(carried_candidates(store, config, episode_id))
    if carrying:
        store.save_carried_memories(episode_id, [
            CarriedMemory(m.candidate.kind, m.candidate.id, m.score)
            for m in memories if m.candidate.kind is not MemoryKind.ENTITY
        ])
    [newest] = store.get_messages(episode_id, last=1) or [None]
    shared = store.message_attachments([newest.id]).get(newest.id, []) if newest is not None else []
    return Recall(memories, recent_episodes, period, reading_notes(shared))
