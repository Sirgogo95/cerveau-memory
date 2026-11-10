"""InMemoryStore: the brain/ports.py contracts on plain dicts.

The reference implementation: fast enough for engine tests and eval runs, and the
spec DjangoMemoryStore must match (both run tests/test_store_contract.py).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from brain.ports import MemoryStore
from brain.text import identity_keys, normalize_key, search_keys
from brain.types import (
    Attachment,
    Candidate,
    CarriedMemory,
    Chunk,
    Node,
    Embedding,
    Entity,
    Episode,
    Fact,
    MemoryKind,
    Message,
    Role,
    Tag,
    ToolRun,
)


def _within(moment: datetime | None, after: datetime | None, before: datetime | None) -> bool:
    """In the time window: `after` inclusive, `before` exclusive (an undated memory is
    only in an open window)."""
    if moment is None:
        return after is None and before is None
    return (after is None or moment >= after) and (before is None or moment < before)


def _recallable(fact: Fact) -> bool:
    return fact.status == "A" and fact.superseded_by is None


def _fact_candidate(fact: Fact, semantic: float | None = None, lexical: float | None = None) -> Candidate:
    return Candidate(
        kind=MemoryKind.FACT, id=fact.id, text=fact.statement, semantic=semantic,
        lexical=lexical, episode_id=fact.episode_id, at=fact.learned_at,
    )


def _chain(fact_id: int, facts: Mapping[int, Fact]) -> list[Fact]:
    """The supersession chain through `fact_id`, oldest first."""
    if fact_id not in facts:
        return []
    replaced = {f.superseded_by: f.id for f in facts.values() if f.superseded_by is not None}
    first = fact_id
    while first in replaced:          # walk back to the oldest
        first = replaced[first]
    chain, current, seen = [], first, set()
    while current is not None and current not in seen and current in facts:
        seen.add(current)
        chain.append(facts[current])
        current = facts[current].superseded_by
    return chain


class _Ids:
    """The next id of a table, like itertools.count but copyable, so a filled store
    can be copied (copy.deepcopy) to run something on it without changing it."""

    def __init__(self) -> None:
        self.next_id = 1

    def __next__(self) -> int:
        value, self.next_id = self.next_id, self.next_id + 1
        return value


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryStore(MemoryStore):
    def __init__(self):
        self._episodes: dict[int, Episode] = {}
        self._messages: dict[int, Message] = {}
        self._message_embeddings: dict[int, Embedding] = {}
        self._episode_embeddings: dict[int, tuple[Embedding, Embedding]] = {}
        self._cache: dict[tuple[str, str], Embedding] = {}  # (model, text_hash) -> vector
        self._episode_ids = _Ids()
        self._message_ids = _Ids()
        # Tags and entities keep a status like the Django models: "A" active, "I" set
        # aside by the system, "Z" forgotten by the user. set_status() is for tests.
        self._tags: dict[int, Tag] = {}
        self._tag_status: dict[int, str] = {}
        self._episode_tags: dict[tuple[int, int], float] = {}
        self._entities: dict[int, Entity] = {}
        self._entity_status: dict[int, str] = {}
        self._tool_calls: list[tuple[int, ToolRun]] = []
        self._mentions: set[tuple[int, int]] = set()
        self._tag_ids = _Ids()
        self._entity_ids = _Ids()
        # Facts keep their status in the record itself (Fact.status).
        self._facts: dict[int, Fact] = {}
        self._fact_entities: dict[int, set[int]] = {}   # fact id -> entities it's about
        self._fact_embeddings: dict[int, Embedding] = {}
        self._tag_embeddings: dict[int, Embedding] = {}
        self._carried: dict[int, list[CarriedMemory]] = {}
        self._imports: dict[tuple[str, str], dict[str, int]] = {}   # (source, conversation) -> ids
        self._activation: dict[Node, tuple[float, datetime]] = {}
        self._fact_ids = _Ids()
        self._attachments: dict[int, Attachment] = {}
        self._attachment_hashes: dict[str, int] = {}
        self._attachment_links: set[tuple[int, int]] = set()   # (message, attachment)
        self._attachment_tags: set[tuple[int, int]] = set()
        self._chunks: dict[int, Chunk] = {}
        self._chunk_embeddings: dict[int, Embedding] = {}
        self._chunk_mentions: set[tuple[int, int]] = set()     # (entity, chunk)
        self._attachment_ids = _Ids()
        self._chunk_ids = _Ids()

    # -- AttachmentStore --------------------------------------------------------

    def _searchable(self, attachment_id: int) -> bool:
        a = self._attachments.get(attachment_id)
        return a is not None and a.status == "A" and a.extraction_status == "done"

    def _chunk_candidate(self, chunk: Chunk, **signals) -> Candidate:
        a = self._attachments[chunk.attachment_id]
        label = ", ".join(p for p in (a.filename, chunk.pages) if p)
        return Candidate(kind=MemoryKind.ATTACHMENT, id=chunk.id, text=chunk.text, episode_id=a.episode_id,
                         at=a.shared_at, parent_id=a.id, label=label, **signals)

    def add_attachment(
        self, filename: str, media_type: str, size_bytes: int, content_hash: str,
        storage_path: str, at: datetime,
    ) -> tuple[Attachment, bool] | None:
        existing = self._attachment_hashes.get(content_hash)
        if existing is not None:
            attachment = self._attachments[existing]
            if attachment.status == "Z":
                return None
            if attachment.status == "I":
                attachment = self._attachments[existing] = replace(attachment, status="A")
            return attachment, False
        attachment = Attachment(id=next(self._attachment_ids), filename=filename, media_type=media_type,
                                size_bytes=size_bytes, storage_path=storage_path)
        self._attachments[attachment.id] = attachment
        self._attachment_hashes[content_hash] = attachment.id
        return attachment, True

    def link_attachment(self, message_id: int, attachment_id: int) -> None:
        self._attachment_links.add((message_id, attachment_id))
        attachment = self._attachments[attachment_id]
        if attachment.shared_at is None:
            message = self._messages[message_id]
            self._attachments[attachment_id] = replace(
                attachment, shared_at=message.created_at, episode_id=message.episode_id, message_id=message.id)

    def get_attachment(self, attachment_id: int) -> Attachment | None:
        a = self._attachments.get(attachment_id)
        return a if a is not None and a.status == "A" else None

    def message_attachments(self, message_ids: Sequence[int]) -> dict[int, list[Attachment]]:
        wanted, found = set(message_ids), {}
        for message_id, attachment_id in sorted(self._attachment_links):
            a = self._attachments[attachment_id]
            if message_id in wanted and a.status == "A":
                found.setdefault(message_id, []).append(a)
        return found

    def pending_attachments(self, limit: int) -> list[Attachment]:
        return [a for a in self._attachments.values()
                if a.status == "A" and a.extraction_status == "pending"][:limit]

    def unsummarized_attachments(self, limit: int) -> list[Attachment]:
        return [a for a in self._attachments.values()
                if a.status == "A" and a.extraction_status == "done" and not a.title][:limit]

    def save_chunks(
        self, attachment_id: int, chunks: Sequence[tuple[Chunk, Embedding]],
        mentions: Sequence[tuple[int, int]], page_count: int | None,
    ) -> None:
        old = {cid for cid, c in self._chunks.items() if c.attachment_id == attachment_id}
        for cid in old:
            del self._chunks[cid]
            self._chunk_embeddings.pop(cid, None)
        self._chunk_mentions = {(e, c) for e, c in self._chunk_mentions if c not in old}
        by_position = {}
        for chunk, embedding in chunks:
            stored = replace(chunk, id=next(self._chunk_ids), attachment_id=attachment_id)
            self._chunks[stored.id] = stored
            self._chunk_embeddings[stored.id] = embedding
            by_position[chunk.position] = stored.id
        for position, entity_id in mentions:
            if position in by_position:
                self._chunk_mentions.add((entity_id, by_position[position]))
        self._attachments[attachment_id] = replace(
            self._attachments[attachment_id], extraction_status="done", error="", page_count=page_count)

    def mark_attachment(self, attachment_id: int, extraction_status: str, error: str = "") -> None:
        self._attachments[attachment_id] = replace(
            self._attachments[attachment_id], extraction_status=extraction_status, error=error)

    def attachment_chunks(self, attachment_id: int) -> list[Chunk]:
        if self.get_attachment(attachment_id) is None:
            return []
        return sorted((c for c in self._chunks.values() if c.attachment_id == attachment_id),
                      key=lambda c: c.position)

    def save_attachment_summary(self, attachment_id: int, title: str, summary_short: str,
                                summary_long: str) -> None:
        self._attachments[attachment_id] = replace(
            self._attachments[attachment_id], title=title, summary_short=summary_short,
            summary_long=summary_long)

    def tag_attachment(self, attachment_id: int, tag_id: int) -> None:
        self._attachment_tags.add((attachment_id, tag_id))

    # -- EpisodeStore ---------------------------------------------------------

    def current_episode(self) -> Episode | None:
        open_episodes = [e for e in self._episodes.values() if e.is_open]
        return max(open_episodes, key=lambda e: e.id, default=None)

    def open_episode(self, at: datetime) -> Episode:
        episode = Episode(id=next(self._episode_ids), started_at=at)
        self._episodes[episode.id] = episode
        return episode

    def close_episode(self, episode_id: int, at: datetime) -> None:
        self._episodes[episode_id] = replace(self._episodes[episode_id], ended_at=at)

    def get_episodes(self, ids: Sequence[int]) -> list[Episode]:
        return [self._episodes[i] for i in ids if i in self._episodes]

    def save_summary(
        self,
        episode_id: int,
        *,
        title: str,
        summary_short: str,
        summary_long: str,
        short_embedding: Embedding,
        long_embedding: Embedding,
    ) -> None:
        self._episodes[episode_id] = replace(
            self._episodes[episode_id],
            title=title,
            summary_short=summary_short,
            summary_long=summary_long,
        )
        self._episode_embeddings[episode_id] = (short_embedding, long_embedding)

    def recent_episodes(self, limit: int, *, exclude_ids: Sequence[int] = ()) -> list[Episode]:
        closed = [
            e for e in self._episodes.values()
            if not e.is_open and e.id not in set(exclude_ids)
        ]
        closed.sort(key=lambda e: (e.ended_at, e.id), reverse=True)
        return closed[:limit]

    def mark_consolidated(self, episode_id: int, at: datetime) -> None:
        self._episodes[episode_id] = replace(self._episodes[episode_id], consolidated_at=at)

    def episode_nodes(self, episode_ids: Sequence[int]) -> dict[int, set[Node]]:
        wanted, found = set(episode_ids), {}
        for (episode_id, tag_id) in self._episode_tags:
            if episode_id in wanted and self._tag_status[tag_id] == "A":
                found.setdefault(episode_id, set()).add((MemoryKind.TAG, tag_id))
        for entity_id, message_id in self._mentions:
            episode_id = self._messages[message_id].episode_id
            if episode_id in wanted and self._entity_status[entity_id] == "A":
                found.setdefault(episode_id, set()).add((MemoryKind.ENTITY, entity_id))
        return found

    def node_episodes(self, nodes: Sequence[Node]) -> dict[Node, set[int]]:
        wanted = set(nodes)
        everything = self.episode_nodes(list(self._episodes))
        found: dict[Node, set[int]] = {}
        for episode_id, episode_nodes in everything.items():
            for node in episode_nodes & wanted:
                found.setdefault(node, set()).add(episode_id)
        return found

    def load_activation(self) -> dict[Node, tuple[float, datetime]]:
        return dict(self._activation)

    def save_activation(self, values: Mapping[Node, tuple[float, datetime]]) -> None:
        self._activation = dict(values)

    def carried_memories(self, episode_id: int) -> list[CarriedMemory]:
        return list(self._carried.get(episode_id, []))

    def save_carried_memories(self, episode_id: int, memories: Sequence[CarriedMemory]) -> None:
        self._carried[episode_id] = list(memories)

    def record_imports(self, source: str, conversation_id: str, messages: Mapping[str, int]) -> None:
        self._imports.setdefault((source, conversation_id), {}).update(messages)

    def imported_conversations(self, source: str) -> set[str]:
        return {conversation for (s, conversation) in self._imports if s == source}

    def imported_messages(self, source: str, conversation_id: str) -> dict[str, int]:
        return dict(self._imports.get((source, conversation_id), {}))

    def save_drift(self, episode_id: int, strikes: int, pending_message_id: int | None) -> None:
        self._episodes[episode_id] = replace(
            self._episodes[episode_id], drift_strikes=strikes, drift_pending_id=pending_message_id)

    def move_messages(self, message_ids: Sequence[int], episode_id: int) -> None:
        moved = set(message_ids)
        sources = {self._messages[m].episode_id for m in moved}
        for m in moved:
            self._messages[m] = replace(self._messages[m], episode_id=episode_id)
        self._tool_calls = [(episode_id if run.message_id in moved else eid, run) for eid, run in self._tool_calls]
        for eid in sources | {episode_id}:
            times = [m.created_at for m in self._messages.values() if m.episode_id == eid]
            self._episodes[eid] = replace(self._episodes[eid], last_message_at=max(times) if times else None)

    def save_running_summary(self, episode_id: int, summary: str, until_message_id: int) -> None:
        self._episodes[episode_id] = replace(
            self._episodes[episode_id],
            running_summary=summary,
            summarized_until=until_message_id,
        )

    def pending_consolidation(self, limit: int) -> list[Episode]:
        pending = [
            e for e in self._episodes.values() if not e.is_open and e.consolidated_at is None
        ]
        pending.sort(key=lambda e: (e.ended_at, e.id))
        return pending[:limit]

    # -- MessageStore ---------------------------------------------------------

    def add_message(
        self,
        episode_id: int,
        role: Role,
        content: str,
        at: datetime,
        embedding: Embedding | None = None,
    ) -> Message:
        episode = self._episodes[episode_id]
        message = Message(
            id=next(self._message_ids),
            episode_id=episode_id,
            role=role,
            content=content,
            created_at=at,
        )
        self._messages[message.id] = message
        if embedding is not None:
            self._message_embeddings[message.id] = embedding
        if episode.last_message_at is None or at > episode.last_message_at:
            self._episodes[episode_id] = replace(episode, last_message_at=at)
        return message

    def set_pinned(self, message_id: int, pinned: bool) -> None:
        self._messages[message_id] = replace(self._messages[message_id], pinned=pinned)

    def pinned_messages(self, episode_id: int) -> list[Message]:
        return sorted((m for m in self._messages.values() if m.episode_id == episode_id and m.pinned),
                      key=lambda m: m.id)

    def get_messages(
        self, episode_id: int, *, last: int | None = None, after: int | None = None
    ) -> list[Message]:
        messages = sorted(
            (
                m for m in self._messages.values()
                if m.episode_id == episode_id and (after is None or m.id > after)
            ),
            key=lambda m: m.id,  # insertion order, not timestamps
        )
        if last is not None:
            messages = messages[-last:] if last > 0 else []
        return messages

    def get_message(self, message_id: int) -> Message | None:
        return self._messages.get(message_id)

    # -- TagStore -------------------------------------------------------------

    def upsert_tag(self, name: str, embedding: Embedding | None = None) -> Tag | None:
        key = normalize_key(name)
        for tag_id, tag in self._tags.items():
            if normalize_key(tag.name) == key:
                if self._tag_status[tag_id] == "Z":
                    return None
                self._tag_status[tag_id] = "A"
                if embedding is not None:
                    self._tag_embeddings.setdefault(tag_id, embedding)
                return tag
        tag = Tag(id=next(self._tag_ids), name=name.strip())
        self._tags[tag.id] = tag
        self._tag_status[tag.id] = "A"
        if embedding is not None:
            self._tag_embeddings[tag.id] = embedding
        return tag

    def similar_tags(self, query: Embedding, limit: int) -> list[tuple[Tag, float]]:
        hits = [
            (replace(self._tags[tid], status=self._tag_status[tid]), cosine(query.vector, e.vector))
            for tid, e in self._tag_embeddings.items()
            if e.model == query.model and self._tag_status[tid] in ("A", "Z")
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:limit]

    def list_tags(self, limit: int) -> list[Tag]:
        uses = {tag_id: 0 for tag_id in self._tags}
        for (_, tag_id) in self._episode_tags:
            uses[tag_id] += 1
        active = [t for t in self._tags.values() if self._tag_status[t.id] == "A"]
        active.sort(key=lambda t: (-uses[t.id], t.name))
        return active[:limit]

    def tag_episode(self, episode_id: int, tag_id: int, relevance: float) -> None:
        self._episode_tags[(episode_id, tag_id)] = relevance

    def keyword_candidates(
        self,
        kind: MemoryKind,
        terms: Sequence[str],
        limit: int,
        *,
        exclude_episode_ids: Sequence[int] = (),
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        wanted = [t.casefold() for t in terms]
        if not wanted:
            return []
        excluded = set(exclude_episode_ids)

        def share(*texts: str) -> float:
            folded = " ".join(texts).casefold()
            return sum(t in folded for t in wanted) / len(wanted)

        hits: list[tuple[float, Candidate]] = []
        if kind is MemoryKind.MESSAGE:
            for m in self._messages.values():
                if m.episode_id not in excluded and _within(m.created_at, after, before):
                    hits.append((share(m.content), Candidate(
                        kind=kind, id=m.id, text=m.content, episode_id=m.episode_id,
                        at=m.created_at, role=m.role)))
        elif kind is MemoryKind.EPISODE:
            for e in self._episodes.values():
                if e.summary_short and e.id not in excluded and _within(e.started_at, after, before):
                    hits.append((share(e.summary_short, e.summary_long), Candidate(
                        kind=kind, id=e.id, text=e.summary_short, episode_id=e.id, at=e.started_at)))
        elif kind is MemoryKind.FACT:
            for f in self._facts.values():
                if _recallable(f) and f.episode_id not in excluded and _within(f.learned_at, after, before):
                    hits.append((share(f.statement), _fact_candidate(f)))
        elif kind is MemoryKind.ATTACHMENT:
            for chunk in self._chunks.values():
                a = self._attachments[chunk.attachment_id]
                if self._searchable(a.id) and a.episode_id not in excluded and _within(a.shared_at, after, before):
                    hits.append((share(chunk.text), self._chunk_candidate(chunk)))
        hits = [(s, c) for s, c in hits if s > 0]
        hits.sort(key=lambda h: (h[0], h[1].at or datetime.min.replace(tzinfo=UTC), h[1].id), reverse=True)
        return [replace(c, keyword=s) for s, c in hits[:limit]]

    def episode_tags(self, episode_ids: Sequence[int]) -> dict[int, dict[int, float]]:
        wanted, found = set(episode_ids), {}
        for (episode_id, tag_id), relevance in self._episode_tags.items():
            if episode_id in wanted and self._tag_status[tag_id] == "A":
                found.setdefault(episode_id, {})[tag_id] = relevance
        return found

    # -- EntityStore ----------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        type: str = "",
        aliases: Sequence[str] = (),
        embedding: Embedding | None = None,
    ) -> Entity | None:
        keys = set(identity_keys(name, aliases))
        for entity_id, entity in self._entities.items():
            known = set(identity_keys(entity.name, entity.aliases))
            if keys & known:
                if self._entity_status[entity_id] == "Z":
                    return None
                self._entity_status[entity_id] = "A"
                new = [a for a in (name, *aliases) if normalize_key(a) not in known]
                if new:
                    entity = replace(entity, aliases=(*entity.aliases, *dict.fromkeys(new)))
                    self._entities[entity_id] = entity
                return entity
        unique = {normalize_key(a): a.strip() for a in aliases if normalize_key(a) != normalize_key(name)}
        entity = Entity(
            id=next(self._entity_ids), name=name.strip(), type=type, aliases=tuple(unique.values())
        )
        self._entities[entity.id] = entity
        self._entity_status[entity.id] = "A"
        return entity

    def add_mention(self, entity_id: int, message_id: int) -> None:
        self._mentions.add((entity_id, message_id))

    def active_entities(self) -> list[Entity]:
        return [e for e in self._entities.values() if self._entity_status[e.id] == "A"]

    def match_entities(self, keys: Collection[str]) -> list[Entity]:
        wanted = set(keys)
        return [
            e for e in self._entities.values()
            if self._entity_status[e.id] == "A"
            and wanted & set(search_keys(e.name, e.aliases, e.type))
        ]

    # -- test helpers ---------------------------------------------------------

    def set_status(self, kind: MemoryKind, item_id: int, status: str) -> None:
        """Set a tag's, entity's, or fact's status ("A", "I", "Z"), as the admin would."""
        if kind is MemoryKind.FACT:
            self._facts[item_id] = replace(self._facts[item_id], status=status)
            return
        if kind is MemoryKind.ATTACHMENT:
            self._attachments[item_id] = replace(self._attachments[item_id], status=status)
            return
        statuses = {MemoryKind.TAG: self._tag_status, MemoryKind.ENTITY: self._entity_status}
        statuses[kind][item_id] = status

    # -- SearchStore ----------------------------------------------------------

    def semantic_candidates(
        self,
        kind: MemoryKind,
        query: Embedding,
        limit: int,
        *,
        exclude_episode_ids: Sequence[int] = (),
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        excluded = set(exclude_episode_ids)
        hits: list[Candidate] = []
        if kind is MemoryKind.MESSAGE:
            for message_id, embedding in self._message_embeddings.items():
                message = self._messages[message_id]
                if (
                    embedding.model != query.model or message.episode_id in excluded
                    or not _within(message.created_at, after, before)
                ):
                    continue
                hits.append(Candidate(
                    kind=kind, id=message.id, text=message.content,
                    semantic=cosine(query.vector, embedding.vector),
                    episode_id=message.episode_id, at=message.created_at, role=message.role,
                ))
        elif kind is MemoryKind.EPISODE:
            for episode_id, pair in self._episode_embeddings.items():
                episode = self._episodes[episode_id]
                similarities = [
                    cosine(query.vector, e.vector) for e in pair if e.model == query.model
                ]
                if (
                    not similarities or episode_id in excluded
                    or not _within(episode.started_at, after, before)
                ):
                    continue
                hits.append(Candidate(
                    kind=kind, id=episode.id, text=episode.summary_short,
                    semantic=max(similarities), episode_id=episode.id, at=episode.started_at,
                ))
        elif kind is MemoryKind.FACT:
            for fact_id, embedding in self._fact_embeddings.items():
                fact = self._facts[fact_id]
                if (
                    embedding.model != query.model or not _recallable(fact)
                    or fact.episode_id in excluded or not _within(fact.learned_at, after, before)
                ):
                    continue
                hits.append(_fact_candidate(fact, semantic=cosine(query.vector, embedding.vector)))
        elif kind is MemoryKind.ATTACHMENT:
            for chunk_id, embedding in self._chunk_embeddings.items():
                chunk = self._chunks[chunk_id]
                a = self._attachments[chunk.attachment_id]
                if (embedding.model != query.model or not self._searchable(a.id)
                        or a.episode_id in excluded or not _within(a.shared_at, after, before)):
                    continue
                hits.append(self._chunk_candidate(chunk, semantic=cosine(query.vector, embedding.vector)))
        else:
            raise NotImplementedError(f"Search over {kind} arrives with its memory kind.")
        hits.sort(key=lambda c: c.semantic or 0.0, reverse=True)
        return hits[:limit]

    def link_fact_entities(self, fact_id: int, entity_ids: Sequence[int]) -> None:
        if fact_id in self._facts:
            self._fact_entities.setdefault(fact_id, set()).update(e for e in entity_ids if e in self._entities)

    def entity_fact_candidates(
        self,
        entity_ids: Sequence[int],
        query: Embedding,
        limit: int,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        wanted = {e for e in entity_ids if self._entity_status.get(e) == "A"}
        hits = []
        for fact_id, linked in self._fact_entities.items():
            fact = self._facts[fact_id]
            embedding = self._fact_embeddings.get(fact_id)
            if (not linked & wanted or embedding is None or embedding.model != query.model
                    or not _recallable(fact) or not _within(fact.learned_at, after, before)):
                continue
            hits.append(_fact_candidate(fact, semantic=cosine(query.vector, embedding.vector)))
        hits.sort(key=lambda c: c.semantic or 0.0, reverse=True)
        return hits[:limit]

    def stored_vectors(self, kind: MemoryKind, ids: Sequence[int], model: str) -> dict[int, list[float]]:
        table: dict[int, Embedding] = {
            MemoryKind.MESSAGE: self._message_embeddings,
            MemoryKind.FACT: self._fact_embeddings,
            MemoryKind.ATTACHMENT: self._chunk_embeddings,
            MemoryKind.EPISODE: {i: pair[0] for i, pair in self._episode_embeddings.items()},
        }.get(kind, {})
        return {i: list(table[i].vector) for i in ids if i in table and table[i].model == model}

    def mention_candidates(
        self,
        kind: MemoryKind,
        entity_ids: Sequence[int],
        limit: int,
        *,
        exclude_episode_ids: Sequence[int] = (),
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        wanted, excluded = set(entity_ids), set(exclude_episode_ids)
        if not wanted:
            return []
        per_message: dict[int, set[int]] = {}
        for entity_id, message_id in self._mentions:
            if entity_id in wanted:
                per_message.setdefault(message_id, set()).add(entity_id)

        if kind is MemoryKind.MESSAGE:
            hits = [
                (len(found), self._messages[mid])
                for mid, found in per_message.items()
                if self._messages[mid].episode_id not in excluded
                and _within(self._messages[mid].created_at, after, before)
            ]
            hits.sort(key=lambda h: (h[0], h[1].id), reverse=True)
            return [
                Candidate(
                    kind=kind, id=m.id, text=m.content, lexical=n / len(wanted),
                    episode_id=m.episode_id, at=m.created_at, role=m.role,
                )
                for n, m in hits[:limit]
            ]
        if kind is MemoryKind.EPISODE:
            per_episode: dict[int, set[int]] = {}
            for mid, found in per_message.items():
                per_episode.setdefault(self._messages[mid].episode_id, set()).update(found)
            episodes = [
                (len(found), self._episodes[eid])
                for eid, found in per_episode.items()
                if eid not in excluded and self._episodes[eid].summary_short
                and _within(self._episodes[eid].started_at, after, before)
            ]
            episodes.sort(key=lambda h: (h[0], h[1].ended_at or h[1].started_at), reverse=True)
            return [
                Candidate(
                    kind=kind, id=e.id, text=e.summary_short, lexical=n / len(wanted),
                    episode_id=e.id, at=e.started_at,
                )
                for n, e in episodes[:limit]
            ]
        if kind is MemoryKind.FACT:
            facts = []
            for fact in self._facts.values():
                found = set().union(*(per_message.get(mid, set()) for mid in fact.source_message_ids))
                if (
                    found and _recallable(fact) and fact.episode_id not in excluded
                    and _within(fact.learned_at, after, before)
                ):
                    facts.append((len(found), fact))
            facts.sort(key=lambda h: (h[0], h[1].id), reverse=True)
            return [_fact_candidate(f, lexical=n / len(wanted)) for n, f in facts[:limit]]
        if kind is MemoryKind.ATTACHMENT:
            per_chunk: dict[int, set[int]] = {}
            for entity_id, chunk_id in self._chunk_mentions:
                if entity_id in wanted and chunk_id in self._chunks:
                    per_chunk.setdefault(chunk_id, set()).add(entity_id)
            chunks = []
            for chunk_id, found in per_chunk.items():
                chunk = self._chunks[chunk_id]
                a = self._attachments[chunk.attachment_id]
                if self._searchable(a.id) and a.episode_id not in excluded and _within(a.shared_at, after, before):
                    chunks.append((len(found), chunk))
            chunks.sort(key=lambda h: (h[0], -h[1].position), reverse=True)
            return [self._chunk_candidate(c, lexical=n / len(wanted)) for n, c in chunks[:limit]]
        raise NotImplementedError(f"Mentions of {kind} are not tracked.")

    # -- FactStore -------------------------------------------------------------

    def add_fact(
        self,
        statement: str,
        type: str,
        *,
        episode_id: int | None,
        message_ids: Sequence[int],
        at: datetime,
        embedding: Embedding,
    ) -> Fact:
        fact = Fact(
            id=next(self._fact_ids), statement=statement.strip(), type=type,
            episode_id=episode_id, learned_at=at, last_confirmed_at=at,
            source_message_ids=tuple(dict.fromkeys(message_ids)),
        )
        self._facts[fact.id] = fact
        self._fact_embeddings[fact.id] = embedding
        return fact

    def confirm_fact(self, fact_id: int, at: datetime, message_ids: Sequence[int] = ()) -> None:
        fact = self._facts[fact_id]
        self._facts[fact_id] = replace(
            fact, times_confirmed=fact.times_confirmed + 1, last_confirmed_at=at,
            source_message_ids=tuple(dict.fromkeys([*fact.source_message_ids, *message_ids])),
        )

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        self._facts[old_id] = replace(self._facts[old_id], superseded_by=new_id)

    def similar_facts(self, query: Embedding, limit: int) -> list[tuple[Fact, float]]:
        hits = [
            (self._facts[fid], cosine(query.vector, e.vector))
            for fid, e in self._fact_embeddings.items()
            if e.model == query.model and self._facts[fid].superseded_by is None
            and self._facts[fid].status in ("A", "Z")
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:limit]

    def get_facts(self, ids: Sequence[int]) -> list[Fact]:
        return [self._facts[i] for i in ids if i in self._facts]

    def fact_history(self, fact_id: int) -> list[Fact]:
        return _chain(fact_id, self._facts)

    # -- ToolCallStore ---------------------------------------------------------

    def add_tool_calls(
        self, episode_id: int, message_id: int | None, runs: Sequence[ToolRun]
    ) -> None:
        self._tool_calls += [(episode_id, replace(r, message_id=message_id)) for r in runs]

    def get_tool_calls(self, episode_id: int) -> list[ToolRun]:
        return [run for eid, run in self._tool_calls if eid == episode_id]

    # -- EmbeddingCacheStore --------------------------------------------------

    def get_cached(self, keys: Sequence[str], model: str) -> dict[str, Embedding]:
        return {k: self._cache[(model, k)] for k in keys if (model, k) in self._cache}

    def put_cached(self, entries: Mapping[str, Embedding]) -> None:
        for key, embedding in entries.items():
            self._cache[(embedding.model, key)] = embedding
