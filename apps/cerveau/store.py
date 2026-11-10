"""DjangoMemoryStore: brain/ports.py implemented with the Django ORM.

The single choke point between the engine and the database. Every read goes through
`active_objects`, so inactive (I) and forgotten (Z) memory never reaches the engine.
It must pass the same contract tests as InMemoryStore
(tests/test_store_contract.py).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from django.db import transaction
from django.db.models import Count, F, Max, Q, QuerySet
from django.db.models.functions import Least
from django.utils import timezone
from pgvector.django import CosineDistance

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

from . import models


def _episodes() -> QuerySet[models.Episode]:
    """Active episodes with `last_message_at` derived from their messages."""
    return models.Episode.active_objects.annotate(last_message_at=Max("messages__created_at"))


def _to_episode(row: models.Episode) -> Episode:
    return Episode(
        id=row.pk,
        started_at=row.started_at,
        ended_at=row.ended_at,
        last_message_at=getattr(row, "last_message_at", None),
        title=row.title,
        summary_short=row.summary_short,
        summary_long=row.summary_long,
        consolidated_at=row.consolidated_at,
        running_summary=row.running_summary,
        summarized_until=row.summarized_until,
        drift_strikes=row.drift_strikes,
        drift_pending_id=row.drift_pending_message_id,
    )


def _to_tag(row: models.Tag) -> Tag:
    return Tag(id=row.pk, name=row.name, status=row.status)


def _to_entity(row: models.Entity) -> Entity:
    return Entity(id=row.pk, name=row.name, type=row.type, aliases=tuple(row.aliases))


def _to_fact(row: models.Fact) -> Fact:
    return Fact(
        id=row.pk, statement=row.statement, type=row.type, episode_id=row.episode_id,
        learned_at=row.learned_at, times_confirmed=row.times_confirmed,
        last_confirmed_at=row.last_confirmed_at, superseded_by=row.superseded_by_id,
        source_message_ids=tuple(sorted(row.sources.values_list("pk", flat=True))),
        status=row.status,
    )


def _recallable_facts() -> QuerySet[models.Fact]:
    """Facts recall may use: active and not replaced by a newer one."""
    return models.Fact.active_objects.filter(superseded_by__isnull=True)


def _fact_candidate(row: models.Fact, **signals) -> Candidate:
    return Candidate(
        kind=MemoryKind.FACT, id=row.pk, text=row.statement, episode_id=row.episode_id,
        at=row.learned_at, **signals,
    )


def _json(text: str) -> Any:
    """A tool result (JSON text) as a JSON value for the database; kept as text if a
    result ever isn't valid JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def _to_attachment(row: models.Attachment) -> Attachment:
    return Attachment(
        id=row.pk, filename=row.filename, media_type=row.media_type, size_bytes=row.size_bytes,
        storage_path=row.storage_path, status=row.status, extraction_status=row.extraction_status,
        error=row.extraction_error, title=row.title, summary_short=row.summary_short,
        summary_long=row.summary_long,
        page_count=row.page_count, shared_at=row.shared_at, episode_id=row.episode_id,
        message_id=row.message_id,
    )


def _to_chunk(row: models.AttachmentChunk) -> Chunk:
    return Chunk(position=row.position, text=row.text, page_start=row.page_start, page_end=row.page_end,
                 heading=row.heading, id=row.pk, attachment_id=row.attachment_id)


def _searchable_attachments() -> QuerySet:
    return models.Attachment.objects.filter(status="A", extraction_status="done")


def _chunk_candidate(row: models.AttachmentChunk, **signals) -> Candidate:
    """A passage as a candidate; `row.attachment` should be select_related."""
    a = row.attachment
    chunk = _to_chunk(row)
    label = ", ".join(p for p in (a.filename, chunk.pages) if p)
    return Candidate(kind=MemoryKind.ATTACHMENT, id=row.pk, text=row.text, episode_id=a.episode_id,
                     at=a.shared_at, parent_id=a.pk, label=label, **signals)


def _window(field: str, after: datetime | None, before: datetime | None) -> Q:
    """A time-window filter on `field`: `after` inclusive, `before` exclusive."""
    q = Q()
    if after is not None:
        q &= Q(**{f"{field}__gte": after})
    if before is not None:
        q &= Q(**{f"{field}__lt": before})
    return q


def _to_message(row: models.Message) -> Message:
    return Message(
        id=row.pk,
        episode_id=row.episode_id,
        role=Role(row.role),
        content=row.content,
        created_at=row.created_at,
        pinned=row.pinned,
    )


class DjangoMemoryStore(MemoryStore):
    # -- EpisodeStore ---------------------------------------------------------

    def current_episode(self) -> Episode | None:
        row = _episodes().filter(ended_at__isnull=True).order_by("-id").first()
        return _to_episode(row) if row else None

    def open_episode(self, at: datetime) -> Episode:
        return _to_episode(models.Episode.objects.create(started_at=at))

    def close_episode(self, episode_id: int, at: datetime) -> None:
        # update() rather than save(): closing is bookkeeping, not an edit worth an
        # audit snapshot.
        models.Episode.objects.filter(pk=episode_id).update(ended_at=at)

    def get_episodes(self, ids: Sequence[int]) -> list[Episode]:
        rows = {row.pk: row for row in _episodes().filter(pk__in=ids)}
        return [_to_episode(rows[i]) for i in ids if i in rows]

    @transaction.atomic
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
        # save(), not update(): summaries are real edits and belong in the audit trail.
        episode = models.Episode.objects.get(pk=episode_id)
        episode.title = title
        episode.summary_short = summary_short
        episode.summary_long = summary_long
        episode.save()
        # One row per (episode, model); the two summaries may come from different models.
        for prefix, embedding in (("summary_short", short_embedding), ("summary_long", long_embedding)):
            models.EpisodeEmbedding.objects.update_or_create(
                episode=episode,
                model=embedding.model,
                defaults={prefix: list(embedding.vector), f"{prefix}_hash": embedding.text_hash},
            )

    def recent_episodes(self, limit: int, *, exclude_ids: Sequence[int] = ()) -> list[Episode]:
        rows = (
            _episodes()
            .filter(ended_at__isnull=False)
            .exclude(pk__in=exclude_ids)
            .order_by("-ended_at", "-id")[:limit]
        )
        return [_to_episode(r) for r in rows]

    def mark_consolidated(self, episode_id: int, at: datetime) -> None:
        models.Episode.objects.filter(pk=episode_id).update(consolidated_at=at)

    def episode_nodes(self, episode_ids: Sequence[int]) -> dict[int, set[Node]]:
        ids = list(episode_ids)
        found: dict[int, set[Node]] = {}
        tagged = models.EpisodeTag.objects.filter(episode_id__in=ids, tag__status="A")
        for episode_id, tag_id in tagged.values_list("episode_id", "tag_id"):
            found.setdefault(episode_id, set()).add((MemoryKind.TAG, tag_id))
        mentioned = models.EntityMention.objects.filter(message__episode_id__in=ids, entity__status="A")
        for episode_id, entity_id in mentioned.values_list("message__episode_id", "entity_id").distinct():
            found.setdefault(episode_id, set()).add((MemoryKind.ENTITY, entity_id))
        return found

    def node_episodes(self, nodes: Sequence[Node]) -> dict[Node, set[int]]:
        tags = [i for kind, i in nodes if kind is MemoryKind.TAG]
        entities = [i for kind, i in nodes if kind is MemoryKind.ENTITY]
        found: dict[Node, set[int]] = {}
        tagged = models.EpisodeTag.objects.filter(tag_id__in=tags, tag__status="A", episode__status="A")
        for tag_id, episode_id in tagged.values_list("tag_id", "episode_id"):
            found.setdefault((MemoryKind.TAG, tag_id), set()).add(episode_id)
        mentioned = models.EntityMention.objects.filter(
            entity_id__in=entities, entity__status="A", message__episode__status="A"
        )
        for entity_id, episode_id in mentioned.values_list("entity_id", "message__episode_id").distinct():
            found.setdefault((MemoryKind.ENTITY, entity_id), set()).add(episode_id)
        return found

    def load_activation(self) -> dict[Node, tuple[float, datetime]]:
        return {
            (MemoryKind(kind), node_id): (value, at)
            for kind, node_id, value, at in models.Activation.objects.values_list("kind", "node_id", "value", "updated_at")
        }

    @transaction.atomic
    def save_activation(self, values: Mapping[Node, tuple[float, datetime]]) -> None:
        models.Activation.objects.all().delete()
        models.Activation.objects.bulk_create([
            models.Activation(kind=kind.value, node_id=node_id, value=value, updated_at=at)
            for (kind, node_id), (value, at) in values.items()
        ])

    # -- AttachmentStore --------------------------------------------------------

    def add_attachment(
        self, filename: str, media_type: str, size_bytes: int, content_hash: str,
        storage_path: str, at: datetime,
    ) -> tuple[Attachment, bool] | None:
        row = models.Attachment.objects.filter(content_hash=content_hash).first()
        if row is None:
            row = models.Attachment.objects.create(
                filename=filename[:255], media_type=media_type, size_bytes=size_bytes,
                content_hash=content_hash, storage_path=storage_path,
            )
            return _to_attachment(row), True
        if row.status == "Z":       # forgotten by you: not relearned
            return None
        if row.status != "A":       # set aside: comes back
            row.reactivate()
        return _to_attachment(row), False

    def link_attachment(self, message_id: int, attachment_id: int) -> None:
        models.MessageAttachment.objects.get_or_create(message_id=message_id, attachment_id=attachment_id)
        message = models.Message.objects.only("created_at", "episode_id").get(pk=message_id)
        # Bookkeeping: when and where it was first shared (update(), not audited).
        models.Attachment.objects.filter(pk=attachment_id, shared_at__isnull=True).update(
            shared_at=message.created_at, episode_id=message.episode_id, message_id=message_id)

    def get_attachment(self, attachment_id: int) -> Attachment | None:
        row = models.Attachment.active_objects.filter(pk=attachment_id).first()
        return _to_attachment(row) if row else None

    def message_attachments(self, message_ids: Sequence[int]) -> dict[int, list[Attachment]]:
        found: dict[int, list[Attachment]] = {}
        links = (models.MessageAttachment.objects
                 .filter(message_id__in=list(message_ids), attachment__status="A")
                 .select_related("attachment").order_by("message_id", "attachment_id"))
        for link in links:
            found.setdefault(link.message_id, []).append(_to_attachment(link.attachment))
        return found

    def pending_attachments(self, limit: int) -> list[Attachment]:
        rows = models.Attachment.active_objects.filter(extraction_status="pending").order_by("added_at", "id")
        return [_to_attachment(r) for r in rows[:limit]]

    def unsummarized_attachments(self, limit: int) -> list[Attachment]:
        rows = (models.Attachment.active_objects.filter(extraction_status="done", title="")
                .order_by("added_at", "id"))
        return [_to_attachment(r) for r in rows[:limit]]

    @transaction.atomic
    def save_chunks(
        self, attachment_id: int, chunks: Sequence[tuple[Chunk, Embedding]],
        mentions: Sequence[tuple[int, int]], page_count: int | None,
    ) -> None:
        models.AttachmentChunk.objects.filter(attachment_id=attachment_id).delete()
        rows = models.AttachmentChunk.objects.bulk_create([
            models.AttachmentChunk(attachment_id=attachment_id, position=c.position, text=c.text,
                                   page_start=c.page_start, page_end=c.page_end, heading=c.heading[:300])
            for c, _ in chunks
        ])
        models.AttachmentChunkEmbedding.objects.bulk_create([
            models.AttachmentChunkEmbedding(chunk=row, model=e.model, embedding=list(e.vector), text_hash=e.text_hash)
            for row, (_, e) in zip(rows, chunks)
        ])
        by_position = {row.position: row for row in rows}
        models.AttachmentMention.objects.bulk_create([
            models.AttachmentMention(entity_id=entity_id, chunk=by_position[position])
            for position, entity_id in dict.fromkeys(mentions) if position in by_position
        ], ignore_conflicts=True)
        models.Attachment.objects.filter(pk=attachment_id).update(
            extraction_status="done", extraction_error="", page_count=page_count, read_at=timezone.now())

    def mark_attachment(self, attachment_id: int, extraction_status: str, error: str = "") -> None:
        models.Attachment.objects.filter(pk=attachment_id).update(
            extraction_status=extraction_status, extraction_error=error, read_at=timezone.now())

    def attachment_chunks(self, attachment_id: int) -> list[Chunk]:
        rows = models.AttachmentChunk.objects.filter(
            attachment_id=attachment_id, attachment__status="A").order_by("position")
        return [_to_chunk(r) for r in rows]

    def save_attachment_summary(self, attachment_id: int, title: str, summary_short: str,
                                summary_long: str) -> None:
        models.Attachment.objects.filter(pk=attachment_id).update(
            title=title[:200], summary_short=summary_short, summary_long=summary_long,
            summarized_at=timezone.now())

    def tag_attachment(self, attachment_id: int, tag_id: int) -> None:
        models.AttachmentTag.objects.get_or_create(attachment_id=attachment_id, tag_id=tag_id)

    def carried_memories(self, episode_id: int) -> list[CarriedMemory]:
        saved = models.Episode.objects.filter(pk=episode_id).values_list("carried", flat=True).first() or []
        return [CarriedMemory(MemoryKind(c["kind"]), int(c["id"]), float(c["score"])) for c in saved]

    def save_carried_memories(self, episode_id: int, memories: Sequence[CarriedMemory]) -> None:
        # update(): rewritten every turn, bookkeeping rather than an edit.
        models.Episode.objects.filter(pk=episode_id).update(
            carried=[{"kind": m.kind.value, "id": m.id, "score": round(m.score, 4)} for m in memories]
        )

    def record_imports(self, source: str, conversation_id: str, messages: Mapping[str, int]) -> None:
        models.ImportedMessage.objects.bulk_create(
            [models.ImportedMessage(source=source, conversation_id=conversation_id, external_id=external,
                                    message_id=message_id) for external, message_id in messages.items()],
            ignore_conflicts=True,
        )

    def imported_conversations(self, source: str) -> set[str]:
        return set(models.ImportedMessage.objects.filter(source=source)
                   .values_list("conversation_id", flat=True).distinct())

    def imported_messages(self, source: str, conversation_id: str) -> dict[str, int]:
        return dict(models.ImportedMessage.objects.filter(source=source, conversation_id=conversation_id)
                    .values_list("external_id", "message_id"))

    def set_pinned(self, message_id: int, pinned: bool) -> None:
        models.Message.objects.filter(pk=message_id).update(pinned=pinned)

    def pinned_messages(self, episode_id: int) -> list[Message]:
        rows = models.Message.objects.filter(episode_id=episode_id, pinned=True).order_by("id")
        return [_to_message(r) for r in rows]

    def save_drift(self, episode_id: int, strikes: int, pending_message_id: int | None) -> None:
        models.Episode.objects.filter(pk=episode_id).update(
            drift_strikes=strikes, drift_pending_message_id=pending_message_id)

    @transaction.atomic
    def move_messages(self, message_ids: Sequence[int], episode_id: int) -> None:
        ids = list(message_ids)
        models.Message.objects.filter(pk__in=ids).update(episode_id=episode_id)
        models.ToolCall.objects.filter(message_id__in=ids).update(episode_id=episode_id)
        models.Attachment.objects.filter(message_id__in=ids).update(episode_id=episode_id)

    def save_running_summary(self, episode_id: int, summary: str, until_message_id: int) -> None:
        # update(): rewritten every ~10 messages, bookkeeping rather than an edit.
        models.Episode.objects.filter(pk=episode_id).update(
            running_summary=summary, summarized_until=until_message_id
        )

    def pending_consolidation(self, limit: int) -> list[Episode]:
        rows = (
            _episodes()
            .filter(ended_at__isnull=False, consolidated_at__isnull=True)
            .order_by("ended_at", "id")[:limit]
        )
        return [_to_episode(r) for r in rows]

    # -- TagStore -------------------------------------------------------------
    # Lookups use `objects` (every status): an inactive tag is reactivated and a
    # forgotten one refused, rather than duplicated.

    def upsert_tag(self, name: str, embedding: Embedding | None = None) -> Tag | None:
        key = normalize_key(name)
        row = models.Tag.objects.filter(key=key).first()
        if row is None:
            row = models.Tag.objects.create(name=name.strip(), key=key)
        elif row.status == "Z":
            return None
        elif row.status != "A":
            row.reactivate()
        if embedding is not None:
            models.TagEmbedding.objects.get_or_create(
                tag=row, model=embedding.model,
                defaults={"embedding": list(embedding.vector), "text_hash": embedding.text_hash},
            )
        return _to_tag(row)

    def similar_tags(self, query: Embedding, limit: int) -> list[tuple[Tag, float]]:
        rows = (
            models.TagEmbedding.objects
            .filter(model=query.model, tag__status__in=["A", "Z"])
            .annotate(distance=CosineDistance("embedding", list(query.vector)))
            .select_related("tag")
            .order_by("distance")[:limit]
        )
        return [(_to_tag(r.tag), 1 - r.distance) for r in rows]  # type: ignore[attr-defined]

    def list_tags(self, limit: int) -> list[Tag]:
        rows = (
            models.Tag.active_objects.annotate(uses=Count("episode_tags"))
            .order_by("-uses", "name")[:limit]
        )
        return [_to_tag(r) for r in rows]

    def tag_episode(self, episode_id: int, tag_id: int, relevance: float) -> None:
        models.EpisodeTag.objects.update_or_create(
            episode_id=episode_id, tag_id=tag_id, defaults={"relevance": relevance}
        )

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
        if not terms:
            return []

        def any_of(*fields: str) -> Q:
            found = Q()
            for term in terms:
                for name in fields:
                    found |= Q(**{f"{name}__icontains": term})
            return found

        def share(*texts: str) -> float:
            folded = " ".join(texts).casefold()
            return sum(t.casefold() in folded for t in terms) / len(terms)

        # A few times the limit, then ranked here: how many terms each one holds is
        # simpler to count in Python than in SQL, and the rows are few.
        room = limit * 4
        if kind is MemoryKind.MESSAGE:
            rows = (models.Message.objects.filter(any_of("content"), episode__status="A")
                    .filter(_window("created_at", after, before))
                    .exclude(episode_id__in=exclude_episode_ids).order_by("-id")[:room])
            hits = [(share(r.content), Candidate(kind=kind, id=r.pk, text=r.content, episode_id=r.episode_id,
                                                 at=r.created_at, role=Role(r.role))) for r in rows]
        elif kind is MemoryKind.EPISODE:
            rows = (models.Episode.active_objects.filter(any_of("summary_short", "summary_long"))
                    .filter(_window("started_at", after, before)).exclude(summary_short="")
                    .exclude(pk__in=exclude_episode_ids).order_by("-started_at")[:room])
            hits = [(share(r.summary_short, r.summary_long), Candidate(
                kind=kind, id=r.pk, text=r.summary_short, episode_id=r.pk, at=r.started_at)) for r in rows]
        elif kind is MemoryKind.FACT:
            rows = (_recallable_facts().filter(any_of("statement"))
                    .filter(_window("learned_at", after, before))
                    .exclude(episode_id__in=exclude_episode_ids).order_by("-id")[:room])
            hits = [(share(r.statement), _fact_candidate(r)) for r in rows]
        elif kind is MemoryKind.ATTACHMENT:
            rows = (models.AttachmentChunk.objects.filter(any_of("text"), attachment__in=_searchable_attachments())
                    .filter(_window("attachment__shared_at", after, before))
                    .exclude(attachment__episode_id__in=exclude_episode_ids)
                    .select_related("attachment").order_by("-attachment__shared_at", "position")[:room])
            hits = [(share(r.text), _chunk_candidate(r)) for r in rows]
        else:
            raise NotImplementedError(f"Keyword search over {kind} is not supported.")
        hits.sort(key=lambda h: h[0], reverse=True)   # stable: newest first within a share
        return [replace(c, keyword=s) for s, c in hits[:limit]]

    def episode_tags(self, episode_ids: Sequence[int]) -> dict[int, dict[int, float]]:
        found: dict[int, dict[int, float]] = {}
        rows = models.EpisodeTag.objects.filter(episode_id__in=list(episode_ids), tag__status="A")
        for episode_id, tag_id, relevance in rows.values_list("episode_id", "tag_id", "relevance"):
            found.setdefault(episode_id, {})[tag_id] = relevance
        return found

    # -- EntityStore ----------------------------------------------------------

    @transaction.atomic
    def upsert_entity(
        self,
        name: str,
        type: str = "",
        aliases: Sequence[str] = (),
        embedding: Embedding | None = None,
    ) -> Entity | None:
        names = [name.strip(), *(a.strip() for a in aliases if a.strip())]
        keys = identity_keys(names[0], names[1:])
        row = models.Entity.objects.filter(match_keys__overlap=keys).order_by("id").first()
        if row is None:
            unique = {normalize_key(a): a for a in names[1:] if normalize_key(a) != keys[0]}
            row = models.Entity.objects.create(
                name=names[0], type=type, aliases=list(unique.values()), match_keys=keys,
                search_keys=search_keys(names[0], unique.values(), type),
            )
            return _to_entity(row)
        if row.status == "Z":
            return None
        if row.status != "A":
            row.reactivate()
        known = set(row.match_keys)
        new = {normalize_key(n): n for n in names if normalize_key(n) not in known}
        if new:
            # A real edit (the entity gained names), so save() and keep it audited.
            row.aliases = [*row.aliases, *new.values()]
            row.match_keys = [*row.match_keys, *new]
            row.search_keys = search_keys(row.name, row.aliases, row.type)
            row.save()
        return _to_entity(row)

    def active_entities(self) -> list[Entity]:
        return [_to_entity(r) for r in models.Entity.active_objects.order_by("id")]

    def match_entities(self, keys: Collection[str]) -> list[Entity]:
        if not keys:
            return []
        rows = models.Entity.active_objects.filter(search_keys__overlap=list(keys))
        return [_to_entity(r) for r in rows.order_by("id")]

    def add_mention(self, entity_id: int, message_id: int) -> None:
        models.EntityMention.objects.get_or_create(entity_id=entity_id, message_id=message_id)

    # -- MessageStore ---------------------------------------------------------

    @transaction.atomic
    def add_message(
        self,
        episode_id: int,
        role: Role,
        content: str,
        at: datetime,
        embedding: Embedding | None = None,
    ) -> Message:
        # last_message_at needs no update: it is derived from the messages.
        row = models.Message.objects.create(
            episode_id=episode_id, role=str(role), content=content, created_at=at
        )
        if embedding is not None:
            models.MessageEmbedding.objects.update_or_create(
                message=row,
                model=embedding.model,
                defaults={"embedding": list(embedding.vector), "text_hash": embedding.text_hash},
            )
        return _to_message(row)

    def get_messages(
        self, episode_id: int, *, last: int | None = None, after: int | None = None
    ) -> list[Message]:
        rows = models.Message.objects.filter(episode_id=episode_id)
        if after is not None:
            rows = rows.filter(id__gt=after)
        if last is None:
            return [_to_message(r) for r in rows.order_by("id")]  # insertion order
        if last <= 0:
            return []
        newest = rows.order_by("-id")[:last]
        return [_to_message(r) for r in reversed(list(newest))]

    def get_message(self, message_id: int) -> Message | None:
        row = models.Message.objects.filter(pk=message_id, episode__status="A").first()
        return _to_message(row) if row else None

    # -- SearchStore ----------------------------------------------------------
    # No vector index in v1: a sequential scan is fast at personal scale
    # (EMBEDDINGS.md section 6). Every query filters on the model and on the source
    # row's status, so only active memory from the same embedding space comes back.

    def link_fact_entities(self, fact_id: int, entity_ids: Sequence[int]) -> None:
        if not models.Fact.objects.filter(pk=fact_id).exists():
            return
        known = models.Entity.objects.filter(pk__in=list(entity_ids)).values_list("pk", flat=True)
        models.FactEntity.objects.bulk_create(
            [models.FactEntity(fact_id=fact_id, entity_id=e) for e in known], ignore_conflicts=True
        )

    def entity_fact_candidates(
        self,
        entity_ids: Sequence[int],
        query: Embedding,
        limit: int,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        rows = (
            models.FactEmbedding.objects
            .filter(model=query.model, fact__in=_recallable_facts(),
                    fact__entity_links__entity_id__in=list(entity_ids),
                    fact__entity_links__entity__status="A")
            .filter(_window("fact__learned_at", after, before))
            .annotate(distance=CosineDistance("embedding", list(query.vector)))
            .select_related("fact")
            .distinct()
            .order_by("distance")[:limit]
        )
        return [_fact_candidate(r.fact, semantic=1 - r.distance) for r in rows]  # type: ignore[attr-defined]

    def stored_vectors(self, kind: MemoryKind, ids: Sequence[int], model: str) -> dict[int, list[float]]:
        table, key, column = {
            MemoryKind.MESSAGE: (models.MessageEmbedding, "message_id", "embedding"),
            MemoryKind.FACT: (models.FactEmbedding, "fact_id", "embedding"),
            MemoryKind.ATTACHMENT: (models.AttachmentChunkEmbedding, "chunk_id", "embedding"),
            MemoryKind.EPISODE: (models.EpisodeEmbedding, "episode_id", "summary_short"),
        }.get(kind, (None, "", ""))
        if table is None or not ids:
            return {}
        rows = table.objects.filter(**{f"{key}__in": list(ids)}, model=model).values_list(key, column)
        return {i: [float(x) for x in vector] for i, vector in rows if vector is not None}

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
        vector = list(query.vector)
        if kind is MemoryKind.MESSAGE:
            rows = (
                models.MessageEmbedding.objects
                .filter(model=query.model, message__episode__status="A")
                .filter(_window("message__created_at", after, before))
                .exclude(message__episode_id__in=exclude_episode_ids)
                .annotate(distance=CosineDistance("embedding", vector))
                .select_related("message")
                .order_by("distance")[:limit]
            )
            return [
                Candidate(
                    kind=kind, id=r.message.pk, text=r.message.content,
                    semantic=1 - r.distance, episode_id=r.message.episode_id,  # type: ignore[attr-defined]
                    at=r.message.created_at, role=Role(r.message.role),
                )
                for r in rows
            ]
        if kind is MemoryKind.EPISODE:
            # LEAST ignores NULLs, so an episode with one summary embedded still ranks.
            rows = (
                models.EpisodeEmbedding.objects
                .filter(model=query.model, episode__status="A")
                .filter(_window("episode__started_at", after, before))
                .exclude(episode_id__in=exclude_episode_ids)
                .annotate(distance=Least(
                    CosineDistance("summary_short", vector),
                    CosineDistance("summary_long", vector),
                ))
                .filter(distance__isnull=False)
                .select_related("episode")
                .order_by("distance")[:limit]
            )
            return [
                Candidate(
                    kind=kind, id=r.episode.pk, text=r.episode.summary_short,
                    semantic=1 - r.distance, episode_id=r.episode.pk,  # type: ignore[attr-defined]
                    at=r.episode.started_at,
                )
                for r in rows
            ]
        if kind is MemoryKind.FACT:
            rows = (
                models.FactEmbedding.objects
                .filter(model=query.model, fact__in=_recallable_facts())
                .filter(_window("fact__learned_at", after, before))
                .exclude(fact__episode_id__in=exclude_episode_ids)
                .annotate(distance=CosineDistance("embedding", vector))
                .select_related("fact")
                .order_by("distance")[:limit]
            )
            return [_fact_candidate(r.fact, semantic=1 - r.distance) for r in rows]  # type: ignore[attr-defined]
        if kind is MemoryKind.ATTACHMENT:
            rows = (
                models.AttachmentChunkEmbedding.objects
                .filter(model=query.model, chunk__attachment__in=_searchable_attachments())
                .filter(_window("chunk__attachment__shared_at", after, before))
                .exclude(chunk__attachment__episode_id__in=exclude_episode_ids)
                .annotate(distance=CosineDistance("embedding", vector))
                .select_related("chunk__attachment")
                .order_by("distance")[:limit]
            )
            return [_chunk_candidate(r.chunk, semantic=1 - r.distance) for r in rows]  # type: ignore[attr-defined]
        raise NotImplementedError(f"Search over {kind} arrives with its memory kind.")

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
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return []
        if kind is MemoryKind.MESSAGE:
            rows = (
                models.Message.objects
                .filter(mentions__entity_id__in=ids, episode__status="A")
                .filter(_window("created_at", after, before))
                .exclude(episode_id__in=exclude_episode_ids)
                .annotate(named=Count("mentions__entity", filter=Q(mentions__entity_id__in=ids), distinct=True))
                .order_by("-named", "-id")[:limit]
            )
            return [
                Candidate(
                    kind=kind, id=r.pk, text=r.content, lexical=r.named / len(ids),  # type: ignore[attr-defined]
                    episode_id=r.episode_id, at=r.created_at, role=Role(r.role),
                )
                for r in rows
            ]
        if kind is MemoryKind.EPISODE:
            entity_filter = Q(messages__mentions__entity_id__in=ids)
            rows = (
                models.Episode.active_objects
                .filter(entity_filter)
                .filter(_window("started_at", after, before))
                .exclude(pk__in=exclude_episode_ids)
                .exclude(summary_short="")
                .annotate(named=Count("messages__mentions__entity", filter=entity_filter, distinct=True))
                .order_by("-named", "-ended_at")[:limit]
            )
            return [
                Candidate(
                    kind=kind, id=r.pk, text=r.summary_short, lexical=r.named / len(ids),  # type: ignore[attr-defined]
                    episode_id=r.pk, at=r.started_at,
                )
                for r in rows
            ]
        if kind is MemoryKind.FACT:
            entity_filter = Q(sources__mentions__entity_id__in=ids)
            rows = (
                _recallable_facts()
                .filter(entity_filter)
                .filter(_window("learned_at", after, before))
                .exclude(episode_id__in=exclude_episode_ids)
                .annotate(named=Count("sources__mentions__entity", filter=entity_filter, distinct=True))
                .order_by("-named", "-id")[:limit]
            )
            return [_fact_candidate(r, lexical=r.named / len(ids)) for r in rows]  # type: ignore[attr-defined]
        if kind is MemoryKind.ATTACHMENT:
            entity_filter = Q(mentions__entity_id__in=ids)
            rows = (
                models.AttachmentChunk.objects
                .filter(entity_filter, attachment__in=_searchable_attachments())
                .filter(_window("attachment__shared_at", after, before))
                .exclude(attachment__episode_id__in=exclude_episode_ids)
                .annotate(named=Count("mentions__entity", filter=entity_filter, distinct=True))
                .select_related("attachment")
                .order_by("-named", "position")[:limit]
            )
            return [_chunk_candidate(r, lexical=r.named / len(ids)) for r in rows]  # type: ignore[attr-defined]
        raise NotImplementedError(f"Mentions of {kind} are not tracked.")

    # -- FactStore -------------------------------------------------------------

    @transaction.atomic
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
        row = models.Fact.objects.create(
            statement=statement.strip(), type=type, episode_id=episode_id,
            learned_at=at, last_confirmed_at=at,
        )
        row.sources.set(list(dict.fromkeys(message_ids)))
        models.FactEmbedding.objects.create(
            fact=row, model=embedding.model, embedding=list(embedding.vector),
            text_hash=embedding.text_hash,
        )
        return _to_fact(row)

    def confirm_fact(self, fact_id: int, at: datetime, message_ids: Sequence[int] = ()) -> None:
        # Bookkeeping: update(), so it isn't audited.
        models.Fact.objects.filter(pk=fact_id).update(
            times_confirmed=F("times_confirmed") + 1, last_confirmed_at=at
        )
        if message_ids:
            models.Fact.objects.get(pk=fact_id).sources.add(*message_ids)

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        # A real change to curated memory: save(), so it's audited ("Replaced by").
        row = models.Fact.objects.get(pk=old_id)
        row.superseded_by_id = new_id
        row.save()

    def similar_facts(self, query: Embedding, limit: int) -> list[tuple[Fact, float]]:
        rows = (
            models.FactEmbedding.objects
            .filter(model=query.model, fact__superseded_by__isnull=True,
                    fact__status__in=["A", "Z"])
            .annotate(distance=CosineDistance("embedding", list(query.vector)))
            .select_related("fact")
            .order_by("distance")[:limit]
        )
        return [(_to_fact(r.fact), 1 - r.distance) for r in rows]  # type: ignore[attr-defined]

    def get_facts(self, ids: Sequence[int]) -> list[Fact]:
        rows = {r.pk: r for r in models.Fact.objects.filter(pk__in=ids)}
        return [_to_fact(rows[i]) for i in ids if i in rows]

    def fact_history(self, fact_id: int) -> list[Fact]:
        row = models.Fact.objects.filter(pk=fact_id).first()
        if row is None:
            return []
        while (older := models.Fact.objects.filter(superseded_by=row).order_by("id").first()):
            row = older                   # walk back to the oldest
        chain, seen = [], set()
        while row is not None and row.pk not in seen:
            seen.add(row.pk)
            chain.append(_to_fact(row))
            row = row.superseded_by
        return chain

    # -- ToolCallStore ---------------------------------------------------------

    def add_tool_calls(
        self, episode_id: int, message_id: int | None, runs: Sequence[ToolRun]
    ) -> None:
        models.ToolCall.objects.bulk_create([
            models.ToolCall(
                episode_id=episode_id, message_id=message_id, name=r.name,
                arguments=dict(r.arguments), result=_json(r.result),
                duration_ms=r.duration_ms, created_at=r.at,
            )
            for r in runs
        ])

    def get_tool_calls(self, episode_id: int) -> list[ToolRun]:
        return [
            ToolRun(
                name=r.name, arguments=r.arguments,
                result=json.dumps(r.result, ensure_ascii=False),
                duration_ms=r.duration_ms, at=r.created_at, message_id=r.message_id,
            )
            for r in models.ToolCall.objects.filter(episode_id=episode_id).order_by("id")
        ]

    # -- EmbeddingCacheStore --------------------------------------------------

    def get_cached(self, keys: Sequence[str], model: str) -> dict[str, Embedding]:
        rows = models.EmbeddingCache.objects.filter(model=model, text_hash__in=keys)
        return {
            r.text_hash: Embedding(vector=list(r.embedding), model=r.model, text_hash=r.text_hash)
            for r in rows
        }

    def put_cached(self, entries: Mapping[str, Embedding]) -> None:
        models.EmbeddingCache.objects.bulk_create(
            [
                models.EmbeddingCache(model=e.model, text_hash=key, embedding=list(e.vector))
                for key, e in entries.items()
            ],
            ignore_conflicts=True,  # another request may have cached the same text
        )
