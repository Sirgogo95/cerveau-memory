"""Erasing memory for good (docs/DJANGO_SETUP.md, "The purge command").

Forgetting (status Z) and setting aside (I) never erase anything: the row stays, and
ChangeLog keeps snapshots of it. Purging does erase, for memory that has been
forgotten or set aside for a while:

1. The rows themselves, with everything that cascades from them: an episode's
   messages, vectors, tags, mentions, and tool calls; a tag's or entity's links.
2. Their ChangeLog snapshots, which hold copies of the content.
3. Facts learned only from purged episodes (a fact with sources elsewhere stays),
   and the history of every purged fact (the older facts it replaced).
4. Tool calls in other episodes whose stored arguments or results name a purged
   memory (a search result quoting a purged message holds its text).
5. The whole embedding cache, and activation rows for purged tags and entities.
   The cache is keyed by a hash of each text, and some of its texts are stored
   nowhere else (the model's search queries, searches typed on the memory page),
   so the entries for purged memory can't all be found; and a vector can be
   partly turned back into its text. It only saves time: stored memory keeps its
   own vectors, and a text needed again is embedded again.
6. Shared files: the row, its passages, vectors, mentions, and tags, and the file on
   disk (the most literal copy of what was shared). A file shared only in purged
   episodes goes with them.

What it can't reach: other conversations that repeat the content in their own words,
and copies of the database (backups, the eval database).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.db import transaction
from django.db.models import Count, Q

from brain.types import MemoryKind
from kernel.base_app.models import ChangeLog

from . import models
from .attachments import attachments_dir

KINDS = ("episode", "fact", "entity", "tag", "attachment")


@dataclass
class PurgePlan:
    """What a purge would erase: ids per model (and, always, the embedding cache)."""

    episodes: list[int] = field(default_factory=list)
    facts: list[int] = field(default_factory=list)
    entities: list[int] = field(default_factory=list)
    tags: list[int] = field(default_factory=list)
    derived_facts: list[int] = field(default_factory=list)   # only sources go, or history of one that goes
    attachments: list[int] = field(default_factory=list)     # forgotten, or shared only in purged episodes
    files: list[str] = field(default_factory=list)           # their paths on disk
    tool_calls: list[int] = field(default_factory=list)      # in other episodes, quoting purged memory

    @property
    def empty(self) -> bool:
        return not (self.episodes or self.facts or self.entities or self.tags or self.derived_facts
                    or self.attachments)

    def summary(self) -> str:
        return (f"{len(self.episodes)} episode(s), {len(self.facts) + len(self.derived_facts)} fact(s) "
                f"({len(self.derived_facts)} learned only from them or replaced by one), {len(self.entities)} "
                f"entit(ies), {len(self.tags)} tag(s), {len(self.attachments)} file(s), "
                f"{len(self.tool_calls)} tool call(s) quoting them")


def _due(model, statuses: Sequence[str], before: datetime, only: Sequence[int] | None) -> list[int]:
    rows = model.objects.filter(status__in=statuses, updated_at__lte=before)
    if only is not None:
        rows = rows.filter(pk__in=only)
    return list(rows.order_by("pk").values_list("pk", flat=True))


def _ids_named(value: Any) -> Iterable[str]:
    """Every memory id ("E3", "M12", "F7") a stored tool call names."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "id" and isinstance(item, str):
                yield item
            else:
                yield from _ids_named(item)
    elif isinstance(value, list):
        for item in value:
            yield from _ids_named(item)


def plan_purge(
    statuses: Sequence[str] = ("Z",),
    older_than: timedelta = timedelta(days=30),
    kinds: Sequence[str] = KINDS,
    *,
    now: datetime,
    episode_ids: Sequence[int] | None = None,
) -> PurgePlan:
    """What would be erased: memory of these kinds and statuses, unchanged for at
    least `older_than` (the grace period). `episode_ids` narrows episodes to those."""
    before = now - older_than
    plan = PurgePlan()
    if "episode" in kinds:
        plan.episodes = _due(models.Episode, statuses, before, episode_ids)
    if "fact" in kinds:
        plan.facts = _due(models.Fact, statuses, before, None)
    if "entity" in kinds:
        plan.entities = _due(models.Entity, statuses, before, None)
    if "tag" in kinds:
        plan.tags = _due(models.Tag, statuses, before, None)

    if plan.episodes:
        # Facts whose sources all lie in purged episodes (or that have none and came
        # from one): nothing else said them.
        elsewhere = Count("sources", filter=~Q(sources__episode_id__in=plan.episodes))
        derived = (models.Fact.objects.annotate(elsewhere=elsewhere)
                   .filter(Q(sources__episode_id__in=plan.episodes) | Q(episode_id__in=plan.episodes),
                           elsewhere=0)
                   .exclude(pk__in=plan.facts).distinct())
        plan.derived_facts = sorted(derived.values_list("pk", flat=True))
    # The history of a purged fact goes with it: an older fact it replaced would
    # otherwise lose its "replaced by" link and come back as current.
    going = {*plan.facts, *plan.derived_facts}
    frontier = set(going)
    while frontier:
        older = set(models.Fact.objects.filter(superseded_by_id__in=frontier)
                    .values_list("pk", flat=True)) - going
        going |= older
        frontier = older
    plan.derived_facts = sorted(going - set(plan.facts))
    facts = [*plan.facts, *plan.derived_facts]
    attachments = set(_due(models.Attachment, statuses, before, None)) if "attachment" in kinds else set()
    if plan.episodes:
        # Shared only in purged episodes: no other message links to it.
        shared_elsewhere = Count("message_links", filter=~Q(message_links__message__episode_id__in=plan.episodes))
        attachments |= set(models.Attachment.objects.annotate(elsewhere=shared_elsewhere)
                           .filter(message_links__message__episode_id__in=plan.episodes, elsewhere=0)
                           .values_list("pk", flat=True))
    plan.attachments = sorted(attachments)
    plan.files = list(models.Attachment.objects.filter(pk__in=plan.attachments)
                      .values_list("storage_path", flat=True))

    named = ({f"E{i}" for i in plan.episodes} | {f"F{i}" for i in facts} | {f"A{i}" for i in plan.attachments}
             | {f"M{i}" for i in models.Message.objects.filter(episode_id__in=plan.episodes)
                .values_list("pk", flat=True)})
    if named:
        for call in models.ToolCall.objects.exclude(episode_id__in=plan.episodes):
            if named & {*_ids_named(call.result), *_ids_named(call.arguments)}:
                plan.tool_calls.append(call.pk)
    return plan


@transaction.atomic
def execute_purge(plan: PurgePlan) -> None:
    """Erase what the plan lists. QuerySet.delete() is a real delete (the models'
    own delete() only sets status I)."""
    facts = [*plan.facts, *plan.derived_facts]
    models.ToolCall.objects.filter(pk__in=plan.tool_calls).delete()
    models.Fact.objects.filter(pk__in=facts).delete()
    models.Episode.objects.filter(pk__in=plan.episodes).delete()
    models.Entity.objects.filter(pk__in=plan.entities).delete()
    models.Tag.objects.filter(pk__in=plan.tags).delete()
    models.Attachment.objects.filter(pk__in=plan.attachments).delete()
    for model_name, ids in (("Episode", plan.episodes), ("Fact", facts),
                            ("Entity", plan.entities), ("Tag", plan.tags), ("Attachment", plan.attachments)):
        if ids:
            ChangeLog.objects.filter(model_name=model_name, instance_id__in=ids).delete()
    models.Activation.objects.filter(
        Q(kind=MemoryKind.TAG.value, node_id__in=plan.tags)
        | Q(kind=MemoryKind.ENTITY.value, node_id__in=plan.entities)
    ).delete()
    models.EmbeddingCache.objects.all().delete()
    # Files last, once the rows are gone: a failure above rolls the rows back and
    # leaves the files.
    folder = attachments_dir()
    transaction.on_commit(lambda: [(folder / path).unlink(missing_ok=True) for path in plan.files])
