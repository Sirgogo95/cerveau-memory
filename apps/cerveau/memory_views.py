"""The memory viewer: what Cerveau knows, why it recalls what it does, and the means
to make it forget.

Django-side pages: they read the models directly (like the admin), except the lens,
which runs the engine's own retrieval pieces so it can't drift from what the model
sees. Changes go through the models' status methods, so each one is audited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import FileResponse, Http404, HttpRequest, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from brain.activation import FLOOR, update_activation
from brain.config import all_sources
from brain.embedding import embed_cached
from brain.keywords import keyword_terms
from brain.names import named_entities
from brain.periods import find_period
from brain.ports import EmbedderError
from brain.retrieval import (
    matched_tags,
    merge,
    score,
    semantic_kept,
    with_activation,
    with_topics,
    worth_recalling,
)
from brain.turn import utcnow
from brain.types import FACT_TYPES, Candidate, MemoryKind, Node

from . import models
from .attachments import attachments_dir
from .engine import get_engine

PAGE_SIZE = 30
LENS_LIMIT = 12
STATUSES = {"active": "A", "set_aside": "I", "forgotten": "Z"}
STATUS_CHOICES = [("active", "Active"), ("set_aside", "Set aside"), ("forgotten", "Forgotten"), ("all", "All")]


def describe_call(name: str, arguments: dict) -> str:
    """A tool call in words, like the chat page's recall trace."""
    if name == "search_memory":
        details = []
        if arguments.get("entities"):
            details.append("about " + ", ".join(map(str, arguments["entities"])))
        if arguments.get("after"):
            details.append(f"from {arguments['after']}")
        if arguments.get("before"):
            details.append(f"before {arguments['before']}")
        if len(arguments.get("kinds") or []) == 1:
            details.append(f"{arguments['kinds'][0]}s only")
        text = f"Searched memory for “{arguments.get('query', '')}”"
        return " · ".join([text, *details])
    if name == "read_memory":
        target = str(arguments.get("id", ""))
        what = {"M": f"message {target}", "F": f"fact {target}"}.get(target[:1].upper(), target)
        return f"Opened {what}" + (" · its messages" if arguments.get("detail") == "messages" else "")
    return f"Used {name}"


def _call_view(call: models.ToolCall) -> dict:
    result = call.result if isinstance(call.result, dict) else {}
    found = result.get("results")
    return {
        "label": describe_call(call.name, call.arguments or {}),
        "found": len(found) if isinstance(found, list) else None,
        "error": result.get("error", ""),
        "duration_ms": call.duration_ms,
        "pretty": json.dumps(call.result, indent=2, ensure_ascii=False),
    }


def _status_filter(request: HttpRequest) -> tuple[str, Q]:
    """The ?status= filter: active by default, or set aside, forgotten, all."""
    chosen = request.GET.get("status", "active")
    if chosen == "all":
        return chosen, Q()
    return chosen, Q(status=STATUSES.get(chosen, "A"))


# -- Timeline -----------------------------------------------------------------------


def timeline(request: HttpRequest):
    status, where = _status_filter(request)
    query = request.GET.get("q", "").strip()
    episodes = models.Episode.objects.filter(where)
    if query:
        episodes = episodes.filter(
            Q(title__icontains=query) | Q(summary_short__icontains=query)
            | Q(summary_long__icontains=query) | Q(messages__content__icontains=query)
        ).distinct()
    episodes = episodes.annotate(
        message_count=Count("messages", distinct=True),
        search_count=Count("tool_calls", distinct=True),
    ).order_by("-started_at").prefetch_related("episode_tags__tag")
    page = Paginator(episodes, PAGE_SIZE).get_page(request.GET.get("page"))
    current = get_engine().store.current_episode()
    return render(request, "cerveau/memory/timeline.html", {
        "tab": "timeline", "page": page, "status": status, "query": query,
        "status_choices": STATUS_CHOICES, "current_id": current.id if current else None,
    })


def episode_detail(request: HttpRequest, pk: int):
    episode = get_object_or_404(models.Episode, pk=pk)
    messages = list(models.Message.objects.filter(episode=episode).order_by("id"))
    calls_by_reply: dict[int | None, list[dict]] = {}
    for call in models.ToolCall.objects.filter(episode=episode).order_by("id"):
        calls_by_reply.setdefault(call.message_id, []).append(_call_view(call))
    entities = (
        models.Entity.objects.filter(mentions__message__episode=episode)
        .annotate(here=Count("mentions", filter=Q(mentions__message__episode=episode)))
        .order_by("-here", "name").distinct()
    )
    current = get_engine().store.current_episode()
    return render(request, "cerveau/memory/episode.html", {
        "tab": "timeline", "episode": episode,
        "turns": [{"message": m, "calls": calls_by_reply.get(m.pk, [])} for m in messages],
        "orphan_calls": calls_by_reply.get(None, []),
        "tags": models.EpisodeTag.objects.filter(episode=episode).select_related("tag").order_by("-relevance"),
        "entities": entities,
        "facts": models.Fact.objects.filter(episode=episode).order_by("id"),
        "files": models.Attachment.objects.filter(message_links__message__episode=episode).distinct().order_by("pk"),
        "is_current": current is not None and current.id == episode.pk,
    })


# -- People and things -------------------------------------------------------------


def entity_list(request: HttpRequest):
    status, where = _status_filter(request)
    query = request.GET.get("q", "").strip()
    kind = request.GET.get("type", "")
    entities = models.Entity.objects.filter(where)
    if query:
        entities = entities.filter(Q(name__icontains=query) | Q(aliases__icontains=query))
    if kind:
        entities = entities.filter(type=kind)
    entities = entities.annotate(mention_count=Count("mentions")).order_by("-mention_count", "name")
    page = Paginator(entities, PAGE_SIZE * 2).get_page(request.GET.get("page"))
    types = (
        models.Entity.objects.exclude(type="").values_list("type", flat=True)
        .distinct().order_by("type")
    )
    return render(request, "cerveau/memory/entities.html", {
        "tab": "entities", "page": page, "status": status, "query": query,
        "kind": kind, "types": types, "status_choices": STATUS_CHOICES,
    })


def entity_detail(request: HttpRequest, pk: int):
    entity = get_object_or_404(models.Entity, pk=pk)
    mentions = (
        models.Message.objects.filter(mentions__entity=entity)
        .select_related("episode").order_by("-created_at")
    )
    return render(request, "cerveau/memory/entity.html", {
        "tab": "entities", "entity": entity, "mentions": mentions[:100],
        "mention_total": mentions.count(),
    })


# -- Facts ---------------------------------------------------------------------------


def fact_list(request: HttpRequest):
    status, where = _status_filter(request)
    query = request.GET.get("q", "").strip()
    kind = request.GET.get("type", "")
    show_history = request.GET.get("history") == "1"
    facts = models.Fact.objects.filter(where).select_related("episode", "superseded_by")
    if not show_history:
        facts = facts.filter(superseded_by__isnull=True)
    if query:
        facts = facts.filter(statement__icontains=query)
    if kind:
        facts = facts.filter(type=kind)
    page = Paginator(facts.order_by("-learned_at"), PAGE_SIZE * 2).get_page(request.GET.get("page"))
    return render(request, "cerveau/memory/facts.html", {
        "tab": "facts", "page": page, "status": status, "query": query, "kind": kind,
        "types": FACT_TYPES, "status_choices": STATUS_CHOICES, "show_history": show_history,
    })


def fact_detail(request: HttpRequest, pk: int):
    fact = get_object_or_404(models.Fact.objects.select_related("episode"), pk=pk)
    chain = get_engine().store.fact_history(fact.pk)
    sources = models.Message.objects.filter(facts=fact).select_related("episode").order_by("created_at")
    return render(request, "cerveau/memory/fact.html", {
        "tab": "facts", "fact": fact, "sources": sources,
        "history": chain if len(chain) > 1 else [],
    })


# -- Files ---------------------------------------------------------------------------


def file_list(request: HttpRequest):
    status, where = _status_filter(request)
    query = request.GET.get("q", "").strip()
    files = models.Attachment.objects.filter(where).select_related("episode")
    if query:
        files = files.filter(Q(filename__icontains=query) | Q(title__icontains=query)
                             | Q(summary_short__icontains=query))
    page = Paginator(files.order_by("-added_at"), PAGE_SIZE * 2).get_page(request.GET.get("page"))
    return render(request, "cerveau/memory/files.html", {
        "tab": "files", "page": page, "status": status, "query": query, "status_choices": STATUS_CHOICES,
    })


def file_detail(request: HttpRequest, pk: int):
    attachment = get_object_or_404(models.Attachment.objects.select_related("episode"), pk=pk)
    chunks = models.AttachmentChunk.objects.filter(attachment=attachment).order_by("position")
    return render(request, "cerveau/memory/file.html", {
        "tab": "files", "file": attachment, "chunks": chunks,
        "tags": models.Tag.objects.filter(attachment_tags__attachment=attachment).order_by("name"),
        "shared_in": (models.Message.objects.filter(attachment_links__attachment=attachment)
                      .select_related("episode").order_by("created_at")),
        "has_original": attachment_file(attachment).exists(),
    })


def attachment_file(attachment: models.Attachment):
    return attachments_dir() / attachment.storage_path


def file_original(request: HttpRequest, pk: int):
    """The file as it was shared. Text and PDF open in the browser; others download.
    A forgotten file isn't served."""
    attachment = get_object_or_404(models.Attachment, pk=pk, status__in=["A", "I"])
    path = attachment_file(attachment)
    if not path.exists():
        raise Http404("The file isn't on disk any more.")
    inline = attachment.media_type.startswith("text/") or attachment.media_type == "application/pdf"
    content_type = "text/plain; charset=utf-8" if attachment.media_type.startswith("text/") else attachment.media_type
    response = FileResponse(path.open("rb"), content_type=content_type, as_attachment=not inline,
                            filename=attachment.filename)
    response["X-Content-Type-Options"] = "nosniff"
    return response


# -- Lens: what a question would recall ----------------------------------------------


@dataclass(frozen=True)
class LensRow:
    candidate: Candidate
    score: float
    automatic: bool       # recalled before every reply
    searchable: bool      # found by search_memory
    current: bool         # in the conversation going on now (automatic recall skips it)
    episode_title: str = ""

    @property
    def memory_id(self) -> str:
        if self.candidate.kind is MemoryKind.ATTACHMENT:
            return f"A{self.candidate.parent_id}"
        prefix = {MemoryKind.EPISODE: "E", MemoryKind.FACT: "F"}.get(self.candidate.kind, "M")
        return f"{prefix}{self.candidate.id}"

    @property
    def percent(self) -> float:
        return round(max(0.0, min(1.0, self.score)) * 100, 1)


def lens(request: HttpRequest):
    """Score the best memories for a question against both thresholds: automatic
    recall's (`min_score`) and the search tool's (`search_min_score`). Same signals
    and scoring as the engine; nothing is excluded, so near misses show too."""
    engine = get_engine()
    query = request.GET.get("q", "").strip()
    include_current = request.GET.get("current") == "1"
    context = {
        "tab": "lens", "query": query, "include_current": include_current,
        "min_score": engine.config.messages.min_score,
        "search_min_score": engine.config.search_min_score,
        "short_reply_chars": engine.config.short_reply_chars,
        "embedding": engine.embedder.model,
    }
    if not query:
        return render(request, "cerveau/memory/lens.html", context)
    store = engine.store
    try:
        [vector] = embed_cached([query], engine.embedder, store)
    except EmbedderError as exc:
        context["error"] = str(exc)
        return render(request, "cerveau/memory/lens.html", context)
    named = named_entities(query, store, engine.config.fuzzy_names)
    topics = matched_tags(vector, store, engine.config)
    terms = keyword_terms(query)
    # What this question would bring to mind, from what's on it now (not saved).
    activation: dict[Node, float] = {}
    if activating(engine.config):
        matched = {(MemoryKind.ENTITY, e.id): 1.0 for e in named}
        matched.update({(MemoryKind.TAG, t.id): s for t, s in topics})
        activation = update_activation(store.load_activation(), matched, store, utcnow(), engine.config)
    current = store.current_episode()
    skip = [current.id] if current is not None and not include_current else []
    # A period in the question filters, as in a real turn (brain/periods.py).
    tz = ZoneInfo(engine.config.timezone) if engine.config.timezone else None
    period = find_period(query, utcnow(), tz)
    window = {"after": period.start, "before": period.end} if period else {}
    context["period"] = period
    rows: list[LensRow] = []
    for kind, source in ((MemoryKind.FACT, engine.config.facts),
                         (MemoryKind.EPISODE, engine.config.episodes),
                         (MemoryKind.MESSAGE, engine.config.messages),
                         (MemoryKind.ATTACHMENT, engine.config.attachments)):
        candidates = semantic_kept(store, kind, vector, LENS_LIMIT, engine.config,
                                   exclude_episode_ids=skip, **window)
        if named:
            candidates = merge(candidates, [
                c for c in store.mention_candidates(
                    kind, [e.id for e in named], LENS_LIMIT, exclude_episode_ids=skip, **window)
                if worth_recalling(c, engine.config)
            ])
        if terms and source.w_keyword > 0:
            candidates = merge(candidates, [
                c for c in store.keyword_candidates(kind, terms, LENS_LIMIT, exclude_episode_ids=skip, **window)
                if worth_recalling(c, engine.config)
            ])
        if kind is MemoryKind.EPISODE and topics and source.w_tag > 0:
            candidates = with_topics(candidates, topics, store)
        if activation and source.w_act > 0:
            candidates = with_activation(candidates, activation, store)
        for c in candidates:
            value = score(c, source)
            rows.append(LensRow(
                candidate=c, score=value,
                automatic=period is not None or value >= source.min_score,
                searchable=value >= engine.config.search_min_score,
                current=current is not None and c.episode_id == current.id,
            ))
    rows.sort(key=lambda r: r.score, reverse=True)
    rows = rows[:LENS_LIMIT]
    titles = dict(
        models.Episode.objects.filter(pk__in={r.candidate.episode_id for r in rows})
        .values_list("pk", "title")
    )
    context.update({
        "rows": [LensRow(**{**r.__dict__, "episode_title": titles.get(r.candidate.episode_id) or ""})
                 for r in rows],
        "named": named,
        "topics": topics,
        "on_mind": mind_rows({n: (v, utcnow()) for n, v in activation.items()}, utcnow(),
                             engine.config.activation_half_life_hours)[:6],
    })
    return render(request, "cerveau/memory/lens.html", context)


# -- On my mind -----------------------------------------------------------------------


def activating(config) -> bool:
    return any(s.w_act > 0 for s in all_sources(config))


@dataclass(frozen=True)
class MindRow:
    name: str
    kind: str          # "topic", or the entity's type
    value: float       # faded to now
    updated_at: datetime
    url: str = ""

    @property
    def percent(self) -> float:
        return round(min(1.0, self.value) * 100, 1)


def mind_rows(activation: dict[Node, tuple[float, datetime]], now: datetime, half_life_hours: float) -> list[MindRow]:
    """Active tags and entities, strongest first, each value faded by the hours since
    it was set (a turn's fading only happens when a message arrives)."""
    tags = models.Tag.active_objects.in_bulk([i for k, i in activation if k is MemoryKind.TAG])
    entities = models.Entity.active_objects.in_bulk([i for k, i in activation if k is MemoryKind.ENTITY])
    rows = []
    for (kind, node_id), (value, at) in activation.items():
        faded = value * 0.5 ** (max((now - at).total_seconds(), 0) / 3600 / half_life_hours)
        if faded < FLOOR:
            continue
        if kind is MemoryKind.TAG and node_id in tags:
            rows.append(MindRow(tags[node_id].name, "topic", faded, at))
        elif kind is MemoryKind.ENTITY and node_id in entities:
            entity = entities[node_id]
            rows.append(MindRow(entity.name, entity.type or "thing", faded, at,
                                reverse("cerveau:entity", args=[node_id])))
    return sorted(rows, key=lambda r: r.value, reverse=True)


def mind(request: HttpRequest):
    """What Cerveau has in mind now: the activation graph's active nodes."""
    engine = get_engine()
    config = engine.config
    rows = mind_rows(engine.store.load_activation(), utcnow(), config.activation_half_life_hours)
    return render(request, "cerveau/memory/mind.html", {
        "tab": "mind", "rows": rows, "enabled": activating(config),
        "turn_decay": round(config.activation_turn_decay * 100),
        "half_life": config.activation_half_life_hours,
    })


# -- Changing status ------------------------------------------------------------------

STATUS_MODELS = {"episode": models.Episode, "entity": models.Entity, "tag": models.Tag,
                 "fact": models.Fact, "attachment": models.Attachment}
ACTIONS = {"forget": "set_archived", "set_aside": "set_inactive", "restore": "reactivate"}


@require_POST
def change_status(request: HttpRequest, kind: str, pk: int):
    """Forget (Z: never used or relearned), set aside (I: unused, can come back), or
    restore (A). Through the model's own methods, so the change is audited."""
    model = STATUS_MODELS.get(kind)
    action = ACTIONS.get(request.POST.get("action", ""))
    if model is None or action is None:
        raise Http404("Unknown memory or action.")
    item = get_object_or_404(model, pk=pk)
    user = request.user if request.user.is_authenticated else None
    getattr(item, action)(user)
    target = request.POST.get("next", "")
    if not url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}):
        target = "/memory/"
    return HttpResponseRedirect(target)
