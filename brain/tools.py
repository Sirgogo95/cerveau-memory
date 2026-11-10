"""Deliberate recall: the memory tools the model can call during a turn.

Automatic recall (retrieval.py) searches once, with the user's message as the query,
before every reply. These tools let the model search again on purpose, with its own
query, a time window, or a person's name (`search_memory`), and open a memory it was
shown by its ID (`read_memory`). See docs/SEARCH_MEMORY_TOOL.md.

Read-only: the tools never change memory. They see exactly what automatic recall
sees (active rows only), and their results reach the model as reference material.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, tzinfo
from typing import Any

from .config import EngineConfig, ModelProfile
from .dating import annotate_dates
from .embedding import embed_cached
from .ports import Embedder, MemoryStore
from .names import fuzzy_names
from .prompt import day, local
from .keywords import keyword_terms
from .retrieval import merge, score, semantic_kept, worth_recalling
from .text import phrase_keys
from .types import Candidate, Fact, MemoryKind, Message, Role, ToolCall, ToolDefinition

SEARCH_MEMORY = ToolDefinition(
    name="search_memory",
    description=(
        "Search your long-term memory of past conversations with the user. Returns "
        "matching facts, episode summaries, messages, and passages of files the user shared, "
        "with IDs and dates. Use it when the "
        "user asks about the past and the notes you were given don't answer it, or "
        "asks for everything about someone or something."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in any language."},
            "kinds": {
                "type": "array",
                "items": {"type": "string", "enum": ["fact", "episode", "message", "attachment"]},
                "description": "What to search: facts, episode summaries, messages, passages of shared files (default: all).",
            },
            "after": {
                "type": "string",
                "description": "Only memories from this date on (YYYY-MM-DD).",
            },
            "before": {
                "type": "string",
                "description": "Only memories from before this date (YYYY-MM-DD).",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only memories mentioning these people, places, or projects.",
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": 20,
                "description": "Maximum number of results (default 8).",
            },
            "include_history": {
                "type": "boolean",
                "description": "With facts: also say what each one replaced, and when. For how something changed.",
            },
        },
        "required": ["query"],
    },
)

READ_MEMORY = ToolDefinition(
    name="read_memory",
    description=(
        "Open a memory by its ID, as shown in your notes or in search results: an "
        "episode (E12), a message (M34), a fact (F7), or a shared file (A3). Use it when you need exact "
        "details that a summary or snippet doesn't give, a fact's sources and history, or a file's passages."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The memory's ID, for example E12, M34, F7, or A3."},
            "detail": {
                "type": "string",
                "enum": ["short", "long", "messages", "passages"],
                "description": ("For an episode: its short or long summary, or its messages (default long). "
                                "For a file: its summary, or its passages."),
            },
            "message_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": ("With detail=messages: first and last message numbers to read, from 1. "
                                "With detail=passages: first and last passage numbers."),
            },
        },
        "required": ["id"],
    },
)

TOOLS = (SEARCH_MEMORY, READ_MEMORY)

DEFAULT_LIMIT = 8
MAX_LIMIT = 20
SNIPPET_CHARS = 400        # a message in search results
FULL_MESSAGE_CHARS = 2000  # a message opened with read_memory
NEIGHBOURS = 2             # messages shown around an opened message, on each side
NO_RESULTS = "No stored memory matches this search."
SEARCHABLE = (MemoryKind.FACT, MemoryKind.EPISODE, MemoryKind.MESSAGE, MemoryKind.ATTACHMENT)
KINDS_ERROR = 'kinds can only contain "fact", "episode", "message", and "attachment".'
LIMIT_REACHED = (
    "Tool call limit for this reply reached: answer with what you already have."
)
_ID = re.compile(r"^\s*([EMFA])(\d+)\s*$", re.IGNORECASE)
_SHOWN_ID = re.compile(r"\[([EMNFA]\d+) ·")


@dataclass(frozen=True)
class ToolContext:
    """What a tool needs during one turn.

    - `shown`: memory IDs the prompt already shows (E12, M34), reported as "already
      in context" instead of repeated.
    - `skip_message_ids`: the current conversation's messages that are in the prompt
      word for word (the new message among them); search leaves them out. The rest of
      the current episode (folded into its running summary) stays searchable.
    """

    store: MemoryStore
    embedder: Embedder
    config: EngineConfig
    profile: ModelProfile
    now: datetime
    tz: tzinfo | None = None
    shown: Collection[str] = ()
    skip_message_ids: Collection[int] = ()


class ToolError(ValueError):
    """Bad arguments from the model; reported back to it so it can try again."""


def shown_ids(texts: Sequence[str]) -> set[str]:
    """The memory IDs a prompt already shows (`[E12 · ...`, `- [M34 · ...`)."""
    return {match for text in texts for match in _SHOWN_ID.findall(text)}


def run_tool(call: ToolCall, ctx: ToolContext) -> str:
    """Run one tool call and return its result as JSON text for the model. Never
    raises for the model's mistakes: errors are results too, so it can recover."""
    try:
        if call.name == SEARCH_MEMORY.name:
            result = search_memory(ctx, **_arguments(call, SEARCH_MEMORY))
        elif call.name == READ_MEMORY.name:
            result = read_memory(ctx, **_arguments(call, READ_MEMORY))
        else:
            result = {"error": f"Unknown tool {call.name!r}. Available: search_memory, read_memory."}
    except ToolError as exc:
        result = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def _arguments(call: ToolCall, tool: ToolDefinition) -> dict[str, Any]:
    allowed = set(tool.parameters["properties"])
    unknown = set(call.arguments) - allowed
    if unknown:
        raise ToolError(f"Unknown arguments for {tool.name}: {', '.join(sorted(unknown))}.")
    return dict(call.arguments)


# -- search_memory ---------------------------------------------------------------


def search_memory(
    ctx: ToolContext,
    *,
    query: Any = "",
    kinds: Any = None,
    after: Any = None,
    before: Any = None,
    entities: Any = None,
    limit: Any = DEFAULT_LIMIT,
    include_history: Any = False,
) -> dict[str, Any]:
    """The best matching memories that aren't already in the prompt.

    Same signals as automatic recall (meaning, plus the names the query mentions),
    with a lower threshold (`search_min_score`): the model asked explicitly, so
    coverage matters more than precision. A filter replaces the threshold: with
    `entities` only memories mentioning them are returned, and with `after` /
    `before` only memories from that window, ranked by similarity but not dropped
    for it ("what did we talk about in September?" has no query to be close to).
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("search_memory needs a non-empty query.")
    wanted = _kinds(kinds)
    start, end = _date(after, "after", ctx.tz), _date(before, "before", ctx.tz)
    count = _limit(limit)
    names = [str(n) for n in entities or () if str(n).strip()] if isinstance(entities, list) else []

    restricted, unresolved = [], []
    for name in names:
        found = ctx.store.match_entities(phrase_keys(name))
        if not found and ctx.config.fuzzy_names:
            # A name asked for on purpose: a typo counts even without a capital.
            found = fuzzy_names(name, ctx.store.active_entities(), require_capital=False)
        (restricted.extend(found) if found else unresolved.append(name))
    # Names in the query itself add the name signal, as in automatic recall.
    mentioned = [e.id for e in ctx.store.match_entities(phrase_keys(query))]
    terms = keyword_terms(query)
    filtered = bool(names) or start is not None or end is not None
    skip = set(ctx.skip_message_ids)

    [vector] = embed_cached([query], ctx.embedder, ctx.store)
    scored: list[tuple[float, Candidate]] = []
    for kind in wanted:
        source = {MemoryKind.MESSAGE: ctx.config.messages, MemoryKind.EPISODE: ctx.config.episodes,
                  MemoryKind.ATTACHMENT: ctx.config.attachments,
                  MemoryKind.FACT: ctx.config.facts}[kind]
        # Room for the skipped messages, so they don't use up the limit.
        room = count + (len(skip) if kind is MemoryKind.MESSAGE else 0)
        if names:
            if not restricted:
                continue
            # Only memories mentioning a requested entity, with their similarity.
            candidates = ctx.store.mention_candidates(kind, [e.id for e in restricted], room, after=start, before=end)
            similarity = {c.id: c.semantic for c in ctx.store.semantic_candidates(
                kind, vector, MAX_LIMIT * 5, after=start, before=end)}
            candidates = [replace(c, semantic=similarity.get(c.id)) for c in candidates]
        else:
            candidates = semantic_kept(ctx.store, kind, vector, room, ctx.config, after=start, before=end)
            if mentioned:
                candidates = merge(candidates, ctx.store.mention_candidates(kind, mentioned, room, after=start, before=end))
            # Exact terms in the query (an invoice number, a setting) are looked up
            # literally too, as in automatic recall.
            if terms and source.w_keyword > 0:
                candidates = merge(candidates, ctx.store.keyword_candidates(kind, terms, room, after=start, before=end))
        if kind is MemoryKind.MESSAGE:
            candidates = [c for c in candidates if c.id not in skip and worth_recalling(c, ctx.config)]
        for candidate in candidates:
            value = score(candidate, source)
            if filtered or value >= ctx.config.search_min_score:
                scored.append((value, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    results, truncated = _fill(
        [_result(c, s, ctx, history=include_history is True) for s, c in scored[:count]],
        _budget_chars(ctx),
    )
    out: dict[str, Any] = {"query": query, "results": results}
    if not results:
        out["note"] = NO_RESULTS
    if unresolved:
        out["unknown_names"] = unresolved
    if truncated:
        out["truncated"] = True
    return out


def _kinds(kinds: Any) -> list[MemoryKind]:
    if kinds is None or kinds == []:
        return list(SEARCHABLE)
    if not isinstance(kinds, list):
        raise ToolError('kinds must be a list, for example ["fact", "episode"].')
    try:
        chosen = [MemoryKind(str(k).lower()) for k in kinds]
    except ValueError as exc:
        raise ToolError(KINDS_ERROR) from exc
    if any(k not in SEARCHABLE for k in chosen):
        raise ToolError(KINDS_ERROR)
    return list(dict.fromkeys(chosen))


def _date(value: Any, name: str, tz: tzinfo | None) -> datetime | None:
    """A date (or date-time) from the model, as the start of that day in the user's
    timezone."""
    if value in (None, ""):
        return None
    try:
        text = str(value).strip()
        if len(text) == 10:
            parsed = datetime.combine(date.fromisoformat(text), time())
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolError(f"{name} must be a date like 2026-10-01.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz) if tz else parsed.astimezone()
    return parsed


def _limit(limit: Any) -> int:
    try:
        return max(1, min(MAX_LIMIT, int(limit)))
    except (TypeError, ValueError) as exc:
        raise ToolError("limit must be a number from 1 to 20.") from exc


def _result(candidate: Candidate, value: float, ctx: ToolContext, history: bool = False) -> dict[str, Any]:
    if candidate.kind is MemoryKind.ATTACHMENT:
        memory_id = f"A{candidate.parent_id}"
    else:
        prefix = {MemoryKind.EPISODE: "E", MemoryKind.FACT: "F"}.get(candidate.kind, "M")
        memory_id = f"{prefix}{candidate.id}"
    out: dict[str, Any] = {"id": memory_id, "kind": candidate.kind.value}
    if candidate.at is not None:
        out["date"] = day(local(candidate.at, ctx.tz))
    if memory_id in ctx.shown:
        out["note"] = "already in context"
        return out
    if candidate.kind is MemoryKind.FACT:
        out["statement"] = candidate.text
        if history:
            older = [f for f in ctx.store.fact_history(candidate.id) if f.id != candidate.id
                     and f.superseded_by is not None]
            if older:
                out["replaced"] = [_fact_line(f, ctx) for f in older]
    elif candidate.kind is MemoryKind.ATTACHMENT:
        out["file"] = candidate.label
        out["text"] = _clip(candidate.text, SNIPPET_CHARS)
    elif candidate.kind is MemoryKind.EPISODE:
        [episode] = ctx.store.get_episodes([candidate.id]) or [None]
        if episode is not None and episode.title:
            out["title"] = episode.title
        out["summary"] = candidate.text
    else:
        out["speaker"] = _speaker(candidate.role)
        text = candidate.text if candidate.at is None else annotate_dates(candidate.text, local(candidate.at, ctx.tz).date())
        out["text"] = _clip(text, SNIPPET_CHARS)
        if candidate.episode_id is not None:
            out["episode"] = f"E{candidate.episode_id}"
    out["score"] = round(value, 2)
    return out


# -- read_memory -----------------------------------------------------------------


def read_memory(
    ctx: ToolContext, *, id: Any = "", detail: Any = "long", message_range: Any = None,
) -> dict[str, Any]:
    """An episode (summary or messages), a message with the ones around it, or a fact
    with its sources and history."""
    match = _ID.match(str(id))
    if not match:
        raise ToolError("id must look like E12 (an episode), M34 (a message), F7 (a fact), or A3 (a file).")
    prefix, number = match.group(1).upper(), int(match.group(2))
    if prefix == "M":
        return _read_message(ctx, number)
    if prefix == "A":
        return _read_attachment(ctx, number, str(detail or "long"), message_range)
    if prefix == "F":
        return _read_fact(ctx, number)
    return _read_episode(ctx, number, str(detail or "long"), message_range)


def _read_episode(ctx: ToolContext, episode_id: int, detail: str, message_range: Any) -> dict[str, Any]:
    if detail not in ("short", "long", "messages"):
        raise ToolError('detail must be "short", "long", or "messages".')
    [episode] = ctx.store.get_episodes([episode_id]) or [None]
    if episode is None:
        raise ToolError(f"There is no memory E{episode_id}.")
    out: dict[str, Any] = {"id": f"E{episode.id}", "date": day(local(episode.started_at, ctx.tz))}
    if episode.title:
        out["title"] = episode.title
    if detail == "messages" or not episode.summary_short:
        messages = ctx.store.get_messages(episode.id)
        first, last = _range(message_range, len(messages))
        lines = [_message(m, n, ctx) for n, m in enumerate(messages, 1) if first <= n <= last]
        out["messages"], truncated = _fill(lines, _budget_chars(ctx))
        out["total_messages"] = len(messages)
        if not episode.summary_short and detail != "messages":
            out["note"] = "Not summarized yet; here are its messages."
        if truncated:
            out["truncated"] = True
            out["note"] = "More messages than fit; use message_range to read the rest."
        return out
    out["summary"] = episode.summary_short if detail == "short" else (
        episode.summary_long or episode.summary_short
    )
    return out


def _read_attachment(ctx: ToolContext, attachment_id: int, detail: str, passage_range: Any) -> dict[str, Any]:
    """A shared file: its summary, or its passages (all that fit, or a range)."""
    if detail not in ("short", "long", "passages"):
        raise ToolError('For a file, detail must be "short", "long", or "passages".')
    attachment = ctx.store.get_attachment(attachment_id)
    if attachment is None:
        raise ToolError(f"There is no memory A{attachment_id}.")
    out: dict[str, Any] = {"id": f"A{attachment.id}", "file": attachment.filename}
    if attachment.shared_at is not None:
        out["date"] = day(local(attachment.shared_at, ctx.tz))
    if attachment.page_count:
        out["pages"] = attachment.page_count
    if attachment.extraction_status != "done":
        out["note"] = {"pending": "Still being read.", "unsupported": "Cerveau can't read this kind of file.",
                       "failed": attachment.error or "It couldn't be read."}.get(attachment.extraction_status, "")
        return out
    if attachment.title:
        out["title"] = attachment.title
    if detail != "passages" and attachment.summary_short:
        out["summary"] = attachment.summary_short if detail == "short" else (
            attachment.summary_long or attachment.summary_short)
        return out
    chunks = ctx.store.attachment_chunks(attachment.id)
    first, last = _range(passage_range, len(chunks))
    lines = [{"passage": c.position + 1, **({"where": c.pages} if c.pages else {}),
              **({"heading": c.heading} if c.heading else {}), "text": c.text}
             for c in chunks if first <= c.position + 1 <= last]
    out["passages"], truncated = _fill(lines, _budget_chars(ctx))
    out["total_passages"] = len(chunks)
    if not attachment.summary_short and detail != "passages":
        out["note"] = "Not summarized yet; here are its passages."
    if truncated:
        out["truncated"] = True
        out["note"] = "More passages than fit; use message_range to read the rest."
    return out


def _range(message_range: Any, total: int) -> tuple[int, int]:
    if message_range is None:
        return 1, total
    try:
        first, last = (int(n) for n in message_range)
    except (TypeError, ValueError) as exc:
        raise ToolError("message_range must be two numbers, for example [1, 10].") from exc
    if first > last:
        first, last = last, first
    return max(1, first), min(total, last)


def _read_fact(ctx: ToolContext, fact_id: int) -> dict[str, Any]:
    [fact] = ctx.store.get_facts([fact_id]) or [None]
    if fact is None or fact.status != "A":
        raise ToolError(f"There is no memory F{fact_id}.")
    sources = []
    for message_id in fact.source_message_ids:
        message = ctx.store.get_message(message_id)
        if message is not None:
            sources.append(_message(message, None, ctx))
    out: dict[str, Any] = {**_fact_line(fact, ctx), "type": fact.type,
                           "times_confirmed": fact.times_confirmed, "sources": sources}
    if fact.superseded_by is not None:
        out["note"] = f"Outdated: replaced by F{fact.superseded_by}."
    chain = ctx.store.fact_history(fact_id)
    if len(chain) > 1:
        out["history"] = [{**_fact_line(f, ctx), **({"current": True} if f.superseded_by is None else {})}
                          for f in chain]
    return out


def _fact_line(fact: Fact, ctx: ToolContext) -> dict[str, Any]:
    out: dict[str, Any] = {"id": f"F{fact.id}", "statement": fact.statement}
    if fact.learned_at is not None:
        out["date"] = day(local(fact.learned_at, ctx.tz))
    return out


def _read_message(ctx: ToolContext, message_id: int) -> dict[str, Any]:
    message = ctx.store.get_message(message_id)
    if message is None:
        raise ToolError(f"There is no memory M{message_id}.")
    siblings = ctx.store.get_messages(message.episode_id)
    position = next(i for i, m in enumerate(siblings) if m.id == message.id)
    around = siblings[max(0, position - NEIGHBOURS): position + NEIGHBOURS + 1]
    return {
        "id": f"M{message.id}",
        "episode": f"E{message.episode_id}",
        "messages": [
            {**_message(m, None, ctx), **({"opened": True} if m.id == message.id else {})}
            for m in around
        ],
    }


def _message(message: Message, number: int | None, ctx: ToolContext) -> dict[str, Any]:
    moment = local(message.created_at, ctx.tz)
    out: dict[str, Any] = {"id": f"M{message.id}"}
    if number is not None:
        out["n"] = number
    out.update({
        "date": f"{day(moment)} {moment:%H:%M}",
        "speaker": _speaker(message.role),
        "text": _clip(annotate_dates(message.content, moment.date()), FULL_MESSAGE_CHARS),  # as memory_line does
    })
    return out


# -- shared ------------------------------------------------------------------------


def _speaker(role: Role | None) -> str:
    return "Cerveau" if role is Role.ASSISTANT else "user"


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _budget_chars(ctx: ToolContext) -> int:
    """Characters one tool result may use: `tool_result_share` of the memory budget."""
    tokens = ctx.profile.memory_budget * ctx.config.tool_result_share
    return math.floor(tokens * ctx.profile.chars_per_token)


def _fill(items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], bool]:
    """Items, in order, while they fit in `budget` characters of JSON."""
    kept, used = [], 0
    for item in items:
        size = len(json.dumps(item, ensure_ascii=False))
        if kept and used + size > budget:
            return kept, True
        kept.append(item)
        used += size
    return kept, False


def tool_limit_result() -> str:
    return json.dumps({"error": LIMIT_REACHED})
