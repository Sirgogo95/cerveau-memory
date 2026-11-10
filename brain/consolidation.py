"""Consolidation, the sleep cycle (PIPELINE.md, Consolidation): when an episode
closes, turn its raw messages into durable memory.

v1 writes a title, a short and a long summary (embedded, so episode search can find
them), tags, and entities with the messages that mention them. Facts, tag edges, and
episode links are v2.

One structured call to the background model does the extraction; Ollama constrains
the reply to EXTRACTION_SCHEMA. Episodes too long for one call are first condensed
part by part into notes, then extracted from the notes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

from .config import ModelProfile
from .dating import absolute_dates
from .language import main_language
from .embedding import embed_cached
from .ports import ChatProvider, ChatProviderError, Embedder, MemoryStore
from .prompt import day, estimate_tokens, local
from .text import contains_phrase, normalize_key, phrase_keys
from .attachments import mark_shared_files
from .tags import MAX_TAGS, MAX_TAG_LENGTH, clean_tags, resolve_tags  # noqa: F401  # re-exported
from .types import ChatMessage, Message, Role

if TYPE_CHECKING:
    from .facts import FactReport, FactSettings

ENTITY_TYPES = ("person", "project", "place", "product", "organization", "other")
# A single pasted log or file shouldn't crowd out the rest of the episode. The
# assistant's replies are cut shorter: they are long and mostly derived from what
# the user said, and left whole they dominate the extraction (the model starts
# attributing Cerveau's suggestions to the user).
MAX_USER_CHARS = 2000
MAX_ASSISTANT_CHARS = 600
# Existing tags shown to the model so it reuses the vocabulary instead of inventing
# near-duplicates ("coding" next to "programming").

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        # First on purpose: the model generates fields in this order, so naming the
        # language before writing makes the title and summaries follow it (without
        # it, a small model drifts to English for the short fields). Not stored.
        "language": {"type": "string"},
        "title": {"type": "string"},
        "summary_short": {"type": "string"},
        "summary_long": {"type": "string"},
        # Named for its language: after Spanish summaries, a field called "tags"
        # came out in Spanish half the time, despite the instructions.
        "english_tags": {"type": "array", "items": {"type": "string"}},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type", "aliases"],
            },
        },
    },
    "required": ["language", "title", "summary_short", "summary_long", "english_tags", "entities"],
}

SYSTEM = (
    "You consolidate a finished conversation between a user and their assistant, "
    "Cerveau, into long-term memory. Report only what the conversation contains; "
    "never invent details. Write the title and summaries in the language the user "
    "wrote in (the one they used most, if several); write tags in English."
)

INSTRUCTIONS = """\
Attribute carefully. Lines starting with "User:" are what the user said; lines \
starting with "Cerveau:" are the assistant's replies.
- Only write that the user said, decided, or plans something when a User line shows it.
- Cerveau's suggestions are not the user's decisions. Mention one only if the user \
accepted it, and say so ("the user agreed to ...").
When User lines begin with a name and a colon ("Caroline: ..."), several people \
share the conversation: say who said, did, or plans what by name, never "the user".
{dates}
Produce JSON with:
- language: the language the user wrote in (the one they used most, if several), for \
example "Spanish". Write title, summary_short, and summary_long in that language.
- title: 3 to 8 words naming what the conversation was about.
- summary_short: 1 or 2 sentences: the topic and any outcome.
- summary_long: up to about 150 words: decisions made, things the user stated about \
themselves or their work, and open questions. Refer to the user as "the user" (in \
the summary's language: "el usuario", "l'utilisateur").
- english_tags: the broad topics this conversation is mainly about, usually 1 to 3, at most \
{max_tags}, in English and lowercase whatever the conversation's language (for example \
"programming", "health", "travel"). Leave out anything mentioned only in passing. \
Topics only: never the names of specific people, projects, places, or products.
- entities: the specific named things discussed: people, projects, places, products, \
organizations. For each: its name, a type ({types}), and aliases: other names or ways \
the user referred to it (for example "my sister"). Leave out generic things, and \
names that only Cerveau brought up.

{material}"""

# The date rule, in the prompt only when the transcript carries its date (dated=True,
# the default). dated=False gives the prompts as they were before 2026-10-19, so
# evals can rebuild memory consolidated without dates from cached replies.
DATE_RULE = (
    'When the conversation says when something happened or will happen ("yesterday", "last '
    'week", "next month"), write the date it means, worked out from the conversation\'s date '
    'given above it ("on 7 May 2023", "in the week before 8 May 2023"). Keep those dates in '
    'the long summary.\n'
)

CONDENSE = """\
Condense this part of a longer conversation into notes of at most 200 words. Keep \
names, decisions, facts the user stated, and open questions. Write in {language}.

{transcript}"""


def user_language(messages: Sequence[Message]) -> str | None:
    """The language the user wrote in, worked out in code (brain/language.py), not left
    to the model: a small model named the wrong one, or wrote in another (Portuguese
    summaries of English chats, 2026-11-06)."""
    return main_language([m.content for m in messages if m.role is Role.USER])


def language_rule(language: str | None, what: str) -> str:
    """The prompt's first line when the language is known: which one to write `what` in."""
    return f"The user wrote in {language}: write {what} in {language}.\n\n" if language else ""


def with_language(schema: dict, language: str | None) -> dict:
    """The reply schema with its language field fixed to `language`, when known: the
    model names the language before writing, so it can't name another."""
    if not language or "language" not in schema["properties"]:
        return schema
    return {**schema, "properties": {**schema["properties"], "language": {"type": "string", "enum": [language]}}}


class ConsolidationError(Exception):
    """The background model's reply could not be used (not JSON, missing fields)."""


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    name: str
    type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Extraction:
    title: str
    summary_short: str
    summary_long: str
    tags: tuple[str, ...] = ()
    entities: tuple[ExtractedEntity, ...] = ()


@dataclass(slots=True)
class ConsolidationReport:
    """What consolidation stored, for logs and the management command."""

    episode_id: int
    title: str = ""
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    mentions: int = 0
    skipped: list[str] = field(default_factory=list)  # forgotten tags and entities
    unsupported: list[str] = field(default_factory=list)  # entities the user never named
    merged: list[tuple[str, str]] = field(default_factory=list)  # (new tag, existing it became)
    facts: FactReport | None = None  # when facts were learned
    fact_error: str = ""       # why fact learning failed (the rest was still saved)


# -- transcript --------------------------------------------------------------


def format_message(message: Message) -> str:
    assistant = message.role is Role.ASSISTANT
    limit = MAX_ASSISTANT_CHARS if assistant else MAX_USER_CHARS
    text = " ".join(message.content.split())
    if len(text) > limit:
        text = text[:limit] + " [...]"
    return f"{'Cerveau' if assistant else 'User'}: {text}"


def split_transcript(lines: Sequence[str], budget: int, chars_per_token: float) -> list[str]:
    """Consecutive lines grouped into parts of at most `budget` tokens each."""
    parts: list[list[str]] = [[]]
    used = 0
    for line in lines:
        cost = estimate_tokens(line, chars_per_token)
        if parts[-1] and used + cost > budget:
            parts.append([])
            used = 0
        parts[-1].append(line)
        used += cost
    return ["\n".join(p) for p in parts if p]


def when(messages: Sequence[Message], tz: tzinfo | None = None) -> str:
    """When a conversation took place, in the user's timezone, so the model can turn
    "yesterday" into a date: "Mon 2023-05-08 13:56", or a range across midnight."""
    first, last = local(messages[0].created_at, tz), local(messages[-1].created_at, tz)
    start = f"{day(first)} {first:%H:%M}"
    if last.date() == first.date():
        return start
    return f"{start} to {day(last)} {last:%H:%M}"


def material_for(
    messages: Sequence[Message], llm: ChatProvider, profile: ModelProfile, tz: tzinfo | None = None,
    dated: bool = True, language: str | None = None,
) -> str:
    """The text to extract from: the transcript if it fits the profile's budget,
    otherwise notes condensed from consecutive parts of it. Either way it opens with
    the conversation's date."""
    lines = [format_message(m) for m in messages]
    header = f"Conversation of {when(messages, tz)}" if dated else "Conversation"
    # Leave room for the instructions and the reply.
    instructions = INSTRUCTIONS.replace("{dates}", DATE_RULE if dated else "")
    budget = profile.memory_budget - estimate_tokens(instructions, profile.chars_per_token)
    transcript = "\n".join(lines)
    if estimate_tokens(transcript, profile.chars_per_token) <= budget:
        return f"{header}:\n{transcript}"

    notes = []
    for part in split_transcript(lines, budget, profile.chars_per_token):
        reply = llm.chat([
            ChatMessage(Role.SYSTEM, SYSTEM),
            ChatMessage(Role.USER, CONDENSE.format(transcript=f"{header}:\n{part}" if dated else part,
                                                   language=language or "the conversation's own language")),
        ])
        notes.append(reply.content.strip())
    joined = "\n\n".join(f"Part {i}:\n{n}" for i, n in enumerate(notes, 1))
    return f"Notes from consecutive parts of the {header[0].lower()}{header[1:]}:\n{joined}"


# -- extraction --------------------------------------------------------------


def clean_entities(raw: Sequence[object]) -> tuple[ExtractedEntity, ...]:
    """Drop malformed items, merge duplicates by key (keeping the first spelling and
    type), normalize the type, and drop aliases that repeat the name."""
    merged: dict[str, tuple[str, str, list[str]]] = {}  # key -> (name, type, aliases)
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        kind = str(item.get("type", "other")).strip().lower()
        _, _, aliases = merged.setdefault(
            normalize_key(name), (name, kind if kind in ENTITY_TYPES else "other", [])
        )
        aliases.extend(str(a).strip() for a in item.get("aliases") or [] if str(a).strip())

    entities = []
    for key, (name, kind, aliases) in merged.items():
        unique: dict[str, str] = {}
        for alias in aliases:
            if normalize_key(alias) != key:
                unique.setdefault(normalize_key(alias), alias)
        entities.append(ExtractedEntity(name, kind, tuple(unique.values())))
    return tuple(entities)


def parse_extraction(content: str) -> Extraction:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"Reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConsolidationError("Reply is not a JSON object.")
    summary_short = str(data.get("summary_short", "")).strip()
    if not summary_short:
        raise ConsolidationError("Reply has no summary_short.")
    return Extraction(
        title=str(data.get("title", "")).strip()[:200],
        summary_short=summary_short,
        summary_long=str(data.get("summary_long", "")).strip() or summary_short,
        tags=clean_tags(data.get("english_tags") or data.get("tags") or []),
        entities=clean_entities(data.get("entities") or []),
    )


def extract(
    messages: Sequence[Message],
    llm: ChatProvider,
    profile: ModelProfile,
    tz: tzinfo | None = None,
    dated: bool = True,
    pin_language: bool = True,
) -> Extraction:
    """Ask the background model for the episode's title, summaries, tags, and
    entities."""
    # pin_language=False: the prompts from before the language fix (cached LoCoMo replies).
    language = user_language(messages) if pin_language else None
    prompt = language_rule(language, "the title and summaries") + INSTRUCTIONS.format(
        max_tags=MAX_TAGS,
        types=", ".join(ENTITY_TYPES),
        material=material_for(messages, llm, profile, tz, dated, language),
        dates=DATE_RULE if dated else "",
    )
    reply = llm.chat(
        [ChatMessage(Role.SYSTEM, SYSTEM), ChatMessage(Role.USER, prompt)],
        schema=with_language(EXTRACTION_SCHEMA, language),
    )
    extraction = parse_extraction(reply.content)
    if not dated or not messages:
        return extraction
    # "On Friday Caroline passed..." with the date it means (brain/dating.py).
    said = messages[0].created_at.astimezone(tz).date()
    return replace(extraction, summary_short=absolute_dates(extraction.summary_short, said),
                   summary_long=absolute_dates(extraction.summary_long, said))


# -- applying ----------------------------------------------------------------


def own_aliases(extracted: ExtractedEntity, others: Sequence[ExtractedEntity], store: MemoryStore) -> list[str]:
    """The entity's aliases without those that belong to someone else: another
    entity's name or alias, in this reply or already in memory. The model once gave
    Caroline the aliases "Melanie" and "Mel" (LoCoMo), which blended two people:
    every mention of Melanie then counted as one of Caroline."""
    own = normalize_key(extracted.name)
    taken = {normalize_key(n) for e in others if normalize_key(e.name) != own for n in (e.name, *e.aliases)}
    kept = []
    for alias in extracted.aliases:
        key = normalize_key(alias)
        if key in taken:
            continue
        if any(normalize_key(e.name) != own and key in {normalize_key(n) for n in (e.name, *e.aliases)}
               for e in store.match_entities([key])):
            continue
        kept.append(alias)
    return kept


def mentioning(messages: Sequence[Message], names: Sequence[str]) -> list[Message]:
    """Messages that contain any of `names` as whole words (accents and case
    ignored)."""
    return [m for m in messages if any(contains_phrase(m.content, n) for n in names)]


def consolidate_episode(
    episode_id: int,
    *,
    store: MemoryStore,
    llm: ChatProvider,
    embedder: Embedder,
    profile: ModelProfile,
    now: datetime,
    force: bool = False,
    facts: FactSettings | None = None,
    tag_same_score: float | None = None,
    tz: tzinfo | None = None,
    dated: bool = True,
    group_facts: bool = False,
    pin_language: bool = True,
) -> ConsolidationReport | None:
    """Consolidate one closed episode. Returns None when there is nothing to do: the
    episode is unknown, inactive, still open, or already consolidated (unless
    `force`). Safe to run twice: summaries are overwritten, tags and mentions upserted.

    Raises ConsolidationError if the model's reply is unusable; the episode stays
    unconsolidated, so the sweep retries it later.

    With `tag_same_score`, a new tag at least that close in meaning to an existing
    one reuses it ("programming" for "coding").

    With `facts` (a facts.FactSettings), facts are learned too, in a separate model
    call. Best effort: if that call fails, the summaries, tags, and entities are kept
    and the error is reported (`extract_facts` can redo it later). `group_facts`: the
    episode is a conversation between several named people (facts.GROUP_FACT_INSTRUCTIONS).
    """
    found = store.get_episodes([episode_id])
    if not found:
        return None
    episode = found[0]
    if episode.is_open or (episode.consolidated_at is not None and not force):
        return None

    report = ConsolidationReport(episode_id)
    messages = store.get_messages(episode_id)
    if not messages:
        store.mark_consolidated(episode_id, now)
        return report
    # The transcript says which messages shared a file (its passages were read on
    # their own: brain/attachments.py).
    messages = mark_shared_files(messages, store)

    # Existing tags are not shown to the model: a 4B model copied them into
    # unrelated conversations ("pets" on a piano lesson). Tags that mean the same
    # are merged by meaning below instead.
    result = extract(messages, llm, profile, tz, dated, pin_language)

    short_vec, long_vec = embed_cached(
        [result.summary_short, result.summary_long], embedder, store
    )
    store.save_summary(
        episode_id,
        title=result.title,
        summary_short=result.summary_short,
        summary_long=result.summary_long,
        short_embedding=short_vec,
        long_embedding=long_vec,
    )
    report.title = result.title

    resolved = resolve_tags(result.tags, store, embedder, tag_same_score)
    report.skipped.extend(resolved.skipped)
    report.merged.extend(resolved.merged)
    for tag in resolved.tags:
        store.tag_episode(episode_id, tag.id, relevance=1.0)
        report.tags.append(tag.name)

    user_messages = [m for m in messages if m.role is Role.USER]
    for extracted in result.entities:
        # Enforced here, not left to the model: an entity must appear in something
        # the user wrote. Names only Cerveau brought up are its suggestions.
        if not mentioning(user_messages, [extracted.name, *extracted.aliases]):
            report.unsupported.append(extracted.name)
            continue
        aliases = own_aliases(extracted, result.entities, store)
        entity = store.upsert_entity(extracted.name, extracted.type, aliases)
        if entity is None:
            report.skipped.append(extracted.name)
            continue
        report.entities.append(entity.name)
        for message in mentioning(messages, [entity.name, *entity.aliases]):
            store.add_mention(entity.id, message.id)
            report.mentions += 1

    # Facts learned before an entity existed (or before this alias) get linked now.
    from .entity_links import link_entity_facts
    linked = [e for e in (store.match_entities(phrase_keys(" ".join(report.entities))) if report.entities else [])]
    link_entity_facts(store, linked)

    if facts is not None:
        from .facts import learn_facts  # facts imports this module

        try:
            report.facts = learn_facts(
                episode_id, messages, store=store, llm=llm, embedder=embedder,
                profile=profile, settings=facts, now=now, tz=tz, dated=dated, group=group_facts,
                pin_language=pin_language,
            )
        except (ConsolidationError, ChatProviderError) as exc:
            report.fact_error = str(exc)

    store.mark_consolidated(episode_id, now)
    return report

