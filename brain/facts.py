"""Facts: learning short, standalone statements the user made (docs/FACTS.md).

At consolidation, after the summaries, one separate model call extracts facts from
the episode's transcript. Each fact names the user lines that state it; code checks
that they are user lines (only facts the user stated, never Cerveau's suggestions).
Then each new fact is deduplicated against what's already known:

- very close to an active fact (`same_score`): the same fact, confirmed again;
- close (`related_score` and above): the model is asked: same, update,
  contradiction, or distinct. An update or contradiction replaces the old fact,
  which stays as history (`supersede_fact`);
- otherwise: a new fact.

A fact matching a forgotten one (Z) is dropped: forgotten means forgotten. The
thresholds come from the embedding profile, calibrated with `python -m evals.fact_pairs`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, tzinfo

from .config import EngineConfig, ModelProfile
from .dating import absolute_dates
from .consolidation import (
    ConsolidationError,
    format_message,
    language_rule,
    split_transcript,
    user_language,
    when,
    with_language,
)
from .embedding import embed_cached
from .entity_links import link_fact
from .ports import ChatProvider, Embedder, MemoryStore
from .prompt import estimate_tokens
from .types import FACT_TYPES, ChatMessage, Fact, Message, Role

MAX_FACTS = 8
MAX_STATEMENT_CHARS = 300
RELATIONS = ("same", "update", "contradiction", "distinct")

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        # First on purpose, as in consolidation: the model names the language before
        # writing, so the statements follow it.
        "language": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "type": {"type": "string", "enum": list(FACT_TYPES)},
                    "lines": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["statement", "type", "lines"],
            },
        },
    },
    "required": ["language", "facts"],
}

FACT_SYSTEM = (
    "You keep a person's long-term memory. From a finished conversation between the "
    "user and their assistant, Cerveau, you note the durable facts the user stated "
    "about themselves, their life, and their work. Report only what the user said; "
    "never invent or guess."
)

# The date rule (see consolidation.DATE_RULE): left out with dated=False.
FACT_DATE_RULE = (
    'When a fact says when something happened or will happen, write the date it means, worked '
    'out from the conversation\'s date given above it: "Caroline went to an LGBTQ support group '
    'on 7 May 2023", not "yesterday"; "the user\'s race is on Saturday 13 May 2023", not "this '
    'Saturday".\n\n'
)

FACT_INSTRUCTIONS = """\
The conversation's lines are numbered. Lines with "User:" are what the user said; \
lines with "Cerveau:" are the assistant's.

A fact must pass three tests:
1. Durable: still true weeks from now. Not a task for today or tonight, a passing \
mood, or a question.
2. Self-contained: understandable alone, months from now. Name the people and things \
it's about.
3. Useful: knowing it would change a future answer.

When User lines begin with a name and a colon ("Caroline: ..."), several people \
share the conversation: say who said, did, or plans what by name, never "the user".

{dates}Write each fact as it is now, not as an event: "Pedro's wedding is on March 3", not \
"the user moved Pedro's wedding to March 3".

Only facts the user stated or confirmed in a User line. Cerveau's suggestions are not \
facts unless the user accepted them. Not facts: questions, hypotheticals, options \
considered but not chosen, general knowledge.

Produce JSON with:
- language: the language the user wrote in (the one they used most, if several).
- facts: at most {max_facts}. For each: statement (one short sentence in that \
language; refer to the user as "the user", "el usuario", or "l'utilisateur"), type, \
and lines: the numbers of the User lines that state it.
  Types: person (someone in the user's life and who they are), preference (likes, \
dislikes, habits), plan (something planned, with its date), project (work or \
projects and their state), health, place (where the user or someone lives, works, \
goes), other.
If nothing passes the tests, return an empty list.

{transcript}"""

# Conversations between several named people (an imported group chat, a meeting
# transcript, LoCoMo): the rules above, written for one user, made the small local
# models return few facts or none, while going speaker by speaker found most of what
# gpt-oss found (gemma4:e4b on 20 LoCoMo sessions: 131 of 172 reference facts, 1
# wrong; the prompt above: 83). Chosen by the caller (`group`), never guessed from
# the text. Worded as measured, the one-person sentence included.
GROUP_FACT_INSTRUCTIONS = """\
Find the people who speak in this conversation. For each person, list what they said
about their own life that would be worth remembering months from now: events and their dates,
plans, family and friends, likes, work, health. Start each statement with the person's name. \
When User lines begin with a name and a colon ("Caroline: ..."), several people share the \
conversation: name the person in every fact, never "the user". Otherwise one person talks with \
Cerveau: call them "the user" and keep how others relate to them ("the user's daughter Sofia"); \
what Cerveau says is not a fact unless the user accepted it.

{dates}Answer with JSON: {{"people": [{{"name": ..., "facts": [{{"statement": ..., "lines": [line numbers]}}]}}]}}
At most {max_facts} facts in all.

{transcript}"""

GROUP_FACT_SCHEMA = {
    "type": "object",
    "properties": {"people": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "facts": {"type": "array", "items": {
                "type": "object",
                "properties": {"statement": {"type": "string"},
                               "lines": {"type": "array", "items": {"type": "integer"}}},
                "required": ["statement", "lines"],
            }},
        },
        "required": ["name", "facts"],
    }}},
    "required": ["people"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"relation": {"type": "string", "enum": list(RELATIONS)}},
    "required": ["relation"],
}

JUDGE = """\
Two statements about the same user, an old one already remembered and a new one:

Old: {old}
New: {new}

Decide in this order:
1. same: do they say the same thing, perhaps in other words or another language?
2. update: is it the same thing with a value that changed (a date, a place, a \
number, a choice), so the old one is no longer true?
3. contradiction: does the new one say the old one is false?
4. distinct: otherwise: they are about different things, and both stay true.

Examples:
- "Lucía lives in Montreal." / "Lucía, the user's sister, is based in Montreal." -> same
- "Lucía lives in Montreal." / "Lucía moved to Toronto." -> update
- "The user has a dog." / "The user has no pets." -> contradiction
- "Lucía lives in Montreal." / "Lucía is a nurse." -> distinct
- "The user wants a funny speech." / "The user will tell a childhood story in the \
speech." -> distinct

Answer with JSON: {{"relation": "same" | "update" | "contradiction" | "distinct"}}"""


@dataclass(frozen=True)
class FactSettings:
    """How facts are learned: the deduplication thresholds (from the embedding
    profile) and how many facts one conversation may give."""

    same_score: float = 0.97
    related_score: float = 0.70
    max_facts: int = MAX_FACTS


def fact_settings(config: EngineConfig) -> FactSettings | None:
    """The settings for learning facts from the engine config, or None when off."""
    if not config.learn_facts:
        return None
    return FactSettings(same_score=config.fact_same_score, related_score=config.fact_related_score,
                        max_facts=config.max_facts)


@dataclass(frozen=True)
class ExtractedFact:
    statement: str
    type: str
    message_ids: tuple[int, ...]


@dataclass
class FactReport:
    """What fact learning did, for logs and the management command."""

    added: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    updated: list[tuple[str, str]] = field(default_factory=list)  # (old, new)
    skipped: list[str] = field(default_factory=list)              # matches a forgotten fact
    unsupported: list[str] = field(default_factory=list)          # no user line states it


# -- extraction ----------------------------------------------------------------


def numbered(messages: Sequence[Message]) -> list[str]:
    """Transcript lines numbered from 1, as the extraction prompt shows them."""
    return [f"[{n}] {format_message(m)}" for n, m in enumerate(messages, 1)]


def parse_facts(
    content: str, messages: Sequence[Message], limit: int = MAX_FACTS, tz: tzinfo | None = None,
    dates: bool = True,
) -> tuple[list[ExtractedFact], list[str]]:
    """The facts in the model's reply, each mapped to the user messages its line
    numbers point to. Facts citing no user line are returned apart: unsupported.
    Relative dates ("last Friday") are made absolute from the date of the first line
    a fact cites, in `tz` (brain/dating.py); not with `dates` False (the undated runs)."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"Fact reply is not JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("facts", []), list):
        raise ConsolidationError("Fact reply has no list of facts.")
    facts, unsupported, seen = [], [], set()
    for item in data.get("facts", []):
        if not isinstance(item, dict):
            continue
        statement = " ".join(str(item.get("statement", "")).split())[:MAX_STATEMENT_CHARS]
        if not statement or statement.casefold() in seen:
            continue
        seen.add(statement.casefold())
        kind = str(item.get("type", "other"))
        lines = item.get("lines") or []
        # Enforced here, not left to the model: a fact must come from what the user
        # wrote. Lines pointing at Cerveau's replies, or nowhere, don't count.
        ids = tuple(dict.fromkeys(
            messages[n - 1].id for n in lines
            if isinstance(n, int) and 1 <= n <= len(messages) and messages[n - 1].role is Role.USER
        ))
        if not ids:
            unsupported.append(statement)
            continue
        facts.append(ExtractedFact(statement, kind if kind in FACT_TYPES else "other", ids))
    speakers = speaker_names(messages)
    if speakers:
        facts = [replace(f, statement=named(f.statement, {speakers[i] for i in f.message_ids})) for f in facts]
    if dates:
        said = {m.id: m.created_at for m in messages}
        facts = [replace(f, statement=absolute_dates(f.statement, said[f.message_ids[0]].astimezone(tz).date()))
                 for f in facts]
    return facts[:limit], unsupported


SPEAKER = re.compile(r"([A-Z][\w'-]*(?: [A-Z][\w'-]*)?): ")
THE_USER = re.compile(r"\b(?:[Tt]he user|[Ee]l usuario|[Ll]'utilisateur)\b")


def speaker_names(messages: Sequence[Message]) -> dict[int, str]:
    """Who wrote each user message, when several people share the conversation:
    every user line starts with a name and a colon ("Caroline: ...") and there are at
    least two names. Empty otherwise, so a one-user chat ("Note: buy milk") is left alone."""
    names = {}
    for m in messages:
        if m.role is not Role.USER:
            continue
        match = SPEAKER.match(m.content)
        if match is None:
            return {}
        names[m.id] = match.group(1)
    return names if len(set(names.values())) >= 2 else {}


def named(statement: str, speakers: set[str]) -> str:
    """"The user" made a name, when the lines a fact cites are all one speaker's:
    small models write "the user" despite the prompt's rule, which leaves the fact
    without its person (and out of that person's entity block)."""
    if len(speakers) != 1:
        return statement
    [name] = speakers
    return THE_USER.sub(name, statement)


def extract_facts(
    messages: Sequence[Message], llm: ChatProvider, profile: ModelProfile, limit: int = MAX_FACTS,
    tz: tzinfo | None = None, dated: bool = True, group: bool = False, pin_language: bool = True,
) -> tuple[list[ExtractedFact], list[str]]:
    """Ask the background model for the conversation's facts. A long conversation is
    read in parts that fit the budget, keeping the global line numbers. `group`: a
    conversation between several named people (GROUP_FACT_INSTRUCTIONS)."""
    instructions, schema = (GROUP_FACT_INSTRUCTIONS, GROUP_FACT_SCHEMA) if group else (FACT_INSTRUCTIONS, FACT_SCHEMA)
    lines = numbered(messages)
    rule = FACT_DATE_RULE if dated else ""
    budget = profile.memory_budget - estimate_tokens(instructions.replace("{dates}", rule), profile.chars_per_token)
    facts: list[ExtractedFact] = []
    unsupported: list[str] = []
    language = user_language(messages) if pin_language else None
    schema = with_language(schema, language)
    for part in split_transcript(lines, budget, profile.chars_per_token):
        prompt = language_rule(language, "each statement") + instructions.format(
            max_facts=limit, types=", ".join(FACT_TYPES), dates=rule,
            transcript=f"Conversation of {when(messages, tz)}:\n{part}" if dated else f"Conversation:\n{part}",
        )
        reply = llm.chat(
            [ChatMessage(Role.SYSTEM, FACT_SYSTEM), ChatMessage(Role.USER, prompt)],
            schema=schema,
        )
        content = by_person(reply.content) if group else reply.content
        found, rejected = parse_facts(content, messages, limit, tz, dated)
        facts += found
        unsupported += rejected
    return facts[:limit], unsupported


def by_person(content: str) -> str:
    """A group reply ({"people": [{"name", "facts"}]}) as the one-list reply
    parse_facts reads; an unreadable reply is passed on for it to reject."""
    try:
        data = json.loads(content)
        people = data.get("people", [])
        items = [{**f, "type": "other"} for p in people if isinstance(p, dict)
                 for f in p.get("facts", []) if isinstance(f, dict)]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return content
    return json.dumps({"facts": items}, ensure_ascii=False)


# -- deduplication ---------------------------------------------------------------


def judge(old: Fact, new: str, llm: ChatProvider) -> str:
    """The model's call on two close facts: same, update, contradiction, distinct.
    An unusable answer counts as distinct (the safe choice: nothing is replaced)."""
    reply = llm.chat(
        [ChatMessage(Role.USER, JUDGE.format(old=old.statement, new=new))], schema=JUDGE_SCHEMA,
    )
    try:
        relation = json.loads(reply.content).get("relation")
    except (json.JSONDecodeError, AttributeError):
        return "distinct"
    return relation if relation in RELATIONS else "distinct"


def learn_facts(
    episode_id: int,
    messages: Sequence[Message],
    *,
    store: MemoryStore,
    llm: ChatProvider,
    embedder: Embedder,
    profile: ModelProfile,
    settings: FactSettings,
    now: datetime,
    tz: tzinfo | None = None,
    dated: bool = True,
    group: bool = False,
    pin_language: bool = True,
) -> FactReport:
    """Extract the episode's facts and fold them into what's already known."""
    report = FactReport()
    extracted, report.unsupported = extract_facts(messages, llm, profile, settings.max_facts, tz, dated, group, pin_language)
    if not extracted:
        return report
    vectors = embed_cached([f.statement for f in extracted], embedder, store)
    for fact, vector in zip(extracted, vectors):
        similar = store.similar_facts(vector, 3)
        if any(f.status == "Z" and s >= settings.same_score for f, s in similar):
            report.skipped.append(fact.statement)
            continue
        close = [(f, s) for f, s in similar if f.status == "A" and s >= settings.related_score]
        outcome = "new"
        for old, similarity in close:
            relation = "same" if similarity >= settings.same_score else judge(old, fact.statement, llm)
            if relation == "same":
                store.confirm_fact(old.id, now, fact.message_ids)
                report.confirmed.append(old.statement)
                outcome = "confirmed"
                break
            if relation in ("update", "contradiction"):
                new = _add(store, fact, episode_id, now, vector)
                store.supersede_fact(old.id, new.id)
                report.updated.append((old.statement, new.statement))
                outcome = "updated"
                break
        if outcome == "new":
            _add(store, fact, episode_id, now, vector)
            report.added.append(fact.statement)
    return report


def _add(store: MemoryStore, fact: ExtractedFact, episode_id: int, now: datetime, vector) -> Fact:
    added = store.add_fact(
        fact.statement, fact.type, episode_id=episode_id, message_ids=fact.message_ids,
        at=now, embedding=vector,
    )
    link_fact(store, added)       # the entities it's about, for entity blocks
    return added

