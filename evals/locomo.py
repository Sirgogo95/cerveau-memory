"""LoCoMo: the public long-term conversational memory benchmark (Maharana et al.,
2024), run through Cerveau's memory.

    python -m evals.locomo --download                 # once: evals/locomo_data/
    python -m evals.locomo                            # retrieval: are the evidence turns recalled?
    python -m evals.locomo --answer --limit 10        # answers, graded by the background model

10 conversations between two people (about 600 turns each, in dated sessions) and
about 2,000 questions, each with the turns that hold its answer ("evidence") and a
category: single-hop, multi-hop, temporal, open-domain, adversarial (about the other
speaker; the right answer is that it wasn't said). Like the published comparisons,
the adversarial category is left out unless asked for.

Each session becomes a closed episode at its date; each turn a message ("Caroline:
Hey Mel!...", shared photos by their caption), stored as the user's, so Cerveau's
short-reply rule doesn't apply. With --summaries the dataset's own session
summaries stand in for consolidation's (not Cerveau's: off by default).

Retrieval mode asks each question the way the retrieval eval does (python -m
evals): automatic recall only, no model. Answer mode runs a real turn per question
on a fresh copy of the memory, then the background model grades the answer against
the gold one (generous: same fact, any wording), as LLM-judge scores are reported;
token F1, the dataset's own metric, is printed too.

Vectors are kept in evals/locomo_data/vectors-<model>.pkl, so only the first run
embeds the 5,882 turns.
"""

from __future__ import annotations

import argparse
import copy
import math
import json
import pickle
import random
import re
import statistics
import string
import threading
import time
import urllib.request
from array import array
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.config import (
    EngineConfig,
    ModelProfile,
    load_embedding_profile,
    load_engine_config,
    load_model_profile,
    with_embedding,
)
from brain.consolidation import ConsolidationError, consolidate_episode
from brain.embedding import embed_cached
from brain.facts import fact_settings
from brain.hashing import text_hash
from brain.prompt import (
    MEMORY_FOOTER,
    MEMORY_HEADER,
    PINNED_FOOTER,
    PINNED_HEADER,
    RECENT_FOOTER,
    RECENT_HEADER,
    SO_FAR_FOOTER,
    SO_FAR_HEADER,
)
from brain.turn import run_turn
from brain.ports import ChatProvider, ChatProviderError
from brain.types import ChatMessage, ChatResponse, Embedding, MemoryKind, Role, Usage
from dotenv import load_dotenv
from adapters.stores.in_memory import InMemoryStore

from .cases import QueryCase, Scenario
from .runner import Prepared, QueryResult, query
from .turns import with_settings

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evals" / "locomo_data"
URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADVERSARIAL = 5
# An adversarial question asks about something the conversation never says (it was
# said about the other speaker, or not at all). The dataset gives no answer for it,
# only `adversarial_answer`: the plausible wrong one a model may be lured into.
NOT_MENTIONED = "Not mentioned in the conversation"


@dataclass(frozen=True, slots=True)
class Turn:
    dia_id: str          # "D3:12": session 3, turn 12
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class Session:
    number: int
    at: datetime
    turns: tuple[Turn, ...]
    summary: str = ""


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    answer: str
    evidence: tuple[str, ...]
    category: int
    trap: str = ""          # adversarial questions: the wrong answer they lure toward


@dataclass(frozen=True, slots=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]

    @property
    def end(self) -> datetime:
        return self.sessions[-1].at


def parse_time(value: str) -> datetime:
    """"1:56 pm on 8 May, 2023" (the dataset's format), as UTC."""
    return datetime.strptime(value.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


def turn_text(raw: dict) -> str:
    text = str(raw.get("text", "")).strip()
    if raw.get("blip_caption"):
        text = f"{text} [shares a photo: {raw['blip_caption']}]".strip()
    return f"{raw['speaker']}: {text}"


def parse(data: list[dict]) -> list[Conversation]:
    conversations = []
    for item in data:
        conv = item["conversation"]
        summaries = item.get("session_summary") or {}
        sessions = []
        for key, turns in conv.items():
            match = re.fullmatch(r"session_(\d+)", key)
            if not match or not turns:
                continue
            n = int(match.group(1))
            sessions.append(Session(
                n, parse_time(conv[f"session_{n}_date_time"]),
                tuple(Turn(t["dia_id"], t["speaker"], turn_text(t)) for t in turns),
                summaries.get(f"session_{n}_summary", ""),
            ))
        sessions.sort(key=lambda s: s.number)
        questions = tuple(
            Question(q["question"],
                     NOT_MENTIONED if int(q["category"]) == ADVERSARIAL else str(q.get("answer", "")),
                     tuple(e.strip() for ev in q.get("evidence", []) for e in str(ev).split(";") if e.strip()),
                     int(q["category"]), str(q.get("adversarial_answer", "")))
            for q in item["qa"]
        )
        conversations.append(Conversation(item["sample_id"], (conv["speaker_a"], conv["speaker_b"]),
                                          tuple(sessions), questions))
    return conversations


def load(path: Path) -> list[Conversation]:
    return parse(json.loads(path.read_text(encoding="utf-8")))


class DiskEmbedder:
    """Remembers every vector in a file, keyed by text hash: the turns are embedded
    once, then every run and every rebuilt store reads them back."""

    def __init__(self, inner, path: Path):
        self.inner, self.model, self.path = inner, inner.model, path
        self._known: dict[str, array] = pickle.loads(path.read_bytes()) if path.exists() else {}
        self._dirty = False
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        keys = [text_hash(t) for t in texts]
        with self._lock:
            missing = {k: t for k, t in zip(keys, texts) if k not in self._known}
            if missing:
                items = list(missing.items())
                for start in range(0, len(items), 64):
                    batch = items[start:start + 64]
                    for (key, _), vector in zip(batch, self.inner.embed([t for _, t in batch])):
                        self._known[key] = array("f", vector.vector)
                self._dirty = True
            return [Embedding(list(self._known[k]), self.model, k) for k in keys]

    def save(self) -> None:
        if self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self._known))
            self._dirty = False


class DiskLLM:
    """Remembers every reply of a model in a file, keyed by the prompt: consolidating
    the 272 sessions calls the model once, and every rebuild after that reads the
    replies back (the same memory every time, and no cost)."""

    def __init__(self, inner, path: Path, temperature: float | None = None):
        # `temperature` joins the key when given, so replies at another temperature
        # aren't mixed in; without it, keys stay as they were (the existing caches).
        self.inner, self.model, self.path, self.temperature = inner, inner.model, path, temperature
        self._known: dict[str, str] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._known[row["key"]] = row["content"]

    def chat(self, messages, *, schema=None, tools=()):
        parts: list = [self.model, [(str(m.role), m.content) for m in messages], schema]
        if self.temperature is not None:
            parts.append({"temperature": self.temperature})
        key = text_hash(json.dumps(parts, ensure_ascii=False, sort_keys=True))
        if key not in self._known:
            content = patiently(lambda: self.inner.chat(messages, schema=schema, tools=tools)).content
            with self._lock:
                self._known[key] = content
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")
        return ChatResponse(content=self._known[key], model=self.model, usage=Usage(0, 0))

    def stream(self, messages, *, tools=()):
        return self.inner.stream(messages, tools=tools)


@dataclass(frozen=True, slots=True)
class Consolidator:
    """What consolidates each session as the app does when an episode closes:
    title, summaries, tags, entities and their mentions, and facts."""

    llm: ChatProvider
    profile: ModelProfile
    config: EngineConfig
    dated: bool = True     # False: the prompts from before the date fix (runs A10, A11)
    # LoCoMo's sessions are two people talking: True asks for facts speaker by speaker
    # (facts.GROUP_FACT_INSTRUCTIONS). False, the one-person prompt, as runs before L4.
    group_facts: bool = False
    pin_language: bool = True   # False: the prompts from before the language fix (2026-11-06), cached


def build(conversation: Conversation, embedder, summaries: bool = False,
          consolidator: Consolidator | None = None) -> Prepared:
    """The conversation's memory: one closed episode per session, one message per
    turn, each session consolidated when it closes if a `consolidator` is given.
    References: "D3:12" for a turn, "S3" for a session's summary."""
    store = InMemoryStore()
    refs: dict[tuple[MemoryKind, int], str] = {}
    for session in conversation.sessions:
        episode = store.open_episode(session.at)
        vectors = embed_cached([t.text for t in session.turns], embedder, store)
        for i, (turn, vector) in enumerate(zip(session.turns, vectors)):
            message = store.add_message(episode.id, Role.USER, turn.text, session.at + timedelta(seconds=30 * i), vector)
            refs[(MemoryKind.MESSAGE, message.id)] = turn.dia_id
        end = session.at + timedelta(seconds=30 * len(session.turns))
        store.close_episode(episode.id, end)
        if consolidator is not None:
            try:
                consolidate_episode(episode.id, store=store, llm=consolidator.llm, embedder=embedder,
                                    profile=consolidator.profile, now=end,
                                    facts=fact_settings(consolidator.config),
                                    tag_same_score=consolidator.config.tag_same_score,
                                    tz=UTC,   # session times are stored as given, in UTC
                                    dated=consolidator.dated, group_facts=consolidator.group_facts,
                                    pin_language=consolidator.pin_language)
            except ConsolidationError as exc:
                print(f"  {conversation.sample_id} session {session.number}: not consolidated ({exc})")
            [consolidated] = store.get_episodes([episode.id])
            if consolidated.summary_short:
                refs[(MemoryKind.EPISODE, episode.id)] = f"S{session.number}"
        elif summaries and session.summary:
            [vector] = embed_cached([session.summary], embedder, store)
            store.save_summary(episode.id, title=f"Session {session.number}", summary_short=session.summary,
                               summary_long=session.summary, short_embedding=vector, long_embedding=vector)
            refs[(MemoryKind.EPISODE, episode.id)] = f"S{session.number}"
    return Prepared(Scenario(conversation.sample_id, "locomo", ()), store, refs)


def build_all(conversations: Sequence[Conversation], embedder, summaries: bool = False,
              consolidator: Consolidator | None = None, workers: int = 1) -> dict[str, Prepared]:
    """Every conversation's memory, by sample id; with `workers` above 1, built side
    by side (the conversations are independent; each one's sessions stay in order)."""
    if workers <= 1:
        return {c.sample_id: build(c, embedder, summaries, consolidator) for c in conversations}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        built = pool.map(lambda c: build(c, embedder, summaries, consolidator), conversations)
        return {c.sample_id: p for c, p in zip(conversations, built)}


def pick(questions: Sequence[Question], categories: set[int], limit: int | None, seed: int) -> list[Question]:
    """The questions of these categories that name their evidence; at most `limit`,
    sampled with `seed` but kept in the dataset's order."""
    chosen = [q for q in questions if q.category in categories and q.evidence]
    if limit is not None and limit < len(chosen):
        keep = set(random.Random(seed).sample(range(len(chosen)), limit))
        chosen = [q for i, q in enumerate(chosen) if i in keep]
    return chosen


# -- Retrieval ------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Recalled:
    conversation: str
    question: Question
    result: QueryResult

    @property
    def session_hit(self) -> bool:
        """An evidence turn's session came back, by its summary or any of its turns."""
        sessions = {e.split(":")[0][1:] for e in self.question.evidence}
        kept = {k[1:] if k.startswith("S") else k.split(":")[0][1:] for k in self.result.kept}
        return bool(sessions & kept)


def recall_questions(conversation: Conversation, prepared: Prepared, questions: Sequence[Question],
                     embedder, config: EngineConfig, reranker=None) -> list[Recalled]:
    known = set(prepared.refs.values())
    vectors: dict[str, Embedding] = {}
    out = []
    for q in questions:
        expect = tuple(e for e in q.evidence if e in known)
        if not expect:
            continue
        out.append(Recalled(conversation.sample_id, q,
                            query(prepared, QueryCase(q.text, expect), embedder, config, vectors, reranker)))
    return out


def groups(rows, category_of) -> list[tuple[str, list]]:
    """Rows by category, then "all" over categories 1 to 4 (adversarial questions are
    reported apart, as in published comparisons)."""
    present = sorted({category_of(r) for r in rows})
    out = [(CATEGORIES[c], [r for r in rows if category_of(r) == c]) for c in present]
    main = [r for r in rows if category_of(r) != ADVERSARIAL]
    return out + ([("all (1-4)", main)] if main else [])


def interval(right: int, n: int) -> float:
    """Half-width of the 95% confidence interval of a share (normal approximation)."""
    p = right / n
    return 1.96 * math.sqrt(p * (1 - p) / n)


def sign_test(gained: int, lost: int) -> float:
    """Two-sided exact sign test: how likely a split at least this uneven is by chance,
    counting only the questions whose verdict changed."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def recall_report(rows: Sequence[Recalled]) -> str:
    lines = [f"  {'category':12} {'questions':>9} {'turn hit':>9} {'turn recall':>12} {'session hit':>12} {'kept':>5}"]
    for name, mine in groups(rows, lambda r: r.question.category):
        lines.append(
            f"  {name:12} {len(mine):9} {sum(r.result.hit for r in mine) / len(mine):9.0%} "
            f"{statistics.fmean(r.result.recall for r in mine):12.0%} "
            f"{sum(r.session_hit for r in mine) / len(mine):12.0%} "
            f"{statistics.fmean(len(r.result.kept) for r in mine):5.1f}"
        )
    return "\n".join(lines)


# -- Answers --------------------------------------------------------------------------

JUDGE = """You grade answers to questions about past conversations.

Question: {question}
Gold answer: {gold}
Answer given: {answer}

Label the answer CORRECT if it gives the same fact as the gold answer, even in other
words, longer, or with more detail; a date or time in another format, or relative
("last Tuesday") but pointing to the same day, period, or year, is CORRECT. Label it
WRONG if it gives a different fact, misses the point, or says it doesn't know.

Reply with one word: CORRECT or WRONG."""

# Published LoCoMo graders are more generous (the Mem0 paper's grader counts an answer
# that touches on the gold answer's topic): `--lenient` regrades with this rule, to
# show how much the grader alone moves a score. Paraphrased, not Mem0's exact prompt.
LENIENT_JUDGE = """You grade answers to questions about past conversations.

Question: {question}
Gold answer: {gold}
Answer given: {answer}

Be generous, as published LoCoMo graders are: label the answer CORRECT if it touches on
the same topic as the gold answer, even if it says more, says it differently, or gives
only part of it. For a question about time, label it CORRECT if it points to the same
date or period, in any format or relative wording. Label it WRONG only if it is about
something else, contradicts the gold answer, or says it doesn't know.

Reply with one word: CORRECT or WRONG."""

ADVERSARIAL_JUDGE = """You grade answers to questions about past conversations.

Question: {question}
The conversations never say this: the question is a trap, and the right answer is that
it isn't mentioned (a plausible wrong answer would be: {trap}).
Answer given: {answer}

Label the answer CORRECT if it says the conversations don't mention it, that it doesn't
know, or that it can't find it, or if it rejects the question's premise (for example,
that it was the other person). Label it WRONG if it answers as if the premise were true,
with the plausible wrong answer or any other.

Reply with one word: CORRECT or WRONG."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["CORRECT", "WRONG"]}},
    "required": ["label"],
}


def verdict(content: str) -> bool:
    """The label, from JSON when the provider kept to the schema, else the last
    CORRECT or WRONG in the text (some cloud models ignore the schema)."""
    try:
        return json.loads(content).get("label") == "CORRECT"
    except (json.JSONDecodeError, AttributeError):
        labels = re.findall(r"\b(CORRECT|WRONG)\b", content.upper())
        return bool(labels) and labels[-1] == "CORRECT"


def patiently[T](call: Callable[[], T], tries: int = 4) -> T:
    """A remote model can time out or rate-limit: try again, waiting longer each time."""
    for attempt in range(tries - 1):
        try:
            return call()
        except ChatProviderError:
            time.sleep(5 * 2 ** attempt)
    return call()


def judge(question: Question, answer: str, llm, lenient: bool = False) -> bool:
    given = answer.strip() or "(no answer)"
    if question.category == ADVERSARIAL:
        prompt = ADVERSARIAL_JUDGE.format(question=question.text, trap=question.trap or "(none given)", answer=given)
    else:
        rule = LENIENT_JUDGE if lenient else JUDGE
        prompt = rule.format(question=question.text, gold=question.answer, answer=given)
    response = patiently(lambda: llm.chat([ChatMessage(Role.USER, prompt)], schema=JUDGE_SCHEMA))
    return verdict(response.content)


def tokens(text: str) -> list[str]:
    text = text.casefold().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in {"a", "an", "the", "and"}]


def f1(answer: str, gold: str) -> float:
    """Token F1, the dataset's own metric (articles and punctuation ignored)."""
    a, g = tokens(answer), tokens(gold)
    common = sum((Counter(a) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(g)
    return 2 * precision * recall / (precision + recall)


NOTES_JUDGE = """You check whether notes contain the answer to a question.

Question: {question}
Gold answer: {gold}

Notes:
{notes}

Do the notes contain the gold answer? ALL if every part of it can be found or worked
out from the notes (a relative date with its message date counts); PART if some parts
can but not all (for an answer that lists several things, some of them); NONE if the
notes don't contain it.

Reply with one word: ALL, PART, or NONE."""

IN_NOTES = ("ALL", "PART", "NONE")


class RecordingLLM:
    """Passes everything to the chat model and keeps what it was shown, so the notes
    a turn saw can be checked afterwards (the answer-in-notes metric)."""

    def __init__(self, inner):
        self.inner, self.model = inner, inner.model
        self.seen: list[list[ChatMessage]] = []

    def chat(self, messages, *, schema=None, tools=()):
        self.seen.append(list(messages))
        return self.inner.chat(messages, schema=schema, tools=tools)

    def stream(self, messages, *, tools=()):
        self.seen.append(list(messages))
        return self.inner.stream(messages, tools=tools)


def notes_seen(prompts: Sequence[Sequence[ChatMessage]]) -> str:
    """What the model was shown from memory during a turn: the recent conversations
    and the summary so far (system message), the <memory> block, and the results of
    any memory search, without the instructions or the question."""
    parts: list[str] = []
    for messages in prompts[:1]:
        for m in messages:
            # The sections' exact opening lines: the instructions mention the same tags.
            for start, end in ((RECENT_HEADER, RECENT_FOOTER), (SO_FAR_HEADER, SO_FAR_FOOTER),
                               (PINNED_HEADER, PINNED_FOOTER), (MEMORY_HEADER, MEMORY_FOOTER)):
                i = m.content.find(start)
                if i != -1:
                    j = m.content.find(end, i)
                    parts.append(m.content[i:j + len(end)] if j != -1 else m.content[i:])
    for messages in prompts:
        parts += [f"<search result>\n{m.content}\n</search result>" for m in messages if m.role is Role.TOOL]
    return "\n\n".join(dict.fromkeys(parts))


def in_notes(question: Question, notes: str, llm) -> str:
    """ALL, PART, or NONE: how much of the gold answer the notes hold."""
    if question.category == ADVERSARIAL or not notes:
        return "NONE" if not notes else ""
    prompt = NOTES_JUDGE.format(question=question.text, gold=question.answer, notes=notes)
    content = patiently(lambda: llm.chat([ChatMessage(Role.USER, prompt)])).content.upper()
    labels = re.findall(r"\b(ALL|PART|NONE)\b", content)
    return labels[-1] if labels else ""


@dataclass(frozen=True, slots=True)
class Answered:
    conversation: str
    question: Question
    answer: str
    judged: dict[str, bool]      # grader model -> right or not
    f1: float
    seconds: float
    searches: int = 0            # memory tool calls the model made
    in_notes: str = ""           # ALL, PART, NONE: the gold answer in what the model was shown (--notes)
    notes: str = ""


FULL_CONTEXT = """Below is the whole history of conversations between {a} and {b}, session by
session, with dates. Answer the question from it.

{transcript}"""


def transcript(conversation: Conversation) -> str:
    return "\n\n".join(
        f"Session {s.number} ({s.at:%A %Y-%m-%d %H:%M}):\n" + "\n".join(t.text for t in s.turns)
        for s in conversation.sessions
    )


def full_context_answer(conversation: Conversation, question: Question, llm) -> str:
    """The ceiling: no memory system, the whole conversation in the prompt."""
    system = FULL_CONTEXT.format(a=conversation.speakers[0], b=conversation.speakers[1],
                                 transcript=transcript(conversation))
    today = conversation.end + timedelta(days=1)
    messages = [ChatMessage(Role.SYSTEM, system),
                ChatMessage(Role.USER, f"(Today is {today:%A %Y-%m-%d}.) {question.text}")]
    return patiently(lambda: llm.chat(messages)).content


EMPTY_TRIES = 3


def in_utc(config: EngineConfig) -> EngineConfig:
    """Session times are stored as given, in UTC, so the answering prompt shows them in
    UTC too. Left to the machine's timezone (until 2026-11-03), a session at
    12:37 am appeared as the evening before: 17 of the 288 sessions are before 4 am."""
    return replace(config, timezone="UTC")


def answer_question(conversation: Conversation, question: Question, *, embedder, llm, judges, profile,
                    config: EngineConfig, memory: Prepared | None = None, full_context: bool = False,
                    reranker=None, notes_judge=None) -> Answered:
    """One question in a fresh conversation on a copy of the memory (nothing one
    question says reaches the next), or with the whole conversation in the prompt."""
    started = time.monotonic()
    # gpt-oss:120b on Ollama Cloud sometimes replies with nothing at all (2 to 4% of the
    # final runs' answers): an empty reply is a failed call, asked again on a fresh copy.
    for _ in range(EMPTY_TRIES):
        store = None if full_context or memory is None else copy.deepcopy(memory.store)
        searches = 0
        if store is None:
            answer = full_context_answer(conversation, question, llm)
        else:
            now = conversation.end + timedelta(days=1)
            recording = RecordingLLM(llm)
            answer = patiently(lambda fresh=store: "".join(c.content for c in run_turn(
                question.text, now, store=fresh, llm=recording, embedder=embedder, profile=profile, config=config,
                reranker=reranker)))
            current = store.current_episode()
            searches = len(store.get_tool_calls(current.id)) if current else 0
        if answer.strip():
            break
    seconds = time.monotonic() - started
    notes = notes_seen(recording.seen) if store is not None and notes_judge is not None else ""
    return Answered(conversation.sample_id, question, answer,
                    {j.model: judge(question, answer, j) for j in judges}, f1(answer, question.answer), seconds,
                    searches, in_notes(question, notes, notes_judge) if notes_judge is not None else "", notes)


def answer_report(rows: Sequence[Answered]) -> str:
    """Judged right per grader and category, with the 95% interval, token F1, how
    often the model searched its memory, and the median and 90th-percentile time."""
    graders = list(rows[0].judged) if rows else []
    head = "".join(f" {g[:20]:>20}" for g in graders)
    lines = [f"  {'category':12} {'questions':>9}{head} {'F1':>6} {'searched':>9} {'median':>7} {'p90':>6}"]
    for name, mine in groups(rows, lambda r: r.question.category):
        right = "".join(
            f" {f'{sum(r.judged[g] for r in mine) / len(mine):.0%} +/-{interval(sum(r.judged[g] for r in mine), len(mine)):.0%}':>20}"
            for g in graders)
        times = sorted(r.seconds for r in mine)
        lines.append(f"  {name:12} {len(mine):9}{right} {statistics.fmean(r.f1 for r in mine):6.2f} "
                     f"{sum(r.searches > 0 for r in mine) / len(mine):9.0%} {statistics.median(times):6.1f}s "
                     f"{times[int(0.9 * (len(times) - 1))]:5.1f}s")
    return "\n".join(lines)


def graders(names: str, models: Path) -> list:
    """Grader providers from comma-separated profile names, always at temperature 0:
    the same answer should get the same verdict (at 0.2, gpt-oss graded identical
    answers differently on two days)."""
    from adapters.llm.factory import provider_from_profile

    out = []
    for name in names.split(","):
        profile = load_model_profile(models / f"{name.strip()}.toml")
        out.append(provider_from_profile(replace(profile, options={**profile.options, "temperature": 0})))
    return out


def regrade(source: Path, judges: Sequence, out: Path, lenient: bool = False) -> list[Answered]:
    """Grade the answers saved in `source` again, without answering again, and save
    them with the new verdicts in `out` (the old ones are replaced); `lenient`, with
    LENIENT_JUDGE."""
    rows = []
    for r in map(json.loads, source.read_text(encoding="utf-8").splitlines()):
        question = Question(r["question"], r["gold"], (), int(r["category"]), r.get("trap", ""))
        rows.append(Answered(r["conversation"], question, r["answer"],
                             {j.model: judge(question, r["answer"], j, lenient) for j in judges},
                             float(r["f1"]), float(r["seconds"]), int(r.get("searches", 0))))
    save(rows, out)
    return rows


def notes_report(rows: Sequence[Answered], grader: str) -> str:
    """Where answers are lost: the gold answer in the notes the model was shown (all
    of it, part, none) against the answer judged right or wrong by `grader`."""
    rows = [r for r in rows if r.in_notes]
    if not rows:
        return ""
    lines = [f"  answer in the notes (graded by the first grader), {len(rows)} questions:",
             f"  {'in the notes':14} {'questions':>9} {'judged right':>13} {'judged wrong':>13}"]
    for label in IN_NOTES:
        mine = [r for r in rows if r.in_notes == label]
        right = sum(r.judged.get(grader, False) for r in mine)
        lines.append(f"  {label:14} {len(mine):9} {right:13} {len(mine) - right:13}")
    wrong = [r for r in rows if not r.judged.get(grader, False)]
    if wrong:
        missing = sum(r.in_notes == "NONE" for r in wrong)
        partial = sum(r.in_notes == "PART" for r in wrong)
        lines.append(f"  of {len(wrong)} wrong answers: {missing} had none of the answer in the notes (memory), "
                     f"{partial} part of it, {len(wrong) - missing - partial} all of it (the model)")
    return "\n".join(lines)


def compare(before: Path, after: Path) -> str:
    """Two saved runs on the same questions: per grader, the answers that turned right
    and wrong, and the sign test on them."""
    def rows(path: Path) -> dict[tuple[str, str, int], dict]:
        # A conversation can ask the same question twice: the n-th time is its own key.
        seen: Counter[tuple[str, str]] = Counter()
        keyed = {}
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            seen[(r["conversation"], r["question"])] += 1
            keyed[(r["conversation"], r["question"], seen[(r["conversation"], r["question"])])] = r
        return keyed
    a, b = rows(before), rows(after)
    shared = [k for k in a if k in b]
    lines = [f"{len(shared)} questions in both runs"]
    for grader in a[shared[0]]["judged"] if shared else []:
        if grader not in b[shared[0]]["judged"]:
            continue
        gained = sum(1 for k in shared if not a[k]["judged"][grader] and b[k]["judged"][grader])
        lost = sum(1 for k in shared if a[k]["judged"][grader] and not b[k]["judged"][grader])
        right_a = sum(a[k]["judged"][grader] for k in shared) / len(shared)
        right_b = sum(b[k]["judged"][grader] for k in shared) / len(shared)
        lines.append(f"  {grader}: {right_a:.0%} -> {right_b:.0%}, {gained} turned right, {lost} turned wrong, "
                     f"sign test p = {sign_test(gained, lost):.3f}")
    return "\n".join(lines)


def row_json(r: Answered) -> str:
    return json.dumps({"conversation": r.conversation, "category": r.question.category,
                       "question": r.question.text, "gold": r.question.answer, "answer": r.answer,
                       "judged": r.judged, "f1": round(r.f1, 3), "seconds": round(r.seconds, 2),
                       "searches": r.searches, "trap": r.question.trap,
                       "in_notes": r.in_notes, "notes": r.notes}, ensure_ascii=False)


def save(rows: Sequence[Answered], path: Path) -> None:
    """One JSON line per answer, to look at or grade again later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row_json(r) + "\n" for r in rows), encoding="utf-8")


def load_rows(path: Path) -> list[Answered]:
    """The answers saved in `path`, verdicts included (a partial line from a crash
    is skipped)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = Question(r["question"], r["gold"], (), int(r["category"]), r.get("trap", ""))
        rows.append(Answered(r["conversation"], question, r["answer"], dict(r["judged"]), float(r["f1"]),
                             float(r["seconds"]), int(r.get("searches", 0)), r.get("in_notes", ""), r.get("notes", "")))
    return rows


def question_keys(items) -> list[tuple[str, str, int]]:
    """(conversation, question, n) for each item: a question asked twice in one
    conversation is its own key the second time."""
    seen: Counter[tuple[str, str]] = Counter()
    keys = []
    for conversation, text in items:
        seen[(conversation, text)] += 1
        keys.append((conversation, text, seen[(conversation, text)]))
    return keys


# -- Command line ---------------------------------------------------------------------

def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        path.write_bytes(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.locomo", description="LoCoMo through Cerveau's memory.")
    parser.add_argument("--download", action="store_true", help=f"Fetch the dataset into {DATA}.")
    parser.add_argument("--data", default=str(DATA / "locomo10.json"))
    parser.add_argument("--conversations", default="", help="Only these, by position (0-9), comma-separated.")
    parser.add_argument("--categories", default="1,2,3,4", help="1 multi-hop, 2 temporal, 3 open-domain, "
                        "4 single-hop, 5 adversarial (default 1-4).")
    parser.add_argument("--limit", type=int, default=None, help="At most this many questions per conversation.")
    parser.add_argument("--seed", type=int, default=0, help="Which questions --limit samples.")
    parser.add_argument("--summaries", action="store_true", help="Use the dataset's session summaries.")
    parser.add_argument("--consolidate", default="", metavar="PROFILE",
                        help="Consolidate each session with this model profile, as the app does (replies kept on disk).")
    parser.add_argument("--answer", action="store_true", help="Answer with the chat model and grade.")
    parser.add_argument("--chat-model", default="qwen3.5-4b")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Answer at this temperature instead of the profile's (0 makes runs repeatable).")
    parser.add_argument("--judge-model", default="qwen3.5-4b-background",
                        help="Grader profile(s), comma-separated: each grades every answer.")
    parser.add_argument("--full-context", action="store_true",
                        help="No memory: the whole conversation in the prompt (needs a long-context model).")
    parser.add_argument("--out", default="", help="Save every answer as JSON lines here.")
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--details", action="store_true", help="Print every question's result.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Conversations built side by side (default: 10 with a cloud consolidator, else 1).")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two runs saved with --out: turned right, turned wrong, sign test.")
    parser.add_argument("--group-facts", action="store_true",
                        help="Learn facts with the prompt for conversations between several people.")
    parser.add_argument("--no-language-pin", action="store_true",
                        help="Consolidate with the prompts from before the language fix (2026-11-06), reading the cached replies.")
    parser.add_argument("--undated", action="store_true",
                        help="Consolidate with the prompts from before the date fix (cached replies of runs A10, A11).")
    parser.add_argument("--consolidation-temperature", type=float, default=None,
                        help="Consolidate at this temperature instead of the profile's (0 for eval runs).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the answers already in --out and answer only the rest (after a crash).")
    parser.add_argument("--notes", action="store_true",
                        help="Also check whether the notes the model was shown hold the answer (one grading call more).")
    parser.add_argument("--reranker", default="", metavar="MODEL",
                        help="Rerank recall's shortlist with this cross-encoder (e.g. BAAI/bge-reranker-v2-m3).")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--regrade", nargs=2, metavar=("ANSWERS", "OUT"),
                        help="Grade saved answers again with --judge-model, without answering again.")
    parser.add_argument("--lenient", action="store_true",
                        help="With --regrade: the generous rule of published graders (same topic counts).")
    parser.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.compare:
        print(compare(Path(args.compare[0]), Path(args.compare[1])))
        return 0

    load_dotenv(ROOT / ".env")     # OLLAMA_API_KEY, for cloud model profiles
    if args.regrade:
        rows = regrade(Path(args.regrade[0]), graders(args.judge_model, ROOT / "configs" / "models"),
                       Path(args.regrade[1]), args.lenient)
        print(f"Graded again: {len(rows)} answers\n{answer_report(rows)}")
        return 0
    path = Path(args.data)
    if args.download:
        download(path)
        print(f"Saved {path}")
        return 0
    if not path.exists():
        parser.error(f"{path} not found: run python -m evals.locomo --download first.")

    embedding = load_embedding_profile(ROOT / "configs" / "embeddings" / f"{args.embedding}.toml")
    config = in_utc(with_settings(with_embedding(load_engine_config(args.config), embedding), args.set))
    if args.fake_embedder:
        from tests.fakes import FakeEmbedder
        embedder = FakeEmbedder()
    else:
        from adapters.embeddings.factory import embedder_from_profile
        embedder = DiskEmbedder(embedder_from_profile(embedding), path.parent / f"vectors-{embedding.model}.pkl")

    reranker = None
    if args.reranker:
        from adapters.rerankers.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(args.reranker, device=args.reranker_device)

    consolidator = None
    if args.consolidate:
        from adapters.llm.factory import provider_from_profile
        background = load_model_profile(ROOT / "configs" / "models" / f"{args.consolidate}.toml")
        if args.consolidation_temperature is not None:
            background = replace(background, options={**background.options,
                                                      "temperature": args.consolidation_temperature})
        consolidator = Consolidator(DiskLLM(provider_from_profile(background),
                                            path.parent / f"llm-{background.name.replace(':', '-')}.jsonl",
                                            args.consolidation_temperature),
                                    background, config, dated=not args.undated, group_facts=args.group_facts,
                                    pin_language=not args.no_language_pin)

    conversations = load(path)
    if args.conversations:
        conversations = [conversations[int(i)] for i in args.conversations.split(",")]
    categories = {int(c) for c in args.categories.split(",")}
    chosen = {c.sample_id: pick(c.questions, categories, args.limit, args.seed) for c in conversations}
    print(f"{len(conversations)} conversation(s), {sum(map(len, chosen.values()))} questions, "
          f"embedding model {embedding.model}{', dataset summaries' if args.summaries else ''}"
          f"{f', consolidated by {consolidator.profile.name}' if consolidator else ''}"
          f"{f', reranked by {reranker.model}' if reranker else ''}")

    try:
        workers = args.workers or (10 if consolidator and consolidator.profile.base_url else 1)
        memories = {} if args.answer and args.full_context else build_all(
            conversations, embedder, args.summaries, consolidator, workers)
        if not args.answer:
            rows = []
            for conversation in conversations:
                prepared = memories[conversation.sample_id]
                rows += recall_questions(conversation, prepared, chosen[conversation.sample_id], embedder, config,
                                         reranker)
                if args.details:
                    for r in rows[-len(chosen[conversation.sample_id]):]:
                        print(f"  {'HIT ' if r.result.hit else 'MISS'} [{r.conversation}] {r.question.text}\n"
                              f"        evidence {list(r.result.query.expect)}; kept {list(r.result.kept)}")
            print(f"\nmin_score {config.messages.min_score:.2f}, automatic recall:\n{recall_report(rows)}")
            return 0

        from adapters.llm.factory import provider_from_profile
        models = ROOT / "configs" / "models"
        profile = load_model_profile(models / f"{args.chat_model}.toml")
        if args.temperature is not None:   # 0: the same answer every run, so runs compare
            profile = replace(profile, options={**profile.options, "temperature": args.temperature})
        llm = provider_from_profile(profile)
        judges = graders(args.judge_model, models)
        # Save as you go: each answer is written when it's graded, so a crash (Ollama
        # and Docker have stopped on their own) loses nothing; --resume skips what's done.
        out = Path(args.out) if args.out else None
        answered = load_rows(out) if out is not None and args.resume and out.exists() else []
        done = set(question_keys((r.conversation, r.question.text) for r in answered))
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(row_json(r) + "\n" for r in answered), encoding="utf-8")
        if answered:
            print(f"Resuming: {len(answered)} answers already saved in {out}")
        todo = [(c, q) for c in conversations for q in chosen[c.sample_id]]
        keys = question_keys((c.sample_id, q.text) for c, q in todo)
        for (conversation, question), key in zip(todo, keys):
            if key in done:
                continue
            row = answer_question(conversation, question, embedder=embedder, llm=llm, judges=judges,
                                  profile=profile, config=config, memory=memories.get(conversation.sample_id),
                                  full_context=args.full_context, reranker=reranker,
                                  notes_judge=judges[0] if args.notes else None)
            answered.append(row)
            if out is not None:
                with out.open("a", encoding="utf-8") as saved:
                    saved.write(row_json(row) + "\n")
            if args.details:
                marks = " ".join("RIGHT" if v else "WRONG" for v in row.judged.values())
                print(f"  [{marks}] [{row.conversation}] {CATEGORIES[question.category]}: {question.text}\n"
                      f"        gold: {question.answer}\n        said: {row.answer.strip()[:300]}", flush=True)
        how = "the whole conversation in the prompt" if args.full_context else "Cerveau's memory"
        print(f"\nAnswers by {llm.model} from {how}, judged right by each grader:\n{answer_report(answered)}")
        if args.notes:
            print(notes_report(answered, judges[0].model))
        return 0
    finally:
        if isinstance(embedder, DiskEmbedder):
            embedder.save()


if __name__ == "__main__":
    raise SystemExit(main())
