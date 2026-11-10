"""Plain data types shared by the engine and the Protocols it depends on.

All datetimes are timezone-aware (UTC). Ids are the store's own integer keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # a tool's result, sent back to the model; never stored


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool the model may call: a name, what it does, and its arguments as a JSON
    Schema. Provider-agnostic; each ChatProvider translates it."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asking to run a tool with these arguments."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolRun:
    """A tool call that ran during a turn, kept for debugging and for the page:
    what the model asked (`name`, `arguments`), what it got back (`result`, the JSON
    text it received), and how long it took. `message_id` is the reply it led to
    (None if the turn ended without one); set by the store when read back."""

    name: str
    arguments: Mapping[str, Any]
    result: str
    duration_ms: int
    at: datetime
    message_id: int | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message of a prompt. An assistant message may carry the `tool_calls` it
    made; a TOOL message carries one result, for the tool named `tool_name`."""

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts reported by the provider for one call."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    model: str
    usage: Usage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One piece of a streamed reply. Only the final chunk carries `usage`. A chunk
    with `tool_calls` means the model wants tools run before it answers."""

    content: str
    usage: Usage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Embedding:
    """A vector together with the model that produced it.

    They travel as a pair because vectors from different models are not comparable:
    stores save `model` on every row and only compare a query against rows embedded
    by the same model.

    `text_hash` is `brain.hashing.text_hash()` of the exact text that was embedded.
    Stores save it next to the vector so a stale vector (its text was rewritten) can
    be detected. Query embeddings, which are never stored, may leave it empty.
    """

    vector: Sequence[float]
    model: str
    text_hash: str = ""

    def __deepcopy__(self, memo) -> Embedding:
        # Never changed once made, so a copied store can share it (a deep copy of
        # every vector would cost seconds for a large store).
        return self


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Episode:
    """A segment of the one continuous conversation. Summaries are empty until
    consolidation has processed the episode.

    `running_summary` covers the episode's earlier messages while it is still going,
    up to and including message `summarized_until`; the messages after it are shown
    verbatim. Empty until the episode outgrows the recent window."""

    id: int
    started_at: datetime
    ended_at: datetime | None = None
    last_message_at: datetime | None = None
    title: str = ""
    summary_short: str = ""
    summary_long: str = ""
    consolidated_at: datetime | None = None
    running_summary: str = ""
    summarized_until: int | None = None
    # Drift (brain/segmentation.py): consecutive messages unlike the conversation so
    # far, and the first of them (where a split would start the new episode).
    drift_strikes: int = 0
    drift_pending_id: int | None = None

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    episode_id: int
    role: Role
    content: str
    created_at: datetime
    # Pinned by the user: stays in the prompt verbatim after the window moves past it.
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class CarriedMemory:
    """A memory recalled in a turn, kept for the next ones (docs/ACTIVATION.md, idea
    B): which one, and the score it had, which fades each turn."""

    kind: MemoryKind
    id: int
    score: float


@dataclass(frozen=True, slots=True)
class Tag:
    """A topic: broad and reusable (`programming`, `travel`). `status` is A, I, or Z
    (forgotten: never relearned)."""

    id: int
    name: str
    status: str = "A"


@dataclass(frozen=True, slots=True)
class Entity:
    """A specific named thing: a person, project, place, or product."""

    id: int
    name: str
    type: str = ""
    aliases: tuple[str, ...] = ()


class MemoryKind(StrEnum):
    """The memory sources the retrieval engine can search. Skills join in v3."""

    EPISODE = "episode"
    MESSAGE = "message"
    TAG = "tag"
    ENTITY = "entity"
    FACT = "fact"
    ATTACHMENT = "attachment"   # a passage (chunk) of a file the user shared


# A node of the activation graph: a tag or an entity (brain/activation.py).
Node = tuple[MemoryKind, int]


# What a fact is about; a short fixed list keeps the model consistent and the viewer
# filterable (docs/FACTS.md).
FACT_TYPES = ("person", "preference", "plan", "project", "health", "place", "other")


@dataclass(frozen=True, slots=True)
class Fact:
    """A short, standalone statement the user made, true beyond one conversation:
    "The user's sister Lucía lives in Montreal." Learned at consolidation from
    `source_message_ids` (always user messages), so it can be checked against the
    original words. Never deleted: an outdated fact points to its replacement
    (`superseded_by`) and stays as history. `status` is A (active), I (set aside),
    or Z (forgotten: never relearned)."""

    id: int
    statement: str
    type: str = "other"
    episode_id: int | None = None
    learned_at: datetime | None = None
    times_confirmed: int = 1
    last_confirmed_at: datetime | None = None
    superseded_by: int | None = None
    source_message_ids: tuple[int, ...] = ()
    status: str = "A"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A search hit with its raw signals. The store reports what it found; the
    engine combines the signals into a score using its config.

    - `text`: what would go into the prompt (an episode's short summary, a message's
      content, a tag or entity name).
    - `semantic`: cosine similarity to the query embedding, higher is closer.
    - `lexical`: name-match signal from 0 to 1: the share of the entities named in
      the message that this memory mentions.
    - `topical`: tag signal, for episodes: the similarity between the message and the
      closest of the episode's tags that matched it (brain/retrieval.matched_tags).
    - `activation`: the highest activation among the tags and entities of the
      memory's episode (brain/activation.py).
    - `keyword`: the share of the message's exact terms (brain/keywords.py) that this
      memory contains literally.
    - `episode_id`: the episode a message belongs to (or the episode itself).
    - `at`: when the memory happened (a message's time, an episode's start), so the
      prompt can date it.
    - `role`: who said it, for messages.
    - `parent_id`, `label`: for an attachment passage, the attachment's id and where
      the passage is ("lease.pdf, p.4"); the prompt shows them.
    """

    kind: MemoryKind
    id: int
    text: str
    semantic: float | None = None
    lexical: float | None = None
    topical: float | None = None
    activation: float | None = None
    keyword: float | None = None
    episode_id: int | None = None
    at: datetime | None = None
    role: Role | None = None
    parent_id: int | None = None
    label: str = ""


# -- Attachments (docs/ATTACHMENTS.md) ------------------------------------------------

EXTRACTION_STATUSES = ("pending", "done", "failed", "unsupported")


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file the user shared, stored once (by the hash of its bytes) and read in the
    background into passages. `storage_path` is relative to the attachments folder.
    `extraction_status`: pending (not read yet), done, failed (`error` says why), or
    unsupported (stored, not searchable). `shared_at`, `episode_id`, `message_id`: when,
    where, and with which message it was first shared."""

    id: int
    filename: str
    media_type: str
    size_bytes: int
    storage_path: str
    status: str = "A"
    extraction_status: str = "pending"
    error: str = ""
    title: str = ""
    summary_short: str = ""
    summary_long: str = ""
    page_count: int | None = None
    shared_at: datetime | None = None
    episode_id: int | None = None
    message_id: int | None = None   # the message that first shared it


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A piece of a document as read: a paragraph, a page, or a whole code file, with
    its page (when the format has pages) and the heading it falls under."""

    text: str
    page: int | None = None
    heading: str = ""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    blocks: tuple[TextBlock, ...]
    page_count: int | None = None
    code: bool = False   # split by lines, not by paragraphs


@dataclass(frozen=True, slots=True)
class Chunk:
    """A passage of an attachment, sized for the embedding model."""

    position: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    heading: str = ""
    id: int = 0
    attachment_id: int = 0

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end in (None, self.page_start):
            return f"p.{self.page_start}"
        return f"pp.{self.page_start}-{self.page_end}"
