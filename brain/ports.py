"""Every contract the engine depends on. Implementations live outside brain/:
pure-Python ones in adapters/, the Django ORM store in apps/cerveau/.

The rule for every store method: the store answers questions and records facts; it
never makes decisions. "Has the conversation been quiet for two hours?" is decided in
brain/; "when was the last message?" is asked of the store.

Each Protocol was derived from what an engine function actually calls, so a method
exists only once something needs it. The Django side (admin, views) queries its models
directly and does not go through these.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .types import (
    Attachment,
    Candidate,
    CarriedMemory,
    ChatChunk,
    ChatMessage,
    ChatResponse,
    Chunk,
    Embedding,
    Entity,
    Episode,
    ExtractedDocument,
    Fact,
    MemoryKind,
    Message,
    Node,
    Role,
    Tag,
    ToolDefinition,
    ToolRun,
)

# ---------------------------------------------------------------------------
# LLMs and embeddings
# ---------------------------------------------------------------------------


class ChatProviderError(Exception):
    """Raised by any ChatProvider when the call fails (unreachable server, unknown
    model, bad response), so callers handle one exception type for every provider."""


class EmbedderError(Exception):
    """Raised by an Embedder that runs elsewhere (a server) when the call fails."""


@runtime_checkable
class ChatProvider(Protocol):
    """An LLM the engine can talk to. One implementation per provider in adapters/llm/."""

    model: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatResponse:
        """Send the full prompt and return the complete reply. With `schema` (a JSON
        Schema), the reply is JSON matching it: constrained by the provider where it
        supports structured output, requested in the prompt otherwise. With `tools`,
        the reply may carry tool calls instead of (or besides) text."""
        ...

    def stream(
        self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolDefinition] = ()
    ) -> Iterator[ChatChunk]:
        """Send the full prompt and yield the reply as it is generated. With `tools`,
        a chunk may carry tool calls."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. The engine embeds before calling a store, so stores
    only save and compare vectors and never need a model loaded."""

    model: str

    @property
    def dimensions(self) -> int:
        """Length of the vectors this model produces."""
        ...

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """One Embedding per text, in the same order. Batched: embedding many texts
        in one call is much faster than one at a time."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-reads recall's shortlist against the question: scores how well each text
    answers `query`, reading the two together (an embedding sees each alone). Only
    the order matters, not the scale. One implementation in adapters/rerankers/."""

    model: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in the same order; higher answers the query better."""
        ...


# ---------------------------------------------------------------------------
# Memory stores
# ---------------------------------------------------------------------------


class EpisodeStore(Protocol):
    def current_episode(self) -> Episode | None:
        """The open episode, or None when there is none (first message ever, or the
        last one was closed)."""
        ...

    def open_episode(self, at: datetime) -> Episode:
        ...

    def close_episode(self, episode_id: int, at: datetime) -> None:
        """Mark the episode ended at `at` (normally its last message's time)."""
        ...

    def get_episodes(self, ids: Sequence[int]) -> list[Episode]:
        """Episodes by id, in the order given. Batched so retrieval loads its hits in
        one query, not one per episode. Ids that don't exist or aren't active
        (including forgotten ones) are skipped."""
        ...

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
        """Written by consolidation, with one embedding per summary (each carries its
        own text_hash, since the two are rewritten independently). Episode search
        compares a query against both and keeps the better match."""
        ...

    def recent_episodes(self, limit: int, *, exclude_ids: Sequence[int] = ()) -> list[Episode]:
        """Closed, active episodes, most recently ended first: what the prompt's
        "recent conversations" section shows, whether or not they match the message."""
        ...

    def mark_consolidated(self, episode_id: int, at: datetime) -> None:
        """Record that consolidation finished for this episode."""
        ...

    def save_drift(self, episode_id: int, strikes: int, pending_message_id: int | None) -> None:
        """Record the episode's drift state (bookkeeping, never audited)."""
        ...

    def move_messages(self, message_ids: Sequence[int], episode_id: int) -> None:
        """Move messages, with their tool calls, to another episode (a split by
        change of subject starts the new episode at an earlier message)."""
        ...

    def save_running_summary(self, episode_id: int, summary: str, until_message_id: int) -> None:
        """Store the summary of the episode's messages up to and including
        `until_message_id`. Rewritten every ~10 messages: bookkeeping, not an
        audited edit."""
        ...

    def pending_consolidation(self, limit: int) -> list[Episode]:
        """Closed, active episodes not consolidated yet, oldest first: what the
        consolidation sweep should process."""
        ...


class MessageStore(Protocol):
    def add_message(
        self,
        episode_id: int,
        role: Role,
        content: str,
        at: datetime,
        embedding: Embedding | None = None,
    ) -> Message:
        """Append a message to an episode. Also advances the episode's
        `last_message_at`, so the two can never disagree."""
        ...

    def get_messages(
        self, episode_id: int, *, last: int | None = None, after: int | None = None
    ) -> list[Message]:
        """An episode's messages in the order they were added, oldest first. Order
        follows insertion, not timestamps, so a clock that jumps back can't reorder
        the conversation. `after` keeps only messages added after that message id
        (those the running summary doesn't cover yet); `last` then keeps only the
        most recent N. None for both returns all (consolidation)."""
        ...

    def get_message(self, message_id: int) -> Message | None:
        """One message by id, or None if it doesn't exist or its episode isn't
        active (the memory tools open messages by id)."""
        ...

    def set_pinned(self, message_id: int, pinned: bool) -> None:
        """Pin a message (it survives the recent window) or unpin it."""
        ...

    def pinned_messages(self, episode_id: int) -> list[Message]:
        """The episode's pinned messages, oldest first."""
        ...


class ConversationStore(EpisodeStore, MessageStore, Protocol):
    """What one turn of the conversation needs: episodes and their messages."""


class TagStore(Protocol):
    def upsert_tag(self, name: str, embedding: Embedding | None = None) -> Tag | None:
        """Return the tag with this name, creating it if needed. Names are compared
        by brain.text.normalize_key, so "Databases" and "databases" are one tag. A tag
        the system set aside is reactivated. Returns None when the user has forgotten
        it: callers skip it, so forgotten memory is never silently re-learned. With
        `embedding` (of the name), the tag keeps that vector for its model."""
        ...

    def similar_tags(self, query: Embedding, limit: int) -> list[tuple[Tag, float]]:
        """The tags closest in meaning to `query` (embedded with the same model),
        active or forgotten (each with its status), closest first, with their cosine
        similarity: so consolidation reuses "programming" for "coding", and never
        relearns a forgotten topic under another name."""
        ...

    def list_tags(self, limit: int) -> list[Tag]:
        """Active tags, most used first: the vocabulary consolidation reuses."""
        ...

    def tag_episode(self, episode_id: int, tag_id: int, relevance: float) -> None:
        """Attach a tag to an episode; doing it again updates the relevance."""
        ...


class EntityStore(Protocol):
    def upsert_entity(
        self,
        name: str,
        type: str = "",
        aliases: Sequence[str] = (),
        embedding: Embedding | None = None,
    ) -> Entity | None:
        """Same rules as upsert_tag, matching by identity keys
        (brain.text.identity_keys: the name and aliases, normalized) against those of
        existing entities. New aliases are added to the matched entity, and its
        search keys (brain.text.search_keys) kept up to date."""
        ...

    def match_entities(self, keys: Collection[str]) -> list[Entity]:
        """Active entities with a search key among `keys` (brain.text.phrase_keys of
        a message): the people, places, and projects the message names."""
        ...

    def link_fact_entities(self, fact_id: int, entity_ids: Sequence[int]) -> None:
        """Record that a fact is about these entities (its statement names them).
        Linking twice is harmless."""
        ...

    def active_entities(self) -> list[Entity]:
        """Every active entity, oldest first (brain/names.py compares a message's
        words with their names to catch typos)."""
        ...

    def add_mention(self, entity_id: int, message_id: int) -> None:
        """Record that a message mentions an entity (its episode is implied).
        Recording the same mention twice is harmless."""
        ...


class SearchStore(Protocol):
    """Candidate generation for the retrieval engine. Only active memory is ever
    returned; scoring and budget filling happen in brain/."""

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
        """Nearest neighbours of `query` among rows embedded with the same model,
        closest first, with `semantic` set. Episodes are compared on both summaries,
        keeping the closer one; facts only while active and not superseded.
        `exclude_episode_ids` drops memory from those episodes (the current one is
        already in the prompt) before `limit` applies. `after` (inclusive) and
        `before` (exclusive) keep only memory from that time window: a message's time,
        an episode's start, a fact's learning."""
        ...

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
        """Messages (or episode summaries, or facts learned from such messages) that
        mention any of these entities, with `lexical` set to the share of them
        mentioned (1.0 = all). Most mentioned first, then newest. Episodes without a
        summary are skipped. `after` and `before` as in semantic_candidates."""
        ...

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
        """Messages (or episode summaries, or facts) containing any of these terms
        literally, case ignored, with `keyword` set to the share of the terms they
        contain. Most terms first, then newest. `after` and `before` as in
        semantic_candidates."""
        ...

    def entity_fact_candidates(
        self,
        entity_ids: Sequence[int],
        query: Embedding,
        limit: int,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]:
        """The facts linked to any of these entities, closest to `query` first (the
        same model's vectors only): active, not superseded, active entities only,
        learned within `after` and `before` when given."""
        ...

    def episode_tags(self, episode_ids: Sequence[int]) -> dict[int, dict[int, float]]:
        """Each episode's active tags, as {episode_id: {tag_id: relevance}}. Episodes
        without any are absent."""
        ...

    def stored_vectors(self, kind: MemoryKind, ids: Sequence[int], model: str) -> dict[int, list[float]]:
        """The stored vectors of these memories under `model` (an episode's short
        summary), as {id: vector}, for comparing candidates with each other (lateral
        inhibition). Memories without one are absent."""
        ...


class EmbeddingCacheStore(Protocol):
    """Lookup table from `(model, text_hash)` to vector, keys computed with
    brain.hashing.text_hash. The engine decides when to use the cache; the store only
    keeps the entries."""

    def get_cached(self, keys: Sequence[str], model: str) -> dict[str, Embedding]:
        """The cached embeddings for those keys under that model; misses are absent."""
        ...

    def put_cached(self, entries: Mapping[str, Embedding]) -> None:
        ...


class FactStore(Protocol):
    """Facts (docs/FACTS.md). The store records; deciding whether a new fact is the
    same as, an update of, or distinct from an old one happens in brain/facts.py."""

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
        """Store a new active fact with its sources and its vector."""
        ...

    def confirm_fact(self, fact_id: int, at: datetime, message_ids: Sequence[int] = ()) -> None:
        """The fact was stated again: one more confirmation, and more sources."""
        ...

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        """`new_id` replaces `old_id`, which stays as history but is no longer recalled."""
        ...

    def similar_facts(self, query: Embedding, limit: int) -> list[tuple[Fact, float]]:
        """The closest facts embedded with the same model that aren't superseded,
        active or forgotten (each with its status), closest first, with their cosine
        similarity. Set-aside facts are skipped. Used to deduplicate new facts, and to
        refuse relearning forgotten ones."""
        ...

    def get_facts(self, ids: Sequence[int]) -> list[Fact]:
        """Facts by id, in that order, whatever their status (missing ids skipped)."""
        ...

    def fact_history(self, fact_id: int) -> list[Fact]:
        """The chain this fact belongs to, oldest first: what it replaced, and what
        replaced it."""
        ...


class TextExtractor(Protocol):
    """Reads one kind of file into text (adapters/extractors/)."""

    def supports(self, media_type: str) -> bool:
        ...

    def extract(self, path: str, media_type: str) -> ExtractedDocument:
        """The document's text as blocks. Raises on a file it can't parse."""
        ...


class AttachmentStore(Protocol):
    """Files the user shared (docs/ATTACHMENTS.md). Stored once per content hash;
    read in the background into passages, which retrieval searches like other memory
    (MemoryKind.ATTACHMENT in semantic_candidates, keyword_candidates, and
    mention_candidates). Only active attachments that were read are ever searched."""

    def add_attachment(
        self, filename: str, media_type: str, size_bytes: int, content_hash: str,
        storage_path: str, at: datetime,
    ) -> tuple[Attachment, bool] | None:
        """The attachment with this content hash, created if new (True) or reused
        (False; one set aside comes back). None when the user forgot it: a forgotten
        file isn't relearned because it was shared again."""
        ...

    def link_attachment(self, message_id: int, attachment_id: int) -> None:
        """Record that a message shared the attachment (twice is harmless)."""
        ...

    def get_attachment(self, attachment_id: int) -> Attachment | None:
        """An active attachment, or None."""
        ...

    def message_attachments(self, message_ids: Sequence[int]) -> dict[int, list[Attachment]]:
        """The active attachments each message shared. Messages with none are absent."""
        ...

    def pending_attachments(self, limit: int) -> list[Attachment]:
        """Active attachments not read yet (extraction pending), oldest first."""
        ...

    def unsummarized_attachments(self, limit: int) -> list[Attachment]:
        """Active, read attachments without a title and summary yet, oldest first."""
        ...

    def save_chunks(
        self, attachment_id: int, chunks: Sequence[tuple[Chunk, Embedding]],
        mentions: Sequence[tuple[int, int]], page_count: int | None,
    ) -> None:
        """Replace the attachment's passages (with their vectors) and their mentions,
        as (chunk position, entity id) pairs, and mark it read (done)."""
        ...

    def mark_attachment(self, attachment_id: int, extraction_status: str, error: str = "") -> None:
        """Record that reading failed or that the type isn't supported."""
        ...

    def attachment_chunks(self, attachment_id: int) -> list[Chunk]:
        """An active attachment's passages in document order."""
        ...

    def save_attachment_summary(self, attachment_id: int, title: str, summary_short: str,
                                summary_long: str) -> None:
        ...

    def tag_attachment(self, attachment_id: int, tag_id: int) -> None:
        ...


class CarryStore(Protocol):
    """What the last turns of an episode recalled, so the next turn can keep it in
    mind (docs/ACTIVATION.md, idea B). Bookkeeping: rewritten every turn, never
    audited."""

    def carried_memories(self, episode_id: int) -> list[CarriedMemory]:
        """What was saved for this episode, as saved (empty if nothing)."""
        ...

    def save_carried_memories(self, episode_id: int, memories: Sequence[CarriedMemory]) -> None:
        """Replace what is saved for this episode."""
        ...


class GraphStore(Protocol):
    """The activation graph (docs/ACTIVATION.md, idea C). Nodes are tags and entities;
    which episodes they appear in is all the graph needs. Only active nodes and
    episodes count. The activation values themselves are bookkeeping, rewritten
    every turn, never audited."""

    def episode_nodes(self, episode_ids: Sequence[int]) -> dict[int, set[Node]]:
        """Each episode's tags and the entities its messages mention. Episodes with
        none are absent."""
        ...

    def node_episodes(self, nodes: Sequence[Node]) -> dict[Node, set[int]]:
        """The episodes each node appears in. Nodes in none are absent."""
        ...

    def load_activation(self) -> dict[Node, tuple[float, datetime]]:
        """Every saved activation value, with when it was set."""
        ...

    def save_activation(self, values: Mapping[Node, tuple[float, datetime]]) -> None:
        """Replace all saved activation values."""
        ...


class ImportStore(Protocol):
    """Which messages came from an imported chat history (brain/importing.py), by the
    export's own ids: a chat already imported is skipped, and an eval question that
    points at an export's message finds the stored one. Bookkeeping, never recalled."""

    def record_imports(self, source: str, conversation_id: str, messages: Mapping[str, int]) -> None:
        """Remember that the export's messages (its id to the stored message's id) of
        one conversation were imported."""
        ...

    def imported_conversations(self, source: str) -> set[str]:
        """The conversation ids already imported from this source."""
        ...

    def imported_messages(self, source: str, conversation_id: str) -> dict[str, int]:
        """The export's message ids of one imported conversation, to the stored ids."""
        ...


class RetrievalStore(SearchStore, GraphStore, Protocol):
    """What retrieval reads: candidates, and the graph for the activation signal."""


class ToolCallStore(Protocol):
    """A log of the memory tools' calls. Written once per turn, after the reply;
    never read by consolidation (tool results are memory that is already stored, and
    re-extracting them would re-learn old things as new)."""

    def add_tool_calls(
        self, episode_id: int, message_id: int | None, runs: Sequence[ToolRun]
    ) -> None:
        """Record the calls of one turn, linked to the reply they led to (None when
        the turn ended without a reply)."""
        ...

    def get_tool_calls(self, episode_id: int) -> list[ToolRun]:
        """An episode's tool calls, oldest first, each with its `message_id`."""
        ...


class TurnStore(ConversationStore, SearchStore, EmbeddingCacheStore, Protocol):
    """What one turn with recall needs: the conversation, search, and the embedding
    cache."""


@runtime_checkable
class MemoryStore(TurnStore, TagStore, EntityStore, FactStore, ToolCallStore, CarryStore, GraphStore,
                  AttachmentStore, ImportStore, Protocol):
    """Everything the engine needs from storage. DjangoMemoryStore and InMemoryStore
    implement this; engine functions type their argument with the narrowest Protocol
    they use (e.g. EpisodeStore), which keeps each function's needs visible."""
