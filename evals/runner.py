"""Running eval cases against the engine's real retrieval.

Each scenario becomes an InMemoryStore filled the way the app would fill it: messages
embedded through the cache, closed episodes, summaries embedded, tags with their vectors, entities with their
mentions. Each query then goes through exactly what a turn does before the prompt:
`named_entities` (exact and misspelled names), `matched_tags`, activation from those
(as for a first message), then `retrieve`. Only search quality is measured
here (the recent-conversations block and the prompt budget are not involved).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.activation import update_activation
from brain.config import EngineConfig, all_sources, with_min_score
from brain.consolidation import mentioning
from brain.embedding import embed_cached
from brain.entity_links import without_names
from brain.attachments import document_title, embedding_texts
from brain.chunking import blocks_from_text, chunk_document
from brain.hashing import text_hash
from brain.keywords import keyword_terms
from brain.text import phrase_keys
from brain.ports import Embedder, Reranker
from brain.retrieval import matched_tags, retrieve
from brain.names import named_entities
from brain.types import Embedding, ExtractedDocument, MemoryKind, Role

from .cases import QueryCase, Scenario

T0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class QueryResult:
    scenario: str
    query: QueryCase
    kept: tuple[str, ...]  # references of the memories retrieve kept, best first

    @property
    def negative(self) -> bool:
        """A question that should recall nothing."""
        return not self.query.expect

    @property
    def found(self) -> set[str]:
        return set(self.kept) & set(self.query.expect)

    @property
    def hit(self) -> bool:
        """At least one expected memory came back."""
        return bool(self.found)

    @property
    def recall(self) -> float:
        return len(self.found) / len(self.query.expect) if self.query.expect else 1.0

    @property
    def rejected(self) -> set[str]:
        return set(self.kept) & set(self.query.reject)


@dataclass(frozen=True, slots=True)
class Metrics:
    queries: int
    hit_rate: float      # questions with an answer: share where an expected memory came back
    recall: float        # questions with an answer: share of expected memories that came back
    precision: float     # of everything kept for them, the share that was expected
    noise: float         # questions with no answer: average memories wrongly kept
    rejected: int        # memories marked "reject" that came back


@dataclass
class Prepared:
    """A scenario loaded into a store, ready to be queried."""

    scenario: Scenario
    store: InMemoryStore
    refs: dict[tuple[MemoryKind, int], str]
    passages: dict[str, str] = field(default_factory=dict)   # "lease#3" -> its text, casefolded

    def resolve(self, ref: str, case: QueryCase) -> list[str]:
        """A kept passage ("lease#3") as the expected or rejected references whose phrase
        it contains ("lease/deposit is 60,000"), or itself."""
        if "#" not in ref:
            return [ref]
        key, text = ref.split("#")[0], self.passages[ref]
        named = [r for r in (*case.expect, *case.reject)
                 if r.startswith(f"{key}/") and r.split("/", 1)[1].casefold() in text]
        return named or [ref]


def prepare(scenario: Scenario, embedder: Embedder, chunk_chars: int = 1200, overlap: int = 150) -> Prepared:
    store = InMemoryStore()
    refs: dict[tuple[MemoryKind, int], str] = {}
    passages: dict[str, str] = {}
    first_messages: dict[str, int] = {}
    for n, case in enumerate(scenario.episodes):
        start = T0 + timedelta(days=n)
        episode = store.open_episode(start)
        texts = [text for _, text in case.messages]
        vectors = embed_cached(texts, embedder, store)
        stored = []
        for i, ((role, text), vector) in enumerate(zip(case.messages, vectors)):
            message = store.add_message(
                episode.id, Role(role), text, start + timedelta(minutes=i), vector
            )
            refs[(MemoryKind.MESSAGE, message.id)] = f"{case.key}:{i}"
            stored.append(message)
            first_messages.setdefault(case.key, message.id)
        store.close_episode(episode.id, start + timedelta(minutes=len(texts)))
        if case.summary:
            long = case.summary_long or case.summary
            short_vec, long_vec = embed_cached([case.summary, long], embedder, store)
            store.save_summary(
                episode.id, title=case.key, summary_short=case.summary, summary_long=long,
                short_embedding=short_vec, long_embedding=long_vec,
            )
            refs[(MemoryKind.EPISODE, episode.id)] = case.key
        if case.tags:
            for name, vector in zip(case.tags, embed_cached(list(case.tags), embedder, store)):
                tag = store.upsert_tag(name, vector)
                if tag is not None:
                    store.tag_episode(episode.id, tag.id, 1.0)
        for entity_case in case.entities:
            entity = store.upsert_entity(entity_case.name, entity_case.type, entity_case.aliases)
            if entity is None:
                continue
            for message in mentioning(stored, [entity.name, *entity.aliases]):
                store.add_mention(entity.id, message.id)
    # Shared files last, so the names they mention are known (as when a file is read).
    for case in scenario.episodes:
        for file in case.attachments:
            found = store.add_attachment(file.filename, "text/markdown", len(file.text),
                                         text_hash(file.text), file.filename, T0)
            assert found is not None
            attachment = found[0]
            store.link_attachment(first_messages[case.key], attachment.id)
            document = ExtractedDocument(blocks_from_text(file.text, markdown=file.filename.endswith(".md")))
            chunks = chunk_document(document, chunk_chars, overlap)
            texts = embedding_texts(chunks, document_title(document, file.filename), file.filename,
                                    case.messages[0][1])
            vectors = embed_cached(texts, embedder, store)
            mentions = [(c.position, e.id) for c in chunks for e in store.match_entities(phrase_keys(c.text))]
            store.save_chunks(attachment.id, list(zip(chunks, vectors)), mentions, None)
            for chunk in store.attachment_chunks(attachment.id):
                refs[(MemoryKind.ATTACHMENT, chunk.id)] = f"{file.key}#{chunk.position}"
                passages[f"{file.key}#{chunk.position}"] = chunk.text.casefold()
    return Prepared(scenario, store, refs, passages)


def query(
    prepared: Prepared, case: QueryCase, embedder: Embedder, config: EngineConfig,
    vectors: dict[str, Embedding], reranker: Reranker | None = None,
) -> QueryResult:
    if case.text not in vectors:
        [vectors[case.text]] = embedder.embed([case.text])
    named = named_entities(case.text, prepared.store, config.fuzzy_names)
    tags = matched_tags(vectors[case.text], prepared.store, config)
    activation = None
    if any(s.w_act > 0 for s in all_sources(config)):
        # A first message, as in a turn: activation from its own names and topics only.
        matched = {(MemoryKind.ENTITY, e.id): 1.0 for e in named}
        matched.update({(MemoryKind.TAG, t.id): s for t, s in tags})
        activation = update_activation({}, matched, prepared.store, T0, config)
    entity_query = None
    if config.entity_block.enabled and config.entity_block.without_name_weight > 0 and named:
        bare = without_names(case.text, named[:config.entity_block.max_entity_blocks])
        if bare:
            [entity_query] = embedder.embed([bare])
    results = retrieve(vectors[case.text], prepared.store, config, entities=named, tags=tags,
                       activation=activation, terms=keyword_terms(case.text),
                       reranker=reranker, text=case.text, entity_query=entity_query)
    kept = tuple(dict.fromkeys(
        resolved
        for r in results
        if (r.candidate.kind, r.candidate.id) in prepared.refs
        for resolved in prepared.resolve(prepared.refs[(r.candidate.kind, r.candidate.id)], case)
    ))
    return QueryResult(prepared.scenario.name, case, kept)


def run(
    prepared: Sequence[Prepared], embedder: Embedder, config: EngineConfig,
    vectors: dict[str, Embedding] | None = None,
) -> list[QueryResult]:
    vectors = {} if vectors is None else vectors
    return [query(p, q, embedder, config, vectors) for p in prepared for q in p.scenario.queries]


def line(metrics: Metrics) -> str:
    return (
        f"hit rate {metrics.hit_rate:5.0%}   recall {metrics.recall:5.0%}   "
        f"precision {metrics.precision:5.0%}   noise {metrics.noise:4.1f}   "
        f"rejected {metrics.rejected}"
    )


def summarize(results: Sequence[QueryResult]) -> Metrics:
    answered = [r for r in results if not r.negative]
    negatives = [r for r in results if r.negative]
    kept = sum(len(r.kept) for r in answered)
    return Metrics(
        queries=len(results),
        hit_rate=sum(r.hit for r in answered) / len(answered) if answered else 1.0,
        recall=sum(r.recall for r in answered) / len(answered) if answered else 1.0,
        precision=sum(len(r.found) for r in answered) / kept if kept else 1.0,
        noise=sum(len(r.kept) for r in negatives) / len(negatives) if negatives else 0.0,
        rejected=sum(len(r.rejected) for r in results),
    )


def sweep(
    prepared: Sequence[Prepared], embedder: Embedder, config: EngineConfig,
    thresholds: Sequence[float],
) -> list[tuple[float, Metrics]]:
    """Metrics for each `min_score`, with everything else from `config`."""
    vectors: dict[str, Embedding] = {}
    return [
        (t, summarize(run(prepared, embedder, with_min_score(config, t), vectors)))
        for t in thresholds
    ]
