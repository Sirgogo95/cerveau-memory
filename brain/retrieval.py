"""Retrieval (PIPELINE.md steps 6 and 7): find candidates in each memory source, score
them with the source's config, keep the ones worth showing.

Three signals so far: meaning (semantic similarity to the message), names (the
message names an entity, so memories that mention it count, however differently
they're worded), and tags (an episode whose topic is close to the message gets a
small boost). Activation joins later; each adds a signal to `score()`.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

from .config import EngineConfig, SourceConfig
from .periods import Period
from .ports import ConversationStore, GraphStore, Reranker, RetrievalStore, SearchStore, TagStore
from .types import Candidate, Embedding, Entity, Episode, MemoryKind, Message, Node, Role, Tag

# Messages shown for a recent episode that has no summary yet (it was just closed and
# consolidation hasn't finished): its last few messages stand in for the summary.
PREVIEW_MESSAGES = 4
# Entities named in a message are shown as short cards; more than this is noise.
MAX_ENTITY_CARDS = 5
# Tags matched to a message: its few closest topics.
MAX_TAG_MATCHES = 3
# Passages of one shared file per turn: a long document mustn't fill the memory block.
MAX_PASSAGES_PER_FILE = 2


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    score: float


def _sources(config: EngineConfig) -> list[tuple[MemoryKind, SourceConfig]]:
    return [
        (MemoryKind.FACT, config.facts),
        (MemoryKind.EPISODE, config.episodes),
        (MemoryKind.MESSAGE, config.messages),
        (MemoryKind.ATTACHMENT, config.attachments),
    ]


def worth_recalling(candidate: Candidate, config: EngineConfig) -> bool:
    """False for Cerveau's short replies ("Noted.", "Classic."): nothing to remember
    in them, and their generic wording scores high against unrelated questions."""
    return not (
        candidate.kind is MemoryKind.MESSAGE and candidate.role is Role.ASSISTANT
        and len(candidate.text.strip()) < config.short_reply_chars
    )


def semantic_kept(
    store: SearchStore, kind: MemoryKind, query: Embedding, wanted: int, config: EngineConfig,
    **filters,
) -> list[Candidate]:
    """The nearest `wanted` candidates worth recalling. When short replies are
    dropped, the store is asked again with twice the room (up to 8x) until enough
    are left or it has no more: four "Noted." replies closest to the query must not
    push the real memory out."""
    room = wanted
    while True:
        found = store.semantic_candidates(kind, query, room, **filters)
        kept = [c for c in found if worth_recalling(c, config)]
        if len(kept) >= wanted or len(found) < room or room >= wanted * 8:
            return kept[:max(wanted, len(kept))]
        room *= 2


def score(candidate: Candidate, source: SourceConfig) -> float:
    return (
        source.w_semantic * (candidate.semantic or 0.0)
        + source.w_lexical * (candidate.lexical or 0.0)
        + source.w_tag * (candidate.topical or 0.0)
        + source.w_act * (candidate.activation or 0.0)
        + source.w_keyword * (candidate.keyword or 0.0)
    )


def with_activation(
    candidates: Sequence[Candidate], activation: Mapping[Node, float], store: GraphStore
) -> list[Candidate]:
    """Each candidate gets `activation`: the highest value among its episode's tags
    and entities (none if they're all at rest)."""
    per_episode = store.episode_nodes(sorted({c.episode_id for c in candidates if c.episode_id is not None}))
    activated = []
    for c in candidates:
        values = [activation[n] for n in per_episode.get(c.episode_id or 0, ()) if n in activation]
        activated.append(replace(c, activation=max(values)) if values else c)
    return activated


def with_carried(scored: Sequence[ScoredCandidate], carried: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    """One entry per memory: the higher of its score now and its carried score."""
    best = {s.candidate.id: s for s in scored}
    for s in carried:
        found = best.get(s.candidate.id)
        if found is None or s.score > found.score:
            best[s.candidate.id] = ScoredCandidate(found.candidate if found else s.candidate, s.score)
    return list(best.values())


def per_file(scored: Sequence[ScoredCandidate], limit: int) -> list[ScoredCandidate]:
    """At most `limit` passages of each file, best first."""
    counts: dict[int | None, int] = {}
    kept = []
    for s in scored:
        counts[s.candidate.parent_id] = counts.get(s.candidate.parent_id, 0) + 1
        if counts[s.candidate.parent_id] <= limit:
            kept.append(s)
    return kept


def matched_tags(query: Embedding, store: TagStore, config: EngineConfig) -> list[tuple[Tag, float]]:
    """The tags close enough to the message (`tag_match_score`), closest first, at
    most MAX_TAG_MATCHES. Forgotten tags never match."""
    return [
        (tag, similarity) for tag, similarity in store.similar_tags(query, MAX_TAG_MATCHES)
        if tag.status == "A" and similarity >= config.tag_match_score
    ]


def with_topics(
    candidates: Sequence[Candidate], tags: Sequence[tuple[Tag, float]], store: SearchStore
) -> list[Candidate]:
    """Episodes carrying a matched tag get `topical`: the best similarity among
    them (times the tag's relevance to the episode)."""
    matched = {tag.id: similarity for tag, similarity in tags}
    per_episode = store.episode_tags([c.id for c in candidates if c.kind is MemoryKind.EPISODE])
    topped = []
    for c in candidates:
        found = [matched[t] * r for t, r in per_episode.get(c.id, {}).items() if t in matched]
        topped.append(replace(c, topical=max(found)) if found and c.kind is MemoryKind.EPISODE else c)
    return topped


def merge(semantic: Sequence[Candidate], named: Sequence[Candidate]) -> list[Candidate]:
    """One candidate per memory: found by meaning, by name, by exact terms, or several
    (then it carries each signal it was found with)."""
    by_id = {c.id: c for c in semantic}
    for c in named:
        found = by_id.get(c.id)
        by_id[c.id] = replace(
            found,
            lexical=c.lexical if c.lexical is not None else found.lexical,
            keyword=c.keyword if c.keyword is not None else found.keyword,
        ) if found else c
    return list(by_id.values())


def rerank(passing: Sequence[ScoredCandidate], text: str, reranker: Reranker) -> list[ScoredCandidate]:
    """The candidates in the order the reranker puts them for the message `text`
    (it reads the two together); their scores stay as they were."""
    scores = reranker.score(text, [s.candidate.text for s in passing])
    order = sorted(range(len(passing)), key=lambda i: scores[i], reverse=True)
    return [passing[i] for i in order]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def inhibit(passing: Sequence[ScoredCandidate], keep: int, strength: float,
            vectors: Mapping[int, Sequence[float]]) -> list[ScoredCandidate]:
    """Lateral inhibition: `keep` candidates picked one at a time, each time the one
    whose score minus `strength` x its closest similarity to those already picked is
    highest, so a near-duplicate of a kept memory gives way to a different one. A
    candidate without a vector is never lowered. Scores are returned unchanged."""
    unit = {i: [x / norm for x in v] for i, v in vectors.items() if (norm := math.sqrt(sum(x * x for x in v)))}
    closest = {id(s): 0.0 for s in passing}          # each candidate's highest similarity to the picked
    pool = list(passing)
    chosen: list[ScoredCandidate] = []
    while pool and len(chosen) < keep:
        best = max(pool, key=lambda s: s.score - strength * closest[id(s)])
        pool.remove(best)
        chosen.append(best)
        if (picked := unit.get(best.candidate.id)) is not None:
            for s in pool:
                if (vector := unit.get(s.candidate.id)) is not None:
                    closest[id(s)] = max(closest[id(s)], sum(a * b for a, b in zip(vector, picked)))
    return chosen


def entity_card(entity: Entity) -> Candidate:
    """A named entity shown in the memory block: "Dr. Marta Peña (person; also: my
    colleague)"."""
    details = [entity.type] if entity.type else []
    if entity.aliases:
        details.append("also: " + ", ".join(entity.aliases))
    text = f"{entity.name} ({'; '.join(details)})" if details else entity.name
    return Candidate(kind=MemoryKind.ENTITY, id=entity.id, text=text, lexical=1.0)


@dataclass(frozen=True, slots=True)
class RecentEpisode:
    """An episode for the prompt's "recent conversations" section. `preview` holds
    its last messages when it has no summary yet."""

    episode: Episode
    preview: list[Message] = field(default_factory=list)


def recent_context(
    store: ConversationStore, config: EngineConfig, *, current_episode_id: int | None
) -> list[RecentEpisode]:
    """The most recent closed episodes, newest first, shown in every prompt whatever
    the message says. Similarity can't answer "what did we talk about last?" (the
    question resembles nothing that was said); recency can."""
    if config.recent_episodes == 0:
        return []
    exclude = [current_episode_id] if current_episode_id is not None else []
    return [
        RecentEpisode(
            episode,
            [] if episode.summary_short else store.get_messages(episode.id, last=PREVIEW_MESSAGES),
        )
        for episode in store.recent_episodes(config.recent_episodes, exclude_ids=exclude)
    ]


def retrieve(
    query: Embedding,
    store: RetrievalStore,
    config: EngineConfig,
    *,
    current_episode_id: int | None = None,
    shown_episode_ids: Collection[int] = (),
    entities: Sequence[Entity] = (),
    tags: Sequence[tuple[Tag, float]] = (),
    carried: Sequence[ScoredCandidate] = (),
    activation: Mapping[Node, float] | None = None,
    terms: Sequence[str] = (),
    period: Period | None = None,
    reranker: Reranker | None = None,
    text: str = "",
    entity_query: Embedding | None = None,
) -> list[ScoredCandidate]:
    """Relevant memory from outside the current episode, best first.

    Each source contributes at most `max_items` candidates scoring at least
    `min_score`, found by meaning and, when the message names `entities`, by the
    memories that mention them. Those entities are added as cards. Episodes carrying
    one of `tags` (from matched_tags) get the tag signal. `carried` memories (from
    earlier turns, already scored) compete with what is found now; a memory in both
    keeps the higher score. With `activation` (from brain/activation.py), memories
    whose episode carries an active node get the activation signal. Memories
    containing `terms` (brain/keywords.py) literally are found too, with the keyword
    signal. The current
    episode is excluded: its recent messages are already in the prompt. Summaries of
    `shown_episode_ids` (the recent episodes the prompt already shows) are skipped
    too; their messages can still match.
    """
    exclude = [current_episode_id] if current_episode_id is not None else []
    entity_ids = [e.id for e in entities]
    kept: list[ScoredCandidate] = [
        ScoredCandidate(entity_card(e), 1.0) for e in entities[:MAX_ENTITY_CARDS]
    ]
    for kind, source in _sources(config):
        if not source.enabled or source.max_items == 0:
            continue
        # A period in the message ("in August") is a filter: only memories from then,
        # ranked by score but none dropped for it, as in the search tool: the
        # question resembles nothing that was said then (brain/periods.py).
        window = {"after": period.start, "before": period.end} if period else {}
        # A file shared in this conversation isn't in the prompt (only the message
        # that shared it is), so its passages are searched too.
        skip = [] if kind is MemoryKind.ATTACHMENT else exclude
        reranking = reranker is not None and bool(text) and (
            config.rerank_sources == "all" or kind.value in config.rerank_sources.split(","))
        pool = source.max_items * (config.rerank_pool if reranking
                                   else config.inhibition_pool if config.inhibition > 0 else 1)
        candidates = semantic_kept(store, kind, query, pool, config, exclude_episode_ids=skip, **window)
        if entity_ids and source.w_lexical > 0:
            candidates = merge(candidates, [
                c for c in store.mention_candidates(
                    kind, entity_ids, source.max_items, exclude_episode_ids=skip, **window
                ) if worth_recalling(c, config)
            ])
        if terms and source.w_keyword > 0:
            candidates = merge(candidates, [
                c for c in store.keyword_candidates(
                    kind, terms, source.max_items, exclude_episode_ids=skip, **window
                ) if worth_recalling(c, config)
            ])
        if kind is MemoryKind.EPISODE:
            candidates = [c for c in candidates if c.id not in shown_episode_ids]
            if tags and source.w_tag > 0:
                candidates = with_topics(candidates, tags, store)
        if activation and source.w_act > 0:
            candidates = with_activation(candidates, activation, store)
        scored = [ScoredCandidate(c, score(c, source)) for c in candidates]
        if carried:
            scored = with_carried(scored, [
                s for s in carried if s.candidate.kind is kind
                and s.candidate.episode_id not in exclude
                and not (kind is MemoryKind.EPISODE and s.candidate.id in shown_episode_ids)
            ])
        passing = [s for s in scored if period is not None or s.score >= source.min_score]
        passing.sort(key=lambda s: s.score, reverse=True)
        if kind is MemoryKind.ATTACHMENT:
            passing = per_file(passing, MAX_PASSAGES_PER_FILE)
        if reranking and len(passing) > source.max_items:
            assert reranker is not None
            passing = rerank(passing[:pool], text, reranker)
        if config.inhibition > 0 and len(passing) > source.max_items:
            passing = inhibit(passing, source.max_items, config.inhibition,
                              store.stored_vectors(kind, [s.candidate.id for s in passing], query.model))
        kept.extend(passing[:source.max_items])
    if config.entity_block.enabled and entities:
        kept = with_entity_blocks(kept, query, store, config, entities, period, entity_query)
    return sorted(kept, key=lambda s: s.score, reverse=True)


def with_entity_blocks(
    kept: list[ScoredCandidate], query: Embedding, store: RetrievalStore, config: EngineConfig,
    entities: Sequence[Entity], period: Period | None = None, entity_query: Embedding | None = None,
) -> list[ScoredCandidate]:
    """Each named entity (at most `max_entity_blocks`, in the order named) brings its
    own facts, ranked by the message, above a threshold below the facts' (the scope
    already says they're about the right entity). A block fact carries its entity as
    `parent_id`, so the prompt shows it under the entity's card; a fact found both
    ways appears once, in the block."""
    block = config.entity_block
    threshold = config.facts.min_score + block.min_score_offset
    window = {"after": period.start, "before": period.end} if period else {}
    blocks: list[ScoredCandidate] = []
    taken: set[int] = set()
    weight = block.without_name_weight if entity_query is not None else 0.0
    for entity in entities[:block.max_entity_blocks]:
        # A wider pool, so the mixed ranking can pick from more than the top few.
        pool = block.max_items * (3 if weight else 1)
        asked = store.entity_fact_candidates([entity.id], query, pool, **window)
        bare = ({c.id: c.semantic or 0.0 for c in store.entity_fact_candidates([entity.id], entity_query, pool, **window)}
                if weight and entity_query is not None else {})
        ranked = sorted(asked, key=lambda c: (1 - weight) * (c.semantic or 0.0) + weight * bare.get(c.id, 0.0),
                        reverse=True)
        kept_here = 0
        for c in ranked:
            if kept_here >= block.max_items:
                break
            if c.id in taken or (period is None and (c.semantic or 0.0) < threshold):
                continue
            taken.add(c.id)
            kept_here += 1
            blocks.append(ScoredCandidate(replace(c, parent_id=entity.id), c.semantic or 0.0))
        if block.expand_seeds and block.expand_items:
            seeds = [s.candidate.id for s in blocks if s.candidate.parent_id == entity.id][:block.expand_seeds]
            blocks += neighbor_facts(store, entity.id, seeds, query, taken, block.expand_items,
                                     config.facts.min_score, window)
    others = [s for s in kept if not (s.candidate.kind is MemoryKind.FACT and s.candidate.id in taken)]
    return others + blocks


def neighbor_facts(
    store: RetrievalStore, entity_id: int, seeds: Sequence[int], query: Embedding, taken: set[int],
    limit: int, min_similarity: float, window: Mapping,
) -> list[ScoredCandidate]:
    """The second search of an entity block: the entity's facts closest to each seed
    fact (not to the question), at least `min_similarity` to it, up to `limit` new
    ones. The pieces of a list resemble each other more than they resemble the
    question ("What activities does Melanie do?")."""
    vectors = store.stored_vectors(MemoryKind.FACT, list(seeds), query.model)
    found: dict[int, ScoredCandidate] = {}
    for seed in seeds:
        if seed not in vectors:
            continue
        near = store.entity_fact_candidates([entity_id], Embedding(vectors[seed], query.model), limit + len(seeds) + 1,
                                            **window)
        for c in near:
            similarity = c.semantic or 0.0
            if c.id in taken or similarity < min_similarity:
                continue
            if c.id not in found or found[c.id].score < similarity:
                found[c.id] = ScoredCandidate(replace(c, parent_id=entity_id), similarity)
    added = sorted(found.values(), key=lambda s: s.score, reverse=True)[:limit]
    taken.update(s.candidate.id for s in added)
    return added
