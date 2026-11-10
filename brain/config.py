"""How the engine is tuned (EngineConfig), what each LLM can hold (ModelProfile), and
how each embedding model is used (EmbeddingProfile).

Values live in TOML files under configs/ and are loaded with load_engine_config(),
load_model_profile() and load_embedding_profile(). Only brain/ reads these types; the
app turns profiles into a provider and an embedder with the factories in adapters/.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ConfigError(ValueError):
    """A config file or value is invalid. Raised at load time, never mid-conversation."""


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """How one memory source is searched and scored (PIPELINE.md steps 6 and 7).

    score = w_semantic × semantic similarity + w_lexical × name match + w_tag × tag
    match (+ activation in v2). The name match is the share of the entities named in
    the message that a memory mentions, so with the defaults a memory mentioning
    everything named scores at least 0.4 and passes `min_score` on that alone. The tag
    match (episodes only) is how close the message is to the episode's closest
    matching tag: a small nudge that breaks near-ties. Candidates scoring below
    `min_score` are dropped; at most `max_items` are kept. The threshold depends on
    the embedding model, so the app sets it from the EmbeddingProfile (with_min_score).
    """

    enabled: bool = True
    w_semantic: float = 1.0
    w_lexical: float = 0.4
    w_tag: float = 0.0
    w_act: float = 0.0
    w_keyword: float = 0.0
    min_score: float = 0.35
    max_items: int = 5

    def __post_init__(self):
        if not 0 <= self.min_score <= 1:
            raise ConfigError("min_score must be between 0 and 1.")
        if self.max_items < 0:
            raise ConfigError("max_items cannot be negative.")


@dataclass(frozen=True, slots=True)
class EntityBlockConfig:
    """Entity blocks (docs/features/ENTITY_PROFILES.md, step 1): when a message names
    a person, place, or project, recall searches that entity's own facts and shows
    them as a block under its card, in date order. Searching only its facts removes
    the competition with everything else that resembles the question."""

    enabled: bool = False
    max_entity_blocks: int = 2        # at most this many entities get a block, in the order named
    max_items: int = 10               # facts per block
    min_score_offset: float = -0.10   # the block's threshold, relative to the facts' min_score
    # Inside a block every fact names the entity, and so does the question, so the
    # shared name dominates similarity. The block is ranked by a mix: this weight on
    # the question without the entity's names, the rest on the question as asked
    # (0 = as asked only). The threshold stays on the question as asked.
    without_name_weight: float = 0.5
    # A second search for the missing pieces of a list: the block's best
    # `expand_seeds` facts look for their closest neighbors among the same entity's
    # facts ("went camping" finds "went swimming", which may not resemble the
    # question), adding up to `expand_items` of them. 0 turns it off.
    expand_seeds: int = 0
    expand_items: int = 0

    def __post_init__(self):
        if (self.max_entity_blocks < 0 or self.max_items < 0 or not -1 <= self.min_score_offset <= 0
                or not 0 <= self.without_name_weight <= 1 or self.expand_seeds < 0 or self.expand_items < 0):
            raise ConfigError("entity_block: max_entity_blocks and max_items at least 0; "
                              "min_score_offset between -1 and 0; without_name_weight between 0 and 1.")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Engine tuning. Grows one field at a time as features land (see PIPELINE.md)."""

    episode_gap: timedelta = timedelta(hours=2)
    # Messages of the current episode kept verbatim before the oldest are folded into
    # its running summary...
    recent_window: int = 20
    # ...this many at a time. Folding in batches keeps the start of the prompt
    # identical for several turns (prefix caching) and costs one model call per batch.
    summary_chunk: int = 10
    # Share of ModelProfile.memory_budget reserved for retrieved memory; whatever it
    # doesn't use goes to the recent messages.
    memory_share: float = 0.3
    messages: SourceConfig = field(default_factory=SourceConfig)
    episodes: SourceConfig = field(default_factory=lambda: SourceConfig(max_items=3))
    facts: SourceConfig = field(default_factory=lambda: SourceConfig(max_items=5))
    # Passages of shared files (brain/attachments.py); at most MAX_PASSAGES_PER_FILE
    # of one file per turn (brain/retrieval.py).
    attachments: SourceConfig = field(default_factory=lambda: SourceConfig(max_items=3))
    # Cerveau's replies shorter than this ("Noted.", "Classic.") are never recalled as
    # messages: they hold nothing of their own, and short, generic texts score high
    # against unrelated questions (measured: 0.58 for "A good ritual. What do you
    # order?" against "who asked me to be best man?"). 0 recalls them all.
    short_reply_chars: int = 60
    # Lateral inhibition: each source picks its memories one at a time, and a
    # candidate's score is lowered by `inhibition` x its closest similarity to those
    # already picked, so near-duplicates give way to different memories (several
    # messages saying the same thing filled every slot on LoCoMo). It chooses among
    # the best `inhibition_pool` x max_items that pass the threshold; scores stay as
    # they were. 0 turns it off.
    inhibition: float = 0.0
    inhibition_pool: int = 3
    # With a reranker (brain.ports.Reranker; CERVEAU_RERANKER in the app), each
    # source reads the best `rerank_pool` x max_items candidates that pass the
    # threshold and keeps the max_items the reranker puts first. Scores and
    # thresholds stay as they were.
    rerank_pool: int = 4
    # Which sources the reranker reorders: "all", or kinds separated by commas
    # ("message"). Reordering within facts dropped the key fact on LoCoMo.
    rerank_sources: str = "all"
    entity_block: EntityBlockConfig = field(default_factory=EntityBlockConfig)
    # Find names typed with a typo ("Lucai" for Lucía; brain/names.py).
    fuzzy_names: bool = True
    # Conversation-aware recall (docs/ACTIVATION.md, idea A): the search vector adds
    # the previous `context_messages` user messages of the conversation, the k-th
    # most recent weighted context_weight^k, so a follow-up ("can I still play on
    # Saturday?") keeps the subject of the message before. 0 searches with the new
    # message alone.
    context_messages: int = 2
    context_weight: float = 0.0
    # Carry-over (docs/ACTIVATION.md, idea B): what a turn recalled stays a candidate
    # in the next turns, its score multiplied by carry_decay each turn, so it stays
    # while it still clears min_score. 0 forgets everything between turns.
    carry_decay: float = 0.0
    # ...but only when the new message finds nothing scoring carry_gate or more on its
    # own: a message with a clear subject of its own ("where should I get coffee?")
    # has moved on; a vague follow-up ("what were my numbers?") hasn't.
    carry_gate: float = 1.0
    # A message at least this similar to the one before is still on its subject:
    # carry-over applies, and activation carries over (brain/recall.py).
    subject_similarity: float = 0.0
    # Activation (docs/ACTIVATION.md, idea C; brain/activation.py): each turn a node's
    # value fades by activation_turn_decay and halves every activation_half_life_hours;
    # matched nodes are boosted, and activation_spread of the boost goes to their
    # neighbours. Memories gain SourceConfig.w_act x their episode's highest value.
    activation_turn_decay: float = 0.3
    activation_half_life_hours: float = 6.0
    activation_spread: float = 0.5
    # Shared files are split into passages of about this many characters, the next
    # one repeating up to attachment_overlap of the last (brain/chunking.py); never
    # more than the embedding model reads (attachment_chunk_size()).
    # Drift (PIPELINE.md step 3, brain/segmentation.py): a message whose similarity to
    # the average of the episode's last drift_window user messages is under
    # drift_threshold is a strike; drift_strikes in a row start a new episode at the
    # first of them. Messages under drift_min_words, and episodes with fewer than
    # drift_min_messages user messages, are never checked. 0 turns it off.
    drift_threshold: float = 0.0
    drift_strikes: int = 2
    drift_window: int = 2
    drift_min_words: int = 5
    drift_min_messages: int = 3
    # Carry-over (PIPELINE.md step 2): after a split by subject, the previous
    # episode's last carryover_messages lead the new episode's prompt, one fewer each
    # turn (3, then 2, then 1...), against a split that was wrong. 0 turns it off.
    carryover_messages: int = 0
    # Pinned messages the window has moved past stay in the prompt, up to this share
    # of the memory budget (oldest pins dropped first if they don't all fit).
    pinned_share: float = 0.2
    attachment_chunk_chars: int = 1200
    attachment_overlap: int = 150
    # How many recent episodes the prompt always shows (title, dates, short summary),
    # so "where were we?" works without any similarity to the past.
    recent_episodes: int = 3
    # Timezone for dates shown to the model, e.g. "America/Santo_Domingo". None uses
    # the machine's local timezone (Cerveau runs locally, on the user's machine).
    timezone: str | None = None
    # Deliberate recall (brain/tools.py): offer the memory tools to models that
    # support tools; at most this many tool calls per reply.
    memory_tools: bool = True
    max_tool_calls: int = 3
    # When a reply opens by declining and the model hasn't searched, search its memory
    # with the user's words and answer again (brain/turn.py, step 9c).
    fallback_search: bool = True
    # Share of ModelProfile.memory_budget one tool result may use.
    tool_result_share: float = 0.15
    # Threshold for search_memory: lower than automatic recall's, since the model
    # asked explicitly. Depends on the embedding model, so the app sets it from the
    # EmbeddingProfile (with_min_score).
    search_min_score: float = 0.25
    # Facts (brain/facts.py): learned at consolidation when on. The deduplication
    # thresholds depend on the embedding model, so the app sets them from the
    # EmbeddingProfile (with_embedding).
    learn_facts: bool = True
    # At most this many facts from one conversation (one model call).
    max_facts: int = 8
    fact_same_score: float = 0.97
    fact_related_score: float = 0.70
    # Tags at least this close in meaning are one topic: consolidation reuses the
    # existing tag ("programming" for "coding"). From the embedding profile.
    tag_same_score: float = 0.85
    # A tag at least this close to the message matches it: the episodes carrying it
    # get the tag signal (SourceConfig.w_tag). From the embedding profile.
    tag_match_score: float = 0.5

    @property
    def tz(self) -> tzinfo | None:
        """The user's timezone; None means the machine's local one."""
        return ZoneInfo(self.timezone) if self.timezone else None

    def __post_init__(self):
        if self.episode_gap <= timedelta(0):
            raise ConfigError("episode_gap must be positive.")
        if self.recent_window < 1:
            raise ConfigError("recent_window must be at least 1.")
        if not 1 <= self.summary_chunk <= self.recent_window:
            raise ConfigError("summary_chunk must be between 1 and recent_window.")
        if not 0 <= self.memory_share < 1:
            raise ConfigError("memory_share must be at least 0 and below 1.")
        if self.recent_episodes < 0:
            raise ConfigError("recent_episodes cannot be negative.")
        if self.short_reply_chars < 0:
            raise ConfigError("short_reply_chars cannot be negative.")
        if not 0 <= self.inhibition <= 1 or self.inhibition_pool < 1:
            raise ConfigError("inhibition must be between 0 and 1; inhibition_pool at least 1.")
        if self.rerank_pool < 1:
            raise ConfigError("rerank_pool must be at least 1.")
        kinds = {"message", "episode", "fact", "attachment"}
        if self.rerank_sources != "all" and not set(self.rerank_sources.split(",")) <= kinds:
            raise ConfigError(f"rerank_sources must be 'all' or kinds from {sorted(kinds)}, comma-separated.")
        if self.context_messages < 0:
            raise ConfigError("context_messages cannot be negative.")
        if not 0 <= self.context_weight <= 1:
            raise ConfigError("context_weight must be between 0 and 1.")
        if not 0 <= self.carry_decay < 1:
            raise ConfigError("carry_decay must be at least 0 and below 1.")
        if not (0 <= self.carry_gate <= 1 and 0 <= self.subject_similarity <= 1):
            raise ConfigError("carry_gate and subject_similarity must be between 0 and 1.")
        if not (0 <= self.activation_turn_decay <= 1 and 0 <= self.activation_spread <= 1):
            raise ConfigError("activation_turn_decay and activation_spread must be between 0 and 1.")
        if self.carryover_messages < 0 or not 0 <= self.pinned_share < 1:
            raise ConfigError("carryover_messages can't be negative; pinned_share must be at least 0 and below 1.")
        if not 0 <= self.drift_threshold <= 1 or self.drift_strikes < 1 or self.drift_window < 1:
            raise ConfigError("drift_threshold must be between 0 and 1, drift_strikes and drift_window at least 1.")
        if not 0 <= self.attachment_overlap < self.attachment_chunk_chars:
            raise ConfigError("attachment_overlap must be at least 0 and below attachment_chunk_chars.")
        if self.activation_half_life_hours <= 0:
            raise ConfigError("activation_half_life_hours must be positive.")
        if self.max_tool_calls < 0:
            raise ConfigError("max_tool_calls cannot be negative.")
        if not 0 < self.tool_result_share < 1:
            raise ConfigError("tool_result_share must be above 0 and below 1.")
        if not 0 <= self.search_min_score <= 1:
            raise ConfigError("search_min_score must be between 0 and 1.")
        if self.max_facts < 1:
            raise ConfigError("max_facts must be at least 1.")
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ConfigError(f"Unknown timezone {self.timezone!r}.") from exc


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One LLM's limits and capabilities: what the prompt must fit into.

    - `context_window`: tokens the model actually runs with (Ollama's num_ctx), which
      may be less than the model's maximum.
    - `output_reserve`: tokens kept free for the reply (Ollama's num_predict).
    - `memory_budget`: tokens for the current-episode slot plus retrieved memory. Set
      well below what fits: long prompts are slower and less reliably used.
    - `chars_per_token`: for estimating tokens without a tokenizer.
    - `supports_tools`: whether the model can call tools (gates deliberate recall).
    - `think`: whether a reasoning model reasons before answering. Off by default:
      reasoning multiplies latency and its tokens count against `output_reserve`.
      None leaves the provider's default (which, for reasoning models, is on).
    - `options`: generation settings passed to the provider (temperature...).
    - `base_url`: a remote server for this model (Ollama Cloud: https://ollama.com);
      empty means the local one the app is configured with. Evaluation only: the
      chat stays on your machine.
    """

    name: str
    provider: str
    context_window: int
    output_reserve: int
    memory_budget: int
    chars_per_token: float = 4.0
    supports_tools: bool = False
    think: bool | None = False
    options: Mapping[str, Any] = field(default_factory=dict)
    base_url: str = ""

    def __post_init__(self):
        for name in ("context_window", "output_reserve", "memory_budget"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{self.name}: {name} must be positive.")
        if self.chars_per_token <= 0:
            raise ConfigError(f"{self.name}: chars_per_token must be positive.")
        if self.memory_budget + self.output_reserve >= self.context_window:
            raise ConfigError(
                f"{self.name}: memory_budget ({self.memory_budget}) + output_reserve "
                f"({self.output_reserve}) must leave room in context_window "
                f"({self.context_window}) for the system prompt and the new message."
            )


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """One embedding model and how to use it.

    - `provider`: "ollama" or "sentence-transformers".
    - `model`: the provider's name for it; also stored with every vector, so vectors
      from different models are never compared.
    - `min_score`: the retrieval threshold calibrated for this model with
      `python -m evals --sweep`. Each model spreads similarities differently (unrelated
      text scores ~0.2 with one model and ~0.75 with another), so a threshold only
      means something for the model it was measured on.
    - `search_min_score`: the lower threshold for the search_memory tool (coverage
      over precision: the model asked). Defaults to `min_score` minus 0.08.
    - `fact_same_score`, `fact_related_score`: fact deduplication (brain/facts.py):
      above the first, two facts are the same; between the two, the model decides.
      Calibrated with `python -m evals.fact_pairs`.
    - `tag_same_score`: two tags this close in meaning are one topic.
    - `tag_match_score`: a tag this close to a message matches it (recall).
    - `max_tokens`: the longest input the model reads; longer text is truncated.
    - `prefix`: text some models expect before every input (e5: "query: ").
    """

    name: str
    provider: str
    model: str
    min_score: float
    max_tokens: int
    prefix: str = ""
    search_min_score: float | None = None
    fact_same_score: float = 0.95
    fact_related_score: float = 0.75
    tag_same_score: float = 0.85
    tag_match_score: float = 0.5

    def __post_init__(self):
        if self.search_min_score is None:
            object.__setattr__(self, "search_min_score", max(0.0, round(self.min_score - 0.08, 2)))
        if self.provider not in ("ollama", "sentence-transformers"):
            raise ConfigError(f"{self.name}: unknown provider {self.provider!r}.")
        if not 0 <= self.min_score <= 1:
            raise ConfigError(f"{self.name}: min_score must be between 0 and 1.")
        if self.max_tokens <= 0:
            raise ConfigError(f"{self.name}: max_tokens must be positive.")


def with_thinking(profile: ModelProfile) -> ModelProfile:
    """The profile with the model's thinking on, and room for it: reasoning tokens count
    against the reply, so the reserve is raised (to 4,096) as far as the context allows.
    The chat page's Think switch uses it for one turn (off by default: on LoCoMo,
    gemma4:e4b with thinking accepted false premises more often, 2026-11-07)."""
    room = profile.context_window - profile.memory_budget - 1024
    return replace(profile, think=True, output_reserve=max(profile.output_reserve, min(4096, room)))


def with_embedding(config: EngineConfig, profile: EmbeddingProfile) -> EngineConfig:
    """The config with every threshold that depends on the embedding model taken
    from its profile: recall, the search tool, fact deduplication, and tags."""
    assert profile.search_min_score is not None
    return replace(
        with_min_score(config, profile.min_score, search=profile.search_min_score),
        fact_same_score=profile.fact_same_score,
        fact_related_score=profile.fact_related_score,
        tag_same_score=profile.tag_same_score,
        tag_match_score=profile.tag_match_score,
    )


def with_min_score(
    config: EngineConfig, threshold: float, search: float | None = None
) -> EngineConfig:
    """The same config with one `min_score` for every memory source (and, given,
    the search tool's threshold)."""
    return replace(
        config,
        messages=replace(config.messages, min_score=threshold),
        episodes=replace(config.episodes, min_score=threshold),
        facts=replace(config.facts, min_score=threshold),
        attachments=replace(config.attachments, min_score=threshold),
        search_min_score=config.search_min_score if search is None else search,
    )


SOURCES = ("messages", "episodes", "facts", "attachments")


def all_sources(config: EngineConfig) -> tuple[SourceConfig, ...]:
    return (config.messages, config.episodes, config.facts, config.attachments)


def load_engine_config(path: str | Path) -> EngineConfig:
    data = _read_toml(path)
    if "episode_gap_hours" in data:
        data["episode_gap"] = timedelta(hours=data.pop("episode_gap_hours"))
    for key in ("search_min_score", "fact_same_score", "fact_related_score", "tag_same_score",
                "tag_match_score"):
        if key in data:
            raise ConfigError(
                f"{path}: {key} depends on the embedding model; "
                "set it in its profile under configs/embeddings/."
            )
    for source in SOURCES:
        if "min_score" in data.get(source, {}):
            # It would be silently replaced by the embedding profile's threshold.
            raise ConfigError(
                f"{path} [{source}]: min_score depends on the embedding model; "
                "set it in its profile under configs/embeddings/."
            )
        if source in data:
            # Start from the field's default so a table only overrides what it names.
            default = EngineConfig.__dataclass_fields__[source].default_factory()  # type: ignore[misc]
            overrides = _build(SourceConfig, data[source], f"{path} [{source}]")
            data[source] = _merge(default, overrides, data[source])
    if "entity_block" in data:
        data["entity_block"] = _build(EntityBlockConfig, data["entity_block"], f"{path} [entity_block]")
    return _build(EngineConfig, data, path)


def _merge(default: SourceConfig, overrides: SourceConfig, named: dict[str, Any]) -> SourceConfig:
    return SourceConfig(**{
        f.name: getattr(overrides if f.name in named else default, f.name)
        for f in fields(SourceConfig)
    })


def load_model_profile(path: str | Path) -> ModelProfile:
    return _build(ModelProfile, _read_toml(path), path)


def load_embedding_profile(path: str | Path) -> EmbeddingProfile:
    return _build(EmbeddingProfile, _read_toml(path), path)


def _read_toml(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc


def _build(cls, data: dict[str, Any], path: str | Path):
    # Unknown keys are almost always typos; fail loudly instead of ignoring them.
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:  # a required key is missing
        raise ConfigError(f"{path}: {exc}") from exc


def attachment_chunk_size(config: EngineConfig, profile: EmbeddingProfile) -> tuple[int, int]:
    """Passage size and overlap in characters: the config's, but never more than the
    embedding model reads (about 3 characters per token, to be safe), since text past
    its limit is silently dropped from the vector."""
    limit = profile.max_tokens * 3
    size = min(config.attachment_chunk_chars, limit)
    return size, min(config.attachment_overlap, size // 4)
