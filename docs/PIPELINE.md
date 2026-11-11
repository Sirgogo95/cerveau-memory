# The message pipeline

What happens, step by step, from the moment a user message arrives until the reply is stored, plus the background consolidation that runs when an episode closes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the architecture, [DJANGO_SETUP.md](DJANGO_SETUP.md) for how it maps onto Django, [EMBEDDINGS.md](EMBEDDINGS.md) for how vectors are stored, and [SEARCH_MEMORY_TOOL.md](SEARCH_MEMORY_TOOL.md) for deliberate recall on top of this pipeline.

**Phases.** Every step and sub-step is labeled with the roadmap phase that builds it (**v1**, **v2**, **v3**). v1 is the time-gap, embeddings-and-lexical pipeline; anything labeled later is designed here but must not creep into v1.

---

## Overview

| # | Step | Phase | LLM calls |
|---|---|---|---|
| 1 | Message arrives, embed it | v1 | None |
| 2 | Time-gap check | v1 | None |
| 3 | Drift check | v2 | Rarely one (gray zone) |
| 4 | Gather signals | v1 | None |
| 5 | Update activation | v2 | None |
| 6 | Retrieve and score candidates | v1, extended in v2 | None |
| 7 | Thresholds and the confidence gate | v1 (gate: v2) | None |
| 8 | Fill the token budget | v1 | None |
| 9 | Assemble the prompt, call the LLM | v1 | The reply |
| 10 | After the reply | v1, extended in v2 | None inline |
| C | Consolidation (background, on episode close) | v1, extended in v2 | 1 or 2 with the cheap model |

**The rule behind the design:** the per-message path stays fast and nearly LLM-free; the expensive thinking happens in consolidation, off the clock.

### Running example

The user is in an episode about Django settings. It has been 20 minutes since their last message, and they write:

> "Back to the bookshop, I want to add an orders table."

---

## 1. Message arrives, embed it (v1)

**Goal:** turn the text into a vector, as cheaply as possible.

1. Normalize the text (trim whitespace).
2. Compute `text_hash(text)` (`brain/hashing.py`: sha256 of the trimmed text).
3. Look up `(model, text_hash)` in the embedding cache. On a miss, call the `Embedder` and store the result. The model is part of the lookup, not of the hash (see [EMBEDDINGS.md section 7](EMBEDDINGS.md#7-relation-to-the-embedding-cache)).

**Output:** the text, its `Embedding` (vector, model, hash), and a timestamp. The message is not saved yet, because its episode is decided in steps 2 and 3.

**Cost:** about 30 ms with bge-m3 on Ollama (GPU). No LLM.

## 2. Time-gap check (v1)

**Goal:** a long silence means a new episode, regardless of topic.

```python
gap = now - episode.last_message_at
if gap > config.episode_gap:  # default 2 hours
    close_episode(episode)
    episode = open_episode(started_at=now)
```

`close_episode`:

1. Sets `ended_at` to the last message's timestamp. **(v1)**
2. Queues consolidation (see [Consolidation](#consolidation)). **(v1)**
3. Marks the last ~3 messages for **carry-over**: they stay in the prompt for the next couple of turns, decaying out, to protect against false splits. **(v2)**

**As built (carry-over, 2026-10-18):** nothing is marked; `carried_over` (GUIDE 4.7) finds the previous episode's last messages when the prompt is built: 3 on the new episode's first turn, then 2, then 1, and only when the episode was split off by subject (it began within `episode_gap` of the last one's end).

Activation is **not** reset. Time-based decay has already faded it naturally. **(v2)**

`last_message_at` is derived from the episode's messages (their latest `created_at`), not stored as a column that every message rewrites; see [DJANGO_SETUP.md section 3](DJANGO_SETUP.md#keep-high-churn-fields-out-of-the-audit-trail).

**Example:** 20 minutes is under 2 hours, so continue to step 3.

## 3. Drift check (v2)

**Goal:** detect a topic change within a continuous conversation. In v1, only the time gap splits episodes.

1. **Skip** if the message has fewer than ~5 words, or the episode has fewer than ~3 messages.
2. Compute the **recent centroid**: the average vector of the last ~5 messages. Comparing against the whole episode would split too often, since conversations naturally wander.
3. Compare: `sim = cosine(message_vector, centroid)`.

| Similarity | Action |
|---|---|
| Above the soft threshold (e.g. 0.5) | Same topic: reset strikes to 0 |
| Between hard and soft (e.g. 0.35 to 0.5) | Gray zone: ask the cheap model "is this a new topic?" |
| Below the hard threshold (e.g. 0.35) | Strike |

The episode tracks `drift_strikes` and `drift_pending_message_id`:

- **First strike:** the message stays in the current episode, marked as pending.
- **Second consecutive strike:** close the episode and open a new one, moving the pending message into it along with the current one. The assistant reply to the pending message moves with it, so the new episode starts with a complete exchange. This needs a store method to move messages between episodes.

**Example:** Django settings vs bookshop orders scores 0.28, so strike one. The message is saved to the current episode as pending. If the next message is also about the bookshop, the split happens and both exchanges move to the new episode.

**As built (2026-10-18):** [brain/segmentation.py](../brain/segmentation.py) `follow_subject`, after the message is saved. The centroid is the last **2** user messages (not 5), the threshold 0.48 for bge-m3, two strikes; no gray-zone model call. Measured with `python -m evals.segments`: 8 of 9 topic changes found, no split inside a topic (GUIDE 4.7).

**Pitfall:** threshold values depend entirely on the embedding model. Calibrate them on real conversations, and again after switching models.

After this step (or after step 2 in v1), the message is **saved** with its `episode_id`.

## 4. Gather signals (v1)

**Goal:** compute once everything retrieval needs to know about this message. No LLM.

- **Query vector:** from step 1.
- **Entity matches:** trigram and alias search over entity names, with `unaccent` for accents. "the bookshop" matches the entity *Harbor Books* through its alias.
- **Tag matches:** tag names found in the text, plus tags whose embedding is close to the message vector (e.g. similarity above 0.6). "orders table" matches `databases` and `retail`.
- **Keywords:** for full-text search with Postgres `tsvector` / `ts_rank`.

```python
@dataclass
class Signals:
    query: Embedding
    entities: dict[int, float]   # entity_id -> match strength
    tags: dict[int, float]       # tag_id -> match strength
    keywords: list[str]
```

Every later step reads from this object instead of recomputing.

**As built (entity matches):** the message is split into words (lowercase, no accents or punctuation) and every run of 1 to 4 words becomes a key; entities whose **search keys** contain one are matched (`match_entities`). Search keys are an entity's name and aliases plus, for people only, each name part, so "Marta" finds *Dr. Marta Peña*; titles and parts under 3 letters don't count, and places never match by a part ("domingo" is not *Santo Domingo*). Merging at consolidation uses separate **identity keys** (name and aliases only), so *Marta Gómez* never merges into *Marta Peña*. Typos are matched too ([brain/names.py](../brain/names.py), 2026-10-17): a capitalized word one letter off a name (two for long names) finds it, in pure Python rather than trigrams. Tag matches and keywords are not built yet.

**As built (keywords, 2026-10-18):** a message's exact terms (identifiers like `INV-2026-00417`, words in quotes) are looked up literally, without Postgres full-text search: plain `icontains` is enough for terms this specific, and works in every language ([brain/keywords.py](../brain/keywords.py), GUIDE 4.19).

**As built (activation, 2026-10-18):** step 5 exists ([brain/activation.py](../brain/activation.py), GUIDE 4.18), on at `w_act = 0.05`: edges from shared episodes, fading per turn and over hours, a boost for matched nodes spreading one hop with the fan effect, a fresh start on a new subject. Measured, it adds a little over the conversation-aware search and carry-over below, at the cost of some noise.

**As built (the conversation, 2026-10-18):** the query vector of steps 6 and 7 is the new message plus the user's previous messages in the episode, each weaker the further back (`context_weight` 0.25), so follow-ups keep their subject ([brain/recall.py](../brain/recall.py), GUIDE 4.17). Names and periods still come from the new message alone. What the previous turns recalled is carried into the next ones, fading, while the conversation stays on that subject (carry-over, GUIDE 4.17); PIPELINE's v3 "warm cache" is this, built early.

**As built (tag matches, 2026-10-18):** tags whose embedding is close to the message vector (0.50 for bge-m3, at most 3) are matched (`matched_tags`); tag names found in the text are not used. The episodes carrying them get a small boost in step 7 (`w_tag` 0.05), measured on the retrieval eval: better hit rate, precision, and noise than without it (GUIDE 4.8).

**As built (time expressions, 2026-10-17):** a period named in the message ("in August", "last week", "la semana pasada", "le mois dernier") becomes a date window ([brain/periods.py](../brain/periods.py)). It filters steps 6 and 7 (only memory from then, none dropped for its score) and the recent conversations, and the memory block notes what's from then, or that nothing is. A month still ahead this year, without a year, is treated as a plan, not a period.

## 5. Update activation (v2)

**Goal:** decide what is "on your mind" after this message.

1. **Load the subgraph:** nodes that currently have activation, nodes matched in step 4, and their neighbors one or two hops out.
2. **Time decay**, applied lazily from `updated_at`: `value × 0.5^(hours / half_life)`.
3. **Turn decay:** `value × (1 - δ)`.
4. **Boost** matched nodes: `value += boost × match_strength`.
5. **Spread**, 2 or 3 iterations: each node sends `S × weight × its_energy ÷ its_fan` to each neighbor. Dividing by fan (number of connections) stops broad hubs from flooding the graph.
6. **Inhibition** (optional): the strongest nodes suppress weaker competitors.
7. **Squash** with a sigmoid into the 0 to 1 range.

**Squash does not remove anything.** It reshapes values along an S-curve, pushing low values toward 0 and high values toward 1, like a soft on/off switch. Every node still exists afterwards.

**Activation and embedding similarity are separate signals.** Activation lives on graph nodes (tags, entities) and reflects what has been on your mind across turns. Embedding similarity compares the current message to stored memories right now. They only meet in step 6, where both contribute to each candidate's score.

**Example:**

| Node | Before | After |
|---|---|---|
| tag `django` | 0.85 | 0.52 (decayed, not mentioned) |
| entity `Harbor Books` | 0.10 | 0.93 (boosted) |
| tag `retail` | 0.05 | 0.61 (boosted + spread from Harbor Books) |
| tag `databases` | 0.30 | 0.74 (boosted) |

Results stay in memory; they are saved in step 10.

## 6. Retrieve and score candidates (v1, extended in v2)

**Goal:** find everything that might be relevant, then rank it.

Sources: episodes, messages, entities, and tags in **v1**; facts join in **v2**, skills in **v3**. For each source, gather candidates, then merge and deduplicate:

1. **Vector search (v1):** top-k nearest by embedding, active rows only, live embedding model only. Episodes are compared on both summaries, keeping the better match.
2. **Lexical hits (v1):** rows mentioning matched entities or keywords.
3. **Graph (v2):** episodes and facts linked to highly active tags and entities.
4. **Warm cache (v3):** memories loaded in recent turns.

Hits already in the prompt as part of the current episode are skipped (each `Candidate` carries its `episode_id`).

**As built (names):** when the message names entities, `mention_candidates` adds the messages and episode summaries that mention them, with a lexical signal equal to the share of the named entities each one mentions. A memory found both ways carries both signals; one found only by name can pass on that alone (with the defaults, `w_lexical = 0.4` clears `min_score = 0.35`). The named entities themselves are added as short cards (`[N7] Dr. Marta Peña (person; also: my colleague)`). In the live check, the message about Dr. Peña scored 0.34 on meaning alone, just under the threshold, and 0.74 with the name.

Score each candidate:

```
score = w_sem × semantic_similarity
      + w_lex × lexical_match           # share of the named entities it mentions
      + w_act × activation of its tags and entities (e.g. the max)    # v2
```

Weights come from `SourceConfig` and differ per source. In v1, `w_act` is 0.

**Example** (v2, with activation):

| Candidate | Semantic | Lexical | Activation | Score |
|---|---|---|---|---|
| Episode: bookshop customer schema | 0.78 | 1.0 | 0.93 | 0.86 |
| Fact: the bookshop uses Django | 0.55 | 1.0 | 0.93 | 0.75 |
| Fact: the bookshop is trilingual | 0.40 | 1.0 | 0.93 | 0.68 |
| Episode: Django settings (current) | n/a | n/a | n/a | always included |
| Episode: weekend hiking plans | 0.10 | 0 | 0.02 | 0.06 |

## 7. Thresholds and the confidence gate (v1, gate in v2)

**Goal:** decide which candidates are eligible at all.

1. **`min_score` per source (v1)** drops weak matches. With `min_score = 0.3`, the hiking episode (0.06) is dropped.
2. **`max_items` per source (v1)** caps how many survive, so one source can't flood the budget.
**As built (2026-10-18):** no gate yet, because nothing needed it: questions about things never said already get "I don't remember" (8 of 8 in the tool eval), since nothing is recalled and the system prompt says so. What failed were near misses, where the subject is stored but not the detail asked for ("who is Pedro marrying?"): the right memory comes back and the model invents the rest. A rule in the system prompt fixed that (GUIDE 9.2). The gate below stays planned for when an eval shows weak matches being trusted.

3. **The confidence gate (v2)** looks at the best candidate overall. If even the best is below the gate (e.g. 0.12), the system concludes it has no relevant memory and says so explicitly in the prompt ("No stored memory matches this question"). When the user asks "what did I decide about X?" and nothing was ever said, the model answers that it has no record instead of inventing one.

The gate matters for recall-style questions. For ordinary chat ("explain HNSW"), having no relevant memory is normal, and the model simply answers.

**Example:** the best candidate scores 0.86, so the gate passes.

## 8. Fill the token budget (v1)

**Goal:** fit the best memories into the model's budget (`ModelProfile.memory_budget`).

1. **Reserve the current-episode slot first:** pinned messages (v2), the running summary, then the recent window of messages, verbatim. The current conversation is always present.
2. **Split the remainder** by source shares (e.g. facts 30%, episodes 50%, entities 10%, skills 10%).
3. **Fill each share greedily** by score until its tokens run out.
4. **Short or long summary per episode:** long if the score is high (e.g. above 0.75) and it fits, otherwise short.
5. **Redistribute leftovers:** unused tokens from one source go to the next source with candidates waiting.
6. **Skills (v3):** description by default; the full skill file only above a high threshold.

**The recent window moves in chunks, not one message at a time.** When the window is full, the oldest ~10 messages are folded into the running summary at once (step 10.5), instead of dropping one message per turn. The start of the prompt then stays identical for about ten turns, which is what makes prefix caching work (step 9).

```python
def fill(candidates, budget):
    chosen, used = [], 0
    for item in sorted(candidates, key=lambda c: c.score, reverse=True):
        if used + item.tokens <= budget:
            chosen.append(item)
            used += item.tokens
    return chosen
```

Tokens are counted with the provider's tokenizer when available, otherwise approximated as characters ÷ 4, with a ~10% safety margin.

**Example:** the bookshop schema episode (0.86) gets its long summary; both facts fit; the Harbor Books entity's one-line description fits.

## 9. Assemble the prompt and call the LLM (v1)

Order: everything that stays the same from turn to turn first, what changes every turn last.

```
[System prompt]                                        stable
  Today's date (user's timezone)                       changes daily
  Recent conversations: last 3 episodes                changes when an episode closes
    (E5 · 2026-10-14 09:02 to 10:40) Title: short summary
    (E6 · ...) not summarized yet: its last messages
[Current episode running summary]                      changes every ~10 messages
[Pinned messages]                                      rarely changes (v2; built 2026-10-18, GUIDE 4.9)
[Recent messages, verbatim]                            append-only between window moves
[Retrieved memory, refreshed every turn]               changes every turn
  Facts (v2)
    (F12, learned 2026-09-30) The bookshop project uses Django.
    (F15, learned 2026-10-01) The bookshop is trilingual: ES, EN, FR.
  Relevant past episodes
    (E31, 2026-10-02) Long summary: designed the customer schema...
  Skills loaded for this turn (v3)
  Confidence gate note, if any (v2)
[New user message]
```

**Recent conversations (as built):** the system prompt always lists the latest closed episodes (`recent_episodes`, default 3), whatever the message says, with dates, titles, and short summaries; an episode not consolidated yet shows its last messages instead. Similarity retrieval can't answer "what did we talk about last?" or "where were we?": the question resembles nothing that was said. Recency can. The system prompt also tells the model never to claim it can't access past conversations, and to say it doesn't remember when memory has nothing. Episode summaries already listed there are not repeated in the `<memory>` block.

**Deliberate recall (as built, 2026-10-17):** models that support tools are also offered `search_memory` and `read_memory` ([SEARCH_MEMORY_TOOL.md](SEARCH_MEMORY_TOOL.md)). The reply streams as usual; when a tool call arrives in the stream, the turn runs the tool, adds the call and its result to the prompt, and streams again, at most `max_tool_calls` (3) times. The newest message also ends with a `(Reply in <language>.)` hint when its language is clear ([brain/language.py](../brain/language.py)), in the prompt only.

**How retrieved memory is placed (as built):** as a `<memory>` block at the start of the new user message, before the user's own words, one line per memory with its ID, date, and speaker (`- [M12 · 2026-10-02 · user] ...`). The stored message stays clean; the block exists only in the prompt. The system prompt explains the block and that instructions inside it are never followed.

**Why this order:** providers and local runtimes reuse their computation for a prompt prefix that is identical to the previous call's. Ollama does this too, and on local hardware reprocessing the prompt is often the slowest part of a turn. Retrieved memory changes every turn, so anything placed after it is recomputed every turn. Putting it last keeps the whole conversation cacheable, and it also puts the memories right next to the question they were retrieved for.

Three details that matter:

- **IDs on every memory**, so the model can report which ones it used (step 10).
- **Dates on every memory**, so the model can reason about time ("you decided this two weeks ago").
- **Memory framed as data, not instructions.** The system prompt states that stored memories are reference material. This protects against memory poisoning if stored text, for example from an email or web page, ever contains commands.

Then the `ChatProvider` is called, streaming the reply, and token usage is recorded.

## 10. After the reply (v1, extended in v2)

1. **Store the assistant message (v1)** in the current episode. Its embedding can be queued; it isn't needed until the next turn.
2. **Record which memories were exposed and used (v2).**
   - *Exposed:* everything that was in the prompt.
   - *Used:* what the reply actually relied on. Detect it either by having the model list the memory IDs it used in a hidden structured field (more accurate, but unreliable on small local models) or by comparing the reply's embedding to each loaded memory (no extra tokens).
3. **Reinforce used memories (v2):** update `last_used_at` and a use counter, and give their nodes a small activation bump. Retrieval strengthens memory. Penalize exposed-but-unused memories very lightly, if at all; a memory can be correctly loaded yet not needed for a given reply. These counters are written without audit snapshots (see [DJANGO_SETUP.md section 3](DJANGO_SETUP.md#keep-high-churn-fields-out-of-the-audit-trail)).
4. **Persist activation (v2):** write changed values back in one batch upsert and delete rows below ~0.01. This is what lets working memory survive a restart.
5. **Move the window (v1):** when the recent window is full, queue a job for the cheap model with the previous running summary plus the ~10 oldest messages in the window. Store the result on the episode along with the ID of the last message it covers; those messages then leave the window together.

   **As built** ([brain/summary.py](../brain/summary.py)): after the reply is stored, `run_turn` calls `on_summary_due(episode_id)` when more than `recent_window` (20) messages aren't covered by the summary; the app queues `update_running_summary` on the worker. The job folds the oldest `summary_chunk` (10) uncovered messages per model call, repeating until at most 20 remain, saving after each batch. The summary is capped at about 250 words (older details compressed), keeps the user's own words for preferences and decisions, follows the same attribution rules as consolidation, and is written in the conversation's language (it isn't embedded). It appears in the system message as a `<this_conversation_so_far>` block, and the prompt shows only the messages after it. Without the worker, nothing breaks: uncovered messages stay in the prompt until the token budget trims the oldest. Verified live 2026-10-16: a fact from the first message of a 26-message episode was answered from the summary alone.

---

## Consolidation

### Queuing (v1)

"Queue" means: don't do the work now; record it as a job and let a separate worker process run it.

```python
from django.tasks import task

@task
def consolidate_episode(episode_id: int):
    ...

consolidate_episode.enqueue(episode_id=42)
```

Django 6 ships this task API, but its built-in backends only run tasks immediately or discard them. **As built:** the `django-tasks-db` backend stores jobs in Postgres, and `python manage.py db_worker` runs them in a separate process ([apps/cerveau/tasks.py](../apps/cerveau/tasks.py)).

When jobs are queued:

- **When a message closes an episode** (the time gap), through `run_turn`'s `on_episode_closed` callback. The engine only announces the closing; the app decides to queue.
- **At web-server start, a sweep** closes the open episode if it has gone stale, then queues every closed episode without `consolidated_at`. Yesterday's conversation is then consolidated before the first message of the day, not while you're chatting.
- **By hand:** `python manage.py consolidate` (runs in place, no worker needed; `--episode`, `--force`, `--queue`).

Required properties:

- **Non-blocking:** consolidation can take 10 to 30 seconds; the user never waits for it.
- **Retries:** if the model times out or replies with unusable JSON, the episode stays without `consolidated_at`, so the next sweep queues it again.
- **Idempotent:** running it twice must not duplicate anything. It sets `episode.consolidated_at` at the end and skips episodes where it is already set; fact deduplication also protects against duplicates.

### The job

Uses the cheap background model, usually one structured-JSON call ([brain/consolidation.py](../brain/consolidation.py)):

1. **Summaries (v1):** short (1 or 2 sentences) and long.
2. **Tags (v1):** the model proposes names, seeing the most used existing tags so it reuses them; each is matched against existing tags by normalized name (embedding similarity for near-duplicates comes later), then upserted. `I` tags reactivate; `Z` tags are skipped.
3. **Entities (v1):** extracted with type and aliases, matched against existing entities by normalized name and aliases (accents and case ignored), upserted; `entity_mentions` rows created for every message that names the entity (whole-word match).
4. **Facts (v2):** extracted with `source` and `confidence`, deduplicated against active facts (confirm, update, supersede, or insert) and against `Z` facts (matches are dropped).
   **As built (2026-10-17)** ([brain/facts.py](../brain/facts.py), [FACTS.md](FACTS.md)): a separate, best-effort model call on the numbered transcript; only facts citing user lines are kept; 0.97 and above confirms, 0.55 to 0.97 asks the model (same, update, contradiction, distinct), an update or contradiction supersedes. Facts join retrieval as a third source (current ones only).
5. **Edge weights (v2):** co-occurrence counts incremented for every tag pair on the episode; Jaccard weights recomputed.
6. **Episode links (v2):** connections to past episodes with similar summaries.
7. **Embeddings (v1):** both summaries embedded into `EpisodeEmbedding`, each with its own text hash for staleness checks ([EMBEDDINGS.md](EMBEDDINGS.md)).
8. **Done (v1):** `consolidated_at` is set.

**As built, details that matter:**

- **Structured output:** the extraction call passes a JSON Schema; Ollama constrains the reply to it, which a 4B model needs to answer reliably. Replies are still validated and cleaned (tags lowercased and capped at 5, entity types checked, duplicates merged).
- **Attribution:** the prompt separates `User:` lines from `Cerveau:` lines and forbids presenting Cerveau's suggestions as the user's decisions. Cerveau's replies are cut to 600 characters in the transcript (the user's to 2,000), because left whole they dominate and get misattributed. And an entity is kept only if a **user** message names it: enforced in code, since the model ignored the instruction in testing.
- **Long episodes:** if the transcript doesn't fit the background profile's `memory_budget`, consecutive parts are first condensed into notes (one call each), then the extraction runs on the notes.
- **Language:** the title and summaries are written in the conversation's language (the user's; the one used most if mixed), tags in English. The extraction schema asks for a `language` field first, so the model names the language before writing (as built, 2026-10-17).

---

## Where removal happens

Nothing is removed during activation. Removal happens in three places:

| What gets removed | Where | Why |
|---|---|---|
| Weak candidates from this prompt | Step 7 thresholds | Not relevant enough right now |
| Activation rows | Step 10 pruning (below ~0.01) | Fully faded from mind |
| Cached memories | Cache eviction, driven by decay | Not likely to be needed soon |

## Where the time goes

| Path | LLM calls | Target time |
|---|---|---|
| Steps 1 to 8 | None (rarely one gray-zone drift check) | Well under a second |
| Step 9 | The reply itself | Depends on the model; prefix caching keeps prompt processing short |
| Step 10 | None inline | Milliseconds |
| Consolidation | 1 or 2 with the cheap model | Seconds, in the background |

All thresholds and weights above are starting values, to be tuned against the personal eval set and LoCoMo.
