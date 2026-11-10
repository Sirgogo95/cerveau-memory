# Facts (v2): the plan, and the decisions to start from

**Status:** built (2026-10-17)
**Builds on:** [README: Facts](../README.md#facts) (the design), [PIPELINE.md](PIPELINE.md) step 6 and consolidation step 4, [SEARCH_MEMORY_TOOL.md](SEARCH_MEMORY_TOOL.md) (`fact` kind, `include_history`)

This note records what was recommended on 2026-10-17, before building, so the work starts from decisions already made. The README remains the design; this note adds what was learned since (bge-m3, the 4B model's attribution limits, the memory viewer and tools).

---

## As built so far (2026-10-17)

- **Data**: `Fact` and `FactStore` in brain/, both stores, Django `Fact` (audited; its source messages; superseded facts kept), `FactEmbedding`.
- **Learning** ([brain/facts.py](../brain/facts.py)): a separate call at consolidation, on the background profile (same qwen at temperature 0.2: at 0.7 the judge answered the same pair differently). The transcript is numbered; each fact cites the user lines that state it, and code drops facts citing none. Facts are written as they are now ("Pedro's wedding is on March 3"), not as events.
- **Deduplication thresholds** (decision 2), measured with `python -m evals.fact_pairs`: rewordings score 0.92 to 0.99, but an update can reach 0.96 and a loosely worded one go down to 0.60. So `fact_same_score = 0.97` (confirmed without asking) and `fact_related_score = 0.55` (the model judges from there). In the bge-m3 profile.
- **The judge prompt**, measured with `--judge` on 47 labelled pairs: asking in order (same? update? contradiction? else distinct) with one example each got 46 right. A first rule ("if both can be true at once, it's distinct") made it call rewordings distinct, since those are both true too.
- **Only facts the user stated** (decision 3): `source` is always `user_stated`.
- **Backfill**: `python manage.py extract_facts`, once per episode (`Episode.facts_learned_at`).

- **In use**: a third recall source (`[F45 · date · fact]` in the prompt, current facts only), the search tool's `fact` kind and `include_history`, `read_memory` on `F` IDs, and the viewer's Facts tab with each fact's sources and history.
- **Verified end to end** on the eval database: a later conversation moved Pedro's wedding; consolidation superseded the old fact; "when is Pedro's wedding?" answered March 3 in 4 of 4 runs, noting the old date.

Known limits: non-durable facts sometimes slip in, some facts are phrased as events, a judge miss can leave a near-duplicate, and the same fact in two languages isn't always merged.

## What a fact is

A short, standalone statement that passes the README's three tests:

1. **Durable**: still true beyond this conversation.
2. **Self-contained**: understandable read cold, without the conversation.
3. **Useful**: knowing it would change a future answer.

Examples: "The user's sister Lucía lives in Montreal." "The user's favorite café is La Cafetera on Calle El Conde." Not facts: questions, hypotheticals, options considered but not chosen, Cerveau's own suggestions unless the user accepted them.

Why facts at all: retrieval is weakest on *updates*. When the user moves, changes a plan, or reverses a decision, both the old and the new conversation come back, and the model has to guess which is current. A fact that is replaced by its successor (and kept as history) answers "what's true now" directly.

## The plan

1. **Extraction at consolidation.** Each fact carries its `statement`, `type`, `source`, `confidence`, and the messages it came from (`source_message_ids`), so it can always be checked against the original words.
2. **Deduplication against existing facts**, by embedding similarity:
   - very close: the same fact, so `times_confirmed` goes up;
   - close: ask the model whether it's a duplicate, an update, a contradiction, or distinct;
   - otherwise: a new fact.
3. **Never deleted.** An outdated fact is marked `superseded_by` its replacement: history stays, and mistakes can be traced.
4. **Status-aware, like tags and entities.** A fact the user forgot (Z) is never re-learned: new extractions matching it are dropped.
5. **Retrieval:** a third memory source next to messages and episodes, with its own `SourceConfig` and share of the budget, shown in the prompt as `[F12 · date] statement`. Only facts that aren't superseded are recalled.
6. **The search tool:** `kinds: ["fact"]`, and `include_history: true` to see how a fact changed (superseded facts marked as outdated).
7. **The memory viewer:** a Facts tab, each fact with its sources (linked messages) and its history chain, plus Forget / Set aside / Restore.
8. **A backfill command** to extract facts from conversations that are already consolidated.

## Decisions (recommended 2026-10-17)

### 1. A separate model call for facts

Not the same JSON as the title, summaries, tags, and entities. Adding facts to that one schema makes a 4B model do more at once, and the extraction prompt already needs careful attribution rules. A second call doubles consolidation time, but consolidation runs on the worker, in the background, so nobody waits for it. Expected to be more accurate; measured before deciding otherwise.

### 2. Calibrate the deduplication thresholds for bge-m3

The README's 0.92 (same fact) and 0.75 (ask the model) were set before bge-m3, whose similarities spread differently. Calibrate them the way the retrieval threshold was chosen: a small eval of fact pairs labelled *same*, *update*, *contradiction*, and *unrelated*, in the three languages, then a sweep. Store the result in the embedding profile (`configs/embeddings/`), since, like `min_score`, a similarity threshold only means something for the model it was measured on.

### 3. Only facts the user stated, at first

Keep `source = user_stated` only; leave `inferred` facts out of v2's first version. The 4B model already struggles with attribution (it once credited the user with Cerveau's suggestion; the entity check exists because of it), and an inferred fact presented as memory is worse than no fact. The same guard as entities applies: a fact must be supported by a user message, checked in code against `source_message_ids`. Inferred facts, with lower confidence, can come later if missing ones are noticed.

## Open questions for the build

- **Fact types:** a short fixed list (`person`, `preference`, `plan`, `project_state`, `health`, `place`, `other`) or free text? A fixed list helps the viewer's filters and the model's consistency.
- **Budget share:** how much of the memory budget facts get versus messages and episodes; start with the README's example (facts 30%) and measure with the eval set extended with fact cases.
- **Reinforcement:** `last_used_at` and `times_confirmed` are in the schema; using them for ranking belongs with activation (v2), not the first version.
