# Feature: entity blocks and entity profiles

**Status:** step 1 built and measured: on all 282 multi-hop questions, graded by deepseek-v4-pro (2026-10-22), +8 points with qwen3.5:4b (p = 0.0003) and +7 with gpt-oss:120b (p = 0.010), the whole answer in the notes about 20 more times; nothing else drops; **on by default since 2026-10-22**. A larger block (20 facts) and a second search among the entity's facts were tried for the partial notes that remain and kept off (no more whole answers in the notes); what worked was 16 facts per session at consolidation together with blocks of 20 (multi-hop 53% to 59%, p = 0.04, with qwen3.5:4b), the defaults since 2026-10-22. Step 2 not needed so far. Two additions from measuring: aliases that belong to another entity are dropped, and a block is ranked half on the question without the entity's name (CERVEAU_LOCOMO_RESULTS.md, section 6.3, item 17)
**Target:** multi-hop questions, the open gap (40 to 44% with every model and memory combination, against 68% with the whole conversation in the prompt)
**Related:** [CERVEAU_LOCOMO_RESULTS.md](../CERVEAU_LOCOMO_RESULTS.md) (section 6.3, item 15), [LOCOMO_FINAL_RUNS.md](../LOCOMO_FINAL_RUNS.md), [FACTS.md](../FACTS.md)

---

## 1. Summary

When a question names a person, place, or project, Cerveau brings a **block of what it knows about that entity**: the facts about it, ranked by the question, in date order, under its card. Optionally, consolidation also keeps a short **written profile** per entity.

It's built in two steps, and the second only if the first isn't enough:

| Step | What | Model calls | Re-consolidation |
|---|---|---|---|
| **1. Entity block** | Link facts to the entities they're about; when a question names an entity, search only its facts and show them as a block | None | None |
| **2. Written profile** | Consolidation writes and updates a short profile per entity from its facts | One per entity that gained facts | Yes |

Brain terms: step 1 is remembering what you know about someone by thinking of them; step 2 is the impression of a person built from many encounters.

## 2. Why

Multi-hop questions need several pieces of evidence from different sessions together. Many of LoCoMo's are about one person across sessions ("What activities does Melanie do?", "Where has Caroline traveled?").

The runs showed:

- The reranker finds **one** piece of evidence for 76% of multi-hop questions.
- But only about **42% of all** the pieces reach the notes (turn recall).
- Ordinary recall ranks every memory against the question and keeps the top few per source. The pieces about one person compete with everything else that resembles the question, and most lose.

Searching only the entity's facts removes that competition: the pieces about Melanie compete only with each other.

## 3. Step 1: the entity block

### 3.1 Linking facts to entities

A new join table records which entities a fact is about:

```python
class FactEntity(models.Model):
    fact = models.ForeignKey("Fact", on_delete=models.CASCADE, related_name="entity_links")
    entity = models.ForeignKey("Entity", on_delete=models.CASCADE, related_name="fact_links")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["fact", "entity"], name="uniq_fact_entity")
        ]
```

Plain `models.Model`: derived, high churn, never audited.

**Why not the link that already exists.** Facts know the messages they were learned from (`FactSource`), and messages know which entities they mention (`EntityMention`); recall already finds "facts learned from messages that mention X" through that path. But a message's mentions say who is *named*, not who the fact is *about*. In LoCoMo every message starts with its speaker's name, so "Caroline: Melanie's pottery is lovely" would file the resulting fact about Melanie under Caroline. Matching the fact's own statement gets "about" right, hence the new table.

**How links are made, without re-running consolidation:** match each fact's statement against entity names and aliases with the same key functions recall uses (`phrase_keys`, `search_keys`, in [brain/text.py](../../brain/text.py)):

- "Caroline went to an LGBTQ support group on 7 May 2023" links to Caroline.
- "Dr. Marta Peña will review the schema" links to Marta Peña (by full name or by a name part, as for people today).
- Typo matching (`fuzzy_names`) is **off** for linking: a wrong link puts a fact in the wrong person's block.

When the link happens:

- At consolidation, right after a fact is added or superseded.
- When an entity is created or gains an alias: re-link its facts (search facts for the new keys).
- A backfill command for existing memory: `python manage.py link_fact_entities` (idempotent, `--dry-run`).

Facts about the user ("the user prefers short replies") have no entity and get no link. That's fine: they're already ordinary recall.

### 3.2 The store contract

Added to the store Protocols in [brain/ports.py](../../brain/ports.py), implemented by both stores and covered by the shared contract tests:

```python
class EntityFactStore(Protocol):
    def link_fact_entities(self, fact_id: int, entity_ids: Sequence[int]) -> None: ...

    def entity_fact_candidates(
        self,
        entity_ids: Sequence[int],
        query: Embedding,
        limit: int,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Candidate]: ...
```

Contract rules:

- Only active facts (`A`), never superseded, never `I` or `Z`; only active entities.
- Ranked by similarity to the query, same model's vectors only.
- The period filter (`find_period`) applies as in ordinary recall.
- A link to a purged fact or entity disappears with it (cascade).

### 3.3 Recall

In [brain/retrieval.py](../../brain/retrieval.py), after the entity cards are built:

1. For each named entity (at most `max_entity_blocks`, default 2), fetch its facts with `entity_fact_candidates`, ranked by the question, up to `entity_block.max_items` (default 10).
2. Keep those above `min_score + entity_block.min_score_offset`, a **lower** threshold than ordinary recall (the scope already guarantees they're about the right entity). Start at an offset of -0.10 and calibrate (section 6).
3. **Deduplicate by fact ID** against the facts ordinary recall found: a fact appears once, in the block.
4. Order the kept facts **by date**, so the block reads as a timeline.

New config, per source like the others:

```toml
[entity_block]
enabled = true
max_entity_blocks = 2
max_items = 10
min_score_offset = -0.10   # below the embedding profile's min_score
budget_share = 0.15
```

The threshold is an **offset**, not a value: thresholds depend on the embedding model, so they live in the embedding profile (`configs/embeddings/bge-m3.toml`), and `default.toml` refuses `min_score`. With bge-m3 (0.53), the block keeps facts from 0.43; switching embedding models keeps the block's threshold in step.

### 3.4 In the prompt

The block sits under the entity's card in the `<memory>` block:

```
- [N7] Melanie (person)
  - [F112 · Sat 2023-05-20] Melanie started pottery classes.
  - [F145 · Sun 2023-07-02] Melanie went camping with her kids at the beach.
  - [F201 · Thu 2023-09-14] Melanie painted a sunset with her children.
```

- The date is the fact's learned date, which is the conversation's date: consolidation runs when the episode closes (in LoCoMo, at the session's end). The statement also carries the event's date when consolidation wrote one (D0's dates fix).
- The block's tokens come from `budget_share` of the memory budget; unused tokens go back to the other sources.
- Facts in a block keep their `F` IDs, so `read_memory` and used-detection work unchanged.

### 3.5 Pitfalls this design handles

| Pitfall | Handling |
|---|---|
| LoCoMo's two speakers are in every message: hundreds of facts each | Ranked by the question and capped (`max_items`), never "all facts about Caroline" |
| A question names someone but asks about something else ("what did Caroline say about the weather?") | The ranking keeps only facts close to the question; the threshold drops the rest |
| The same fact found twice (ordinary recall and the block) | Deduplicated by fact ID |
| Superseded facts | Only current facts, as in recall |
| A wrong link puts a fact in the wrong block | Exact name and alias matching only, no typo matching |
| The block crowds out episodes and messages | Its own `budget_share` |
| Several people named | At most `max_entity_blocks`, in the order named |

## 4. Step 2: the written profile (only if step 1 isn't enough)

### 4.1 What it is

A short text per entity, at most about 120 words, written by the background model from the entity's current facts:

> Melanie is Caroline's friend and a mother of two. She started pottery classes in May 2023, went camping with her kids in July, and paints with her children. She ran a charity race in June 2023.

### 4.2 Storage

On `Entity`: `profile` (text), `profile_fact_hash` (hash of the sorted IDs of the facts it was written from), `profile_updated_at`. `profile` is curated memory, so it joins `CHANGE_FIELDS` and the audit trail; the hash and date are written with `update()`.

An `EntityEmbedding` row stores the profile's vector, with its hash, like episode summaries ([EMBEDDINGS.md](../EMBEDDINGS.md)).

### 4.3 When it's written

- At the end of consolidation, for each entity whose facts changed (the hash no longer matches).
- Not for entities with fewer than 3 facts: the block alone is enough there.
- One model call per entity, capped input (its 30 most recent facts plus the previous profile).

### 4.4 In the prompt

The profile replaces the plain card text; the block still follows with the facts most relevant to the question. The profile gives the overview, the block the specific pieces.

### 4.5 Risks specific to step 2

- **Drift**: a profile rewritten from a previous profile can drift from the facts. Always rewrite from the facts, with the previous profile only as a style reference.
- **Cost**: extra calls at consolidation; on the local model, extra GPU time.
- **Re-consolidation** for LoCoMo (about 55 minutes on the cloud model; replies for summaries and facts stay cached, only profile calls are new).
- **Attribution**: profiles only state what facts say; the facts already follow the user-stated rule.

## 5. The memory viewer

- **Entity page**: its linked facts in date order, and the profile when step 2 exists, with Forget / Set aside on each fact as today.
- **Lens**: shows the entity block a question would bring, marked as such, with each fact's score.

## 6. Evaluation

Measured the same way as every earlier change, on the tuning split only, before the configuration is frozen.

### 6.1 LoCoMo

| Measure | How |
|---|---|
| Answers, 200 questions | qwen3.5:4b answering, temperature 0, memory consolidated with dates, grader gpt-oss:120b at temperature 0; `--compare` against D0 and a sign test |
| Multi-hop retrieval, all 282 questions | Share of each question's evidence that reaches the notes, counting a fact as evidence when its source messages include an evidence turn |
| By category | Multi-hop is the target; single-hop and temporal must not drop |
| Time per answer | Should stay close to D0's (no model calls added in step 1) |

The 200-question sample holds only 25 multi-hop questions, so the retrieval measure on all 282 is the more reliable signal for this feature.

### 6.2 Cerveau's own evals

- Retrieval set: hit rate, recall, and **noise**, especially for questions that name someone but should recall little.
- Personal case, multi-turn eval, tool eval: no regressions.

### 6.3 Calibration

- Sweep `entity_block.min_score_offset` (for example -0.15 to 0) and `max_items` (6, 10, 15) on LoCoMo retrieval and the retrieval set.
- Pick the setting that raises multi-hop evidence recall without raising noise.

### 6.4 Decision rule

- **Keep step 1** if answers improve on the 200 questions (sign test leaning positive) and multi-hop retrieval clearly rises, with no regression in the other categories or in Cerveau's own evals.
- **Try step 2** only if step 1 helps retrieval but answers still lag on multi-hop.
- **Remove** whichever step doesn't help, like the other unused features.

Then freeze the configuration and run the final runs.

## 7. Changes required

### `brain/`

- `brain/types.py`: nothing new for step 1 (facts and entities exist); `Entity.profile` for step 2.
- `brain/ports.py`: `EntityFactStore` (section 3.2), added to `MemoryStore`.
- `brain/facts.py`: link facts after adding or superseding them.
- `brain/entity_links.py` (new): `entities_in_text(text, entities)`, using the existing key functions.
- `brain/retrieval.py`: the entity block (section 3.3), deduplication, date order.
- `brain/prompt.py`: block lines under the card.
- `brain/config.py`: `[entity_block]` settings, validated.
- Step 2: `brain/profiles.py` (new): when a profile is due, the profile call, the hash.

### `adapters/`

- `adapters/stores/in_memory.py`: `EntityFactStore`.

### Django

- `FactEntity` model and migration.
- `DjangoMemoryStore`: `EntityFactStore`.
- `link_fact_entities` management command.
- Viewer: entity page and lens.
- `purge_memory`: nothing extra (cascade), checked by a test.
- Step 2: `Entity` profile fields, migration, `EntityEmbedding` rows, the consolidation hook.

### Evals

- `evals/locomo.py`: multi-hop evidence recall counting facts by their source messages.
- `evals/cases/`: a few questions that name someone but should recall little, to watch noise.

## 8. Checklist

**Step 1**

- [ ] Count the facts that use a pronoun instead of a name (open question 2)
- [ ] `FactEntity` model, migration, store methods on both stores
- [ ] Contract tests: active only, superseded excluded, forgotten excluded, period filter, cascade
- [ ] `entities_in_text` with tests (full names, aliases, name parts for people, no typo matching)
- [ ] Linking at consolidation and on new aliases; `link_fact_entities` backfill
- [ ] Entity block in recall: scoped search, threshold, cap, dedup, date order, budget share
- [ ] Prompt lines; viewer entity page and lens
- [ ] LoCoMo: answers on 200 vs D0, multi-hop evidence recall on 282, by category
- [ ] Own evals: noise, personal case, multi-turn, tool eval
- [ ] Calibration sweep; decision recorded in the results file

**Step 2 (only if decided)**

- [ ] Profile fields, embedding, staleness hash
- [ ] Profile call at consolidation, capped input, rewritten from facts
- [ ] Re-consolidation for LoCoMo; same measurements as step 1
- [ ] Decision recorded; configuration frozen

## 9. Open questions

1. **Facts about two people** ("Caroline and Melanie went hiking"): link to both. Fine for the block, but worth checking it doesn't double their weight elsewhere.
2. **Facts without names** ("she started a new job"), where consolidation wrote a pronoun: no link. The consolidation prompt already asks to name each speaker; **measure how often facts still use pronouns before building** (a count over the LoCoMo facts is enough): if many do, the block misses them, and the fact prompt needs a rule first.
3. **Entities the user names indirectly** ("my sister"): covered by aliases if consolidation recorded them.
4. **Shared files**: should passages mentioning an entity join its block? Not in step 1; facts only.
