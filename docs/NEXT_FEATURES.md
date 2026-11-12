# Next features: a docs site, reach, and ideas from memv

**Status:** proposed, after the launch (none of this changes the published numbers, which describe the current version)
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [FACTS.md](FACTS.md), [CERVEAU_LOCOMO_RESULTS.md](CERVEAU_LOCOMO_RESULTS.md), [PERSONAL_CHATS_EVAL.md](PERSONAL_CHATS_EVAL.md)

---

## 1. Why

[memv](https://vstorm-co.github.io/memv) is an open-source memory library for agents (MIT). It differs from Cerveau in useful ways:

- **Reach:** a developer installs it into an existing agent in minutes, with examples for PydanticAI, LangGraph, LlamaIndex, CrewAI and AutoGen. Cerveau is an app you run; its engine can't yet be used from other tools.
- **Ideas:** extraction that stores only what memory failed to predict ("predict-calibrate", from the [Nemori paper](https://arxiv.org/abs/2508.03341)); facts with two times, when something happened and when it was learned (bi-temporal, as in Graphiti/Zep); rank fusion of vector and keyword search.
- **Documentation:** a site with one page per concept, a sidebar and search.

memv publishes no benchmark results; Cerveau does. The aim here is to take what fits, measure it the same way as every earlier change, and credit the sources.

## 2. The order

| # | Feature | Why first | Size | Changes the numbers? | Section |
|---|---|---|---|---|---|
| 1 | Documentation site | The docs exist; this makes them readable | Small | No | 3 |
| 2 | MCP server | Other tools can use Cerveau's memory | Medium | No | 4 |
| 3 | `brain/` as a library | Developers can build the engine into their own agent | Medium | No | 5 |
| 4 | Bi-temporal facts | Changed decisions and "when was this true" questions | Medium | Possibly (temporal) | 6 |
| 5 | Predict-calibrate extraction | Fewer duplicate facts | Medium, an experiment | Yes, must be measured | 7 |
| 6 | Rank fusion (RRF) in recall | A different way to combine signals | Small, an experiment | Yes, must be measured | 8 |
| 7 | Reinforcement of used memories | Designed in PIPELINE step 10 (v2), never built; what ACT-R and Engram do | Medium, an experiment | Yes, must be measured | 9 |

Items 1 to 3 add reach without touching the engine's behavior. Items 4 to 7 change the engine and follow the usual rule: measured on the LoCoMo tuning sample and the personal chats, kept only if they help without losing trick questions.

## 3. Documentation site

**What:** a site generated from the existing Markdown with [MkDocs](https://www.mkdocs.org/) and the Material theme, published with GitHub Pages from the public repository.

**Pages:**

| Section | Pages | From |
|---|---|---|
| Getting started | Install, first chat, importing a ChatGPT export | README |
| Concepts | Episodes, Consolidation, Facts, Entities and tags, Recall and activation, Search tools, Forgetting and purge | ARCHITECTURE, FACTS, ACTIVATION, the GUIDE |
| Results | LoCoMo results, the scoreboard (rendered as a page), personal chats eval, the write-up | Results files |
| Code | The GUIDE, split by part (story, engine, adapters, Django app, evals) | GUIDE |
| Development | STATUS, the LoCoMo plan, design proposals | Log and plans |

**Notes:**

- The GUIDE's split (item R10 of the LoCoMo plan) happens here: one page per part, with every section link kept working.
- The docs stay in `docs/`; the site is built from them, not a copy.
- A GitHub Actions workflow builds and publishes on each push to `main`.
- Later, the same site can hold a personal page and other projects.

**Checklist:**

- [ ] `mkdocs.yml` with the Material theme, navigation as above, search
- [ ] The GUIDE split into parts, links checked (`mkdocs build --strict` fails on broken links)
- [ ] The scoreboard as a page
- [ ] Publishing workflow; GitHub Pages turned on
- [ ] README links to the site

## 4. MCP server

**What:** an MCP server that exposes Cerveau's memory to other tools (Claude Desktop, coding agents, any MCP client), so they search and add to the same memory the chat uses.

**Tools exposed:**

| Tool | Does | Notes |
|---|---|---|
| `search_memory` | Searches facts, summaries, entities and messages | The same function the chat model calls |
| `read_memory` | Reads one memory by ID, with its source | Same as the chat's |
| `recall` | Returns what automatic recall would bring for a text, within a token budget | The Lens, as a tool |
| `remember` | Adds a note or a message from another tool | Goes through consolidation like any message; tagged with its source |
| `forget` | Sets a memory aside or forgets it | Never purges; purging stays a deliberate command |

**Design rules:**

- **Local only:** the server runs on the user's machine (stdio, or HTTP on `127.0.0.1`), like the app.
- **Same engine, no copy:** it calls `brain/` through the same store as the app.
- **The source is recorded:** memories added through MCP carry the client's name, visible in the viewer, so the user knows what came from where.
- **Never returns forgotten memory** (status `Z`), as everywhere else.
- **Read-only by default:** `remember` and `forget` are off until the user turns them on in config.

**Built with** the official Python MCP SDK, as a management command (`python manage.py mcp_server`).

**Checklist:**

- [ ] Server with the five tools, read-only by default
- [ ] Source recorded on added memories; shown in the viewer
- [ ] Tests: forgotten memory never returned, write tools off by default, budgets respected
- [ ] Setup instructions for Claude Desktop and one coding agent
- [ ] README and docs site

## 5. `brain/` as a library

**What:** the engine published as a Python package, so a developer can give their own agent Cerveau's memory without the Django app.

**Why it's feasible:** `brain/` is already plain Python, with storage behind Protocols (`brain/ports.py`) and an in-memory store passing the same contract tests as the Django one.

**Shape:**

```python
from cerveau import Memory
from cerveau.stores import PostgresStore   # or InMemoryStore
from cerveau.models import OllamaModel     # or any OpenAI-compatible server, once supported

memory = Memory(store=PostgresStore(url), model=OllamaModel("gemma4:e4b"), embedder=...)
memory.add_message("user", "My sister Lena is getting married in Lisbon on June 14.")
notes = memory.recall("When is Lena's wedding?", budget_tokens=2000)
prompt_block = notes.to_prompt()
```

**What it needs:**

- A Postgres store that doesn't depend on Django (plain SQL or SQLAlchemy), passing the existing contract tests. The Django store stays for the app.
- Consolidation callable synchronously or in the background, without Django's task queue.
- A package name, version, and a short "use in your own agent" page.

**Decision to make first:** whether a second store is worth maintaining. If not, the MCP server alone covers most of the reach.

**Checklist:**

- [ ] Decide: library or MCP only
- [ ] Django-free Postgres store, contract tests passing
- [ ] Public API (`Memory`, `recall`, `add_message`, `consolidate`), documented
- [ ] Package build, published
- [ ] Example: one framework integration

## 6. Bi-temporal facts

**What:** each fact records two times, as in memv and Graphiti:

- **Event time:** when what it says was true (`valid_from`, `valid_to`).
- **Learned time:** when Cerveau learned it, and when it stopped believing it (`learned_at`, `invalidated_at`).

**Today:** the event date is written into the fact's text by consolidation ("on 7 May 2023"), and supersession marks an old fact replaced. This was the deferred item 2 of the LoCoMo plan ("event dates on facts, separate from when the fact was learned").

**What it adds:**

- **Changed decisions:** "what was the plan before I changed it?" reads the invalidated fact, with when it changed.
- **Questions about a period:** "where was I working last year?" filters by event time instead of hoping the date is in the text.
- **The viewer:** a fact's history as a timeline.

**Design:**

- New fields on `Fact`: `valid_from`, `valid_to` (nullable dates), `invalidated_at` (set when superseded). `learned_at` is the existing creation time.
- Consolidation's reply schema gains optional event dates; the existing date rules (`brain/dating.py`) fill them from relative expressions.
- Recall's period filter (`find_period`) uses event time when present, the text date otherwise.
- Superseded facts stay hidden from normal recall, as today, but become reachable for questions about the past.

**Measured on:** LoCoMo temporal questions (all 321) and the tuning sample; the personal chats' "when" and "changed decision" questions. Kept only if temporal questions don't drop.

**Checklist:**

- [ ] Fields, migration, contract tests (both stores)
- [ ] Consolidation fills event dates; tests on relative dates
- [ ] Period filter on event time
- [ ] Questions about the past reach invalidated facts
- [ ] Viewer: a fact's history
- [ ] Measured; decision recorded

## 7. Predict-calibrate extraction (an experiment)

**What:** before extracting facts from a closed episode, consolidation retrieves what memory already knows about it, predicts what the episode says, and extracts only what the prediction missed: new facts, updates, and contradictions. Based on the Nemori paper, as memv uses it.

**Why it might help:** fewer duplicate facts, no separate importance scoring, and it fits the brain metaphor (the brain remembers surprises). Cerveau today extracts up to 16 facts per session and deduplicates afterwards.

**The risk, specific to small models:** if gemma predicts badly, it may treat a new fact as already known and skip it. The first episodes about a person extract almost everything (nothing to predict yet), so the risk grows as memory grows.

**Design for the experiment:**

- Behind a config flag (`consolidation.mode = "extract" | "predict_calibrate"`), off by default.
- One extra step per episode: retrieve up to 10 existing facts about the episode's entities, ask for the prediction, then extract the difference, labelled `new`, `update` or `contradiction`.
- `update` and `contradiction` go through the existing supersession.

**Measured on:**

- LoCoMo tuning sample and multi-hop set: answers, trick questions, and number of facts stored.
- Personal chats (both sets): answers, and facts stored.
- Consolidation time per episode on the 6 GB GPU.

**Keep it if** answers hold or improve with fewer facts, trick questions don't drop, and consolidation stays under about a minute per episode. Otherwise record the result and remove it, like the reranker.

**Checklist:**

- [ ] Flag, prediction prompt, difference extraction
- [ ] Tests on made-up episodes (known fact skipped, new fact kept, contradiction supersedes)
- [ ] Re-consolidation of LoCoMo tuning conversations and personal chats with the flag
- [ ] Measured; decision recorded

## 8. Rank fusion in recall (an experiment)

**What:** combine the recall signals by rank (reciprocal rank fusion: each memory scores the sum of 1 / (k + its rank) across signals) instead of by weighted scores.

**Today:** meaning, names, exact words and activation are combined into one score with tuned weights and a threshold (`min_score`).

**Why try it:** rank fusion needs no weights and is robust when signals use different scales. **Why it may not help:** Cerveau's weights and threshold were calibrated on two datasets, and the threshold check on real chats (PERSONAL_CHATS_EVAL 6g) held.

**Measured on** the retrieval set, LoCoMo tuning sample and personal chats, with `recall_sweep`. Lowest priority.

## 9. Reinforcement of used memories (an experiment)

**What:** memories that are recalled and used get stronger over the long term, as in ACT-R's base-level activation and in [Engram](https://pypi.org/project/engramai/): each use is recorded, and a memory's strength sums its past uses, each weighted by how long ago it happened (`ln Σ tᵢ^(-d)`). A memory used 50 times over months stays strong after a quiet week; one used once fades back to its similarity score.

**Today:** it is designed but not built ([PIPELINE.md](PIPELINE.md) step 10, "Reinforce used memories (v2)"; ARCHITECTURE's "Retrieval strengthens memory"). What exists is short-term: activation on tags and entities that fades over turns and hours, and carry-over of recalled memories, which keep 90% of their score each turn (`carry_decay = 0.9`) while the conversation stays on their subject. Nothing records long-term use.

**Design:**

- **Record uses, not exposures:** a memory counts as used when the reply relied on it. Detecting that is the hard part on a 4B model (ACTIVATION.md left "used versus exposed" out for that reason); start with the cheap signal, the reply's embedding compared with each memory in the prompt, and check it by hand on a sample.
- **Storage:** a use log or two columns per memory (`use_count`, a decayed strength with its `updated_at`), written without audit snapshots, like activation.
- **Scoring:** `w_use × strength`, a small weight next to `w_act`, reranking only.
- **Never deletes:** strength affects ranking, never retention. Forgetting stays the user's decision.

**Risk:** a rich-get-richer loop, where memories recalled often get recalled more and crowd out others. Watch noise on unrelated questions and the variety of what's recalled.

**Measured on:** the multi-turn eval (follow-ups, drags, noise), the personal chats, and long live use. LoCoMo's questions are asked one at a time with no history of use, so it can't show the effect.

## 10. Credits

In ARCHITECTURE's related work, when any of these is built:

- **memv** (vstorm): predict-calibrate in practice, bi-temporal facts, the docs layout.
- **Nemori** (arXiv 2508.03341): predict-calibrate.
- **Graphiti / Zep** (Rasmussen et al., 2025): bi-temporal facts.
- **Reciprocal rank fusion** (Cormack et al., 2009).
- **ACT-R** (Anderson, 1983) and **Engram** (engramai): long-term strength from past uses.
