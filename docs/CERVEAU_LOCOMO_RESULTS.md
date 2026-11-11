# Cerveau on LoCoMo: full test log and results

Runs made 2026-10-18 to 2026-10-24. This file is self-contained: it explains what Cerveau is, what the benchmark is, which model did which job in every run, and every number obtained, so another reader (or another AI assistant) can check and compare without the code.

**Final numbers (2026-10-25, section 6.6, the grid with dated message notes; empty replies asked again on 2026-11-03, section 6.8):** fully local **66.9%** (deepseek-v4-pro; 62.4% by gpt-oss:120b), best case **76.2%** (70.7%), the whole conversation 82.7% (78.6%), on all 1,536 gradable questions; under the lenient grader like the published ones, 76.4% fully local and 82.6% best case. **The reported grader is deepseek-v4-pro**, chosen by the user's hand labels (it agrees with them on 85 of 100 answers, gpt-oss:120b on 70; section 6.7); gpt-oss:120b stays beside it as the second grader. **Human-reviewed** (the audits with the user's review, sections 6.7 and 6.8): fully local **66.3% (+/-2.4)** on normal questions and 90.7% on trick questions; best case **76.4%** on normal questions and 84.8% on trick questions (87.0% if its 10 empty replies counted as declining). **How these were reached:** fully local scored 65.3% at the frozen run (section 6.5); 66.9% is after one fix found by diagnosing that run's misses (dates annotated in message notes, item 26), tuned on half the conversations and confirmed on the other half, then the grid rerun on all ten. The number is therefore not an untouched held-out score: the fix was chosen by looking at these questions' misses. Its held-out evidence is the confirmation on five conversations (temporal 69.5% to 74.4%, 8 turned right and 0 wrong).

**First final grid (2026-10-24, section 6.5):** on all 1,536 gradable questions, Cerveau answers **76.0%** (deepseek-v4-pro; 69.5% by gpt-oss:120b) with gpt-oss:120b consolidating and answering, and **65.3%** (61.7%) **fully local** with gemma4:e4b consolidating and answering on a 6 GB laptop GPU; the whole conversation in gpt-oss:120b's prompt scores 80.9% (77.1%). Under a lenient grader like the published ones: 81.6% and 75.6%. The hand labels later chose deepseek-v4-pro as the reported grader (section 6.7).

**Where things stood (2026-10-22):** on a 200-question sample, graded by gpt-oss:120b at temperature 0, Cerveau answers **72%** correctly with the laptop's 4B model answering and memory consolidated by gpt-oss:120b with dates (run D0), and **74%** with gpt-oss:120b answering from the same memory (G1). Raw messages alone give 60%; a big model reading the whole conversation, 80% (the ceiling). A reranker (built, off) raises retrieval from 64% to 78% but not the answers, with either model. Multi-hop questions are the open gap; entity blocks raise them by 7 to 8 points (graded by deepseek-v4-pro), and the remaining losses are mostly notes holding part of a multi-part answer. Next: save-and-resume in the runner, graders checked against hand labels, a fully local consolidation, and final runs on all 1,540 questions (section 10).

## 1. What is being tested

**Cerveau** is a personal AI memory system built by one developer, on Django and Postgres with pgvector, running on a laptop (RTX 3060 Laptop GPU, 6 GB). Conversations are stored as messages grouped into episodes. When an episode closes, a background job "consolidates" it: title, short and long summary, topic tags, the people and things mentioned, and facts. Before each reply, Cerveau recalls relevant memories (messages, episode summaries, facts, file passages) and puts them in the model's prompt. The model can also search memory itself with two tools (`search_memory`, `read_memory`).

**Recall scoring**, per memory: `1.0 x semantic similarity + 0.4 x name match + 0.6 x exact-term match + 0.05 x activation (+ 0.05 x tag match for episodes)`. A memory must score at least `min_score = 0.53` (calibrated for bge-m3). Each source keeps at most `max_items` memories: messages 5 at first, **10 since these tests**; episodes 3; facts 5.

## 2. The benchmark: LoCoMo

LoCoMo (Maharana et al., 2024), file `locomo10.json` from github.com/snap-research/locomo.

- 10 conversations between two people (for example Caroline and Melanie), 19 to 32 dated sessions each, **272 sessions and 5,882 turns** in all (about 600 turns per conversation).
- **1,986 questions**, each listing the turns that hold its answer ("evidence"), in 5 categories:

| Category | Questions | What it asks |
|---|---|---|
| 1 multi-hop | 282 | combine facts from several sessions |
| 2 temporal | 321 | when something happened |
| 3 open-domain | 96 | an inference from what was said |
| 4 single-hop | 841 | one stated fact |
| 5 adversarial | 446 | about the other speaker; right answer is "not said" |

As in the published comparisons, **category 5 is left out**: 1,540 questions in categories 1 to 4 (1,532 whose evidence turns could be located).

**The 200-question sample** used for all answer runs: 20 questions per conversation, sampled with seed 0. It holds 121 single-hop, 45 temporal, 25 multi-hop and 9 open-domain questions. Every answer run uses the same 200.

## 3. How the runner works (`python -m evals.locomo`)

**Building memory:** each session becomes a closed episode at its real date and time; each turn becomes a message stored with the user role, text `"Speaker: what they said"`, shared photos added as `[shares a photo: caption]`. Embeddings: **bge-m3**, run locally by Ollama. Without `--consolidate`, there are no summaries, facts, tags or entities: raw messages only.

**Two modes:**

- **Retrieval mode** (no chat model): each question goes through automatic recall only. Measures:
  - *turn hit*: at least one evidence turn among the recalled memories;
  - *turn recall*: the share of evidence turns recalled;
  - *session hit*: the right session recalled, by any of its turns or its summary.
- **Answer mode**: each question is asked as a real chat turn through Cerveau's full pipeline (recall, prompt, memory tools available), in a fresh conversation, on freshly built memory, "today" being the day after the last session. Then a **grader model** labels the answer CORRECT or WRONG against the gold answer.

**Grader prompt (summary):** CORRECT if the answer gives the same fact as the gold answer, even in other words, longer, or with more detail; a date in another format or relative but pointing to the same day, period or year is CORRECT; WRONG if it gives a different fact, misses the point, or says it doesn't know. Token F1 (the dataset's own metric) is also computed; it stays low because the chat model answers in full sentences.

**Runner details** (as of 2026-10-21): `--undated` consolidates with the prompts from before the date fix (checked identical), so memory without dates rebuilds from the cached replies of A10 and A11. each conversation's memory is built once per run and copied for every question (a few milliseconds), so nothing one question does reaches the next; rebuilding the ten consolidated memories from the cached model replies takes about 6 minutes. A cloud consolidation runs the ten conversations side by side. Each answer records how many memory searches the model made; the report gives a 95% interval per grader and the 90th-percentile time; `--compare` gives turned right, turned wrong and a sign test between two saved runs.

**Full-context mode** (`--full-context`): no memory system at all; the whole conversation (about 26,000 tokens) is put in the prompt with the question. This is the ceiling.

## 4. The models and what each one did

| Model | Where | Used for |
|---|---|---|
| bge-m3 | local, Ollama | all embeddings (messages, summaries, facts, tags, questions) |
| qwen3.5:4b | local, Ollama, 16,384-token context, memory budget 8,000 tokens, thinking off | **answering** in most runs (Cerveau's real chat model); temperature 0.7 by default, 0 in the later runs |
| qwen3.5:4b, background profile | local, temperature 0.2 | **grading** in the first runs (found too lenient, see 6.3) |
| gpt-oss:120b | Ollama Cloud (Pro plan) | **grading** from run A3 on (temperature 0.2 until D1, **0** from D2 on, and for every re-grade); **answering** in runs A4, A5, G1 and G2; **consolidating** the 272 sessions for runs A10 and A11 (without dates) and from D0 on (with dates) |
| bge-reranker-v2-m3 | local, sentence-transformers on the CPU (about 2 GB, downloaded once) | **reranking** recall's shortlist in runs R1, G2 and RM (section 5.5) |
| deepseek-v4-pro | Ollama Cloud, temperature 0 | **grading** from runs N0, N1, GN0, GN1 on (first grader, with gpt-oss as the second), and the **answer-in-notes** check: on three hard cases it judged like kimi-k3, in less than half the time, and rejected "a cup" for the gold answer "pots", which gpt-oss accepted |

Ollama Cloud ignores structured output: neither a JSON Schema nor `format: "json"` changes what the models return (tested on gpt-oss:120b, deepseek-v4.1-flash, glm-5.3, kimi-k2.6 and mistral-large-4; three wrap their JSON in a code fence, mistral-large-4 answers in prose). Pydantic AI's documentation says the same and points to Ollama issue #12362. So for cloud models the schema is put in the prompt, the JSON is extracted from the reply, and an unreadable reply is asked once more. Local Ollama does enforce the schema.

## 5. Retrieval results (no chat model, 1,532 questions)

### 5.1 Message slots (raw messages, no consolidation)

| messages max_items | turn hit | turn recall | session hit | memories kept per question |
|---|---|---|---|---|
| 5 (old default) | 52% | 46% | 80% | 4.9 |
| 10 (new default) | 62% | 56% | 88% | 9.4 |
| 20 | 73% | 65% | 93% | 17.6 |

By category at 5 and 10 slots (turn hit / turn recall / session hit):

| Category | 5 slots | 10 slots |
|---|---|---|
| multi-hop | 41 / 20 / 77 | 55 / 29 / 86 |
| temporal | 58 / 55 / 78 | 69 / 66 / 86 |
| open-domain | 29 / 22 / 57 | 37 / 28 / 70 |
| single-hop | 55 / 54 / 84 | 65 / 64 / 90 |

With the dataset's own session summaries added (not Cerveau's), 5 slots: session hit 88%, turn figures unchanged.

### 5.2 Lateral inhibition (10 slots, raw messages)

A memory's score is lowered by `strength x its closest similarity to memories already picked`, so near-duplicates give way to different memories. Picked from the best 3 x max_items.

| strength | turn hit | turn recall | session hit | multi-hop turn hit |
|---|---|---|---|---|
| 0 (off) | 62% | 56% | 88% | 55% |
| 0.1 | 64% | 57% | 89% | 57% |
| 0.2 | 64% | 57% | 89% | 59% |
| 0.3 | 64% | 57% | 89% | 60% |
| 0.5 | 65% | 57% | 90% | 63% |

### 5.3 Consolidated sessions (10 slots, inhibition off)

| | turn hit | turn recall | session hit | kept |
|---|---|---|---|---|
| raw messages | 62% | 56% | 88% | 9.4 |
| consolidated by gpt-oss:120b | 64% | 57% | 92% | 12.8 |

Consolidated, by category (turn hit / turn recall / session hit): multi-hop 56 / 30 / 93, temporal 72 / 69 / 91, open-domain 42 / 33 / 78, single-hop 66 / 64 / 94.

### 5.4 Consolidated with dates (10 slots)

Re-consolidated with the conversation's date in the transcript and absolute dates asked for (section 6.3, item 8): turn hit 64%, turn recall 57%, session hit 92% (temporal session hit 91% to 93%). Retrieval barely moves: the dates change what summaries and facts say, not how they're found.

### 5.5 The reranker (the 200-question sample, consolidated with dates)

Each source reads 4 x its slots (40 candidates for messages) and keeps the ones bge-reranker-v2-m3 puts first; scores and thresholds unchanged.

| | turn hit | turn recall | session hit |
|---|---|---|---|
| without reranker | 64% | 59% | 96% |
| with reranker | **78%** | **72%** | 96% |

By category (turn hit, without and with): multi-hop 52% and 76%, temporal 78% and 80%, open-domain 22% and 22%, single-hop 65% and 81%. On one question, it scored the dated fact "Caroline went to an LGBTQ support group on 7 May 2023" 1.00, "I went to a support group yesterday" 0.30 and "I love my support group" 0.01 for "When did Caroline go to the support group?". About 1.8 s for 40 pairs on the CPU.

## 6. Answer results (the 200-question sample)

### 6.1 Every run

"Judged right" = share of the 200 answers labeled CORRECT by the grader. Unless said otherwise, memory is raw messages with 10 slots and inhibition off.

| Run | Answered by | Memory | Temp. | Graded by | Judged right | single-hop | temporal | multi-hop | open-domain | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | qwen3.5:4b | messages, **5 slots** | 0.7 | qwen3.5:4b | 57% | 64% | 56% | 40% | 22% | 0.12 |
| A2 | qwen3.5:4b | messages, 10 slots | 0.7 | qwen3.5:4b | 67% | 74% | 62% | 56% | 33% | 0.12 |
| A3 | qwen3.5:4b | messages | 0.7 | **gpt-oss:120b** | **57%** (57% re-graded at 0) | 70% | 49% | 28% | 11% | 0.12 |
| A3 | (same answers) | | | qwen3.5:4b | 70% | 79% | 69% | 52% | 22% | |
| A4 | **gpt-oss:120b** | messages | 0.2 | gpt-oss:120b | **62%** (62% re-graded at 0) | 70% | 60% | 28% | 44% | 0.15 |
| A4 | (same answers) | | | qwen3.5:4b | 78% | 80% | 80% | 64% | 67% | |
| A5 | gpt-oss:120b | **whole conversation, no memory system** | 0.2 | gpt-oss:120b | **80%** | 87% | 76% | 68% | 56% | 0.16 |
| A5 | (same answers) | | | qwen3.5:4b | 92% | 93% | 91% | 96% | 78% | |
| A6 | qwen3.5:4b | messages, inhibition 0.5 | 0.7 | gpt-oss:120b | 60% | 74% | 49% | 28% | 22% | 0.12 |
| A7 | qwen3.5:4b | messages (A3 repeated) | 0.7 | gpt-oss:120b | 56% | 71% | 31% | 32% | 33% | 0.12 |
| A8 | qwen3.5:4b | messages | **0** | gpt-oss:120b | **60%** | 74% | 56% | 20% | 22% | 0.13 |
| A9 | qwen3.5:4b | messages, inhibition 0.5 | 0 | gpt-oss:120b | 59% | 73% | 44% | 28% | 33% | 0.13 |
| A10 | qwen3.5:4b | **consolidated by gpt-oss:120b** | 0 | gpt-oss:120b | **64%** (66% re-graded at 0) | 79% | 40% | 40% | 56% | 0.15 |
| A11 | qwen3.5:4b | consolidated by gpt-oss:120b, inhibition 0.5 | 0 | gpt-oss:120b | **66%** | 79% | 49% | 40% | 56% | 0.15 |
| D0 | qwen3.5:4b | **consolidated by gpt-oss:120b, with dates** (2026-10-19) | 0 | gpt-oss:120b | **71%** (plus or minus 6) | 83% | 64% | 36% | 44% | 0.15 |
| D1 | qwen3.5:4b | consolidated with dates, inhibition 0.5 | 0 | gpt-oss:120b | 65% (plus or minus 7) | 78% | 58% | 28% | 33% | 0.15 |
| D0 re-graded | (D0's answers) | | | gpt-oss:120b at **0**, twice | **72%** both times | 82% | 64% | 44% | 44% | 0.15 |
| D2 | qwen3.5:4b | consolidated with dates; **message lines show their time** (2026-10-20) | 0 | gpt-oss:120b at **0** | 68% (plus or minus 6) | 82% | 58% | 32% | 44% | 0.15 |
| R1 | qwen3.5:4b | consolidated with dates, **reranker** bge-reranker-v2-m3 (2026-10-20) | 0 | gpt-oss:120b at **0** | 72% (plus or minus 6) | 85% | 62% | 44% | 33% | 0.15 |
| G1 | **gpt-oss:120b** | consolidated with dates (2026-10-20) | 0 | gpt-oss:120b at **0** | **74%** (plus or minus 6) | 82% | 73% | 40% | 56% | 0.18 |
| G2 | **gpt-oss:120b** | consolidated with dates, **reranker** | 0 | gpt-oss:120b at **0** | 75% (plus or minus 6) | 87% | 67% | 40% | 56% | 0.18 |
| T0 | qwen3.5:4b | consolidated **without** dates (`--undated`), **all 321 temporal questions** (2026-10-21) | 0 | gpt-oss:120b at **0** | | | **45%** (plus or minus 5) | | | 0.14 |
| M0 | qwen3.5:4b | consolidated with dates, **all 282 multi-hop questions** (2026-10-21) | 0 | gpt-oss:120b at **0** | | | | **35%** (plus or minus 6) | | 0.09 |
| M1 | qwen3.5:4b | the same, **entity blocks** (version 1) | 0 | gpt-oss:120b at **0** | | | | 39% | | 0.09 |
| M2, M3 | qwen3.5:4b | multi-hop, version 2 (alias rule), without and with blocks | 0 | gpt-oss:120b at **0** | | | | 34%, **38%** | | 0.09 |
| GM0 to GM3 | gpt-oss:120b | multi-hop, versions 1 and 2, without and with blocks | 0 | gpt-oss:120b at **0** | | | | 41%, 43%; 44%, **46%** | | 0.10 |
| B0, B1 | qwen3.5:4b | the 200 questions, version 2, without and with blocks | 0 | gpt-oss:120b at **0** | 70%, 70% | 79%, 80% | 64%, 64% | 40%, 40% | 44%, 44% | 0.15 |
| GB0, GB1 | gpt-oss:120b | the 200 questions, version 2, without and with blocks | 0 | gpt-oss:120b at **0** | 72%, 74% | 83%, 84% | 69%, 67% | 36%, 40% | 44%, 56% | 0.18 |
| N0, N1 | qwen3.5:4b | all 282 multi-hop, without and with blocks, **notes checked** (2026-10-22) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | **45%, 53%** (34%, 38%) | | 0.09 |
| GN0, GN1 | gpt-oss:120b | the same | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | **52%, 59%** (41%, 45%) | | 0.10 |
| N2, N3 | qwen3.5:4b | all 282 multi-hop, blocks on: a block of 20 facts; 10 plus a second search (2026-10-22) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | 56%, 54% (37%, 40%) | | 0.09 |
| N4, N5 | qwen3.5:4b | all 282 multi-hop, memory with **16 facts per session**; blocks of 10, of 20 (2026-10-22) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | 55%, **59%** (41%, 41%) | | 0.09 |
| GN4, GN5 | gpt-oss:120b | the same | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | 59%, **63%** (44%, 47%) | | 0.10 |
| L3 | qwen3.5:4b | the 200 questions, qwen3.5:4b's memory re-consolidated at temperature 0 with a rules fact prompt (2026-10-23; the prompt reverted since) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 68% (65%) | 80% (79%) | 51% (51%) | 48% (32%) | 44% (44%) | 0.14 |
| B3 | qwen3.5:4b | the 200 questions, gpt-oss:120b's memory with the same rules prompt at temperature 0 | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 75% (74%) | 83% (86%) | 64% (67%) | 68% (44%) | 33% (44%) | 0.15 |
| L3m, B3m | qwen3.5:4b | all 282 multi-hop on those two memories, notes checked | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | 48%, 57% (33%, 43%) | | 0.08, 0.09 |
| L4 | **gemma4:e4b** | the 200 questions, memory consolidated by **gemma4:e4b** (fully local, temperature 0, facts speaker by speaker) (2026-10-24) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 62% (62%) | 79% (79%) | 31% (33%) | 52% (40%) | 22% (22%) | 0.21 |
| C4 | **gemma4:e4b** | the 200 questions, gpt-oss:120b's memory (B2's), gemma4:e4b answering | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 74% (72%) | 83% (83%) | 73% (71%) | 48% (32%) | 33% (33%) | 0.24 |
| **L5** | **gemma4:e4b** | the 200 questions, L4's memory with **relative dates made absolute** in code (no re-consolidation) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | **70%** (68%) | 80% (79%) | **64%** (64%) | 48% (36%) | 33% (33%) | 0.24 |
| L5m | **gemma4:e4b** | all 282 multi-hop on L5's memory, notes checked | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | | | | 48% (37%) | | 0.16 |
| G5, G5m | gpt-oss:120b | the 200 questions and all 282 multi-hop, gpt-oss:120b's memory with dates made absolute, gpt-oss:120b answering (2026-10-24) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | **83%** (78%) | 88% (87%) | 80% (71%) | 72%; 64% on the 282 (52%; 47%) | 56% (56%) | 0.24 |
| L6, G6 | gemma4:e4b, gpt-oss:120b | L5 and G5 with one more sentence in the tool instructions ("if your notes hold only part of what the question asks, or you're unsure, call search_memory"); reverted | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 70%, 82% (68%, 78%) | | | 48%, 62% on the 282 (37%, 50%) | | |
| L1, L2 | qwen3.5:4b | the 200 questions, memory consolidated by **qwen3.5:4b** (fully local; 16 facts per session, blocks of 20); L2 with speaker names in facts (2026-10-23) | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 64%, **66%** (62%, 63%) | 80%, 80% (77%, 76%) | 44%, 49% (49%, 56%) | 36%, 44% (28%, 24%) | 33%, 33% | 0.12, 0.13 |
| B1r, B2 | qwen3.5:4b | the 200 questions: B1 regraded; 16 facts per session and blocks of 20 | 0 | **deepseek-v4-pro** (gpt-oss) at 0 | 74%, **78%** (71%, 74%) | 80%, 83% (80%, 82%) | 64%, 73% (67%, 73%) | 64%, 76% (40%, 40%) | 44%, 33% (44%, 44%) | 0.15, 0.16 |
| RM | qwen3.5:4b | the 200 questions, reranker on messages only | 0 | gpt-oss:120b at **0** | 73% | 83% | 62% | 56% | 33% | 0.15 |
| T1 | qwen3.5:4b | consolidated **with** dates, all 321 temporal questions | 0 | gpt-oss:120b at **0** | | | **62%** (plus or minus 5) | | | 0.19 |

Median time per answer: about 1.7 to 2.4 s for qwen3.5:4b, 1.3 to 1.5 s for gpt-oss:120b; with the reranker, about 7 to 8 s for either. Runs D0, D2, R1, G1 and G2 compare cleanly with each other (same memory, temperature 0 answers, the same grader at temperature 0, D0 re-graded); earlier runs used the grader at 0.2 (item 9).

### 6.2 Question by question (same 200 questions)

| Comparison | Turned right | Turned wrong | Net |
|---|---|---|---|
| A2 vs A1 (10 slots vs 5, graded by qwen) | 23 | 3 | +20 |
| A6 vs A3 (inhibition, temp 0.7) | 20 | 15 | +5 |
| A7 vs A3 (same setup run twice, temp 0.7) | 14 | 18 | -4 |
| A9 vs A8 (inhibition, temp 0) | 13 | 16 | -3 (172 answers worded differently) |
| A10 vs A8 (consolidated vs messages, temp 0) | 25 | 17 | +8 |
| A11 vs A8 (consolidated + inhibition vs messages) | 26 | 14 | +12 |
| D0 vs A10 (dates vs no dates, consolidated; grader at 0.2) | 22 | 9 | +13, sign test p = 0.029 |
| D0 vs A10, both re-graded at temperature 0 | 22 | 11 | +11, **p = 0.08** (temporal: +11 / -3, p = 0.057) |
| **All 321 temporal questions**, dates vs none (runs T1, T0), both at 0 | 82 | 25 | +57, **p = 0.00000003** |
| Entity blocks vs none, all 282 multi-hop, qwen3.5:4b (M3 vs M2) | 25 | 14 | +11, p = 0.11 |
| Entity blocks vs none, all 282 multi-hop, gpt-oss:120b (GM3 vs GM2) | 20 | 14 | +6, p = 0.39 |
| Entity blocks vs none, the 200, qwen3.5:4b (B1 vs B0) | 10 | 9 | +1 |
| Entity blocks vs none, the 200, gpt-oss:120b (GB1 vs GB0) | 12 | 10 | +2 |
| Reranker on messages only vs none, the 200 (RM vs D0) | 15 | 12 | +3, p = 0.70 |
| **Entity blocks vs none, 282 multi-hop, qwen3.5:4b, graded by deepseek-v4-pro** (N1 vs N0) | 31 | 8 | +23, **p = 0.0003** |
| **Entity blocks vs none, 282 multi-hop, gpt-oss:120b, graded by deepseek-v4-pro** (GN1 vs GN0) | 31 | 13 | +18, **p = 0.010** |
| A4 vs A3, both re-graded at 0 (gpt-oss vs qwen answering, raw messages) | 20 | 11 | +9, p = 0.15 |
| G1 vs D0 (gpt-oss vs qwen answering, consolidated with dates), both at 0 | 15 | 11 | +4, p = 0.56 |
| D1 vs A11 (the same with inhibition) | 18 | 21 | -3, p = 0.75 |
| D0 re-graded vs D0 re-graded (the grader against itself, temperature 0) | 1 | 1 | 0 |
| D2 vs D0, both graded at temperature 0 | 2 | 8 | -6, p = 0.11 |
| R1 vs D0 (reranker vs none, both graded at temperature 0) | 14 | 12 | +2, p = 0.85 |
| G2 vs G1 (reranker vs none, gpt-oss answering) | 11 | 8 | +3, p = 0.65 |

### 6.3 What the runs show

1. **The small grader is lenient.** On the same answers, qwen3.5:4b grades 13 to 16 points higher than gpt-oss:120b (A3: 70% vs 57%; A4: 78% vs 62%; A5: 92% vs 80%), most on the hard categories. A1 and A2 were graded by the small model, so their absolute numbers are high; the 5-to-10-slot comparison between them holds (same grader).
2. **Run-to-run noise at temperature 0.7 is about plus or minus 3 points** on 200 questions (A3 57% vs A7 56%, with 32 verdicts changed). From A8 on, answers use temperature 0, which repeats exactly, so differences between A8 to A11 are real for that sample.
3. **Recall is the main limit, not the answering model.** gpt-oss:120b answering from the same notes (A4, 62%) beats qwen3.5:4b (A3, 57%) by only 5 points, and not at all on single-hop or multi-hop. With everything in the prompt (A5) it reaches 80%. The 18-point gap between 62% and 80% is what better recall can win.
4. **Why answers missed** (the 77 wrong answers of A4):

| Where the evidence was | Answers |
|---|---|
| scored above the threshold but crowded out of the 10 message slots | 28 |
| only part of it recalled (11 of them multi-hop) | 14 |
| scored below the threshold | 19 |
| all recalled, answer still wrong | 16 |

5. **Ten message slots instead of five** was the biggest single gain: retrieval turn hit 52% to 62%, answers +20 net (A1 to A2). Now the default.
6. **Lateral inhibition** improves retrieval a little (turn hit 62% to 65%, multi-hop 55% to 63%) but not answers on raw messages (A8 60%, A9 59%). On consolidated memory it adds 2 points (A10 64%, A11 66%), all on temporal. Built, kept off by default.
7. **Consolidation helps** (A8 60% to A10 64%): multi-hop doubles (20% to 40%), open-domain 22% to 56%, single-hop 74% to 79%. **Temporal drops** (56% to 40%): summaries and facts keep what happened but not when ("Caroline went to a support group", not "on 7 May 2023") and they take slots from the dated messages.
8. **Dates fix the temporal drop** (runs D0 and D1, 2026-10-19). Consolidation had never been told when a conversation took place, so "I went yesterday" could not become a date. Now the transcript it reads opens with the conversation's date and time (in the user's timezone), and the summary and fact prompts ask for absolute dates ("Caroline went to an LGBTQ support group on 7 May 2023"). Re-consolidated (all 272 sessions, none failed, about 55 minutes with the conversations side by side): temporal 40% to 64%, single-hop 79% to 83%, **71% overall**, against 64% before the fix; 22 answers turned right and 9 wrong (p = 0.029) with the grader at 0.2. On the stricter grader (temperature 0, re-graded 2026-10-21): 66% to 72%, 22 turned right and 11 wrong, **p = 0.08**; temporal 47% to 64% (11 right, 3 wrong, p = 0.057). Same direction, smaller margin: not significant on 200 questions. **Confirmed on all 321 temporal questions** (2026-10-21, qwen3.5:4b answering at temperature 0, graded by gpt-oss:120b at 0): 45% without dates, **62% with them**, 82 turned right and 25 wrong, sign test p = 0.00000003; the model also searched its memory half as often (17% to 9% of questions). Inhibition now costs 6 points (D1, 65%), so it stays off. Retrieval barely moved (turn hit 64%, session hit 92%): the dates change what summaries and facts say, not how they are found.
9. **Grader consistency** (2026-10-20). D2 was first graded at the grader profile's temperature (0.2): 66%, with 0 answers turned right and 11 wrong against D0, but 4 of those 11 answers were word for word the same as D0's. Graded twice at temperature 0, D0's own answers get the same verdict 198 times in 200 (72% both times, 71% when first graded). Graders now always run at temperature 0, and `--regrade` grades saved answers again without answering. Earlier comparisons used the 0.2 grader: treat differences of a few points among A3 to D1 with that in mind.
10. **Times on message lines don't help on LoCoMo** (D2). Showing each recalled message's time ("Mon 2023-05-08 13:57"), so the model could tell two messages of the same day apart, gave 68% against 72% for the same memory without times (2 turned right, 8 wrong, p = 0.11): no gain, leaning worse. LoCoMo has no questions about a value changed within one conversation, which is the case times are for, so the change was reverted; the compaction eval (COMPACTION_EVAL.md), which has such questions, will decide.
11. **The reranker finds more, answers the same** (runs R1, G1, G2, 2026-10-20). With the reranker, the message holding the answer reaches the notes in 78% of questions instead of 64% (section 5.5), multi-hop most, but the answers don't follow:

```text
answers (memory consolidated with dates)   without reranker   with reranker
qwen3.5:4b answering (D0 re-graded, R1)          72%               72%    14 turned right, 12 wrong, p = 0.85
gpt-oss:120b answering (G1, G2)                  74%               75%    11 turned right, 8 wrong, p = 0.65
median time per answer                       1.4 to 2.2 s       7 to 8 s
```

   The stronger model doesn't change the picture, so the extra evidence is mostly redundant with what consolidation already put in the notes (a dated fact or a summary). Built, off by default (`CERVEAU_RERANKER` empty in the app, `--reranker` in the runner).
12. **Good memory nearly closes the gap between models.** On raw messages, gpt-oss answering beat qwen3.5:4b by 5 points (A4 62% vs A3 57%, both re-graded at 0: 20 turned right, 11 wrong, p = 0.15); on memory consolidated with dates, by 2 (G1 74% vs D0 72%: 15 right, 11 wrong, p = 0.56). A trend, not a proof at plus or minus 6 points per score; and the memory was consolidated by the large model, so the local-first claim waits for the fully local run (F1).
13. **Why the reranker lost answers** (2026-10-21, R1's 12 answers right in D0 and wrong with the reranker). About a third are grader flips on answers that barely changed ("seven years", "Home Alone"): borderline answers can flip even at temperature 0. The rest come from reranking **within** the facts source: it pushed out the key fact ("Gina lost her job before opening her online store", "Jolene has a pet snake named Susie"), and three dates came out a day off once the dated fact was gone and the model worked "yesterday" out from message dates. The next test: rerank messages only.
14. **Multi-hop misses are mostly partial lists, and the grader is strict about them.** Of D0's 14 wrong multi-hop answers, most give some items of a list answer, not all: Joanna's screenplay emotions (four of five: relief, excitement, anxiety, worry; "hope" missing) was judged wrong. Part of the 40% is the grading rule; the memory often finds some pieces but not all. Published graders may be more generous (the Mem0 paper's grader is described as counting an answer that touches the same topic), which would make Cerveau's numbers conservative by comparison; an optional grader with that prompt would let both be reported.
15. **Multi-hop is the open gap.** It stays at 40 to 44% with every combination (either model, with or without the reranker), against 68% with the whole conversation in the prompt (A5). The reranker finds a first piece of evidence for 76% of multi-hop questions, but they need several pieces from different sessions together, and only about 42% of all their evidence turns reach the notes (turn recall). Candidates: a second search for the missing piece, one search per person or subject named in the question, more slots for multi-part questions.
16. **Rerank messages only** (2026-10-21): with the reranker leaving facts and summaries in their order, 73% against D0's 72% (15 turned right, 12 wrong, p = 0.70), about 4 s more per answer. The losses from reordering facts are gone, but no gain: the reranker stays off.
17. **Entity blocks** (2026-10-21, features/ENTITY_PROFILES.md step 1): a named entity's own facts, ranked by the question, under its card. Two problems found and fixed between version 1 and version 2: the model had given Caroline Melanie's names as aliases (5 of 557 entities carried another entity's name; the only pair of people), and the shared name dominated the ranking inside a block (now half the ranking uses the question without the name). The multi-hop baseline on all 282 multi-hop questions: 35% (qwen3.5:4b), 41% (gpt-oss:120b).

```text
all 282 multi-hop questions        without blocks   with blocks   turned right / wrong
qwen3.5:4b answering, version 1         35%             39%           22 / 12   p = 0.12
qwen3.5:4b answering, version 2         34%             38%           25 / 14   p = 0.11
gpt-oss:120b answering, version 1       41%             43%           24 / 20   p = 0.65
gpt-oss:120b answering, version 2       44%             46%           20 / 14   p = 0.39

the 200-question sample
qwen3.5:4b answering, version 2         70%             70%           10 / 9
gpt-oss:120b answering, version 2       72%             74%           12 / 10
```

   Multi-hop rises in all four comparisons, none significant alone; nothing else drops; qwen3.5:4b's answers take about 1 s more. The alias rule alone is neutral on scores (qwen3.5:4b on the 200: 72% to 70%, 1 turned right, 5 wrong). Retrieval on the multi-hop set without blocks: turn hit 56%, turn recall 30%.
18. **Where multi-hop answers are lost: the answer in the notes** (2026-10-22). `--notes` records what memory showed the model during each answer (recent conversations, recalled notes, search results) and asks the grader whether those notes hold all, part, or none of the gold answer. A stronger grader joined for these runs: **deepseek-v4-pro** (on three hard cases it judged like kimi-k3, at less than half the time, and rejected "a cup" for the gold answer "pots", which gpt-oss accepted), with gpt-oss kept as the second grader.

```text
all 282 multi-hop questions, graded by deepseek-v4-pro (gpt-oss in brackets)
                         without blocks   with blocks   turned right / wrong   sign test
qwen3.5:4b answering       45% (34%)       53% (38%)          31 / 8           p = 0.0003
gpt-oss:120b answering     52% (41%)       59% (45%)          31 / 13          p = 0.010

the gold answer in the notes the model was shown (graded by deepseek-v4-pro)
                         all    part   none
qwen3.5:4b, no blocks    117    143     22
qwen3.5:4b, blocks       141    126     15
gpt-oss:120b, no blocks  120    143     19
gpt-oss:120b, blocks     138    129     15

wrong answers, with blocks: notes held none / part / all of the answer
qwen3.5:4b      132 wrong:  11 none,  82 part,  39 all
gpt-oss:120b    116 wrong:  13 none,  78 part,  25 all
```

   - **The main loss is partial notes**: about 80 wrong answers per run had some pieces of a multi-part answer in the notes, not all. Memory finds nothing for only about 5% of questions; the model fails with the whole answer in front of it 25 (gpt-oss) to 39 (qwen3.5:4b) times.
   - **Entity blocks put the whole answer in the notes about 20 more times**, and with the stronger grader their gain is significant for both answering models (+8 and +7 points). gpt-oss, strict on partial lists, had hidden most of it (+4, p = 0.19).
   - **The graders differ by 11 to 15 points** on the same answers, most on partial lists: hand labels will tell which is right.
   - **Two free fixes for partial notes, tried and kept off** (runs N2 and N3, 2026-10-22, entity blocks on by default since). A larger block (20 facts instead of 10), and a second search: the block's best 3 facts look for their closest neighbors among the same entity's facts (the pieces of a list resemble each other more than the question), adding up to 6. Neither put the whole answer in the notes more often:

   ```text
   all 282 multi-hop questions, qwen3.5:4b answering, entity blocks on
                                      deepseek-v4-pro   gpt-oss   notes held all / part / none
   N1, block of 10 facts                    53%           38%          141 / 126 / 15
   N2, block of 20 facts                    56%           37%          141 / 127 / 14
   N3, 10 facts plus a second search        54%           40%          135 / 132 / 15

   against N1                         deepseek-v4-pro                 gpt-oss
   N2                                 22 right, 13 wrong, p = 0.18     14 right, 15 wrong, p = 1.0
   N3                                 14 right, 12 wrong, p = 0.85     14 right, 6 wrong, p = 0.12
   ```

     The small gains go in opposite directions under the two graders and none is significant. The missing pieces are not waiting just below the block's cut among the entity's facts; most likely they were never learned as facts, which only consolidation can change (more facts kept per session, a re-consolidation). The second search stays in the code, off (`expand_seeds = 0`).
   - **More facts per session, with larger blocks: kept, on by default since 2026-10-22** (runs N4, N5, GN4, GN5, B2). At consolidation, 239 of 562 fact replies (43%) had returned exactly 8 facts, the cap. With the cap at 16 (`max_facts`, a re-consolidation of the facts only: about 270 new fact calls plus their deduplication, summaries from the cache), sessions gave 8.8 facts on average instead of 6.7; half still gave 8 or fewer, and 7 reached 16.

```text
all 282 multi-hop questions          deepseek-v4-pro          gpt-oss    notes held all / part / none
qwen3.5:4b, 8 facts, block 10 (N1)         53%                  38%          141 / 126 / 15
qwen3.5:4b, 16 facts, block 10 (N4)        55%                  41%          127 / 142 / 13
qwen3.5:4b, 16 facts, block 20 (N5)        59%  39/22 p = 0.040  41%          153 / 117 / 12
gpt-oss:120b, 8 facts, block 10 (GN1)      59%                  45%          138 / 129 / 15
gpt-oss:120b, 16 facts, block 10 (GN4)     59%                  44%          129 / 139 / 14
gpt-oss:120b, 16 facts, block 20 (GN5)     63%  36/25 p = 0.20   47%          151 / 119 / 12

the 200-question sample, qwen3.5:4b      deepseek-v4-pro          gpt-oss
8 facts, block 10 (B1, regraded)           74%                  71%
16 facts, block 20 (B2)                    78%  21/11 p = 0.11   74%  16/11 p = 0.44
```

     Neither change works alone: 16 facts with blocks of 10 loses whole answers from the notes (the block shows a smaller share of each person's facts), and blocks of 20 with 8 facts find nothing more (N2). Together, the notes hold the whole answer about 12 more times with either model, and partial notes among wrong answers drop from 82 to 69 (qwen3.5:4b) and 78 to 67 (gpt-oss:120b). On the 200 questions nothing drops: single-hop 97 to 101 right, temporal 29 to 33, multi-hop 16 to 19 (deepseek-v4-pro), open-domain 4 to 3 (of 9). qwen3.5:4b's answers take about 0.6 s more (a longer prompt).
19. **Fully local: memory consolidated by qwen3.5:4b** (runs L1, L2, 2026-10-23). The 272 sessions took about 80 minutes on the laptop's GPU (RTX 3060 Laptop, 6 GB; about 3,000 calls, none failed), with 16 facts per session allowed. qwen3.5:4b answering on its own memory scores 64% on the 200 questions (deepseek-v4-pro; 62% by gpt-oss) against 78% on gpt-oss's memory (B2): single-hop holds (80% against 83%), multi-hop (36% against 76%) and temporal (44% against 73%) fall.
   - **The facts are the gap, not the summaries.** qwen's summaries are dated (244 of 272) and name the speakers. Its facts are few: 3.8 per session against 8.8, none at all for 82 of the 272 sessions; and 75% of them say "the user" instead of a name (gpt-oss: 7%), despite the prompt's rule for named speakers. A fact saying "the user" in a two-person conversation doesn't say who, and isn't linked to that person's entity block.
   - **Speaker names in code** (L2): when every user line starts with a name and a colon and there are at least two names, `parse_facts` replaces "the user" with the speaker of the lines a fact cites, if they're all one speaker's. No new extraction call (the cached replies are parsed again), but about 2,200 new deduplication checks on the renamed facts. 64% to 66% (14 turned right, 10 wrong, p = 0.54); multi-hop 36% to 44%, temporal 44% to 49%. Kept: small, and right either way.
   - **Next for the local memory:** the fact prompt itself, so a 4B model finds more facts (an example with named speakers), with a local re-consolidation of the facts only.
20. **Improving qwen3.5:4b's memory, and where it stopped** (2026-10-23 and 24). After L2, qwen3.5:4b's facts were still few (3.8 per session, none for 82 of 272 sessions) and unstable (at temperature 0, sessions that gave several facts one night gave none the next). What was tried:
   - **A rules prompt** (a clearer "durable" test, "most conversations hold several facts", a "who a fact is about" section; L3 and B3, consolidated at temperature 0): empty sessions 82 to 16 and 4.5 facts per session, but the answers barely moved: 66% to 68% locally (p = 0.70), and on gpt-oss's memory 78% to 75% (p = 0.23) and multi-hop 59% to 57% (p = 0.63). Reverted.
   - **A worked example in the prompt** made qwen3.5:4b return nothing; **no schema, plain text, temperature 0.7, a neutral system prompt** changed nothing.
   - **Five prompt styles** on 20 sessions (10 it had left empty, 10 it had handled), scored by deepseek-v4-pro against gpt-oss's facts for the same sessions (172 reference facts): the committed prompt covered 37; speaker by speaker ("per person") 96 and a fixed list of categories 94, with about 15% of their facts wrong. In one-person chats both added noise (statements copied in the first person, lost language, moods, duplicates).
   - **Thinking**: at temperature 0 it reasoned in circles to the limit and never answered; at 0.6 it finished, in about 2 minutes per call, and rescued the committed prompt on empty sessions (0 to 5 to 12 facts) without beating the other prompts. Without thinking, temperature 0.6 left the same sessions empty: thinking, not temperature, did it. Too slow for the app; stopped after 17 of 30 calls.
   - **Summaries** were not the problem: on 20 sessions judged blind by deepseek-v4-pro (shown the conversation's date), qwen3.5:4b left out 23 important details against gpt-oss's 22, with 9 errors against 5; it dates 241 of 272 summaries.
   - Conclusion: prompt and temperature don't lift a 4B model past its capacity; the next step was another model.
21. **gemma4:e4b replaces qwen3.5:4b as the local model** (2026-10-24). Google's Gemma 4 "E4B" computes like a 4.5B model (8B with its per-layer embeddings) and fits the 6 GB GPU at 16,384 tokens of context (about 4.7 GB in use, bge-m3 alongside), at about 95 tokens/s, twice qwen3.5:4b. On the same 20 sessions:

```text
                                  facts  empty  covered (of 172)  wrong  true extras
qwen3.5:4b, committed prompt        53      9          37             9        9
gemma4:e4b, committed prompt        99      0          83             4       13
gemma4:e4b, per person             163      0         131             1       36
gpt-oss:120b, committed prompt     167      0         135 (its own)  13       22
```

   In one-person chats gemma4:e4b with the committed prompt was clean (the language kept, no moods, nothing from a coding chat or a chat of questions), while "per person" copied first-person sentences and chat activity. So facts are now asked **speaker by speaker for conversations between several named people** (`GROUP_FACT_INSTRUCTIONS`, chosen by the caller, never guessed from the text; the runner's `--group-facts`) and with the committed prompt otherwise. qwen3.5:4b's runs stay here as history; it isn't used any more.
   - **L4**, everything by gemma4:e4b (temperature 0, facts speaker by speaker; about 80 minutes and 5,230 calls for the 272 sessions, 7.6 facts per session, none empty): 62%, temporal **31%**. **C4**, gemma4:e4b answering on gpt-oss's memory: 74% (B2, qwen3.5:4b answering the same memory: 78%, p = 0.14). The gap was gemma's memory, and in it the dates: only 53 of its 2,102 facts carried a year and 368 still said "last Friday" or "last weekend", which recall showed with the date they were learned, so "passed the interviews last Friday" was answered with that Sunday. gemma4:e4b never used `search_memory` in the 200 answers (qwen3.5:4b searched in 6 to 9%).
22. **Relative dates made absolute in code** (L5, 2026-10-24; [brain/dating.py](../brain/dating.py)). Each fact cites the lines that state it, and each line has its date, so "last Friday" in a fact learned on Sunday 22 October 2023 becomes "on Friday 20 October 2023"; spans are anchored as LoCoMo's answer key words them ("the weekend before 22 October 2023", "a few days before ..."); "recently" and a plain "on Friday" are left alone; summaries are dated from their conversation. English, Spanish, and French. No model call and no re-consolidation: the cached replies were parsed again (only the deduplication of the rewritten facts ran again, locally). 320 of gemma's 411 facts with relative words were made absolute (57 of the rest say "recently").

```text
the 200 questions, gemma4:e4b answering        all          temporal (45)
L4, gemma's memory                             62% (62%)       14
L5, the same memory with absolute dates        70% (68%)       29        22 turned right, 6 wrong, p = 0.004
C4, gpt-oss's memory                           74% (72%)       33        L5 against C4: p = 0.25
```

   On all 282 multi-hop questions, L5's memory gives 48% (gpt-oss's memory with qwen3.5:4b answering, N5: 59%); its notes hold the whole answer 135 times (qwen3.5:4b's memory: 114; gpt-oss's: 153), and 42 wrong answers had all of it in the notes: the remaining local gap is now as much in the answering (gemma4:e4b doesn't search) as in the memory.
23. **A sentence asking to search when the notes are partial** (runs L6, G6, 2026-10-24). gemma4:e4b searched in 1 of 200 answers, while most of its multi-hop misses had part of the answer in the notes; the tool instructions say to search "when your notes don't answer it". One more sentence ("If your notes hold only part of what the question asks (a list, several events, a when), or you're unsure, call search_memory before answering") made it search 23 times on the 282 multi-hop questions instead of 1, with the same score (48%, 12 turned right, 13 wrong) and the whole answer in the notes as often (132 against 135): the searches don't find what the memory doesn't hold. gpt-oss:120b: 13 to 16 searches, 64% to 62% (p = 0.61) by deepseek-v4-pro, 47% to 50% by gpt-oss (p = 0.21); on the 200, 83% to 82%. Nothing broke, nothing improved: reverted. G5, the run without the sentence, is also the first measure of gpt-oss:120b answering on the current defaults: 83% on the 200 (78% by gpt-oss), 64% on the 282 multi-hop.
24. **Cost.** The first consolidation of the 272 sessions with gpt-oss:120b took about 2 hours and about 3,600 calls (one for summaries, one for facts, one per new fact to deduplicate), and one session failed (conv-30 session 5, an empty reply). The consolidation with dates took about 55 minutes with the ten conversations side by side, and none failed (an unreadable reply is now asked once more). All cloud work up to 2026-10-19 used **1.7% of the weekly allowance** of an Ollama Cloud Pro plan ($20/month) and 1% of the 5-hour session allowance. Replies are cached on disk, so later rebuilds cost nothing but about 6 minutes of filing.
25. **The final grid and Ollama Cloud's limits** (2026-10-24). F0 to F4 (about 10,000 answers, each graded twice) took about 5 hours, F3, F4, and F0 side by side; with four lenient regrades on top (about 12,000 more grading calls), the Pro plan's 5-hour usage limit was reached and F2's lenient regrade failed with HTTP 429 ("Max is $100/month ... with no 5-hour or weekly caps"). It ran after the limit reset at 21:00 (76.1% / 76.1%); nothing else was lost.
26. **Improving gemma4:e4b's answering, after the freeze** (2026-10-25, outputs in `evals/locomo_data/dev/`; not part of the frozen final runs). F1q showed that a different small answerer gets about 4 points more from the same local memory (10 on multi-hop), so the answering side was the target.
   - **Diagnosis** (F1 with `--notes`, all 1,536 normal questions): of gemma4:e4b's 527 misses, 178 had the whole answer in its notes (reading), 216 part of it (partial recall), 133 none (recall). Temporal misses are half reading (48 of 106), multi-hop mostly partial (97 of 149). When the notes held the answer, it answered right 926 times of 1,104 (84%). Of the 48 temporal reading misses, 38 had the date only in a raw message line ("a pic from last Friday"), which the date rewrite of facts and summaries never touched.
   - **Method:** every change tried on the tuning conversations (0, 2, 4, 6, 8: 786 normal and 232 trick questions), compared with F1 on the same questions by sign test; a change is adopted only if normal questions gain (by both graders, or significantly by deepseek-v4-pro), trick questions lose no more than about 2 points, and the held-out conversations (1, 3, 5, 7, 9) confirm it. A, deepseek-v4-pro and gpt-oss:120b with the notes check; the others deepseek-v4-pro only (Ollama Cloud's weekly allowance was at 80%).

```text
tuning conversations, against F1         normal (786)                trick (232)                 verdict
A   a paragraph on using the notes       65.4% -> 68.2%, 31/9         80.6% -> 74.6%, 0/14       no: trick questions
    (lists, dates, inference)            p = 0.0007 (gpt-oss: +1.2)   p = 0.0001
A2  the list sentence alone              65.4% -> 67.0%, 27/14        80.6% -> 76.3%, 1/11       no: trick questions
                                         p = 0.06                     p = 0.006
B   a search with the question,          65.4% -> 67.0%, 42/29        80.6% -> 81.9%, 10/7       not proven (open-domain
    run by code before the answer        p = 0.15                     p = 0.63                   25% -> 34%); +0.4 s
C   relative dates in message notes      65.4% -> 66.0%, 13/8         80.6% -> 79.3%, 3/6        adopted: temporal
    annotated ("last Friday [Friday      temporal 65.0% -> 69.4%, 8/1, p = 0.04                  confirmed below
    19 May 2023]")
C   held-out conversations               temporal 69.5% -> 74.4%, 8/0, p = 0.008; trick 87.8% -> 86.4%, 0/3, p = 0.25
```

   - **What it shows:** every sentence added to the answer prompt that asks for more (all the items, the date worked out, an inference) also makes gemma4:e4b accept more false premises; A gained 8 points on multi-hop but lost 6 on trick questions without gaining one, and even the list sentence alone lost 4. Changes made in code, which hand the model better notes instead of asking it to try harder, don't have that cost: C fixed temporal answers ("last year" became "in 2022") on both halves, and B (more notes from a search) even gained a little on trick questions. **C is in the code since 2026-10-25** (`annotate_dates` in brain/dating.py, used by `memory_line`); Cerveau's tool eval is unchanged with it. **B and C together** (2026-10-25, C now in the code, so the run is B on top of it; deepseek-v4-pro only): on the tuning conversations normal questions 65.4% to 68.2% (48 turned right, 26 wrong, p = 0.014) and trick questions 80.6% to 82.8%; but **the held-out conversations didn't confirm it**: normal 65.2% to 66.9% (44/31, p = 0.17), trick questions 87.4% to 83.6% (8/16, p = 0.15, past the 2-point allowance). Consistent on both halves: multi-hop +5 (48% to 53%, 44% to 49%). Not consistent: open-domain (+10, then 0; few questions) and trick questions (+2, then minus 4; a search with the question likely brings back the other speaker's matching messages, which is what trick questions exploit). Not adopted; next ideas: search only for list questions, or keep only search results about the person the question names (the entity matching of entity blocks).

### 6.4 Detailed report: entity blocks and the answer in the notes (runs N0, N1, GN0, GN1)

Written 2026-10-22. Entity blocks gain 7 to 8 points on multi-hop with both answering models, and the remaining losses are mostly notes that hold only part of a multi-part answer.

#### What was run

| Run | Who answers | Entity blocks | Questions |
|---|---|---|---|
| **N0** | qwen3.5:4b (local) | off | all 282 multi-hop questions |
| **N1** | qwen3.5:4b | **on** | same 282 |
| **GN0** | gpt-oss:120b (cloud) | off | same 282 |
| **GN1** | gpt-oss:120b | **on** | same 282 |

All four share the same memory (consolidated once by gpt-oss with dates, cached), temperature 0, and the same questions in the same order. Each answer was graded twice: by **deepseek-v4-pro** (the new, stronger grader) and by **gpt-oss** (the grader of every earlier run). Each run also checked whether the notes shown to the model held **all**, **part** or **none** of the official answer.

#### Headline results

| Run | deepseek-v4-pro | gpt-oss grader | Median time per answer | Searched memory |
|---|---|---|---|---|
| N0, qwen, no blocks | 45% | 34% | 2.6 s | 4% |
| **N1, qwen, blocks** | **53%** | 38% | 3.3 s | 8% |
| GN0, gpt-oss, no blocks | 52% | 41% | 1.9 s | 5% |
| **GN1, gpt-oss, blocks** | **59%** | 45% | 2.0 s | 5% |

Blocks against no blocks, question by question (deepseek-v4-pro grading):

| | Turned right | Turned wrong | Net | Sign test |
|---|---|---|---|---|
| qwen (N0 to N1) | 31 | 8 | **+23** | **p = 0.0003** |
| gpt-oss (GN0 to GN1) | 31 | 13 | **+18** | **p = 0.010** |

Both gains are statistically significant. The chance of the qwen result being luck is about 3 in 10,000.

#### What the notes held

| Run | All of the answer | Part | None |
|---|---|---|---|
| N0, qwen, no blocks | 117 | 143 | 22 |
| N1, qwen, blocks | **141** | 126 | 15 |
| GN0, gpt-oss, no blocks | 120 | 143 | 19 |
| GN1, gpt-oss, blocks | **138** | 129 | 15 |

How blocks changed each question's notes (qwen runs):

- Part to **all**: 31 questions
- None to **all**: 6 questions
- None to part: 3 questions
- All to part: 12 questions (the block pushed something out)
- Unchanged: 228 questions

Blocks improved the notes for 40 questions and worsened them for 14, and none of those 14 turned a right answer wrong. The gpt-oss runs show the same pattern (37 improved, 19 worse, 1 turned wrong). Blocks almost never hurt.

Most of the gain isn't visible in these labels, though. Only 9 of qwen's 31 newly right answers came from notes moving up a label. The rest kept the same label (usually "part") but held more pieces, or held them in a more readable order: the block lists a person's facts by date.

#### Where the remaining misses are

Wrong answers with blocks on, by what the notes held:

| | qwen (132 wrong) | gpt-oss (116 wrong) |
|---|---|---|
| None of the answer in the notes (memory found nothing) | 11 | 13 |
| **Part of the answer** (some pieces, not all) | **82** | **78** |
| All of it (the model didn't use it) | 39 | 25 |

1. **Partial notes are the main problem**, about two thirds of the misses. Typical cases:
   - "What books has Melanie read?" (gold: *Nothing is Impossible*, *Charlotte's Web*): only *Charlotte's Web* is in memory.
   - "What events has Caroline participated in to help children?": youth-center volunteering is found, the school speech isn't.
2. **Memory finding nothing is rare:** 15 of 282 questions with blocks.
3. **Reading failures are moderate, and partly the grading.** When qwen's notes held the whole answer (141 questions), qwen got 102 right and gpt-oss 112. The 120B model reads better by about 10 questions out of 141. Some of qwen's "reading failures" are judgment calls: for "What are Melanie's pets' names?", qwen named all three correctly (Oliver, Bailey, Luna) but added "two dogs and one cat", which is wrong, so it was marked wrong.

#### Small model against big model, on the same notes

With blocks on, by deepseek-v4-pro:

| | Questions |
|---|---|
| Both right | 133 |
| gpt-oss right, qwen wrong | 33 |
| qwen right, gpt-oss wrong | 17 |
| Both wrong | 99 |

The big model wins 16 more questions net (59% against 53%): a real but modest edge for a model 30 times larger. 99 questions defeat both models, and those are mostly the partial-notes cases, where no model can answer.

#### The two graders

| Run | Graders agree | Only deepseek says right | Only gpt-oss says right |
|---|---|---|---|
| N0 | 89% | 31 | 1 |
| N1 | 82% | 48 | 4 |
| GN0 | 87% | 35 | 3 |
| GN1 | 84% | 42 | 2 |

- **deepseek-v4-pro is consistently more generous**, and their disagreements are almost one-sided. In N1, 47 of the 52 disagreements are deepseek saying right where gpt-oss says wrong.
- They disagree most on **partial lists** (26 cases in N1) and on **complete answers with extra detail** (21 cases). Example, "What is Caroline's identity?" (gold: "Transgender woman"): "Caroline is transgender..." gets right from deepseek and wrong from gpt-oss.
- **Why blocks looked weak before:** blocks produce fuller, more detailed answers, and the strict gpt-oss grader penalized exactly those. With deepseek grading, the gain is clear.

Which grader is "correct" is a judgment call that only hand labels can settle. Until then, both are reported. (Settled 2026-11-02: deepseek-v4-pro, section 6.7.)

#### By conversation

| Conversation | Questions | qwen (no blocks to blocks) | gpt-oss (no blocks to blocks) |
|---|---|---|---|
| conv-26 | 32 | 53% to 66% | 50% to 56% |
| conv-30 | 11 | 45% to 55% | 64% to 64% |
| conv-41 | 31 | 35% to 45% | 45% to 52% |
| conv-42 | 37 | 38% to 43% | 46% to 57% |
| conv-43 | 31 | 39% to 42% | 48% to 52% |
| conv-44 | 30 | 60% to 73% | 70% to 73% |
| conv-47 | 20 | 35% to 35% | 55% to 60% |
| conv-48 | 21 | 38% to 57% | 43% to 57% |
| conv-49 | 37 | 46% to 51% | 57% to 62% |
| conv-50 | 32 | 56% to 62% | 53% to 59% |

Blocks help in 18 of the 20 cells and leave the other 2 unchanged (conv-30 with gpt-oss, conv-47 with qwen); no conversation gets worse. The biggest gains come in conv-26 (Caroline and Melanie, also the one fixed by the alias rule) and conv-48.

#### Costs

- **Time:** blocks add about 0.7 s per answer for qwen (2.6 to 3.3 s), mostly from a longer prompt. For gpt-oss it's barely measurable (1.9 to 2.0 s).
- **Prompt size:** the notes grow by about 800 characters (3,500 to 4,300), roughly 200 tokens.
- **Memory searches:** qwen searched twice as often with blocks (4% to 8%). Larger notes about a person seem to prompt it to look for the rest.
- **No extra model calls** to build blocks, and **no re-consolidation.**

#### Conclusions

1. **Entity blocks work:** +8 points with qwen, +7 with gpt-oss, both significant, consistent across conversations, and almost never harmful. Worth switching on by default.
2. **The next multi-hop target is partial notes:** about 80 wrong answers per run had some pieces of a multi-part answer, not all. Possible fixes: a larger block for list-type questions ("what activities", "which books"), more facts kept per session at consolidation, or a second search for the missing items. The larger block and the second search were tried since (runs N2 and N3, item 18): no more whole answers in the notes. What worked was both together with consolidation: 16 facts per session and blocks of 20 (N5, GN5, B2), +6 and +4 points on multi-hop, +4 on the 200 questions, now the default.
3. **Model size matters less than memory:** the 120B model adds 6 points over the 4B one on the same notes, while blocks alone add 7 to 8 to either.
4. **The grader choice moves scores by 11 to 15 points** on multi-hop. Hand labels are now the most valuable missing piece before any number is published.

### 6.5 The final runs (2026-10-24, frozen commit d3237d2)

All 1,540 questions of categories 1 to 4 (1,536 can be graded: 4 name no evidence) plus the 446 adversarial ones, reported apart; the configuration frozen in LOCOMO_FINAL_RUNS.md (section 4b): gemma4:e4b as the local model, both memories consolidated at temperature 0 with facts speaker by speaker and dates made absolute in code, answers at temperature 0, every answer graded by deepseek-v4-pro and gpt-oss:120b at temperature 0. No run crashed or needed resuming. **Which grader is reported was provisional** until the hand labels chose (section 9 of LOCOMO_FINAL_RUNS.md: the grader agreeing best with them); they chose deepseek-v4-pro (section 6.7). Both are given throughout.

#### Scores (deepseek-v4-pro / gpt-oss:120b; n per category in brackets)

| Run | Setup | n | All (1 to 4) | single-hop | temporal | multi-hop | open-domain | adversarial | median s | searched |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 | whole conversation, gpt-oss answers | 1536 | **80.9%** / 77.1% (+/-2.0) | 89% / 88% (841) | 73% / 71% (321) | 69% / 55% (282) | 73% / 64% (92) | 49% / 46% (446) | 1.4 | 0% |
| F4 | cloud memory, gpt-oss answers | 1536 | **76.0%** / 69.5% (+/-2.1) | 82% / 78% (841) | 78% / 72% (321) | 64% / 48% (282) | 47% / 45% (92) | 79% / 70% (446) | 1.6 | 6% |
| F3 | local memory, gpt-oss answers | 1536 | **73.7%** / 69.5% (+/-2.2) | 80% / 79% (841) | 73% / 72% (321) | 61% / 46% (282) | 52% / 47% (92) | 81% / 74% (446) | 1.6 | 6% |
| F2 | cloud memory, gemma answers | 1536 | **66.9%** / 62.9% (+/-2.4) | 76% / 74% (841) | 66% / 63% (321) | 52% / 40% (282) | 34% / 29% (92) | 84% / 81% (446) | 2.0 | 0% |
| F1 | local memory, gemma answers (fully local) | 1536 | **65.3%** / 61.7% (+/-2.4) | 75% / 73% (841) | 67% / 64% (321) | 46% / 36% (282) | 27% / 30% (92) | 84% / 83% (446) | 2.0 | 0% |

#### Paired comparisons (categories 1 to 4)

| From | To | What changes | deepseek-v4-pro | gpt-oss:120b |
|---|---|---|---|---|
| F0 | F4 | memory against the whole conversation (gpt-oss answers) | 80.9% to 76.0%, 129 / 203, p = 5.78e-05 | 77.1% to 69.5%, 119 / 236, p = 5.23e-10 |
| F4 | F3 | local memory against cloud memory (gpt-oss answers) | 76.0% to 73.7%, 94 / 130, p = 0.0192 | 69.5% to 69.5%, 125 / 125, p = 1 |
| F2 | F1 | local memory against cloud memory (gemma answers) | 66.9% to 65.3%, 124 / 149, p = 0.146 | 62.9% to 61.7%, 136 / 155, p = 0.291 |
| F4 | F2 | gemma against gpt-oss answering (cloud memory) | 76.0% to 66.9%, 45 / 185, p = 2.4e-21 | 69.5% to 62.9%, 69 / 170, p = 5.13e-11 |
| F3 | F1 | gemma against gpt-oss answering (local memory) | 73.7% to 65.3%, 47 / 176, p = 9.83e-19 | 69.5% to 61.7%, 53 / 173, p = 4.79e-16 |
| F4 | F1 | fully local against the best case | 76.0% to 65.3%, 77 / 242, p = 5.24e-21 | 69.5% to 61.7%, 104 / 224, p = 3.04e-11 |

#### What the grid shows

1. **The memory system costs about 5 points against the whole conversation** (F4 against F0: 80.9% to 76.0% by deepseek-v4-pro, p = 0.00006), with no conversation in the prompt at all, only what memory recalls.
2. **gemma4:e4b builds memory almost as well as gpt-oss:120b**: with gpt-oss:120b answering, local memory costs 2.3 points by deepseek-v4-pro (p = 0.02) and nothing by gpt-oss (69.5% both); with gemma4:e4b answering, 1.6 points (p = 0.15).
3. **The local gap is the answerer**: on the same memory, gemma4:e4b answers 8 to 9 points below gpt-oss:120b (F3 to F1 and F4 to F2, p below 1e-15 by both graders), most on multi-hop (61% to 46% on local memory) and open-domain. It never used `search_memory` (gpt-oss:120b: 6% of answers).
4. **Fully local: 65.3% by deepseek-v4-pro, 61.7% by gpt-oss** (F1), on a 6 GB laptop GPU, 2.0 s per answer.
5. **Memory resists false premises far better than the whole conversation**: on the adversarial questions, 79% to 84% for the memory runs against 49% for F0. With the whole conversation in front of it, the model more often accepts a question's wrong premise (an event of the other speaker's, for instance).
6. **The tuning sample was easier**: every run scores 1.5 to 7.4 points higher on the 200 questions used for tuning than on the other 1,335:

| Run | 200 tuning questions | the other questions |
|---|---|---|
| F0 | 84.6% / 78.6% (201) | 80.3% / 76.9% (1335) |
| F4 | 78.1% / 73.6% (201) | 75.7% / 68.8% (1335) |
| F3 | 80.1% / 78.6% (201) | 72.7% / 68.1% (1335) |
| F2 | 68.2% / 65.2% (201) | 66.7% / 62.5% (1335) |
| F1 | 71.1% / 70.1% (201) | 64.4% / 60.4% (1335) |

   The full-set numbers are the ones to report; the 200-question results of sections 6.1 to 6.3 ran high.
7. **The graders agree on 92% to 95% of answers** (Cohen's kappa 0.80 to 0.85); deepseek-v4-pro is the more generous by 3.6 to 6.5 points, most on multi-hop and open-domain.

8. **A different small answerer on the same memory (F1q, asked for by the user):** qwen3.5:4b answering on gemma4:e4b's memory, F1's exact setup otherwise, all questions. Normal questions 69.1% (deepseek-v4-pro; 64.0% by gpt-oss:120b) against gemma4:e4b's 65.3% (61.7%): 136 turned right, 78 wrong, p = 0.00009 (gpt-oss: p = 0.018); multi-hop 46% to 56%, open-domain 27% to 40%, single-hop 75% to 78%, temporal 67% to 64%. Trick questions the other way: 83.9% to 78.0% (p = 0.003). qwen3.5:4b searches its memory in 8% of answers (gemma4:e4b never) and answers in 2.9 s (gemma4:e4b 2.0 s). The default stays gemma4:e4b (the user's decision, for its speed and caution); F1q sizes what better answering could recover from the same local memory: about 4 points, 10 on multi-hop.

#### A lenient grader, as published runs use

Published LoCoMo graders are more generous (the Mem0 paper's counts an answer that touches on the gold answer's topic). `--regrade --lenient` grades the same saved answers with a paraphrase of that rule, with no new answers:

| Run | strict (deepseek / gpt-oss) | lenient (deepseek / gpt-oss) |
|---|---|---|
| F0 | 80.9% / 77.1% | **84.8%** / 84.9% |
| F4 | 76.0% / 69.5% | **81.6%** / 80.7% |
| F3 | 73.7% / 69.5% | **79.8%** / 79.6% |
| F2 | 66.9% / 62.9% | **76.1%** / 76.1% |
| F1 | 65.3% / 61.7% | **75.6%** / 75.8% |

The rule alone adds 4 to 14 points, most to the small answerer's partial answers, and under it the two graders agree within one point. F2's lenient regrade ran after Ollama Cloud's 5-hour usage limit reset (item 25).

#### How this differs from the app

- **Facts speaker by speaker**: LoCoMo's sessions are two people talking, and Cerveau asks for facts speaker by speaker in conversations between several people; the app's one-person chats use the one-person prompt (with gemma4:e4b on 20 sessions it found 83 of 172 reference facts against 131).
- **Temperatures**: 0 for consolidation and answering; the app consolidates at 0.2 and chats at 1.0.
- **Speakers**: each LoCoMo turn is stored as "Name: text", which the app never does.

### 6.6 The final numbers: the grid with dated message notes (2026-10-25, commit 6d29a48)

The reported LoCoMo numbers (the user's decision, LOCOMO_FINAL_RUNS.md section 4c): the grid of section 6.5 rerun with relative dates annotated in message notes and search results (item 26), everything else unchanged; answered on 2026-10-25, graded after Ollama Cloud's weekly reset by deepseek-v4-pro and gpt-oss:120b at temperature 0. F0 has no memory, so it is reused from section 6.5. All 1,536 gradable questions of categories 1 to 4, plus the 446 trick questions apart; sign tests against the same run of section 6.5.

```text
                                        deepseek-v4-pro                        gpt-oss:120b                 trick      temporal
F1c fully local (gemma4:e4b both)       65.3% -> 66.9%, 38/14, p = 0.001       61.7% -> 62.4%, p = 0.23     84% -> 83%  67% -> 73%
F2c gpt-oss memory, gemma4:e4b answers  66.9% -> 67.6%, 35/25, p = 0.25        62.9% -> 63.9%, p = 0.09     84% -> 83%  66% -> 70%
F3c gemma4:e4b memory, gpt-oss answers  73.7% -> 74.7%, 62/46, p = 0.15        69.5% -> 69.8%, p = 0.70     81% -> 81%  73% -> 75%
F4c gpt-oss for both (best case)        76.0% -> 75.8%, 54/58, p = 0.78        69.5% -> 70.2%, p = 0.31     79% -> 80%  78% -> 77%
F0  the whole conversation (reused)     80.9%                                  77.1%                        49%
```

**Final, strict: fully local 66.9% (deepseek-v4-pro; 62.4% by gpt-oss:120b), best case 75.8% (70.2%), the whole conversation 80.9% (77.1%)** (after the empty replies were asked again, section 6.8: best case 76.2% (70.7%), F3c 75.2% (70.1%), the whole conversation 82.7% (78.6%)). The dated notes help where the small model answers: temporal +6 points fully local (significant overall, p = 0.001), +4 with gpt-oss's memory; with gpt-oss:120b answering nothing changes, since it already works dates out itself. Trick questions within a point. **Lenient** (`--regrade --lenient`, 2026-10-25, the two ends of the grid): fully local **76.4%** (deepseek-v4-pro; 76.6% gpt-oss:120b, from 75.6% / 75.8%), best case **82.0%** (80.8%, from 81.6% / 80.7%); F0 84.8% / 84.9% (reused); F2c and F3c were not regraded leniently. **The human audit on F1c** (2026-10-26, the final fully local run): groups rebuilt from F1c's own grades, the 509 verdicts of unchanged answers carried over from F1, and 324 answers checked anew with the same rule (155 judged wrong, 35 disagreements, and a fresh random 150 of the 1,299 judged right, 134 of them new). Normal questions: human-reviewed **65.8% (+/-2.6)**, 67.5% without the broken questions, against 66.9% by deepseek-v4-pro and 62.4% by gpt-oss:120b (judged right: 93.8% right, 6 of 112 wrongly accepted; disagreements: 72.9%; judged wrong: 10.6% right, 52 wrongly rejected, 30 broken questions). Trick questions: **91.1% (+/-3.9)** against 82.7% and 80.9%. As on F1, deepseek-v4-pro tracks the human standard closely (1.1 points above it here, 0.3 below on F1, both inside the interval) and gpt-oss:120b understates by about 3.5 points. Verdicts in `evals/locomo_data/dev/audit-f1c-verdicts-*.json`, groups in `audit-f1c-groups.json`; the review page `audit-review.html` now shows F1c's 833 checked answers.

### 6.7 Human review: the reported grader (2026-11-02)

**The rule, fixed before labeling** (LOCOMO_FINAL_RUNS.md section 3.3): the grader that agrees best with the user's own labels, by accuracy and Cohen's kappa, is reported.

**The labels.** The user labeled 100 answers drawn at random (seed 20261022) from three earlier runs, B2, N5 and GN5 (sections 6.3 and 6.4), without seeing the graders' verdicts, on a labeling page published privately on claude.ai. The user's rule: right if the answer gives what was asked with nothing false; for a list, most of it; declining when the conversation holds the answer is wrong; recognizing a broken premise is right. The user marked 79 right and 21 unsure. Each unsure answer was then checked against the conversation's evidence and a verdict suggested with its reason; the user accepted all 21 suggestions (5 right, 16 wrong). Final labels: **84 right, 16 wrong**. Labels in `evals/locomo_data/labels-human.json`, keys in `labels-key.json`.

```text
                  agrees with the user   Cohen's kappa   judged right   rejected what the user accepted   accepted what the user rejected
deepseek-v4-pro   85 of 100              0.60            69             15                                0
gpt-oss:120b      70 of 100              0.37            54             30                                0
by category (agreements): multi-hop (80) 70 / 55, single-hop (12) 9 / 9, temporal (8) 6 / 6
```

**Result: deepseek-v4-pro is the reported grader**; gpt-oss:120b stays beside it as the second grader in every table. It matches the audits: on F1 the audit sided with deepseek-v4-pro on 78 of 117 disagreements, and on both F1 and F1c deepseek-v4-pro's strict score fell inside the human-reviewed interval (section 6.6). Both graders are **stricter than the user, never more lenient**: every disagreement is an answer the user accepted and a grader rejected, mostly multi-hop lists that were incomplete with nothing false. The user's standard therefore sits between the strict and the lenient scores, which is why both are reported.

**Limits.**
- The sample is random within the three runs, not over the whole benchmark: 80 multi-hop, 12 single-hop, 8 temporal, no open-domain and no trick questions, because two of the runs answered multi-hop questions only. It measures the graders where they differ most (partial lists); it is not an accuracy estimate, and the user's 84% is not a score of any run.
- 9 questions appear twice (two runs' different answers to the same question): 100 answers, 91 distinct questions.
- One labeler. The 21 unsure answers were settled by accepting suggested verdicts, not labeled blind; counted either way (all right or all wrong), deepseek-v4-pro agrees more (69 against 54, or 84 against 73).
- The candidate second graders of section 3.3 (mistral-large-4, glm-5.3, deepseek-v4.1-flash) were not run.

**The user's review of the audit verdicts** (F1c, 833 answers, section 6.6; done 2026-11-02). Scope agreed with the user: the 58 verdicts that overrule both graders on normal questions (52 judged wrong by both that the audit calls right, 6 judged right by both that it calls wrong), the only ones that move the human-reviewed score away from deepseek-v4-pro's; the other verdicts agree with at least one grader. The user checked all 58 and 11 others (69) and **agreed with 68**. The one disagreement (F1c:1665, "What is a stress reliever for Evan?", an answer naming half the key's list with nothing false) the audit called wrong and the user right, since the question asks for one; the user's view replaces the audit's. **Final human-reviewed estimate, F1c: 66.3% (+/-2.4) on normal questions** (68.1% without the broken questions), against 66.9% by deepseek-v4-pro and 62.4% by gpt-oss:120b; trick questions 91.1% (+/-3.9), unchanged (90.7% after the trick recheck of section 6.8). The estimate script was rebuilt and reproduces the audit's own figures (65.8%, 67.5%, 91.1%) before the user's view is applied. Views in the Grading Desk's `labels` collection (ids `A~F1c:<i>`).

**How the numbers are reported:** three layers side by side, none replacing another: the automatic scores (strict by deepseek-v4-pro, with gpt-oss:120b beside it, and lenient, comparable with published runs); the human-reviewed score with its method; and the caveat that LoCoMo is not Cerveau's use case.

### 6.8 Two runner bugs found by the audits, and the F4c audit (2026-11-03)

**Empty replies.** gpt-oss:120b on Ollama Cloud sometimes returned nothing at all, and the runner saved the empty text as the answer: 42 in F0, 54 in F3c, 70 in F4c (69 and 75 in the first grid's F3 and F4), none where gemma4:e4b answers. Graders counted an empty normal answer wrong, and deepseek-v4-pro counted an empty trick answer as declining (right). **Fix:** `answer_question` now asks again on a fresh copy of the memory when the reply is empty, up to three times (`EMPTY_TRIES`, tests in `tests/test_locomo.py`). With the user's go-ahead, only the empty answers were asked again, at the same settings, with `--resume` on copies of the runs without them, then put back in their places and graded by both graders (strict, and leniently for F4c and F0). Most answered; 14 in F4c, 8 in F3c and 3 in F0 stayed empty after three tries. The old files are kept as `*-empty.jsonl`.

```text
                        strict, deepseek-v4-pro / gpt-oss:120b      trick (deepseek)   lenient
F4c best case           75.8 / 70.2  ->  76.2 / 70.7  (56 refilled)  80.0 -> 78.9      82.0 / 80.8 -> 82.6 / 81.3
F3c                     74.7 / 69.8  ->  75.2 / 70.1  (46 refilled)  80.9 -> 79.8      not run
F0 whole conversation   80.9 / 77.1  ->  82.7 / 78.6  (39 refilled)  48.7 -> 47.8      84.8 / 84.9 -> 86.8 / 87.0
```

The gain is smaller than the empty share suggested, since many of the new answers are wrong too. The ceiling gains most (+1.8): with the whole conversation in the prompt, empty replies were more often on questions it would have answered. The first grid (section 6.5) keeps its empty answers, as the record of that run.

**Dates in the answering prompt shown in the machine's timezone.** The runner stores LoCoMo's session times in UTC as given, but the answering prompt showed them in the engine's timezone, which was unset, so the machine's local one. A session at 12:37 am on 31 October appeared as the evening of 30 October (found by the user on F4c:1299). 17 of the 288 sessions are before 4 am, and 94 of the 1,540 normal questions have evidence in them (18 temporal). It affects every run alike (message dates and the "[date]" notes, not what consolidation wrote), and the Django app not at all, where one timezone is used throughout. **Fix:** the runner now answers in UTC (`in_utc`, with a test). **Decision (the user, 2026-11-03):** keep the current numbers and state the bug; its effect is expected to be well under a point, mostly on temporal questions. Reruns would undo the reviewed audits.

**The F4c audit** (the best case, gpt-oss:120b consolidates and answers), as for F1c: every answer both graders judged wrong, every disagreement, and a random 150 of those both judged right (seed 2025), checked against the conversation with the user's rule by six reviewers in parallel (744 answers), then the 56 refilled answers regrouped by their new grades (29 checked anew, 27 now judged right by both). The user reviewed all 37 verdicts that overrule both graders on normal questions (36 judged wrong by both that the audit calls right, 1 judged right by both that it calls wrong) and agreed with all 37. **Human-reviewed F4c: 76.4% on normal questions** (78.3% without the broken questions) against 76.2% by deepseek-v4-pro and 70.7% by gpt-oss:120b: deepseek-v4-pro tracks the human reading on the best case as on fully local, and the human-reviewed gap between them is about 10 points (66.3% against 76.4%). The interval by the same formula as F1c's (+/-1.1) understates the uncertainty, because the sampled group was right almost every time.

**Trick questions: an invented fact counts as wrong** (the user's ruling, 2026-11-03). Earlier verdicts counted a trick answer right whenever it did not take the trap. Now an answer that states, as true of the person asked, anything the conversation does not support (an invented reason, activity or item), or accepts the question's false setup as fact, is wrong. Every trick answer judged right was rechecked against the whole conversation (not only the evidence lines): F1c 2 of 93 flipped, F4c 10 of 82. Human-reviewed trick questions: **F1c 90.7%** (from 91.1%), **F4c 84.8%** counting its 10 empty replies as no answer (87.0% counting them as declining; from 89.2%). gpt-oss:120b, answering, invents a fact about the wrong person more often than gemma4:e4b, which more often declines. Two caveats on that comparison: F1c and F4c differ in memory as well as in the answering model (gemma's against gpt-oss's), and F4c's 84.8% counts its 10 empty replies as no answer (87.0% counting them as declines).

Verdicts: `evals/locomo_data/dev/audit-f4c-verdicts-*.json` (the trick recheck in `audit-f1c-verdicts-5-trick.json` and `audit-f4c-verdicts-8-trick.json`, applied over the others), groups in `audit-f4c-groups.json`; both audits on the Grading Desk with a run selector.

### 6.9 Changes to the app since the final numbers (2026-11-06 and 07)

The final LoCoMo numbers above were measured before four changes found by the personal chats eval (PERSONAL_CHATS_EVAL.md, sections 6c to 6e), and LoCoMo was not rerun after them:

- **The summary and fact language is worked out in code** and fixed in the reply schema (gemma4:e4b had written 37 of 77 summaries of the user's English chats in Portuguese). It changes the consolidation prompts, so a LoCoMo rerun would consolidate again instead of reading the cached replies.
- **Earlier replies in the notes are clipped and grouped last** under their own heading. LoCoMo's turns are all stored as the user's, so this doesn't reach it.
- **The fallback search**: a reply that only declines becomes a search on the question and a second answer. gemma4:e4b never searched on LoCoMo (item 23), so this could change its fully local score; on Cerveau's tool-use eval it gave 55 of 56 against 52 of 56.

- **Weekday names in the date rules** (2026-11-11): "this Saturday" and "on Friday" are now worked out into dates at consolidation, as "yesterday" already was. It changes the consolidation prompts; checked on the temporal questions of conversations 1, 3 and 5 (gemma4:e4b consolidating and answering, deepseek-v4-pro grading, `evals/locomo_data/dev/weekday/`): 76% before and after, 4 turned right and 4 wrong (p = 1.0).

On the user's own chats, the measure that matters for the app: fully local 44 of 60 against 47 of 60 with gpt-oss:120b consolidating and answering in the cloud, a gap of 3 where LoCoMo's is about 10 (PERSONAL_CHATS_EVAL.md 6e).

## 7. Cerveau's own evaluations (checks against regressions)

| Eval | Before (5 message slots) | After (10 slots) |
|---|---|---|
| Retrieval set: 202 written questions, 41 episodes | hit 92%, recall 81%, noise 1.6 | hit 92%, recall 82%, noise 1.7 |
| Personal case: 21 questions on 4 real conversations | hit 100%, recall 79%, noise 1.8 | hit 100%, recall 92%, noise 2.2 |
| Multi-turn: follow-up questions | 88% | 88% |
| Tool eval (qwen3.5:4b, real turns): absent topics, near misses, memory questions | | 8/8, 12/12, 24/26 |

"Noise" is the average number of memories recalled for questions that should recall nothing. The later changes don't touch these evals: dates change consolidation prompts (the written sets carry their own summaries), and the reranker, inhibition, and times are off.

## 8. Published numbers, for comparison

All published runs use **GPT-4o-mini** to answer and (where judged) to grade, on all questions of categories 1 to 4. Cerveau's final runs (section 6.6) use all 1,536 gradable questions, a local 4B model (or gpt-oss:120b) to answer, and deepseek-v4-pro to grade with its own strict and lenient prompts, so comparisons are rough. The scoreboard leaves these figures out for that reason; they are kept here, with the spread below, so the choice can be checked.

| System | Score | How measured | Source |
|---|---|---|---|
| Mem0 | 66.88% | LLM judge (GPT-4o-mini) | Mem0 paper |
| Zep | 65.99% | LLM judge, run by Mem0's authors | Mem0 paper |
| Zep | 75.14% (plus or minus 0.17) | LLM judge, Zep's own corrected run | Zep blog, rebuttal of the Mem0 paper |
| SYNAPSE | 80.7 (0 to 100 judge score); F1 40.5 | GPT-4o-mini answering; judge model not named | SYNAPSE, Findings of ACL 2026 (arXiv 2601.02744) |
| In SYNAPSE's re-runs | Mem0 57.1, Zep 49.0, MemoryOS 67.7, ENGRAM 77.6 (judge score); Zep F1 39.7 | same setup as SYNAPSE | SYNAPSE paper |
| OpenViking (ByteDance) | 80 to 83% with three agents (native memory: 24 to 57%) | method, answering and grading models not published | OpenViking repo and blog (vendor-reported) |

Notes:

- The same system scores differently depending on who runs it (Zep: 49.0, 66.0 or 75.1), so only runs made side by side really compare.
- An independent audit (Penfield Labs) reports that about 6.4% of LoCoMo's answer key is wrong and that LLM judges accept many deliberately wrong answers; no system can reach 100%, and judged scores are generous.
- Full-context ceilings reported elsewhere are around 80% (for example 80.6 in the Mnemis paper), matching Cerveau's A5 (80%).

## 9. Summary for comparison

### 9.0 The final runs: all questions (2026-10-24)

| Setup | Categories 1 to 4, strict (deepseek-v4-pro / gpt-oss:120b) | Lenient rule (deepseek-v4-pro / gpt-oss:120b) | Adversarial (deepseek-v4-pro) |
|---|---|---|---|
| Whole conversation in gpt-oss:120b's prompt (F0, ceiling) | 80.9% / 77.1% | 84.8% / 84.9% | 49% |
| **Cerveau, gpt-oss:120b consolidates and answers (F4, best case)** | **76.0% / 69.5%** | **81.6% / 80.7%** | 79% |
| Cerveau, gemma4:e4b's memory, gpt-oss:120b answers (F3) | 73.7% / 69.5% | 79.8% / 79.6% | 81% |
| Cerveau, gpt-oss:120b's memory, gemma4:e4b answers (F2) | 66.9% / 62.9% | 76.1% / 76.1% | 84% |
| **Cerveau fully local: gemma4:e4b consolidates and answers (F1)** | **65.3% / 61.7%** | **75.6% / 75.8%** | 84% |

Published, for orientation only (other models, other graders, other setups; section 8): Mem0 66.9% and Zep 66.0% with GPT-4o-mini as run in the Mem0 paper, Zep 75.1% in its own rerun. Only runs made side by side compare.

### 9.1 The 200-question sample (graded by gpt-oss:120b)

| Setup | Judged right |
|---|---|
| Whole conversation in a big model's prompt (ceiling, A5) | 80% |
| Cerveau, big cloud model answering, memory consolidated by a big cloud model with dates, reranker on (G2) | 75% |
| Cerveau, big cloud model answering, the same memory, entity blocks on (GB1, alias rule) | 74% |
| Cerveau, big cloud model answering, memory consolidated by a big cloud model with dates (G1) | 74% |
| **Cerveau, laptop model answering, memory consolidated with 16 facts per session, entity blocks of 20 (B2, the defaults since 2026-10-22)** | **74%** (78% by deepseek-v4-pro) |
| Cerveau, gemma4:e4b answering on the same memory (C4) | 72% (74% by deepseek-v4-pro) |
| **Cerveau fully local: gemma4:e4b consolidates and answers, dates made absolute (L5)** | **68%** (70% by deepseek-v4-pro) |
| Cerveau, laptop model answering, memory consolidated by a big cloud model with dates (D0) | 72% (71% at first grading) |
| Cerveau, laptop model answering, the same memory, reranker on (R1); messages only (RM) | 72%; 73% |
| Cerveau, laptop model answering, the same memory with the alias rule, entity blocks off and on (B0, B1) | 70%, 70% |
| Cerveau, laptop model answering, memory consolidated without dates (A10) | 66% (64% at first grading) |
| Cerveau, big cloud model answering from raw messages (A4) | 62% |
| Cerveau, laptop model answering from raw messages, temperature 0 (A8) | 60% |
| Cerveau, laptop model answering from raw messages, temperature 0.7, the chat default (A3, A7) | 56 to 57% |

### 9.2 Targeted sets (all questions of one category)

| Set | Setup | Judged right |
|---|---|---|
| All 321 temporal questions (gpt-oss grader) | laptop model, memory consolidated without dates (T0) | 45% |
| | laptop model, memory consolidated **with dates** (T1) | **62%** (p = 0.00000003) |
| All 282 multi-hop questions (deepseek-v4-pro grader; gpt-oss in brackets) | laptop model, no entity blocks (N0) | 45% (34%) |
| | laptop model, **entity blocks** (N1) | **53%** (38%), p = 0.0003 |
| | big cloud model, no entity blocks (GN0) | 52% (41%) |
| | big cloud model, **entity blocks** (GN1) | **59%** (45%), p = 0.010 |
| | laptop model, blocks of 20 facts (N2) | 56% (37%), p = 0.18 against N1 |
| | laptop model, blocks plus a second search (N3) | 54% (40%), p = 0.85 against N1 |
| | laptop model, **16 facts per session, blocks of 20** (N5, the defaults since 2026-10-22) | **59%** (41%), p = 0.040 against N1 |
| | fully local, gemma4:e4b consolidates and answers, dates made absolute (L5m) | 48% (37%) |
| | big cloud model, **16 facts per session, blocks of 20** (GN5) | **63%** (47%), p = 0.20 against GN1 |
| | whole conversation in a big model's prompt (A5's multi-hop, gpt-oss grader, 25-question sample) | 68% |

The two graders differ by 11 to 15 points on the same multi-hop answers, mostly on partial lists; hand labels will settle which to report (settled 2026-11-02: deepseek-v4-pro, section 6.7).

## 10. Next steps (as of 2026-10-24)

0. **Entity blocks** (section 6.3, items 17 and 18): on all 282 multi-hop questions, graded by deepseek-v4-pro, +8 points (qwen3.5:4b, p = 0.0003) and +7 (gpt-oss:120b, p = 0.010), nothing else drops; **on by default since 2026-10-22** (Cerveau's own evals unchanged with them). **Next for multi-hop: partial notes**, about 80 wrong answers per run had some pieces of a list, not all. A larger block and a second search didn't help (N2, N3); 16 facts per session at consolidation with blocks of 20 did (N5: 53% to 59%, p = 0.04; GN5: 59% to 63%; the 200 questions 74% to 78%), and are the defaults since 2026-10-22. About 70 wrong answers per run still have part of a list.
1. **Reranker**: built and measured (R1, G1, G2, and messages only, RM): retrieval 64% to 78% turn hit, answers unchanged with either answering model (72% to 72% with qwen3.5:4b, 74% to 75% with gpt-oss), 5.6 s more per answer; off by default. **Multi-hop is the open gap**: gathering every piece of a multi-session answer, not finding one.
2. **Event dates as a separate field on facts**: deferred. Writing the dates into summaries and facts recovered temporal questions (D0); a separate field is only worth a schema change if facts ever need filtering by date.
3. **Save answers as they come, and resume**: done (2026-10-22). Each answer is appended to `--out` as soon as it's graded; after a crash, the same command with `--resume` keeps what's saved and answers the rest.
4. **Brief answers**: one run with a short-answer instruction, as a separate configuration, for a token F1 comparable with SYNAPSE's 40.5.
5. **Graders**: hand-label 80 to 100 answers from earlier runs (blind to the graders' verdicts), run gpt-oss:120b and candidate second graders (mistral-large-4, glm-5.3, deepseek-v4.1-flash) on them, keep gpt-oss plus the best-agreeing grader from another model family. **Done for the two graders in use (2026-11-02, section 6.7):** the user labeled 100 answers; deepseek-v4-pro agrees with them on 85, gpt-oss:120b on 70, so deepseek-v4-pro is reported. The candidate second graders were not run.
6. **A fully local Cerveau**: qwen3.5:4b reached 68% (items 19 and 20); **gemma4:e4b**, the local model since 2026-10-24, reaches **70%** with dates made absolute in code (items 21 and 22), against 78% on gpt-oss's memory. Next: why gemma4:e4b never searches its memory (multi-hop, 42 wrong answers with the whole answer in the notes), then switch the app's chat model to gemma4:e4b.
7. **The final runs**: done (2026-10-24, section 6.5): F0 to F4 on all questions, graded by both graders, plus a lenient regrade. Left: the hand labels choose the reported grader. **To finish: the human review of the misses** (LOCOMO_FINAL_RUNS.md section 5.4): the 572 answers of F1 both graders judged wrong are checked (94 right by a human rule, 32 key errors; an upper bound on its own), and the symmetric half (150 judged right, all 127 disagreements) is done: the human-reviewed estimate is 65.6% (+/-2.3) on normal questions, matching deepseek-v4-pro's 65.3%, and 91.7% on trick questions. Then moved to F1c (section 6.6). The hand labels chose deepseek-v4-pro and the "Human review" section is written (section 6.7); left: the user's review of the 58 verdicts that overrule both graders on normal questions.
8. Chat temperature: 0 scored 60% vs 56 to 57% at 0.7 on raw messages; worth measuring as a default for the app.

Notes:

- **A bug found after these runs, fixed before the final runs:** for adversarial questions the dataset gives only `adversarial_answer`, the plausible wrong answer. The runner used it as the gold answer, so category 5 would have been graded backwards. No run above includes category 5, so no number here is affected. The runner now uses "Not mentioned in the conversation" as the gold answer and grades declining, or rejecting the false premise, as correct.
- Not used in these runs: the answer model was not told to answer briefly (so F1 is low), and consolidation prompts were Cerveau's own (written for one user and the assistant, with one added rule to name each speaker).

## 11. Reproducing

```text
python -m evals.locomo --download                       # dataset into evals/locomo_data/
python -m evals.locomo                                  # retrieval, raw messages
python -m evals.locomo --set inhibition=0.5             # retrieval with inhibition
python -m evals.locomo --consolidate gpt-oss-120b-cloud # retrieval, consolidated (first run builds the cache)
python -m evals.locomo --answer --limit 20 --temperature 0 --judge-model gpt-oss-120b-cloud
python -m evals.locomo --answer --limit 20 --temperature 0 --consolidate gpt-oss-120b-cloud --judge-model gpt-oss-120b-cloud
python -m evals.locomo --answer --limit 20 --chat-model gpt-oss-120b-cloud --full-context --judge-model gpt-oss-120b-cloud
python -m evals.locomo --answer --limit 20 --temperature 0 --consolidate gpt-oss-120b-cloud --reranker BAAI/bge-reranker-v2-m3 --judge-model gpt-oss-120b-cloud
python -m evals.locomo --regrade before.jsonl regraded.jsonl --judge-model gpt-oss-120b-cloud   # grade saved answers again
python -m evals.locomo --compare before.jsonl after.jsonl                                       # turned right, turned wrong, sign test
python -m evals.locomo --answer --categories 2 --temperature 0 --consolidate gpt-oss-120b-cloud --undated --judge-model gpt-oss-120b-cloud   # temporal, without dates
python -m evals.locomo --answer --categories 1 --temperature 0 --consolidate gpt-oss-120b-cloud --judge-model deepseek-v4-pro-cloud,gpt-oss-120b-cloud --notes --out n1.jsonl   # multi-hop, entity blocks (on by default; entity_block.enabled=false for N0), answer in the notes
python -m evals.locomo ... --out n1.jsonl --resume      # after a crash: keep the saved answers, answer the rest
python -m evals.locomo ... --set entity_block.max_items=20                                        # N2
python -m evals.locomo ... --set entity_block.expand_seeds=3 --set entity_block.expand_items=6    # N3
python -m evals.locomo ... --set max_facts=8 --set entity_block.max_items=10                      # N1's memory and blocks (16 and 20 are the defaults since N5)
python -m evals.locomo --answer --limit 20 --temperature 0 --consolidate gemma4-e4b-background --consolidation-temperature 0 --group-facts --chat-model gemma4-e4b --judge-model deepseek-v4-pro-cloud,gpt-oss-120b-cloud   # L5, fully local
```

Cloud runs need `OLLAMA_API_KEY` in `.env`. Answers of each run are saved as JSON lines with `--out`. Graders always run at temperature 0.

Sources: Mem0 figures as cited in the MemR3 paper (arXiv 2512.20237); Zep's response at blog.getzep.com ("Lies, damn lies, statistics: is Mem0 really SOTA in agent memory?"); SYNAPSE at arxiv.org/abs/2601.02744; OpenViking at github.com/volcengine/OpenViking and blog.openviking.ai; Penfield Labs LoCoMo audit on Substack.
