# LoCoMo: plan for the final runs

**Status:** final runs done (2026-10-24 and 25; results in CERVEAU_LOCOMO_RESULTS.md sections 6.5 and 6.6); the hand labels chose deepseek-v4-pro as the reported grader (2026-11-02, section 6.7 there)
**Builds on:** [CERVEAU_LOCOMO_RESULTS.md](CERVEAU_LOCOMO_RESULTS.md) (runs A1 to A11, 2026-10-18 and 19)
**Purpose:** produce LoCoMo numbers solid enough to publish: fully local, best case, and the two mixed cases, on the full question set, graded reliably.

---

## 1. Goal

Answer two questions with numbers others can trust and reproduce:

1. **What does Cerveau score fully on a laptop?** The local model (gemma4:e4b since 2026-10-24) answers, and it also consolidates. This is the number the local-first pitch rests on.
2. **What does the same memory system score with strong models?** A cloud model answers, and a cloud model consolidates.

The two mixed cases separate how much comes from better memory and how much from a better answering model.

## 2. Phase 1: improve the system (before any final run)

Work in this order, measuring each change on the **tuning split** only (section 4).

| # | Change | Targets | Why |
|---|---|---|---|
| 1 | **Absolute dates at consolidation**: summaries and facts resolve "yesterday", "last week" against the session's date ("on 7 May 2023") | Temporal | **Done** (run D0): temporal 40% to 64%, overall 64% to 71%, p = 0.029 with the grader at 0.2; re-graded at 0, 66% to 72%, p = 0.08; **confirmed on all 321 temporal questions: 45% to 62%, p = 0.00000003** |
| 2 | **Event dates on facts**, separate from when the fact was learned, shown in the memory line | Temporal | **Deferred**: item 1 wrote the dates into the facts and recovered temporal; a separate field only if facts need filtering by date |
| 3 | **Reranker**: retrieve 40 to 50 candidates, rescore each against the question with a cross-encoder (bge-reranker-v2-m3 is a multilingual candidate), keep the best 10 | Crowding | 28 of 77 wrong answers had the evidence crowded out of the slots. **Built and measured 2026-10-20** (run R1): turn hit 64% to 78%, answers unchanged at 72%, 5.6 s more per answer; off by default (`--reranker`, `CERVEAU_RERANKER`) |
| 4 | **Inhibition** on consolidated memory, re-checked after 1 to 3 | Temporal, crowding | Rechecked after 1 (D1): 65% against 71% without it; stays off unless the reranker changes the picture |
| 5 | **Brief answers** for one run, to report a token F1 comparable with SYNAPSE (40.5) | F1 | Full-sentence answers keep F1 low |

Rules for this phase:

- Keep each change only if it helps on the tuning split, the multi-turn eval, and the written sets ([evals/](../evals/)), as for every earlier change.
- Re-check that nothing regresses on Cerveau's own evals (retrieval set, personal case, multi-turn, tool eval).
- Record each measurement in CERVEAU_LOCOMO_RESULTS.md, as for A1 to A11.

## 2b. Follow-up from the review (2026-10-21)

A review of runs R1, G1, G2 and the grader finding ("Cerveau LoCoMo review: reranker, graders, multi-hop", 2026-10-20) raised the points below. In order:

| # | Step | Why | Cost |
|---|---|---|---|
| R1 | **Done 2026-10-21.** A3 57%, A4 62%, A10 66% at 0; dates p = 0.08 (was 0.029), model gaps p = 0.15 and 0.56. Dates confirmed on all 321 temporal questions: 45% to 62%, p = 0.00000003. **Re-grade A3, A4 and A10 at temperature 0** (`--regrade`), then redo the dates comparison (D0 vs A10) and the model-gap comparison (A3 vs A4 against D0 vs G1) on the stricter grader | The dates result (p = 0.029) and the model gap were measured with the grader at 0.2 | About 15 minutes, cloud grading only |
| R2 | **Done 2026-10-21**: a third grader flips, the rest the reranker pushing key facts out of the facts source; next, rerank messages only. **Inspect R1's 12 answers that turned wrong**: did the reranker reorder the facts source and drop a dated fact? If so, test reranking messages only | Distinguishes a fixable interaction from plain redundancy | Minutes |
| R3 | **Done 2026-10-21**: mostly partial lists judged wrong (four of five emotions: wrong); add an optional generous, Mem0-style grader to report both. **Inspect a dozen wrong multi-hop answers** with list-shaped gold answers ("pottery, painting, camping, museum..."): does the grader count a partial list as wrong? | Part of the 40% may be the grading rule, not the memory | Minutes |
| R4 | **Done 2026-10-22** (`--notes`, graded by deepseek-v4-pro): the main multi-hop loss is notes holding part of a multi-part answer (about 80 wrong answers per run). **"Answer in the notes" metric**: the grader is given the recalled notes and the gold answer and says whether the notes contain it | Turn hit counts original messages; after consolidation the answer often lives in a fact or a summary, so turn hit misled about the reranker | Small runner change, one grading call per question |
| R5 | **Done 2026-10-22.** **Save and resume**: each answer appended to `--out` when graded; a restarted run skips the questions already there | Ollama and Docker stopped on their own five times in two days | Small runner change |
| R6 | **Done 2026-10-21**: baselines 35% (qwen3.5:4b) and 41% (gpt-oss:120b) on all 282. **A multi-hop tuning set**: all 282 multi-hop questions, since the 200-question sample holds only 25 (plus or minus 19 points) | No multi-hop change can be judged on 25 questions | About 15 minutes per run locally, plus grading |
| R7 | **Built and measured 2026-10-21; on by default since 2026-10-22**: multi-hop +8 (qwen3.5:4b) and +7 (gpt-oss:120b) graded by deepseek-v4-pro, nothing else drops. Designed in [features/ENTITY_PROFILES.md](features/ENTITY_PROFILES.md) (step 1). **A named person brings that person's facts as a block.** When a message names a person (named entities are already detected), recall adds that person's facts together, newest first, within a budget, instead of only the few most similar ones | Many multi-hop questions are about one person across sessions ("What activities has Melanie done with her family?"); their facts are learned, but only the top few by similarity are recalled | Engine change in recall and the store contract; no model calls at question time |
| R8 | Designed in [features/ENTITY_PROFILES.md](features/ENTITY_PROFILES.md) (step 2, only if step 1 isn't enough). **Entity profiles at consolidation.** Each person (and project, place) gets a running profile: a short summary plus dated facts grouped under them, updated when an episode is consolidated, and recalled as one card when the entity is named. Today an entity card is a name, a type, and aliases | Puts in one card what is otherwise scattered over many sessions; the most important entity in real use is the user. Needs a budget: a person in a 600-turn conversation may have 100+ facts, so the profile summarizes and keeps the newest | Larger: a schema change (profile on entities), a consolidation step, a recall card, the viewer |
| R9 | Then, if multi-hop is still short: iterative recall (a second search for what's missing, using the activation graph), question decomposition (one search per person named), more slots for multi-part questions ("both", "all", lists) | Further multi-hop ideas from the review, cheapest last | Each measured on the multi-hop set |
| R10 | **Split the GUIDE by part** (story, engine, adapters, Django app, evals) with an index page, keeping every section link working | About 270 KB is heavy on a phone | Mechanical but large; on its own |

Caveats for the write-up, from the review:

- Retrieval metrics count evidence turns, not whether the answer is in the notes (hence R4); many memory papers report only retrieval.
- The model gap (5 points on raw messages, 2 on consolidated memory) is suggestive, not proven, at plus or minus 6 points per score; and the memory in D0 and G1 was consolidated by the large cloud model, so the local-first claim needs the fully local run (F1).
- The grader's temperature is part of the method and is reported (0 from D2 on).

## 3. Phase 2: choose and validate the graders

### 3.1 Hand-labelled set

1. Take about 50 answers from earlier runs (A4, A8, A11), mixing categories.
2. Include the hard cases on purpose: dates in other formats or relative but correct, partial answers, near misses, longer answers with extra detail, and "I don't know".
3. Label each one CORRECT or WRONG yourself, against the gold answer, using the grader prompt's own rules.
4. Note the LoCoMo answers you believe are wrong in the answer key (an audit reports about 6.4%); keep them, but mark them.

Store the set in `evals/locomo_data/` (git-ignored with the dataset), with the answer text, the gold answer, and your label.

### 3.2 Candidate graders

| Grader | Family | Role |
|---|---|---|
| gpt-oss:120b (Ollama Cloud) | OpenAI | Current grader |
| mistral-large-4 (Ollama Cloud) | Mistral | Candidate second grader |
| glm-5.3 (Ollama Cloud) | Zhipu | Candidate second grader |
| deepseek-v4.1-flash (Ollama Cloud) | DeepSeek | Candidate second grader |
| qwen3.5:4b (local) | Qwen | Side note only: measured 13 to 16 points too lenient |

Confirm availability first: `curl https://ollama.com/api/tags`. The catalog changes often.

### 3.3 Selection

1. Run each candidate on the 50 hand-labelled answers with `--regrade` (at temperature 0, which every grader now uses: at 0.2, gpt-oss graded identical answers differently on two days; at 0 it agrees with itself on 198 of 200).
2. Measure agreement with your labels: accuracy, and Cohen's kappa.
3. **Final graders:** gpt-oss:120b plus the candidate with the best agreement, as long as it comes from a family other than the answering model's.
4. If gpt-oss:120b itself agrees with your labels clearly worse than a candidate, swap their roles.

**Result (2026-11-02, CERVEAU_LOCOMO_RESULTS.md section 6.7):** 100 hand labels instead of 50 (84 right, 16 wrong). deepseek-v4-pro agrees with them on 85 (kappa 0.60), gpt-oss:120b on 70 (kappa 0.37); both are only ever stricter than the labels. deepseek-v4-pro is reported, gpt-oss:120b given beside it; the candidates were not run. deepseek-v4-pro is not from the family of either answering model (gemma4:e4b, gpt-oss:120b), so the family rule holds.

Rule: no answer is graded only by a model from its own family. In the cloud-answering cells, the second grader is what protects against self-preference.

## 4. Freezing the configuration

Tuning and reporting on the same questions inflates the result. To avoid it:

- **Tuning split:** the existing 200-question sample (20 per conversation, seed 0). All of Phase 1 is measured here.
- **Report set:** all 1,540 questions in categories 1 to 4, plus category 5 (adversarial) reported separately.
- **Freeze before the final runs:** commit the configuration, then run nothing but the final runs on that commit. No threshold changes after seeing report-set numbers.
- The 200 tuning questions are part of the 1,540; also report the score on the other 1,340 alone, as a check that tuning didn't overfit.

## 4b. The frozen configuration (2026-10-24)

- **Engine:** `configs/default.toml` at the freeze commit (entity blocks on, 20 facts per block, 16 facts per session, the reranker and inhibition off), bge-m3 embeddings.
- **Local model:** **gemma4:e4b** (`gemma4-e4b` to answer, `gemma4-e4b-background` to consolidate), replacing qwen3.5:4b (CERVEAU_LOCOMO_RESULTS.md, items 20 to 22).
- **Consolidation, both models:** `--consolidation-temperature 0` and `--group-facts` (LoCoMo's sessions are two people talking: facts speaker by speaker), relative dates made absolute in code (brain/dating.py). The local memory is L5's (cached, `llm-gemma4-e4b.jsonl`); the cloud memory was rebuilt the same way on 2026-10-24 (`llm-gpt-oss-120b.jsonl`, temperature-0 keys).
- **Answering:** `--temperature 0`, the tool instructions as committed (a sentence asking to search more was tried and reverted, item 23).
- **Checked before freezing:** Cerveau's tool eval with gemma4:e4b (near misses 12 of 12 with and without tools, memory 24 of 26, as qwen3.5:4b; one vaguer "absent" answer); 50 adversarial questions with gemma4:e4b on its own memory: 86% (deepseek-v4-pro), 80% (gpt-oss), the grading fix working.
- **Graders:** deepseek-v4-pro and gpt-oss:120b at temperature 0, until the hand labels (section 3) choose; grading can be redone on saved answers (`--regrade`), so the runs don't wait for it.

## 4c. The final numbers, second version: with dated message notes (decided 2026-10-25)

The user decided that the reported LoCoMo numbers will be the grid rerun at commit 6d29a48, with relative dates annotated in message notes and search results (CERVEAU_LOCOMO_RESULTS.md item 26), everything else as in section 4b. The first grid (d3237d2) stays in the results file as the record of the frozen run before that change.

| Run | Answers | Memory | Status |
|---|---|---|---|
| F1c | gemma4:e4b | gemma4:e4b (cached) | answered 2026-10-25 (`evals/locomo_data/dev/f1c.jsonl`) |
| F2c | gemma4:e4b | gpt-oss:120b (cached, no new consolidation) | answered 2026-10-25 (`dev/f2c.jsonl`) |
| F3c | gpt-oss:120b | gemma4:e4b (cached) | answered 2026-10-25 with the user's go-ahead (`dev/f3c.jsonl`; gpt-oss:120b answering uses the cloud) |
| F4c | gpt-oss:120b | gpt-oss:120b (cached) | answered 2026-10-25 (`dev/f4c.jsonl`) |
| F0 | gpt-oss:120b, the whole conversation | none | reused from the first grid: without memory, the change can't affect it |

All four were answered on 2026-10-25 (1,982 answers each, no crash, no usage limit; the empty replies of F3c, F4c and F0 were asked again on 2026-11-03, CERVEAU_LOCOMO_RESULTS.md section 6.8). **Graded on 2026-10-25 after the reset** (strict, both graders): F1c 66.9% / 62.4%, F2c 67.6% / 63.9%, F3c 74.7% / 69.8%, F4c 75.8% / 70.2% (CERVEAU_LOCOMO_RESULTS.md section 6.6). Lenient, the two ends: F1c 76.4% / 76.6%, F4c 82.0% / 80.8% (F2c and F3c strict only). F1c to F4c are all graded locally by gemma4:e4b as placeholders only; after the reset every run is graded by deepseek-v4-pro and gpt-oss:120b (`--regrade ANSWERS OUT --judge-model deepseek-v4-pro-cloud,gpt-oss-120b-cloud`, and `--lenient` for the lenient table). F3c and F4c: the commands of section 5.3 with `--out dev/f3c.jsonl` and `dev/f4c.jsonl`, at the same commit.

## 5. The final runs

### 5.1 The grid

| Run | Answers | Consolidates | What it shows |
|---|---|---|---|
| **F1** | gemma4:e4b (local) | gemma4:e4b (local) | **Fully local**: the headline for a laptop |
| **F2** | gemma4:e4b (local) | gpt-oss:120b (cloud) | What stronger consolidation adds for a small answerer |
| **F3** | gpt-oss:120b (cloud) | gemma4:e4b (local) | What weak memory costs a strong answerer |
| **F4** | gpt-oss:120b (cloud) | gpt-oss:120b (cloud) | **Best case** of the memory system |
| **F0** | gpt-oss:120b (cloud) | none: whole conversation in the prompt | Ceiling, for reference |

F1 and F4 are the two runs that answer the goal; F2 and F3 are cheap once both memories exist (the consolidation replies are cached on disk) and make the comparison interpretable.

### 5.2 Held constant across all runs

- The same questions, the same order.
- Temperature 0 for answering.
- bge-m3 embeddings, the same embedding profile and `min_score`.
- The same engine configuration (the frozen commit), prompts, and memory budget.
- The same graders for every run.
- Each consolidation built once and reused (F1 and F3 share the local one; F2 and F4 share the cloud one).

Only the answering model and the consolidating model change.

### 5.3 Running them

- **Save as you go and resume** (built 2026-10-22, `--resume`): local Ollama and Docker each stopped on their own during these tests. Each answer is appended to `--out` when graded, and a restarted run skips the questions already there.
- **Local consolidation** of the 272 sessions on gemma4:e4b: built (L5's cache, about 80 minutes, none failed).
- **Cloud consolidation** for F2 and F4: rebuilt on 2026-10-24 with the same settings as the local one (group facts, temperature 0).
- Save every answer with `--out`, so grading can be redone without answering again.

Commands at the frozen commit:

```text
G="--judge-model deepseek-v4-pro-cloud,gpt-oss-120b-cloud --categories 1,2,3,4,5 --temperature 0"
L="--consolidate gemma4-e4b-background --consolidation-temperature 0 --group-facts"
C="--consolidate gpt-oss-120b-cloud --consolidation-temperature 0 --group-facts --workers 10"

python -m evals.locomo --answer $G $L --chat-model gemma4-e4b --out f1.jsonl
python -m evals.locomo --answer $G $C --chat-model gemma4-e4b --out f2.jsonl
python -m evals.locomo --answer $G $L --chat-model gpt-oss-120b-cloud --out f3.jsonl
python -m evals.locomo --answer $G $C --chat-model gpt-oss-120b-cloud --out f4.jsonl
python -m evals.locomo --answer $G --full-context --chat-model gpt-oss-120b-cloud --out f0.jsonl
```

After a crash, the same command with `--resume`. F3, F4 and F0 answer in the cloud and can run side by side; F1 and F2 share the GPU, one after the other. About 2,000 questions each (1,540 plus the adversarial ones): roughly 2 to 3 hours for each gemma4:e4b run and 1.5 for each cloud one.

Category 5 (adversarial) is included with `--categories 1,2,3,4,5`; it is reported on its own row and kept out of the "all (1-4)" row. Its gold answer is "Not mentioned in the conversation"; the grader counts declining, or rejecting the false premise, as correct.

## 5.4 Human review of the misses (to finish before the report is final)

An important part of the final report: the automatic graders are stricter than a person, and LoCoMo's answer key has errors, so the report also gives a human-reviewed score.

**Done (2026-10-24):** every answer of F1 (fully local) that both graders judged wrong, 572 of them, was checked against the conversation's evidence lines with the user's rule (right: what the key says in any words, most of a list with nothing false, a date pointing to the same day, a premise correctly rejected, or for a trick question a true fact about the person asked instead of the trap; key error: the question or key itself is wrong). 99 by Claude, the rest by helpers given the same rule and those 99 as calibration, spot-checked. Result: 446 really wrong, 94 right by the rule (51 normal questions, 43 trick questions), 32 key errors. **This half alone is one-directional:** checking only answers judged wrong can only raise the score, so the 65.3% to 68.6% it gives on the normal questions (70.0% without the broken questions; trick questions 84% to 93.5%) is an upper bound, not an estimate. Graders also accept wrong answers (the Penfield audit found LLM judges accepting deliberately wrong ones).

**The symmetric half (started 2026-10-24):** the same check on a random 150 of the 1,283 answers both graders judged right, and on all 127 where they disagree (`audit-groups.json` says which group each answer is in). The human-reviewed score is then estimated by group: each group's size times its human-right rate, summed, with an interval. Only that estimate goes into the report.

**Result of the symmetric audit (2026-10-24):** on the normal questions (1,536) the human-reviewed estimate is **65.6% (+/-2.3)**, 67.0% without the broken questions, against 65.3% by deepseek-v4-pro and 61.7% by gpt-oss:120b. By group: of 916 answers both graders judged right, 105 checked, 95.2% right (5 wrongly accepted, mostly a false extra detail such as a wrong date or the other person's words); of 118 where they disagreed, all checked, 72.0% right; of 502 both judged wrong, all checked, 10.2% right (51 wrongly rejected) and 30 broken questions. The wrongly rejected and wrongly accepted answers nearly cancel out: **deepseek-v4-pro grades the normal questions almost exactly like the user's standard**, and gpt-oss:120b understates by about 4 points. On the disagreements the audit sides with deepseek-v4-pro 78 times out of 117 and with gpt-oss:120b 39 times (a strong hint for the grader choice; the user's labels on the random 100 decide). Trick questions (446): human-reviewed **91.7% (+/-3.3)** against 83.9% and 82.7%, the definition difference (a true fact about the right person counts for the user, not for the graders). The review page has all 849 checked answers with a group filter.

**The human audit on F1c** (2026-10-26, the final fully local run): groups rebuilt from F1c's own grades, the 509 verdicts of unchanged answers carried over from F1, and 324 answers checked anew with the same rule (155 judged wrong, 35 disagreements, and a fresh random 150 of the 1,299 judged right, 134 of them new). Normal questions: human-reviewed **65.8% (+/-2.6)**, 67.5% without the broken questions, against 66.9% by deepseek-v4-pro and 62.4% by gpt-oss:120b (judged right: 93.8% right, 6 of 112 wrongly accepted; disagreements: 72.9%; judged wrong: 10.6% right, 52 wrongly rejected, 30 broken questions). Trick questions: **91.1% (+/-3.9)** against 82.7% and 80.9%. As on F1, deepseek-v4-pro tracks the human standard closely (1.1 points above it here, 0.3 below on F1, both inside the interval) and gpt-oss:120b understates by about 3.5 points. Verdicts in `evals/locomo_data/dev/audit-f1c-verdicts-*.json`, groups in `audit-f1c-groups.json`; the review page `audit-review.html` now shows F1c's 833 checked answers.

Files: `evals/locomo_data/final/audit-verdicts-*.json`, the page `evals/locomo_data/audit-review.html` (git-ignored with the dataset).

**To do:**
- [x] Finish the symmetric half (150 judged right, 127 disagreements) and compute the estimate by group (65.6% +/-2.3 on normal questions, 91.7% on trick questions).
- [x] The user reviews the verdicts (on the Grading Desk, F1c's 833); the user's view replaces Claude's where they differ. Scope (2026-11-02): the 58 verdicts that overrule both graders on normal questions, the only ones that move the score. Checked 69 (all 58 and 11 others), disagreed with 1; human-reviewed F1c 66.3% (+/-2.4) on normal questions, 91.1% on trick questions (CERVEAU_LOCOMO_RESULTS.md section 6.7).
- [x] The user's hand labels choose the reported grader (section 3.3): deepseek-v4-pro, 85 of 100 against 70 (2026-11-02). Measured on the **random 100** only (a symmetric sample); the 699 answers at least one grader judged wrong are for studying where the graders disagree, not for measuring which is more accurate.
- [x] Write a "Human review" section in CERVEAU_LOCOMO_RESULTS.md (section 6.7) and a short line on the scoreboard, with the method stated.
- [x] The same review for F4c (best case), 2026-11-03: 744 answers checked, the user agreed with all 37 verdicts that overrule both graders; human-reviewed 76.4% on normal questions against 76.2% by deepseek-v4-pro, so the human-reviewed gap between local and cloud is about 10 points. On the way: empty replies from gpt-oss:120b asked again (F0 82.7%, F3c 75.2%, F4c 76.2%), dates shown in UTC when answering, and invented facts on trick questions counted wrong (CERVEAU_LOCOMO_RESULTS.md section 6.8).

**How to report it (agreed 2026-10-24):**
- Three layers side by side, none replacing another: the automatic scores (strict, and lenient by a rule written to match the published papers' description; other systems' published scores still aren't directly comparable: other judges, prompts and answering models); the human-reviewed score, with the method; and the caveat that LoCoMo is not the use case (two friends talking, synthetic conversations, trick questions built on mixing up the speakers, about 6% of the key wrong, scores moving 10 points with the grader's rule). Like a school test: not the truest measure, but the shared exam that makes comparison with Mem0 and Zep possible.
- Say who reviewed: "verdicts by an AI following a fixed rule, reviewed by the author", with the share the author checked and the disagreement rate. State the rule next to the number: it is closer to the lenient grader than to the strict one (most of a list with nothing false counts).
- Supported by the symmetric audit: deepseek-v4-pro matches the human-reviewed score on normal questions (65.3% against 65.6% +/-2.3), so the review validates it rather than raising the score; gpt-oss:120b understates by about 4 points. Not supported: that the graders understate Cerveau in general. Supported too: on trick questions the difference is a definition (a true fact instead of the trap), not accuracy; the key has real errors; and gemma4:e4b rarely invents: most real misses are "I don't remember" when the answer existed (about 40% in the first 77), the safe failure for a personal assistant, then partial lists (about 17%), dates (13%), wrong or vague facts (22%).
- Not supported, so not claimed: that it "mostly" gives true or partial answers (partial is one part of the misses), or that a prompt making it search would recover the rest (tried, item 23: 23 searches instead of 1, no gain; better recall or a stronger answering model are the likelier levers).
- Whether users get good value is answered by the personal-chats eval (PERSONAL_CHATS_EVAL.md), not by LoCoMo.

## 6. What to record for each run

- Commit hash of the frozen configuration.
- Model names and versions (Ollama digests) for answering, consolidating, embedding, and grading.
- Date, hardware, and the Ollama Cloud plan's usage.
- Number of questions, and any that couldn't run (and why).
- Failed consolidation sessions.
- Median and 90th-percentile time per answer.

## 7. What to report

### 7.1 Main table

Per run: judged right by each grader, both graders' average, and the 95% confidence interval.

| Run | Grader 1 | Grader 2 | Average | 95% CI | single-hop | temporal | multi-hop | open-domain | adversarial |
|---|---|---|---|---|---|---|---|---|---|
| F1 | | | | | | | | | |
| F2 | | | | | | | | | |
| F3 | | | | | | | | | |
| F4 | | | | | | | | | |
| F0 | | | | | | | | | |

Confidence interval for a share p on n questions: about `1.96 x sqrt(p(1-p)/n)`. On 1,540 questions at 65%, that's about ±2.4 points (it was ±6.6 on 200).

### 7.2 Grader reliability

- Each grader's agreement with the hand labels (accuracy and kappa): 100 labels, deepseek-v4-pro 85% (0.60), gpt-oss:120b 70% (0.37).
- Agreement between the two graders on every run (share of identical verdicts, and kappa).
- A sample of disagreements, hand-checked: which grader was right.

### 7.3 Paired comparisons

For F1 vs F2, F3 vs F4, and F2 vs F4: answers turned right and turned wrong on the same questions, with a sign test, so a difference is called real only when it is.

### 7.4 Also report

- Score on the 1,340 questions outside the tuning split.
- Token F1, and, for the brief-answer run, F1 next to SYNAPSE's.
- Token and time costs per answer.
- The comparison table with published systems (CERVEAU_LOCOMO_RESULTS.md section 8), with the setup differences stated plainly.

## 8. How to word the results

- **The headline is F1**, fully local, with its interval: "a 4B model on a laptop, with memory it consolidated itself".
- **Say "in the range of"**, not "beats" or "matches", unless the interval clearly supports it.
- **State the setup differences** with published numbers: different answering and grading models, a different grader prompt.
- **Mention the answer-key errors** and that judged scores run generous, as in section 8 of the results file.
- **Publish the losing cells too.** If fully local is lower than the cloud-consolidated run, that is the finding.

## 9. Checklist

- [x] Phase 1 changes, each measured on the tuning split (items 1, 3 and 4; 2 deferred, 5 not run)
- [x] Follow-up from the review (section 2b): re-grades, inspections, answer-in-notes metric, save and resume, multi-hop set, person fact blocks (R1 to R7; entity profiles R8 to R10 not needed for the final runs)
- [x] Own evals re-checked after Phase 1
- [x] Runner: category 5 included and reported separately (gold answer fixed, 2026-10-19)
- [x] Runner: memory built once per conversation and copied per question; cloud consolidation side by side (`--workers`); tool searches counted; 95% intervals and 90th-percentile time in the report; `--compare` with a sign test
- [x] Hand-labelled answers (100, 2026-11-02)
- [x] Candidate graders run on them (2026-11-02: deepseek-v4-pro agrees with the hand labels on 85 of 100, gpt-oss:120b on 70; deepseek-v4-pro reported, gpt-oss:120b second)
- [x] Configuration frozen and committed (2026-10-24, section 4b)
- [x] Local consolidation built (gemma4:e4b, L5)
- [x] Cloud consolidation rebuilt (group facts, temperature 0, 2026-10-24)
- [x] Own evals re-checked with gemma4:e4b (tool eval) and an adversarial check (50 questions)
- [x] F0 to F4 answered and graded by both graders (2026-10-24)
- [x] Confidence intervals, agreement, paired comparisons, the tuning sample against the rest
- [x] A lenient regrade (`--regrade --lenient`) of all five runs
- [x] Results written up in CERVEAU_LOCOMO_RESULTS.md (section 6.5)
- [x] Reproduction commands updated (section 5.3)
- [x] Human review of the misses (section 5.4): the user's review of the verdicts, then a "Human review" section in the results file and on the scoreboard (2026-11-02)
