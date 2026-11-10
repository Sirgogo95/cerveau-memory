# Evaluation: recoverable compaction

**Status:** proposed
**Related:** [PIPELINE.md](PIPELINE.md) (running summary, step 10.5), [CERVEAU_LOCOMO_RESULTS.md](CERVEAU_LOCOMO_RESULTS.md), [LOCOMO_FINAL_RUNS.md](LOCOMO_FINAL_RUNS.md), the reranker (results file, section 10)

---

## 1. The claim to test

Coding agents and chat tools **compact** a conversation that outgrows the context window: they summarize the history and drop the original. Any detail the summary missed is gone for the rest of the session.

Cerveau never drops anything. Every message stays stored; the running summary is only what's loaded by default, and recall or `read_memory` can bring an original back when a question needs it.

**Claim:** *Cerveau makes compaction recoverable. Details that a summary loses can still be answered, because the originals come back through recall.*

This eval measures how often that's true, where it fails, and how much the reranker helps.

## 2. Questions the eval answers

1. How many planted details does the running summary keep, and how many does it lose?
2. Of the details the summary lost, how many does Cerveau still answer correctly? (the **recovery rate**, the core number)
3. How does Cerveau compare with plain compaction (summary only) and with no memory at all (recent window only)?
4. Which kinds of detail are lost and recovered: identifiers, numbers, names, decisions, preferences, dates, changed values?
5. Does recovery depend on how early the detail was said, or on how many similar details surround it?
6. Does the answer come from automatic recall, or does the model have to search?
7. Does the reranker raise recovery?

## 3. Precondition: recall within the current episode

Today `retrieve()` excludes the current episode: its messages are assumed to be in the prompt already. That was true before the running summary existed. Now, in a long episode, the oldest messages are **folded** into the summary and are no longer in the prompt, yet automatic recall can't reach them. Only the search tool, which skips just the messages in the prompt, can (checked in the code, 2026-10-20: `retrieve` takes `current_episode_id` and excludes the whole episode; `search_memory` takes `skip_message_ids`).

So this eval measures two variants of Cerveau (section 5), and its first finding may be a design change:

- **Folded messages become recallable**: messages of the current episode that the running summary covers and the recent window no longer shows are eligible for automatic recall, like messages from past episodes.
- Everything still in the recent window stays excluded, as now.

That change is small (`retrieve` excludes only the messages actually in the prompt, not the whole episode) and is what makes the claim true without relying on the model to search. It changes the app, not just the eval:

- Behind a setting, off by default, until measured.
- In ordinary long chats, recalled raw messages may repeat what the summary already says, costing prompt tokens. Measure the tokens per turn, and consider skipping a folded message when the summary already holds its content.
- The usual regression checks before turning it on: the retrieval set, the multi-turn eval, the tool eval, and the personal case.

A second gap matters here: **memory lines carry only the day** ("[M12 · Mon 2026-10-19 · user]"). Within one episode, every recalled message has the same day, so when recall brings back both "the table is `rx_items`" and the later "rename it to `rx_lines`", the model can't tell which came last. Memory lines need the time when they share the day of other lines (or of today). **Tried on LoCoMo on 2026-10-20** (every recalled message showing its time): 68% against 72% without (2 turned right, 8 wrong, p = 0.11), so no gain and reverted. LoCoMo has no question about a value changed within one conversation; this eval does, so it decides: run the changed-value questions with and without times (a condition setting, not a code change in the app until it helps).

## 4. The test data

### 4.1 Shape of a conversation

Each test conversation is **one long episode** (no time gaps, no topic split), so everything happens under one running summary:

A **turn** here is one exchange: a user message and Cerveau's reply, so 120 turns are 240 messages.

```
turns 1 to 10      setup: the topic is introduced
turns 11 to 90     planted details, spread out, among ordinary turns
turns 91 to 120    more ordinary turns, so early details are folded long before the questions
questions          asked at the end, each on its own copy of the conversation (below)
```

At the current settings, the prompt shows every message the running summary doesn't cover yet, which is between 20 and 30 messages (`recent_window` 20, folded `summary_chunk` 10 at a time). A detail said in the first 80 turns (160 messages) is folded and out of the window by the time questions start.

**Each question runs on its own copy** of the conversation as it stands after turn 120, as the LoCoMo runner does. Asked one after another in the same episode, earlier questions and answers would sit in the window for later ones, and the added messages could trigger folds that change the summary mid-test.

A second variant asks the same questions **in a new episode** after a time gap, to compare within-episode recovery with cross-episode recall. When the long episode closes, the app consolidates it (summaries, tags, facts), so that variant runs twice: without consolidation (raw messages only), and consolidated by the background model, to see how much recovery then comes from facts rather than from the original messages.

### 4.2 Planted details

About 8 details per conversation, of these kinds:

| Kind | Example | Why it's hard |
|---|---|---|
| Identifier | "the table is called `rx_items`" | Summaries paraphrase names away |
| Number | "the budget is 4,750 dollars" | Summaries round or drop numbers |
| Name | "the reviewer is Dr. Ana Ferreira" | Minor people get dropped |
| Decision | "we'll use UUIDs, not integer ids" | Survives as a topic, loses the choice |
| Preference | "I want replies under 100 words" | Mentioned once, easy to drop |
| Date | "the demo is on 14 November" | Relative or dropped |
| Changed value | "rename it to `rx_lines`" (after `rx_items`) | The summary may keep the old one |
| Aside | a one-line detail inside a long message on another topic | Low salience |

### 4.3 Distractors

Each conversation also includes look-alikes, to test precision, not just recall:

- Similar values that are **not** the answer ("the old budget was 4,500", "a table called `rx_audit` exists too").
- Questions about details **never said** ("what's the staging database called?"), where the right answer is "not mentioned".

### 4.4 Topics and languages

- Mixed topics: a software project, planning a trip, a renovation, a medical appointment schedule, a small business, a study plan.
- English, Spanish, and French, plus a few conversations that switch language, since the running summary is written in the conversation's language.

### 4.5 Generation

- A **scripted generator** places the details, distractors, and questions from templates, with a fixed seed, so the answer key is exact and reproducible.
- The **ordinary turns** (filler) are written once by a cloud model from a topic outline and cached on disk, so they read naturally but never change between runs.
- Every planted detail records the turn it was said in and its exact value; every question records which detail it asks about (or none).
- **Collision check:** the filler is string-checked against every planted value and distractor (numbers, names, identifiers, normalized); a filler turn that contains one is written again. Otherwise the cloud model may invent "the budget" or a table name that changes the answer key.
- Files live in `evals/compaction_cases/`, in the same style as the other eval cases.

Suggested size: **30 conversations × ~120 turns, ~8 details each, ~3 questions per detail kind present**, about 250 questions in all, plus about 40 never-said questions.

## 5. Conditions

All conditions use the same conversations, questions, chat model, memory budget, and temperature 0. Only the memory setup changes.

| Condition | What's in the prompt | Simulates |
|---|---|---|
| **W** Window only | The same window Cerveau shows (the uncovered messages, 20 to 30), no summary | No memory at all |
| **S** Summary only | Running summary + last 20 messages; no recall of the current episode; no tools | Classic compaction (as coding agents do it) |
| **C0** Cerveau today | Summary + window + recall of past episodes + tools (which can search the current episode) | The current design |
| **C1** Cerveau, folded recall | C0, plus automatic recall of folded messages (section 3) | The proposed change |
| **C2** C1 + reranker | C1, with the reranker reordering candidates | The planned reranker |
| **X0** Cross-episode, raw | The questions asked after a time gap, in a new episode, under C1; the long episode not consolidated | Recall from past episodes |
| **X1** Cross-episode, consolidated | X0, with the long episode consolidated by the background model | Recall with summaries and facts |
| **F** Full context | The whole conversation in the prompt (cloud model only, if it fits) | Ceiling |

Run C0 to C2 with **qwen3.5:4b** answering (the real setup), and once with **gpt-oss:120b** answering, to separate memory from model.

The running summary is produced by the same model and prompt as in the app, run synchronously in the eval instead of on the worker, and cached on disk like LoCoMo's consolidation replies.

## 6. Measurements

### 6.1 Per question

| Field | Meaning |
|---|---|
| `in_summary` | Does the running summary at question time contain the detail? (exact value check, then a cloud-model check for paraphrases) |
| `in_window` | Is the message holding the detail still in the recent window? (should be false for planted details) |
| `recalled` | Is the message holding the detail among the recalled memories? Needs a hook: `run_turn` gains an optional callback receiving the recalled memories (used only by evals), so the eval records exactly what the turn recalled rather than re-running retrieval |
| `searched` | Did the model call `search_memory` or `read_memory`, and did the result contain the detail? |
| `answer_correct` | Graded answer (section 7) |
| `used_old_value` | For changed values: did the answer give the value from before the change? |

### 6.2 Headline numbers

| Number | Definition |
|---|---|
| **Summary survival** | Share of details the summary kept (`in_summary`) |
| **Accuracy** | Share of questions answered correctly |
| **Recovery rate** | Among details **not** in the summary: share answered correctly. This is the claim |
| **Recall-driven recovery** | Among recovered details: share where automatic recall found the message (vs the model searching) |
| **False recall** | Never-said questions answered with an invented value |
| **Stale answers** | Changed values answered with the old value |

### 6.3 Breakdowns

- By detail kind (section 4.2).
- By age: turns between the detail and the question (buckets: 30 to 60, 60 to 90, over 90).
- By distractor load: zero, one, or several look-alikes in the conversation.
- By language.

## 7. Grading

- **Exact details** (identifiers, numbers, dates, names): checked by normalized string or number match against the planted value, so most grades need no model. Normalized: case, accents, code formatting (backticks), thousands separators ("4,750" and "4750"), and dates in the usual formats; a number written in words ("four thousand seven hundred fifty") falls back to a grader.
- **Paraphrasable details** (decisions, preferences): graded by the same graders chosen for the LoCoMo final runs, against the planted value.
- **Never-said questions**: correct if the answer declines or says it wasn't mentioned.
- **Changed values**: correct only with the new value; giving the old one counts as stale.

## 8. Hypotheses

Written down before running, so the results can confirm or refute them:

1. **S loses a lot**: summary survival under about 60%, worst for identifiers, numbers, and asides.
2. **W fails almost everything** planted early, as designed.
3. **C0 recovers only when the model searches**, so its recovery rate is moderate and most of it is tool-driven.
4. **C1 recovers far more**, mostly through automatic recall.
5. **C2 adds most on conversations with distractors**, where crowding is the problem.
6. **Changed values are the weakest kind** in every condition, since the old and new values look alike.
7. **X0 (cross-episode, raw) is close to C1**, since both rely on the same recall; X1 is higher, with facts holding some details.

## 9. Runner

```text
python -m evals.compaction --generate                  # builds cases into evals/compaction_cases/ (cached)
python -m evals.compaction --condition W,S,C0,C1        # answers and grades
python -m evals.compaction --condition C2 --reranker bge-reranker-v2-m3
python -m evals.compaction --condition F --chat-model gpt-oss-120b-cloud
python -m evals.compaction --compare c0.jsonl c1.jsonl  # turned right/wrong, sign test
```

Like the LoCoMo runner: each conversation's memory built once and copied per question, answers saved with `--out`, a 95% interval per number, `--set key=value` for settings. C0 and C1 can run before the reranker exists; C2 waits for it.

## 10. Report format

### 10.1 Main table

| Condition | Summary survival | Accuracy | Recovery rate | Recall-driven | False recall | Stale |
|---|---|---|---|---|---|---|
| W | | | | | | |
| S | | | | | | |
| C0 | | | | | | |
| C1 | | | | | | |
| C2 | | | | | | |
| X0 | | | | | | |
| X1 | | | | | | |
| F | | | | | | |

Summary survival is the same for every condition with a summary (the same summaries), so it's reported once; it frames how much there was to recover.

### 10.2 Also report

- Recovery rate by detail kind and by age.
- Tokens per turn and time per answer, per condition (recovery shouldn't cost a huge prompt).
- Ten examples of details lost by the summary and recovered, and ten not recovered, with why.

## 11. Risks and limits

- **Synthetic conversations.** Planted details are cleaner than real ones. Add a few real long episodes from the personal set once enough exist.
- **The summary prompt keeps "the user's own words"**, which may preserve more details than other compaction methods. That's fair to report (it's Cerveau's real behavior), but the S condition shouldn't be presented as how every tool compacts.
- **Small model limits.** If qwen3.5:4b answers wrongly with the detail recalled, that's a model error, not a memory one; `recalled` separates the two.
- **Budget.** The 8,000-token memory budget sets how much can be recovered at once; report it with the results.
- **Cost.** About 30 conversations, roughly 22 folds each: about 660 running-summary calls on qwen3.5:4b (an hour or two locally, cached once), plus the filler written once by a cloud model, plus grading.

## 12. Checklist

- [x] Time on memory lines tried on LoCoMo: no gain (68% vs 72%), reverted; to measure here on changed values
- [ ] Generator: details, distractors, questions, answer key, seeds, collision check
- [ ] Filler turns written by a cloud model, cached
- [ ] Synchronous running-summary folding in the eval, cached
- [ ] `in_summary` check (exact, then paraphrase)
- [ ] Recall hook on `run_turn`; one copy of the conversation per question
- [ ] Conditions W, S, C0
- [ ] Folded-message recall (section 3) behind a setting; condition C1
- [ ] Reranker; condition C2
- [ ] Cross-episode variants X0 and X1; full-context F
- [ ] Grading: exact checks, graders, never-said, stale
- [ ] Report with intervals, breakdowns, examples
- [ ] Results added to the docs, and a section for the blog post

## 13. Order of work

1. ~~Time on memory lines~~: tried on LoCoMo, no gain, reverted (section 3); measured again here, on changed values, as a setting.
2. **Folded-message recall** behind a setting, off by default (section 3), with the regression checks.
3. **Generator and runner**, with one copy per question and the recall hook; then conditions W, S, C0, C1.
4. **C2** once the reranker exists; X0, X1, and F alongside.

Review notes (2026-10-20): questions on copies, the shared-day problem in memory lines, the two cross-episode variants, the recall hook, the true window size, and the collision check were added after a review of the first draft.
