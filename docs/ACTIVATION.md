# Activation (v2): the plan, and how to know it helps

**Status:** built (2026-10-18): the eval, A, B, and C, all on (C at weight 0.05)
**Builds on:** [ARCHITECTURE.md: Activation](ARCHITECTURE.md#activation) and [The tag graph](ARCHITECTURE.md#the-tag-graph) (the design), [PIPELINE.md](PIPELINE.md) steps 5, 6 and 10, the tag signal ([GUIDE](GUIDE.md) 4.8)

This note records what was recommended before building, the same way [FACTS.md](FACTS.md) did. [ARCHITECTURE.md](ARCHITECTURE.md) remains the design. This note adds what Cerveau has learned since: tags and entities exist and are reliable enough to use, a tag signal measured as a small win, and every new signal has to earn its place on an eval.

---

## The problem activation solves

Each turn, retrieval starts from scratch: it looks only at the message just sent. Follow-up messages often mean little on their own:

```text
You (turn 1):  My knee is acting up again.
               -> recalls the earlier episode: a sprained ligament,
                  physio twice a week, "no running or basketball
                  for six weeks"
You (turn 2):  Should I still play on Saturday?
               -> "play on Saturday" resembles nothing about the knee;
                  that episode drops out of the prompt
```

The model still sees turn 1 (recent messages stay in the prompt), but not the memory turn 1 brought up. So it answers without knowing the orthopedist said no basketball.

Activation is the README's answer: what came up in turn 1 stays "in mind" for a few turns, fading, and keeps boosting the memories linked to it.

## Three ways to fix it, cheapest first

The README describes the full version (a graph of tags and entities, energy that spreads and decays). Two much simpler ideas could fix the same example, so they should be measured first:

| | Idea | Cost | Catches |
|---|---|---|---|
| A | **Conversation-aware query**: retrieve with the last 2 or 3 user messages embedded together, not the last one alone | One extra embedding per turn, no new tables | Follow-ups whose subject is in the previous messages |
| B | **Carry-over**: memories recalled in the last turns stay candidates, their score fading each turn (PIPELINE's "warm cache") | A small list on the episode | Exactly the example above |
| C | **Graph activation**: tags and entities matched in a turn get energy that decays over turns and hours and spreads to linked tags and entities; memories carrying active nodes get a boost | New tables (edges, activation), an update step every turn | B's cases, plus related memories never recalled yet (talking about the knee warms up `health`, which nudges the cholesterol results) |

**Decided (2026-10-18): build all three.** They cover different cases: A keeps the subject when a follow-up doesn't name it, B keeps the exact memories already recalled, and only C brings up related memories nothing has recalled yet and gives Cerveau a lasting "on my mind" (for the viewer, reinforcement, and prefetching later). The eval comes first anyway, not to decide whether to build each one but to set their weights and catch interference: B and C both make the old topic linger, and together they could drag it into the next subject. Each has its own weight, so any of them can be turned off with a config value; one that hurts topic changes even after tuning goes to weight 0, with the reason written down.

## Step 1: a multi-turn eval

The retrieval eval asks one question at a time, so it can't see any of this. A new case format: a short conversation, with the memories expected at a given turn.

```json
{
  "episodes": [ ...past conversations, same format as evals/cases... ],
  "conversations": [
    {
      "turns": [
        {"text": "My knee is acting up again.", "expect": ["knee"]},
        {"text": "Should I still play on Saturday?", "expect": ["knee", "knee:2"]}
      ]
    },
    {
      "turns": [
        {"text": "My knee is acting up again."},
        {"text": "Anyway, what's a good recipe for sancocho?", "expect": ["sancocho"], "reject": ["knee"]}
      ]
    }
  ]
}
```

The second conversation matters as much as the first: when the subject changes, what was in mind must let go, or every idea above turns into noise. About 30 conversations, mixed: follow-ups that need the earlier turn, topic changes that must not drag it along, and follow-ups across languages.

Measured the same way as today (hit rate, recall, precision, noise), per turn kind, with each idea alone and all together. **Rule:** follow-ups must improve while switches keep their score; weights are tuned until they do.

**Built (2026-10-18):** `python -m evals.turns` ([evals/turns.py](../evals/turns.py)), 37 conversations and 76 turns in [evals/turn_cases/](../evals/turn_cases/) (life, work, and the three languages), each run through `brain.recall`, the same function a real turn uses. Baseline:

```text
           hit   recall
opener     95%    89%
follow-up  62%    55%
switch    100%   100%
```

The problem is real: follow-ups miss almost 4 times in 10 ("what were my numbers last time?" after "I got a reminder about my retest" finds nothing). Switches are perfect because nothing carries over yet.

## Step 2: A and B

**A, built (2026-10-18):** `conversation_query` in [brain/recall.py](../brain/recall.py) ([GUIDE](GUIDE.md) 4.17). The search vector is the new message plus the previous user messages, the k-th most recent weighted `context_weight^k`:

```text
context_weight  follow-ups  switches dragging the old subject
0 (before)      62%         0 of 7
0.25 (shipped)  81%         1
0.5             91%         5
```

The number of earlier messages made no difference. The drags happen on messages that announce the switch ("By the way...", "Totally unrelated..."); detecting those phrases would fix this eval, but every switch in it was written with one, so the gain would be partly an artifact. Left out for now.

The plan for A, as written before building:

- **A** changes one line of the turn: the query embedding is the mean of the last few user messages, weighted toward the newest (the newest counts most so a change of subject wins quickly). The weight is the one thing to calibrate.
**B, built (2026-10-18):** carry-over in [brain/recall.py](../brain/recall.py) ([GUIDE](GUIDE.md) 4.17), saved as `Episode.carried`. As planned it dragged the old subject into the switches (8 rejected memories at its weakest setting), because nothing told it the subject had changed. Two conditions fixed that: carry only when the new message finds nothing scoring 0.6 or more on its own (a clear subject means it moved on), and when it is at least 0.42 similar to the message before (an unrelated question finds nothing too).

```text
                          follow-ups  dragged into switches
A alone                   81%         1
A + B, no conditions      88%         8
A + B, both (shipped)     84%         1
```

A small gain, at no measured cost; 0.42 rests on 7 switches, so personal cases should confirm it.

The plan for B, as written before building:

- **B** keeps, on the episode, the IDs of the memories recalled in the last turns with the turn they came from. Retrieval adds them as candidates with a `carried` signal that halves each turn (`w_carry × 0.5^turns`). A new episode starts empty. In the store contract it is two methods (save and load), with a small column on `Episode`.

Both are pure Python in brain/ plus a few lines in each store, and both are easy to turn off.

## Step 3: C, graph activation

**C, built (2026-10-18), on:** [brain/activation.py](../brain/activation.py) ([GUIDE](GUIDE.md) 4.18). Edges are computed from shared episodes when needed (no edge table, so no backfill and nothing stale), values are capped at 1 instead of the sigmoid, energy spreads only from nodes boosted this turn, and a message unlike the one before starts fresh (the same `subject_similarity` as carry-over, renamed from `carry_similarity`).

```text
                         follow-ups  drags  noise
activation alone, 0.2    81%         2      2.5
A + B (shipped)          84%         1      0.5
A + B + activation 0.05  88%         2      1.0
A + activation, no B     84%         2      1.0
```

Alone it nearly matches A. On top of A and B it adds one follow-up for one more drag and twice the noise on unrelated questions; faster fading and less spread changed nothing. By the rule above it would ship at weight 0; it was turned on anyway (2026-10-18, your decision) at `w_act = 0.05`. With it on, single questions gain recall (73% to 79%) and carry more noise (1.1 to 1.5 stray memories per unrelated question); `min_score` 0.54 would trade back the noise but also the follow-ups, so 0.53 stays. Longer personal conversations will tell whether the weight is right.

The plan for C, as written before building:

What the README describes, sized to what exists:

- **Nodes**: tags and entities (both already stored, with status).
- **Edges**: learned at consolidation from co-occurrence. Every pair of an episode's tags and entities gets +1; weight is Jaccard (`episodes with both / episodes with either`). One `NodeEdge` table, a backfill command for existing episodes.
- **State**: one `Activation` row per active node (`kind`, `id`, `value`, `updated_at`), rows under 0.01 deleted. Not audited (it changes every turn).
- **Each turn** (pure function in `brain/activation.py`, tested without Django):
  1. Decay lazily from `updated_at`: `value × 0.5^(hours / half_life)`, then `× (1 - δ)` per turn.
  2. Boost the nodes matched this turn: named entities (`named_entities`) and matched tags (`matched_tags`), by match strength.
  3. Spread once to neighbours, divided by each node's number of connections (the fan effect, so `work` doesn't warm up everything).
  4. Squash to 0 to 1.
- **Scoring**: `w_act × the highest activation among a memory's tags and entities`, next to `w_tag` and `w_lexical`. Re-ranking only at first, the way the tag signal works.
- **Seen in the viewer**: the lens and a small "On my mind" list show the active nodes and their values, so you can see why something came up.

Things the README mentions that are left out until measured: lateral inhibition, two or three spread iterations, "used" versus "exposed" memories (detecting what a reply actually used is unreliable with a 4B model), carry-over of messages across an episode boundary.

## Decisions

1. **Order** (decided): eval first, then A, B, and C, each measured alone and together, then tuned together.
2. **How long "in mind" lasts** (starting points, tuned on the eval): a topic not mentioned for about 3 turns is mostly gone; time half-life of a few hours, so coming back the next morning starts fresh.

## Slices

| Slice | What | Shippable alone |
|---|---|---|
| 1 | Multi-turn cases and runner (`python -m evals.turns`), baseline numbers | Done (2026-10-18) |
| 2 | A, measured and weighted | Done (2026-10-18): follow-ups 62% to 81% |
| 3 | B, measured and weighted, then A and B together | Done (2026-10-18): follow-ups 81% to 84% with two conditions |
| 4 | C: edges at consolidation and backfill | Replaced: edges computed from shared episodes when needed |
| 5 | C: activation state, the turn update, scoring, measured, then all three tuned together | Done (2026-10-18): measured, shipped at weight 0 |
| 6 | Lens and "On my mind" | Done (2026-10-18): an "On my mind" tab in the memory viewer; the lens shows what a question would bring to mind |
