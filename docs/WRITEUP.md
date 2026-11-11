# Giving a local LLM a long-term memory, on a 6 GB laptop GPU

Chat assistants forget. Each conversation starts empty, or with a short list of saved notes. I wanted the opposite: one continuous conversation with an assistant that remembers what I told it weeks ago, runs entirely on my laptop, and lets me see and delete everything it knows.

That's Cerveau (French for brain). This is what I built, how I measured it, and what I learned about small models along the way. The code is at [github.com/Sirgogo95/cerveau-memory](https://github.com/Sirgogo95/cerveau-memory).

## What it does

There is no list of chats. You talk to Cerveau in one conversation, and it splits that conversation into episodes on its own, when you go quiet for a couple of hours or change subject.

When an episode ends, the model "sleeps on it": it writes a summary, extracts facts ("the user's sister Lena is getting married in Lisbon on June 14"), people, places and topics, and turns "next Saturday" into a real date. Weeks later, when you ask something, Cerveau searches all of that by meaning, by names, by exact words, and by what has been on your mind lately, and puts the few memories that matter into the prompt, within a fixed token budget. If the notes aren't enough, the model can search its memory itself.

Everything runs locally: gemma4:e4b (about 4B effective parameters) through Ollama, bge-m3 for embeddings, Postgres with pgvector, a Django app. On an RTX 3060 Laptop GPU with 6 GB, both models stay loaded, the first word of a reply comes in about 3 seconds, and consolidating a conversation takes 10 to 30 seconds in the background. A memory viewer shows every summary and fact, and why a memory was recalled; you can edit or forget anything, and a purge command really deletes it.

## How I measured it

A memory system is easy to demo and hard to measure. I used two tests.

**LoCoMo**, a widely used benchmark for long-term conversational memory: ten long conversations between two people, 1,536 questions about them, plus 446 trick questions about things that were never said. I ran every question, not a sample.

Scores on LoCoMo depend heavily on who grades them, so I checked the graders before trusting them. I labelled 100 answers by hand. deepseek-v4-pro agreed with me on 85 of them, gpt-oss:120b on 70, so deepseek-v4-pro became the reported grader. Then I reviewed 833 of its verdicts on the fully local run. My count came out at 66.3% where the grader said 66.9%, and a review of the cloud run gave 76.4% against its 76.2%: close enough to trust it, and I report both.

One rule I set early: on a trick question, an answer that invents a fact counts as wrong, even if it doesn't take the trap.

**My own chat history.** LoCoMo is two friends chatting, not one person with an assistant. So I imported 25 of my ChatGPT conversations (610 messages over 14 months) and wrote 60 questions about them before running anything: what I decided, what changed, when, connections across chats, and things I never said.

## The results

| Setup | Strict | Generous | Trick questions |
|---|---|---|---|
| **Fully local**: gemma4:e4b writes and reads the memory | **66.9%** | 76.4% | 83% |
| Cloud: gpt-oss:120b writes and reads the memory | 76.2% | 82.6% | 79% |
| Reference: gpt-oss:120b reads the whole history, no memory | 82.7% | 86.8% | 48% |

("Generous" counts an answer on the right topic, as published scores usually do. I don't compare these with other systems' published numbers: they use other judges, prompts and answering models, and the same product has been reported anywhere from 49% to 75% depending on who ran it; the published figures and their setups are in [section 8 of the results](CERVEAU_LOCOMO_RESULTS.md#8-published-numbers-for-comparison).)

Three things stand out.

**A small model writes memory almost as well as a big one.** With the big model answering in both cases, notes written by the laptop model cost only about 1 point against notes written by a model thirty times its size. Summarizing and extracting facts is within a small model's reach.

**Reading the memory is where the small model loses.** On the same notes, gemma answers about 8 points below gpt-oss, mostly on questions that need several memories put together. That's the gap between the laptop and the cloud, and it's on the answering side.

**Memory resists false premises; a long context doesn't.** Given the whole conversation, the 120B model scores highest on normal questions, but it goes along with a false premise about half the time (48% on trick questions). With Cerveau's memory, both models decline about 8 times in 10. Probably because retrieval gives the model less material to confuse.

How the score got there, first on a 200-question practice set: 57% at the start (a laptop model reading raw messages), 66% once conversations were consolidated into summaries and facts, then a series of fixes, the largest being dates: "last week" turned into real dates at consolidation time, which moved temporal questions from 45% to 62% (on all 321 temporal questions). The final fully local run scored 65.3%; one more fix found by studying its misses (dates spelled out in the messages the model reads) brought it to 66.9%. I tuned that fix on half the conversations and confirmed it on the other half, but it was still found by looking at these questions, so I say so.

## What my own chats found

On my 60 questions, the fully local setup got 44 right, the cloud model 47, and no memory at all 10 (the questions it got right without memory were the ones it correctly declined).

More useful than the score were the faults. LoCoMo had never shown any of them:

- **Summaries in the wrong language.** My chats are in English, but gemma wrote 37 of 77 summaries in Portuguese. The language is now worked out in code and fixed in the output schema.
- **Suggestions taken for decisions.** ChatGPT's long replies were in memory too. Asked what I had decided on one project, the model gave ChatGPT's suggested options as my decision. ChatGPT's replies, imported as the assistant's side of the conversation, are now clipped in the notes, labelled as suggestions, and grouped last.
- **Never searching.** When the notes didn't have the answer, the model said "I don't know" instead of using its search tool. A reply that only declines now triggers a search first.

On 30 new questions written after the fixes, the score went from 16 to 21: suggestive on 30 questions, not conclusive (7 turned right and 2 wrong). Those questions were drafted by Claude, which knew the fixes, and I didn't review them, so they aren't fully independent. The weakest point remains telling what I decided from what the assistant suggested: asked "did I name it X?", small models say yes too easily when the assistant proposed the name.

## The clearest finding: pushing a small model harder makes it invent

This showed up three times, in three different experiments.

- **Prompt nudges.** Instructions asking the model to dig deeper into its notes helped a little on normal questions and cost points on trick questions.
- **A stronger answerer.** qwen3.5:4b answered normal questions better than gemma on the same memory (69% against 65% on the frozen run, before the last date fix), and trick questions worse (78% against 84%).
- **Thinking.** With reasoning turned on, gemma did better on my own chats (48 of 60 against 44). On a sample of 50 LoCoMo trick questions, it went along with 5 false premises it had declined before (38 right against 42), giving one person's hobby or pet as another's; on 50 normal questions it gained only one (42 against 41).

The sample is small, but the pattern in what went wrong was the same each time: asked to try harder, a small model fills gaps instead of declining. My personal questions couldn't show it, because their never-said questions were about topics never raised, not traps that swap two people. So thinking in Cerveau is a switch you turn on per question ("Think first"), off by default.

Two other checks went the same way. Making recall stricter (fewer, closer memories) answered worse on my chats, 38 and 41 against 44: gemma uses the extra notes more than it gets misled by them. And a larger context window, which gemma supports up to 128K, fits on the 6 GB card but slows every reply (about 90 tokens per second at 16K, 55 at 32K, 44 at 64K) for room the memory doesn't need, so it stays at 16K.

## What I'd tell someone building on a small model

- **Measure on your own data, not only on a benchmark.** My chats found three real bugs in a few days; LoCoMo had shown none of these three.
- **Check your grader before you trust a number.** Two strong LLM judges disagreed with my hand labels on 15 and 30 answers out of 100. A hundred labels, an afternoon's work, settled which one to use.
- **Count invented details as wrong.** A memory that confidently remembers something false is worse than one that admits a gap.
- **Watch the trick questions when you make the model try harder.** Gains on normal questions can hide losses there.
- **Let the small model write the notes.** It's good at it; spend your effort, or a bigger model, on reading them.

## Limits

- The small model is the bottleneck: about 8 points behind a 120B model on the same memory.
- Telling suggestions from decisions is still weak.
- LoCoMo is synthetic conversations between two friends; my own eval is 60 questions on one person's history. Read both as checks, not as proof.
- It's one user on one machine, with no login.

## Try it

Everything is on [GitHub](https://github.com/Sirgogo95/cerveau-memory) under the MIT license: the app, the eval harness, every run and the full results. It needs Python 3.12+, Docker, Ollama, a 6 GB GPU and about 8 GB of disk; the README's quick start takes a few minutes, plus downloading the two models (7.8 GB).

Next on the list: more model providers (anything OpenAI-compatible, and the Claude API), a Claude export importer, and voice through Pipecat, so a voice assistant can have a memory that stays on your machine.

I designed and evaluated Cerveau myself and built it with Claude Code as a coding assistant; the design decisions, the evaluation rules and the human review of the grades are mine. Reports from other machines, especially Linux and macOS, are very welcome.
