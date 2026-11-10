"""The running summary of the current episode (PIPELINE.md step 10.5).

A long session would otherwise lose its beginning: the prompt keeps the episode's
recent messages verbatim, and past `recent_window` the oldest ones drop out. Instead,
the oldest `summary_chunk` messages are folded into a running summary, which the
prompt shows in their place. Folding a batch at a time means the messages leave the
prompt together, so its start stays identical for several turns (prefix caching).

The engine only decides *when* a fold is due (`summary_due`); the app runs
`fold_running_summary` in the background, like consolidation.
"""

from __future__ import annotations

from collections.abc import Sequence

from .config import EngineConfig
from .consolidation import format_message
from .ports import ChatProvider, ConversationStore
from .types import ChatMessage, Message, Role

# Keeps the summary a fixed, small cost in every prompt however long the episode gets:
# older details are compressed as new ones are added.
MAX_SUMMARY_WORDS = 250

SYSTEM = (
    "You keep a running summary of a conversation that is still going on, between a "
    "user and their assistant, Cerveau. Report only what the conversation contains; "
    "never invent details."
)

FOLD = """\
Update the summary with the new messages below.

- Keep what still matters: what the user is working on, decisions, facts the user \
stated, names, numbers, and open questions. Drop small talk.
- Keep the user's own words for what they like, prefer, want, or decided ("their \
favorite", "they decided"): those words are the point, so don't soften them.
- Attribute carefully: lines starting with "User:" are what the user said; lines \
starting with "Cerveau:" are the assistant. Cerveau's suggestions are not the user's \
decisions unless the user accepted them.
- Write in the conversation's own language, in plain sentences, at most \
{max_words} words. Compress older details rather than dropping recent ones.
- Reply with the updated summary only.

Summary so far:
{previous}

New messages:
{transcript}"""


class SummaryError(Exception):
    """The background model returned no usable summary."""


def summary_due(unsummarized: int, config: EngineConfig) -> bool:
    """Whether the episode has outgrown its window: more messages not yet covered by
    the running summary than `recent_window`."""
    return unsummarized > config.recent_window


def fold(
    previous: str, messages: Sequence[Message], llm: ChatProvider
) -> str:
    """One model call: the previous summary plus these messages, as a new summary."""
    prompt = FOLD.format(
        max_words=MAX_SUMMARY_WORDS,
        previous=previous or "(nothing yet: this is the start of the conversation)",
        transcript="\n".join(format_message(m) for m in messages),
    )
    reply = llm.chat([ChatMessage(Role.SYSTEM, SYSTEM), ChatMessage(Role.USER, prompt)])
    summary = reply.content.strip()
    if not summary:
        raise SummaryError("The model returned an empty summary.")
    return summary


def fold_running_summary(
    episode_id: int,
    *,
    store: ConversationStore,
    llm: ChatProvider,
    config: EngineConfig,
) -> int:
    """Fold the episode's oldest uncovered messages into its running summary, a
    batch of `summary_chunk` at a time, until at most `recent_window` remain
    uncovered. Returns how many messages were folded (0 if none were due).

    Safe to run twice or late: it works from what is uncovered *now*. Each batch is
    saved before the next is folded, so a failure keeps the progress made.
    """
    found = store.get_episodes([episode_id])
    if not found:
        return 0
    episode = found[0]
    summary, until, folded = episode.running_summary, episode.summarized_until, 0

    while True:
        uncovered = store.get_messages(episode_id, after=until)
        if not summary_due(len(uncovered), config):
            return folded
        batch = uncovered[:config.summary_chunk]
        summary = fold(summary, batch, llm)
        until = batch[-1].id
        store.save_running_summary(episode_id, summary, until)
        folded += len(batch)
