"""One turn of the conversation: the pipeline written as code.

Each later feature slots in as one more step here (signals, activation, running
summary, reinforcement after the reply). The caller (a Django view, the eval runner,
a test) only supplies the dependencies and streams the chunks.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

from .config import EngineConfig, ModelProfile
from .embedding import embed_cached
from .language import detect_language
from .periods import period_note
from .ports import ChatProvider, Embedder, MemoryStore, Reranker
from .prompt import assemble_prompt, pinned_lines, system_message
from .attachments import mark_shared_files
from .recall import recall
from .segmentation import OnClose, carried_over, ensure_episode, follow_subject
from .summary import summary_due
from .tools import TOOLS, ToolContext, run_tool, shown_ids, tool_limit_result
from .types import ChatChunk, ChatMessage, Message, Role, ToolCall, ToolRun


def utcnow() -> datetime:
    return datetime.now(UTC)


# How much of a reply is held back to see whether it opens by declining (step 9c).
DECLINE_HOLD_CHARS = 40
# A reply that opens by declining and stays this short is only a decline: searched.
DECLINE_MAX_CHARS = 240
DECLINE_START = re.compile(
    r"^[\s*_>#\"'“]*(?:(?:sorry|lo siento|désolée?)[,.!]?\s*)?(?:"
    r"i\s+(?:don['’]?t|do\s+not|can['’]?t|cannot|couldn['’]?t)\s+(?:remember|recall|have|find|see|know)"
    r"|i\s+have\s+no\s+(?:record|information|notes?|memory)"
    r"|there\s+(?:is|['’]s)\s+no(?:thing)?\s+(?:record|mention|information|in\s+my\s+notes)"
    r"|(?:my\s+)?notes\s+(?:don['’]?t|do\s+not)"
    r"|no\s+(?:recuerdo|tengo|encuentro)|no\s+hay\s+(?:registro|información)"
    r"|je\s+ne\s+(?:me\s+souviens|trouve|sais)|je\s+n['’]ai\s+(?:pas|aucune?))",
    re.IGNORECASE,
)


def declines(start: str) -> bool:
    """Whether a reply opens by saying it doesn't remember or have the answer."""
    return bool(DECLINE_START.match(start))


def run_turn(
    text: str,
    now: datetime,
    *,
    store: MemoryStore,
    llm: ChatProvider,
    embedder: Embedder,
    profile: ModelProfile,
    config: EngineConfig,
    clock: Callable[[], datetime] = utcnow,
    on_episode_closed: OnClose | None = None,
    on_summary_due: Callable[[int], None] | None = None,
    on_user_message: Callable[[Message], None] | None = None,
    reranker: Reranker | None = None,
) -> Iterator[ChatChunk]:
    """Answer one user message, yielding the reply as it streams.

    The user message is saved (with its embedding) before the model is called, so it
    is never lost. The reply is saved only once the stream completes; if the provider
    fails mid-stream, ChatProviderError propagates and nothing partial is stored.
    `clock` timestamps the reply (injectable for tests). `on_episode_closed` is
    called with the id of an episode this message closed (the app queues its
    consolidation). `on_summary_due` is called with the episode's id once it has
    outgrown its recent window (the app queues the running-summary fold).
    `on_user_message` is called with the stored user message before anything is
    recalled (the app links the files it shared, and reads the small ones, so this
    reply can already use them).

    When the model supports tools, it may call the memory tools (brain/tools.py)
    before answering: chunks carrying `tool_calls` are yielded as they come (the app
    can show "searching..."), the tools run, and the model continues with their
    results, at most `config.max_tool_calls` times. The calls are stored with the
    reply (ToolCallStore), for debugging and for the page.
    """
    [query] = embed_cached([text], embedder, store)                     # step 1
    episode = ensure_episode(store, now, config, on_episode_closed)     # step 2
    message = store.add_message(episode.id, Role.USER, text, now, query)
    if on_user_message is not None:
        on_user_message(message)
    # A change of subject can start a new episode here (step 3), two messages in.
    episode = follow_subject(text, message, query, episode, store=store, embedder=embedder,
                             config=config, on_close=on_episode_closed)

    tz = config.tz
    recalled = recall(text, query, store, config, embedder=embedder,               # steps 4-7
                      episode_id=episode.id, now=now, tz=tz, reranker=reranker)
    memories, recent_episodes, period = recalled.memories, recalled.recent_episodes, recalled.period
    # The messages the running summary doesn't cover yet. Normally at most
    # recent_window; the extra chunk covers a fold still waiting for the worker.
    recent = store.get_messages(
        episode.id, after=episode.summarized_until,
        last=config.recent_window + config.summary_chunk,
    )
    # After a split by subject, the previous episode's last messages lead the prompt
    # for a couple of turns; pinned messages the window moved past stay in it.
    lead = carried_over(store, episode, config)
    in_window = {m.id for m in recent}
    pins = pinned_lines([m for m in store.pinned_messages(episode.id) if m.id not in in_window],
                        int(profile.memory_budget * config.pinned_share), profile.chars_per_token, tz)
    shown = mark_shared_files([*lead, *recent], store)        # "[shared a file: A3 ...]", prompt only
    tools = TOOLS if config.memory_tools and profile.supports_tools and config.max_tool_calls else ()
    prompt = assemble_prompt(                                            # steps 8-9
        shown, profile, memories=memories, memory_share=config.memory_share,
        system_prompt=system_message(
            now, recent_episodes, tz=tz, running_summary=episode.running_summary,
            pinned=pins, tools=bool(tools),
        ),
        tz=tz,
        reply_language=detect_language(text),
        period_note="\n".join(filter(None, [period_note(
            period, any(period.contains(m.candidate.at) for m in memories)
            or any(period.contains(r.episode.started_at) for r in recent_episodes),
        ) if period else "", *recalled.notes])),
    )
    in_prompt = shown[len(shown) - (len(prompt) - 1):]  # the messages that fit
    context = ToolContext(
        store=store, embedder=embedder, config=config, profile=profile, now=now, tz=tz,
        shown=shown_ids([m.content for m in prompt]),
        skip_message_ids={m.id for m in in_prompt},
    )

    parts: list[str] = []
    runs: list[ToolRun] = []
    calls_made = 0
    while True:
        offered = tools if calls_made < config.max_tool_calls else ()
        calls: list[ToolCall] = []
        said: list[str] = []
        # Until the model has searched, the start of its reply is held back; one that
        # opens with "I don't remember" is held to its end, and if it is only that (a
        # short decline), it is dropped for a search on the user's own words (step 9c).
        # gemma4:e4b never called search_memory on its own, even when the answer was in
        # memory; a longer reply that hedges and then answers is kept as it is (dropping
        # one lost a right answer, 2026-11-06).
        state = "hold" if offered and config.fallback_search and not runs else "stream"
        held: list[ChatChunk] = []
        for chunk in llm.stream(prompt, tools=offered):
            said.append(chunk.content)
            calls.extend(chunk.tool_calls)
            if state == "stream":
                yield chunk
                continue
            held.append(chunk)
            start = "".join(said)
            if state == "hold" and (calls or len(start) >= DECLINE_HOLD_CHARS):
                state = "suspect" if not calls and declines(start) else "stream"
                if state == "stream":
                    yield from held
        reply_so_far = "".join(said)
        if state in ("hold", "suspect"):  # decided at the reply's end
            only_declines = not calls and declines(reply_so_far) and len(reply_so_far) <= DECLINE_MAX_CHARS
            state = "drop" if only_declines else "stream"
            if state == "stream":
                yield from held
        if state == "drop":
            calls, said = [ToolCall("search_memory", {"query": text})], []
            yield ChatChunk("", tool_calls=tuple(calls))      # the page shows the search
        parts.extend(said)
        if not calls or not offered:
            break
        # Run the tools and let the model continue with their results (step 9b).
        prompt = [*prompt, ChatMessage(Role.ASSISTANT, "".join(said), tool_calls=tuple(calls))]
        for call in calls:
            calls_made += 1
            started = time.monotonic()
            result = run_tool(call, context) if calls_made <= config.max_tool_calls else tool_limit_result()
            runs.append(ToolRun(
                call.name, dict(call.arguments), result,
                round((time.monotonic() - started) * 1000), clock(),
            ))
            prompt.append(ChatMessage(Role.TOOL, result, tool_name=call.name))

    reply = "".join(parts)                                              # step 10
    if not reply.strip():
        if runs:
            store.add_tool_calls(episode.id, None, runs)
        return  # an empty reply is a model failure, not part of the conversation
    [reply_embedding] = embed_cached([reply], embedder, store)
    answer = store.add_message(episode.id, Role.ASSISTANT, reply, clock(), reply_embedding)
    if runs:
        store.add_tool_calls(episode.id, answer.id, runs)
    if on_summary_due is not None and summary_due(len(recent) + 1, config):
        on_summary_due(episode.id)                                      # step 10.5
