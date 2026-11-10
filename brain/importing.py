"""Bringing an existing chat history into memory (docs/PERSONAL_CHATS_EVAL.md, 4.3).

An export (ChatGPT's, Claude's) is read by an adapter into `ImportedChat`s; each one
is replayed here through the same rules as a live conversation: a chat starts a new
episode, the time gap and the change-of-subject check split it further, and every
message keeps its original time. No model is called: nothing is recalled and no
reply is written, and each episode is closed for consolidation when its chat ends.
Chats are imported whole, one after another (in order of their first message), not
interleaved by time: switching between chats would look like a change of subject.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .config import EngineConfig
from .embedding import embed_cached
from .ports import Embedder, MemoryStore
from .segmentation import OnClose, ensure_episode, follow_subject
from .types import Role


@dataclass(frozen=True, slots=True)
class ChatLine:
    """One message of an imported chat. Consecutive messages of the same role in the
    export are one line here (a reply split in parts is one reply), so a line can
    stand for several of the export's ids."""

    role: Role
    text: str
    at: datetime
    external_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImportedChat:
    source: str              # "chatgpt", "claude"
    conversation_id: str
    title: str
    lines: tuple[ChatLine, ...]
    do_not_remember: bool = False   # marked so in the source app: left out unless asked

    @property
    def started_at(self) -> datetime:
        return self.lines[0].at


def merge_lines(lines: Sequence[ChatLine]) -> tuple[ChatLine, ...]:
    """Consecutive lines of the same role as one, keeping the first one's time."""
    out: list[ChatLine] = []
    for line in lines:
        if out and out[-1].role is line.role:
            last = out[-1]
            out[-1] = ChatLine(last.role, f"{last.text}\n\n{line.text}", last.at, last.external_ids + line.external_ids)
        else:
            out.append(line)
    return tuple(out)


def close_open_episode(store: MemoryStore, on_close: OnClose | None) -> None:
    current = store.current_episode()
    if current is not None:
        store.close_episode(current.id, current.last_message_at or current.started_at)
        if on_close is not None:
            on_close(current.id)


def import_chat(chat: ImportedChat, *, store: MemoryStore, embedder: Embedder, config: EngineConfig,
                on_close: OnClose | None = None) -> dict[str, int]:
    """Store one chat's messages as a live conversation would have, and record which
    stored message each of the export's ids became. Returns that mapping."""
    close_open_episode(store, on_close)                       # each chat starts a new episode
    vectors = embed_cached([line.text for line in chat.lines], embedder, store)
    mapping: dict[str, int] = {}
    for line, vector in zip(chat.lines, vectors):
        if line.role is Role.USER:
            episode = ensure_episode(store, line.at, config, on_close)
            message = store.add_message(episode.id, Role.USER, line.text, line.at, vector)
            follow_subject(line.text, message, vector, episode, store=store, embedder=embedder,
                           config=config, on_close=on_close)
        else:
            # A reply joins its question's episode (a chat may also open with one).
            episode = store.current_episode() or ensure_episode(store, line.at, config, on_close)
            message = store.add_message(episode.id, Role.ASSISTANT, line.text, line.at, vector)
        mapping.update(dict.fromkeys(line.external_ids, message.id))
    close_open_episode(store, on_close)
    store.record_imports(chat.source, chat.conversation_id, mapping)
    return mapping
