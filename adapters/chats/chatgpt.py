"""ChatGPT's data export (Settings, Data controls, Export data).

The conversations come as `conversations.json`, or split over `conversations-000.json`,
`-001.json`, ... Each one maps node ids to `{id, parent, message}`; edited and
regenerated replies are branches, and the one that was continued ends at
`current_node`, so it is found by walking back through `parent`. Kept: the user's
messages and the assistant's replies, as text (`parts` strings, voice transcriptions),
with a note for an image or a file. Skipped: the model's reasoning (`thoughts`,
`reasoning_recap`), system and tool messages, and anything addressed to a tool.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.importing import ChatLine, ImportedChat, merge_lines
from brain.types import Role

SOURCE = "chatgpt"
ROLES = {"user": Role.USER, "assistant": Role.ASSISTANT}
SKIPPED_CONTENT = {"thoughts", "reasoning_recap", "user_editable_context", "model_editable_context"}


def conversation_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("conversations-*.json"))
    return files or [folder / "conversations.json"]


def continued_branch(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages of the branch that was continued, oldest first."""
    nodes, out, node = conversation.get("mapping") or {}, [], conversation.get("current_node")
    seen: set[str] = set()
    while node and node in nodes and node not in seen:   # `seen`: a malformed export can't loop
        seen.add(node)
        if nodes[node].get("message"):
            out.append(nodes[node]["message"])
        node = nodes[node].get("parent")
    return out[::-1]


def message_text(message: dict[str, Any]) -> str:
    """What a message says, with a note in brackets for what can't be read as text."""
    pieces: list[str] = []
    images = 0
    for part in message.get("content", {}).get("parts") or []:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, dict):
            kind = part.get("content_type")
            if kind == "audio_transcription" and part.get("text"):
                pieces.append(str(part["text"]))
            elif kind == "image_asset_pointer":
                images += 1
    notes = ["[shared an image]"] * images
    for attachment in (message.get("metadata") or {}).get("attachments") or []:
        if str(attachment.get("mime_type", "")).startswith("image/"):
            continue                                      # already noted by its image part
        notes.append(f"[shared a file: {attachment.get('name') or 'unnamed'}]")
    return "\n".join(filter(None, [" ".join(p.strip() for p in pieces if p.strip()), *notes])).strip()


def stamp(value: Any, fallback: datetime) -> datetime:
    """A Unix time in seconds (the export's), or in milliseconds (seen once); the
    fallback when missing or unreadable."""
    try:
        seconds = float(value)
        return datetime.fromtimestamp(seconds / 1000 if seconds > 1e11 else seconds, UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return fallback


def read_conversation(conversation: dict[str, Any]) -> ImportedChat | None:
    """One conversation as an ImportedChat, or None if nothing in it is kept."""
    fallback = stamp(conversation.get("create_time"), datetime.fromtimestamp(0, UTC))
    lines = []
    for message in continued_branch(conversation):
        role = ROLES.get(message.get("author", {}).get("role", ""))
        if role is None or message.get("recipient") not in (None, "all"):
            continue
        if message.get("content", {}).get("content_type") in SKIPPED_CONTENT:
            continue
        text = message_text(message)
        if not text:
            continue
        at = fallback = stamp(message.get("create_time"), fallback)   # a missing time: the one before
        lines.append(ChatLine(role, text, at, (str(message["id"]),)))
    if not lines:
        return None
    return ImportedChat(SOURCE, str(conversation.get("conversation_id") or conversation.get("id")),
                        conversation.get("title") or "", merge_lines(lines),
                        bool(conversation.get("is_do_not_remember")))


def read_chatgpt(folder: str | Path) -> list[ImportedChat]:
    """Every conversation of the export in `folder`, in order of its first message."""
    chats = []
    for path in conversation_files(Path(folder)):
        for conversation in json.loads(path.read_text(encoding="utf-8")):
            chat = read_conversation(conversation)
            if chat is not None:
                chats.append(chat)
    return sorted(chats, key=lambda c: c.started_at)
