"""Files the user shares (docs/ATTACHMENTS.md): reading them into passages, and
summarizing them. Both run in the background; the chat never waits for them.

Reading: the file's text is extracted (adapters/extractors/), split into passages
(brain/chunking.py), each passage embedded, and every known person, place, or
project it names recorded as a mention, with the matching retrieval uses: so a name
on page 40 finds that page. No model call.

Summarizing: one call to the background model writes a title, two summaries, and
topic tags. The input is capped (the headings, the first passages, and an even
sample of the rest) so a long file costs one call; its details stay reachable
through its passages. Summaries say what the document says, attributed to it, never
to the user: a document isn't the user's own statement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .chunking import chunk_document
from .config import ModelProfile
from .embedding import embed_cached
from .ports import ChatProvider, Embedder, MemoryStore, TextExtractor
from .tags import clean_tags, resolve_tags
from .text import phrase_keys
from .types import Attachment, ChatMessage, Chunk, ExtractedDocument, Message, Role

# Embedding many passages at once is faster, but one request shouldn't carry a whole
# book: Ollama holds the batch in memory.
EMBED_BATCH = 32
# How much of the message a file was shared with goes into each passage's context.
CONTEXT_MESSAGE_CHARS = 300
# Share of the background model's memory budget the summary's input may use.
SUMMARY_SHARE = 0.6
MAX_TITLE = 200


@dataclass
class ReadReport:
    status: str                     # done, failed, or unsupported
    chunks: int = 0
    mentions: int = 0
    pages: int | None = None
    error: str = ""


def document_title(document: ExtractedDocument, filename: str) -> str:
    """The document's first heading, or its file name without the extension."""
    return next((b.heading for b in document.blocks if b.heading), filename.rsplit(".", 1)[0])


def embedding_texts(chunks: Sequence[Chunk], title: str, filename: str, shared_with: str) -> list[str]:
    """What is embedded for each passage: the passage with a line of context first,
    `Residential Lease Agreement > Utilities (lease.md; shared with: Here's the lease
    for the Bella Vista apartment)`. A section alone often doesn't say what it's about
    ("The Tenant pays for electricity"): measured, questions naming the apartment found
    those sections only with the context (17 of 17 instead of 15). Only the vector
    gets it; the stored passage, which prompts quote, stays as written."""
    said = " ".join(shared_with.split())[:CONTEXT_MESSAGE_CHARS]
    where = f"{filename}; shared with: {said}" if said else filename
    return [f"{title} > {c.heading} ({where})\n{c.text}" if c.heading else f"{title} ({where})\n{c.text}"
            for c in chunks]


def read_attachment(
    attachment_id: int,
    path: str,
    *,
    store: MemoryStore,
    embedder: Embedder,
    extractors: Sequence[TextExtractor],
    max_chars: int,
    overlap: int,
    force: bool = False,
) -> ReadReport | None:
    """Read one attachment into passages. None if there is nothing to do (unknown,
    forgotten, or already read without `force`)."""
    attachment = store.get_attachment(attachment_id)
    if attachment is None or (attachment.extraction_status != "pending" and not force):
        return None
    extractor = next((e for e in extractors if e.supports(attachment.media_type)), None)
    if extractor is None:
        store.mark_attachment(attachment_id, "unsupported")
        return ReadReport("unsupported")
    try:
        document = extractor.extract(path, attachment.media_type)
    except Exception as exc:  # noqa: BLE001  # any parser error: record it, don't crash the worker
        store.mark_attachment(attachment_id, "failed", f"Couldn't read the file: {exc}")
        return ReadReport("failed", error=str(exc))
    chunks = chunk_document(document, max_chars, overlap)
    if not chunks:
        reason = ("No text found. A scanned PDF has no text layer; reading those (OCR) "
                  "comes later.") if document.page_count else "The file has no text."
        store.mark_attachment(attachment_id, "failed", reason)
        return ReadReport("failed", pages=document.page_count, error=reason)
    message = store.get_message(attachment.message_id) if attachment.message_id is not None else None
    texts = embedding_texts(chunks, document_title(document, attachment.filename), attachment.filename,
                            message.content if message is not None else "")
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embed_cached(texts[start:start + EMBED_BATCH], embedder, store))
    mentions = [
        (chunk.position, entity.id)
        for chunk in chunks
        for entity in store.match_entities(phrase_keys(chunk.text))
    ]
    store.save_chunks(attachment_id, list(zip(chunks, vectors)), mentions, document.page_count)
    return ReadReport("done", chunks=len(chunks), mentions=len(mentions), pages=document.page_count)


# -- Summarizing ------------------------------------------------------------------------

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "title": {"type": "string"},
        "summary_short": {"type": "string"},
        "summary_long": {"type": "string"},
        "english_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["language", "title", "summary_short", "summary_long", "english_tags"],
}

SUMMARY_SYSTEM = (
    "You summarize a document the user shared with their assistant, Cerveau, so it can "
    "be found and recalled later. Report only what the document contains; never invent "
    "details."
)

SUMMARY_INSTRUCTIONS = """\
The document is "{filename}"{pages}. Below are its headings and passages from it \
(from the start, then a sample of the rest; "[...]" marks skipped parts).

Produce JSON with:
- language: the document's main language, for example "Spanish". Write title and \
summaries in it.
- title: 3 to 10 words naming what the document is.
- summary_short: 1 or 2 sentences: what kind of document it is and what it is about.
- summary_long: up to about 150 words: its main points, names, dates, and figures. \
Attribute them to the document ("the document says", "el documento indica"), never \
to the user.
- english_tags: the broad topics it is about, usually 1 to 3, in English and lowercase.

{material}"""


@dataclass
class SummaryReport:
    title: str = ""
    tags: list[str] = field(default_factory=list)
    merged: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def sample_passages(chunks: Sequence[Chunk], budget_chars: int) -> str:
    """The headings, then passages from the start, then an even sample of the rest,
    within the budget; "[...]" where passages were skipped."""
    headings = list(dict.fromkeys(c.heading for c in chunks if c.heading))
    parts = ["Headings: " + "; ".join(headings)] if headings else []
    used = sum(len(p) for p in parts)
    front = budget_chars * 2 // 5
    chosen: list[int] = []
    for i, chunk in enumerate(chunks):
        if used + len(chunk.text) > front:
            break
        chosen.append(i)
        used += len(chunk.text)
    rest = list(range(len(chosen), len(chunks)))
    if rest:
        average = max(1, sum(len(chunks[i].text) for i in rest) // len(rest))
        room = max(0, (budget_chars - used) // average)
        step = max(1, len(rest) // max(1, room)) if room else len(rest) + 1
        for i in rest[::step]:
            if used + len(chunks[i].text) > budget_chars:
                break
            chosen.append(i)
            used += len(chunks[i].text)
    text, previous = [], -1
    for i in sorted(chosen):
        if i != previous + 1:
            text.append("[...]")
        label = f"({chunks[i].pages}) " if chunks[i].pages else ""
        text.append(label + chunks[i].text)
        previous = i
    if previous != len(chunks) - 1:
        text.append("[...]")
    return "\n\n".join(parts + text)


def summarize_attachment(
    attachment_id: int,
    *,
    store: MemoryStore,
    llm: ChatProvider,
    profile: ModelProfile,
    embedder: Embedder,
    tag_same_score: float | None = None,
    force: bool = False,
) -> SummaryReport | None:
    """Title, summaries, and tags for a read attachment, from one model call. None if
    there is nothing to do (unknown, not read, or already summarized without `force`)."""
    attachment = store.get_attachment(attachment_id)
    if attachment is None or attachment.extraction_status != "done" or (attachment.title and not force):
        return None
    chunks = store.attachment_chunks(attachment_id)
    budget = int(profile.memory_budget * profile.chars_per_token * SUMMARY_SHARE)
    pages = f" ({attachment.page_count} pages)" if attachment.page_count else ""
    prompt = SUMMARY_INSTRUCTIONS.format(
        filename=attachment.filename, pages=pages, material=sample_passages(chunks, budget)
    )
    reply = llm.chat([ChatMessage(Role.SYSTEM, SUMMARY_SYSTEM), ChatMessage(Role.USER, prompt)],
                     schema=SUMMARY_SCHEMA)
    try:
        data = json.loads(reply.content)
    except json.JSONDecodeError:
        data = {}
    title = " ".join(str(data.get("title", "")).split())[:MAX_TITLE] or attachment.filename
    short = str(data.get("summary_short", "")).strip()
    long = str(data.get("summary_long", "")).strip() or short
    store.save_attachment_summary(attachment_id, title, short, long)
    resolved = resolve_tags(clean_tags(data.get("english_tags") or []), store, embedder, tag_same_score)
    for tag in resolved.tags:
        store.tag_attachment(attachment_id, tag.id)
    return SummaryReport(title, [t.name for t in resolved.tags], resolved.merged, resolved.skipped)


def mark_shared_files(messages: Sequence[Message], store: MemoryStore) -> list[Message]:
    """The messages with a line naming each file they shared, `[shared a file: A3
    "Naco lease"]`: for the prompt and the consolidation transcript, never stored."""
    shared = store.message_attachments([m.id for m in messages])
    if not shared:
        return list(messages)
    return [
        replace(m, content=m.content + "".join(
            f'\n[shared a file: A{a.id} "{a.title or a.filename}"]' for a in shared[m.id]))
        if m.id in shared else m
        for m in messages
    ]


def reading_notes(attachments: Sequence[Attachment]) -> list[str]:
    """What the prompt should say about files shared in the new message that can't be
    quoted yet: still being read, or unreadable."""
    notes = []
    for a in attachments:
        if a.extraction_status == "pending":
            notes.append(f'The user just shared "{a.filename}" (A{a.id}). It is still being read: its '
                         "content isn't available yet. If they ask about it, say you're still reading it.")
        elif a.extraction_status in ("failed", "unsupported"):
            why = a.error or "Cerveau can't read this kind of file yet."
            notes.append(f"The user shared \"{a.filename}\" (A{a.id}), but it couldn't be read: {why} "
                         "Say so if they ask about it.")
    return notes
