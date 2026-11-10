"""Splitting a document into passages (docs/ATTACHMENTS.md, section 6).

A passage is what gets embedded and what a prompt quotes, so it should hold one
idea, read well alone, and stay under what the embedding model reads. Paragraphs
(or, for code, lines) are packed into passages of at most `max_chars`; a paragraph
longer than that is split at sentence ends, then at spaces, never mid-word. A new
heading starts a new passage, so a section's passages carry its heading. When a
passage is cut for size, the next one repeats the last sentence or two (`overlap`
characters at most), so a sentence across the cut still reads whole somewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Chunk, ExtractedDocument, TextBlock

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page: int | None
    heading: str


def _split_long(text: str, max_chars: int) -> list[str]:
    """Pieces of at most max_chars: at sentence ends, then at spaces."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sentence in SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _units(document: ExtractedDocument, max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    for block in document.blocks:
        if document.code:
            lines = block.text.splitlines()
            units.extend(_Unit(line, block.page, block.heading) for line in lines)
            continue
        text = " ".join(block.text.split())
        if text:
            units.extend(_Unit(p, block.page, block.heading) for p in _split_long(text, max_chars))
    return units


def _tail(units: list[_Unit], overlap: int, joiner: str) -> list[_Unit]:
    """The last units of a passage, at most `overlap` characters together."""
    kept: list[_Unit] = []
    size = 0
    for unit in reversed(units):
        size += len(unit.text) + len(joiner)
        if size > overlap:
            break
        kept.insert(0, unit)
    return kept if len(kept) < len(units) else []


def chunk_document(document: ExtractedDocument, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """The document's passages, in order."""
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be positive and larger than overlap.")
    joiner = "\n" if document.code else " "
    chunks: list[Chunk] = []
    current: list[_Unit] = []

    def flush() -> None:
        text = joiner.join(u.text for u in current).strip("\n")
        if text.strip():
            pages = [u.page for u in current if u.page is not None]
            chunks.append(Chunk(
                position=len(chunks), text=text,
                page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                heading=current[0].heading,
            ))

    for unit in _units(document, max_chars):
        if current and unit.heading != current[-1].heading:
            flush()                      # a new section starts a new passage
            current = []
        size = sum(len(u.text) + len(joiner) for u in current)
        if current and size + len(unit.text) > max_chars:
            flush()
            current = _tail(current, overlap, joiner)
            if sum(len(u.text) + len(joiner) for u in current) + len(unit.text) > max_chars:
                current = []
        current.append(unit)
    if current:
        flush()
    return chunks


def blocks_from_text(text: str, *, markdown: bool = False) -> tuple[TextBlock, ...]:
    """Paragraphs of a plain text (blank lines between them); with `markdown`, lines
    starting with # are headings for what follows."""
    blocks: list[TextBlock] = []
    heading = ""
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = paragraph.strip().splitlines()
        body: list[str] = []
        for line in lines:
            if markdown and line.lstrip().startswith("#"):
                if body:
                    blocks.append(TextBlock(" ".join(body), heading=heading))
                    body = []
                heading = line.lstrip("# ").strip()
            else:
                body.append(line)
        if body:
            blocks.append(TextBlock(" ".join(body), heading=heading))
    return tuple(blocks)
