"""Storing a shared file (docs/ATTACHMENTS.md, section 4.1): fast, in the request,
no model call.

The bytes are checked (size, and a type Cerveau can read, from the name and the
first bytes), hashed, and written once under CERVEAU_ATTACHMENTS_DIR, named by hash
(`ab/cd/abcd...ef.pdf`): the same file shared twice is stored once, and a crafted
filename can't escape the folder. Reading it happens in the background.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath

from django.conf import settings

from adapters.extractors import detect_media_type
from brain.ports import AttachmentStore
from brain.types import Attachment

HEAD_BYTES = 8192
# Text files up to this size are read during the turn that shares them (a few
# passages: embedded in well under a second), so the reply can already quote them.
READ_NOW_BYTES = 50_000


class UploadRefused(ValueError):
    """The file can't be stored: too large, or a kind Cerveau can't read."""


@dataclass(frozen=True)
class Stored:
    attachment: Attachment | None   # None: you forgot this file before; it isn't relearned
    created: bool
    filename: str


def attachments_dir() -> Path:
    return Path(settings.CERVEAU_ATTACHMENTS_DIR)


def attachment_path(attachment: Attachment) -> Path:
    return attachments_dir() / attachment.storage_path


def read_now(attachment: Attachment) -> bool:
    """Read a small text file right away (in the request). True if it was read."""
    from adapters.extractors import MARKDOWN, TEXT, CODE, all_extractors
    from brain.attachments import read_attachment
    from brain.config import attachment_chunk_size

    from .engine import get_engine

    if (attachment.extraction_status != "pending" or attachment.size_bytes > READ_NOW_BYTES
            or attachment.media_type not in (TEXT, MARKDOWN, CODE)):
        return False
    engine = get_engine()
    assert engine.embedding_profile is not None
    size, overlap = attachment_chunk_size(engine.config, engine.embedding_profile)
    report = read_attachment(attachment.id, str(attachment_path(attachment)), store=engine.store,
                             embedder=engine.embedder, extractors=all_extractors(),
                             max_chars=size, overlap=overlap)
    return report is not None and report.status == "done"


def store_upload(filename: str, data: bytes, store: AttachmentStore, now: datetime) -> Stored:
    """Store a shared file (or find it, if the same bytes were shared before)."""
    name = PurePath(filename.replace("\\", "/")).name or "file"
    limit = settings.CERVEAU_MAX_ATTACHMENT_MB * 1024 * 1024
    if len(data) > limit:
        raise UploadRefused(f"{name} is larger than {settings.CERVEAU_MAX_ATTACHMENT_MB} MB.")
    if not data:
        raise UploadRefused(f"{name} is empty.")
    media_type = detect_media_type(name, data[:HEAD_BYTES])
    if media_type is None:
        raise UploadRefused(
            f"Cerveau can't read {name} yet: text, Markdown, code, PDF, and Word files work.")
    digest = hashlib.sha256(data).hexdigest()
    relative = f"{digest[:2]}/{digest[2:4]}/{digest}{PurePath(name).suffix.lower()[:10]}"
    path = attachments_dir() / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    found = store.add_attachment(name, media_type, len(data), digest, relative, now)
    if found is None:
        return Stored(None, False, name)
    attachment, created = found
    return Stored(attachment, created, name)
