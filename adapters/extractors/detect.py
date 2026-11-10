"""What kind of file this is, from its name and its first bytes.

The extension proposes, the bytes confirm: a PDF starts with %PDF, a DOCX is a zip
archive, and text decodes as UTF-8 (or Latin-1) without NUL bytes. A file whose
bytes don't match its name, or of a kind not read yet (images, archives,
spreadsheets), gets None: stored nowhere, refused at upload. Pure Python: no native
library to install on Windows.
"""

from __future__ import annotations

from pathlib import PurePath

TEXT = "text/plain"
MARKDOWN = "text/markdown"
CODE = "text/x-code"
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

TEXT_EXTENSIONS = {".txt", ".text", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
                   ".ini", ".cfg", ".xml", ".html", ".htm", ".rst", ".srt", ".vtt"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".c", ".h",
                   ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".sql", ".sh", ".ps1", ".css",
                   ".scss", ".vue", ".svelte", ".r", ".m", ".lua", ".dart", ".scala"}


def looks_like_text(head: bytes) -> bool:
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError as exc:
        # A multi-byte character cut by the sample's end is still text.
        if exc.start >= len(head) - 3:
            return True
    try:
        head.decode("latin-1")
        return sum(b < 9 or 13 < b < 32 for b in head) < len(head) // 100 + 1
    except UnicodeDecodeError:
        return False


def detect_media_type(filename: str, head: bytes) -> str | None:
    """The media type Cerveau reads this file as, or None if it can't read it.
    `head` is the start of the file (a few kilobytes are enough)."""
    extension = PurePath(filename).suffix.lower()
    if extension == ".pdf":
        return PDF if head.startswith(b"%PDF") else None
    if extension == ".docx":
        return DOCX if head.startswith(b"PK\x03\x04") else None
    if not looks_like_text(head):
        return None
    if extension in MARKDOWN_EXTENSIONS:
        return MARKDOWN
    if extension in CODE_EXTENSIONS:
        return CODE
    if extension in TEXT_EXTENSIONS or extension == "":
        return TEXT
    return None
