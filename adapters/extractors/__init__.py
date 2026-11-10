"""Text extractors for shared files (docs/ATTACHMENTS.md, section 5), one per format,
behind brain.ports.TextExtractor, so brain/ never imports a parsing library."""

from __future__ import annotations

from .detect import CODE, DOCX, MARKDOWN, PDF, TEXT, detect_media_type
from .documents import DocxExtractor, PdfExtractor, PlainTextExtractor, all_extractors

__all__ = [
    "CODE", "DOCX", "MARKDOWN", "PDF", "TEXT",
    "DocxExtractor", "PdfExtractor", "PlainTextExtractor", "all_extractors", "detect_media_type",
]
