"""The extractors: plain text, Markdown, and code (standard library), PDF with a text
layer (pypdf), and Word documents (python-docx). The parsing libraries are imported
inside extract(), so a process that never reads a PDF never loads them."""

from __future__ import annotations

from pathlib import Path

from brain.chunking import blocks_from_text
from brain.types import ExtractedDocument, TextBlock

from .detect import CODE, DOCX, MARKDOWN, PDF, TEXT


def read_text(path: str) -> str:
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class PlainTextExtractor:
    """Text and Markdown by paragraphs (Markdown headings kept); code whole, to be
    split by lines."""

    def supports(self, media_type: str) -> bool:
        return media_type in (TEXT, MARKDOWN, CODE)

    def extract(self, path: str, media_type: str) -> ExtractedDocument:
        text = read_text(path)
        if media_type == CODE:
            return ExtractedDocument((TextBlock(text),), code=True)
        return ExtractedDocument(blocks_from_text(text, markdown=media_type == MARKDOWN))


class PdfExtractor:
    """One block per page. A scanned PDF has no text layer: its pages come back empty,
    and reading it fails with a clear reason (OCR comes later)."""

    def supports(self, media_type: str) -> bool:
        return media_type == PDF

    def extract(self, path: str, media_type: str) -> ExtractedDocument:
        from pypdf import PdfReader

        reader = PdfReader(path)
        blocks = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                blocks.append(TextBlock(text, page=number))
        return ExtractedDocument(tuple(blocks), page_count=len(reader.pages))


class DocxExtractor:
    """Paragraphs, with each one's heading (paragraphs styled "Heading ..." or
    "Title"); tables read row by row."""

    def supports(self, media_type: str) -> bool:
        return media_type == DOCX

    def extract(self, path: str, media_type: str) -> ExtractedDocument:
        import docx

        document = docx.Document(path)
        blocks: list[TextBlock] = []
        heading = ""
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            if style.startswith("Heading") or style == "Title":
                heading = text
            else:
                blocks.append(TextBlock(text, heading=heading))
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(TextBlock(" | ".join(cells), heading=heading))
        return ExtractedDocument(tuple(blocks))


def all_extractors() -> list:
    return [PlainTextExtractor(), PdfExtractor(), DocxExtractor()]
