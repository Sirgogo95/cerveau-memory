"""adapters/extractors: telling what a file is, and reading text, Markdown, code, PDF,
and Word files."""

import tempfile
import unittest
from pathlib import Path

from adapters.extractors import (
    CODE,
    DOCX,
    MARKDOWN,
    PDF,
    TEXT,
    DocxExtractor,
    PdfExtractor,
    PlainTextExtractor,
    detect_media_type,
)


def tiny_pdf(*pages: str) -> bytes:
    """A minimal PDF with one line of Helvetica text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        page_no, content_no = len(objects) + 1, len(objects) + 2
        kids.append(f"{page_no} 0 R")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
                       f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


class DetectTests(unittest.TestCase):
    def test_extensions_are_confirmed_by_the_bytes(self):
        self.assertEqual(detect_media_type("lease.pdf", b"%PDF-1.4 ..."), PDF)
        self.assertIsNone(detect_media_type("lease.pdf", b"not a pdf"))
        self.assertEqual(detect_media_type("plan.docx", b"PK\x03\x04..."), DOCX)
        self.assertIsNone(detect_media_type("plan.docx", b"%PDF"))

    def test_text_markdown_and_code(self):
        self.assertEqual(detect_media_type("notes.txt", "Café con leche".encode()), TEXT)
        self.assertEqual(detect_media_type("README.md", b"# Title"), MARKDOWN)
        self.assertEqual(detect_media_type("app.py", b"import os"), CODE)
        self.assertEqual(detect_media_type("Dockerfile", b"FROM python"), TEXT)

    def test_binary_and_unknown_kinds_are_refused(self):
        self.assertIsNone(detect_media_type("photo.txt", b"\x89PNG\r\n\x1a\n\x00\x00"))
        self.assertIsNone(detect_media_type("photo.png", b"\x89PNG"))
        self.assertIsNone(detect_media_type("archive.zip", b"PK\x03\x04"))


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, name: str, data: bytes) -> str:
        path = Path(self.folder.name) / name
        path.write_bytes(data)
        return str(path)

    def test_markdown_keeps_headings_and_code_stays_whole(self):
        path = self.write("a.md", "# Lease\n\nRent is monthly.\n\n# Deposit\n\n60,000 pesos.".encode())
        doc = PlainTextExtractor().extract(path, MARKDOWN)
        self.assertEqual([(b.heading, b.text) for b in doc.blocks],
                         [("Lease", "Rent is monthly."), ("Deposit", "60,000 pesos.")])
        code = PlainTextExtractor().extract(self.write("a.py", b"def f():\n    return 1\n"), CODE)
        self.assertTrue(code.code)
        self.assertEqual(code.blocks[0].text, "def f():\n    return 1\n")

    def test_latin1_text_is_read_too(self):
        doc = PlainTextExtractor().extract(self.write("a.txt", "Fianza de 60.000 pesos, año".encode("latin-1")), TEXT)
        self.assertIn("año", doc.blocks[0].text)

    def test_pdf_pages_become_blocks_with_their_numbers(self):
        doc = PdfExtractor().extract(self.write("a.pdf", tiny_pdf("Rent is monthly", "Deposit 60000")), PDF)
        self.assertEqual(doc.page_count, 2)
        self.assertEqual([(b.page, b.text.strip()) for b in doc.blocks],
                         [(1, "Rent is monthly"), (2, "Deposit 60000")])

    def test_word_paragraphs_carry_their_heading(self):
        import docx

        document = docx.Document()
        document.add_heading("Deposit", level=1)
        document.add_paragraph("It is 60,000 pesos.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Rent", "38,000"
        path = str(Path(self.folder.name) / "a.docx")
        document.save(path)
        doc = DocxExtractor().extract(path, DOCX)
        self.assertEqual([(b.heading, b.text) for b in doc.blocks],
                         [("Deposit", "It is 60,000 pesos."), ("Deposit", "Rent | 38,000")])


if __name__ == "__main__":
    unittest.main()
