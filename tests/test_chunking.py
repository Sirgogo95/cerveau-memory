"""brain.chunking: passages sized for the embedding model, cut at good places."""

import unittest

from brain.chunking import blocks_from_text, chunk_document
from brain.types import ExtractedDocument, TextBlock

SENTENCE = "The tenant pays the rent on the first day of each month. "


def prose(*blocks: TextBlock) -> ExtractedDocument:
    return ExtractedDocument(tuple(blocks))


class ChunkingTests(unittest.TestCase):
    def test_short_paragraphs_share_a_passage_and_long_ones_split_at_sentences(self):
        chunks = chunk_document(prose(TextBlock("One."), TextBlock("Two.")), 300, 50)
        self.assertEqual([c.text for c in chunks], ["One. Two."])
        long = chunk_document(prose(TextBlock(SENTENCE * 12)), 300, 0)
        self.assertGreater(len(long), 1)
        self.assertTrue(all(len(c.text) <= 300 for c in long))
        self.assertTrue(all(c.text.endswith(".") for c in long))          # never mid-sentence here

    def test_a_cut_repeats_the_last_sentence(self):
        a, b, *_ = chunk_document(prose(TextBlock(SENTENCE * 12)), 300, 80)
        self.assertTrue(b.text.startswith(a.text.split(". ")[-1].rstrip(".")))

    def test_a_word_longer_than_a_passage_is_cut_rather_than_lost(self):
        chunks = chunk_document(prose(TextBlock("x" * 700)), 300, 0)
        self.assertEqual("".join(c.text for c in chunks), "x" * 700)

    def test_a_new_heading_starts_a_new_passage_and_pages_are_kept(self):
        doc = prose(TextBlock("Rent is monthly.", page=1, heading="Rent"),
                    TextBlock("Deposit is 60,000.", page=2, heading="Deposit"),
                    TextBlock("Returned in 30 days.", page=3, heading="Deposit"))
        chunks = chunk_document(doc, 500, 50)
        self.assertEqual([(c.heading, c.pages) for c in chunks], [("Rent", "p.1"), ("Deposit", "pp.2-3")])
        self.assertEqual([c.position for c in chunks], [0, 1])

    def test_code_is_split_by_lines(self):
        code = ExtractedDocument((TextBlock("\n".join(f"x{i} = {i}" for i in range(80))),), code=True)
        chunks = chunk_document(code, 120, 20)
        self.assertTrue(all(len(c.text) <= 120 for c in chunks))
        self.assertTrue(all(c.text.startswith("x") for c in chunks))     # whole lines only
        self.assertIn("x79 = 79", chunks[-1].text)

    def test_empty_documents_and_bad_sizes(self):
        self.assertEqual(chunk_document(prose(TextBlock("   ")), 300, 0), [])
        with self.assertRaises(ValueError):
            chunk_document(prose(), 100, 100)

    def test_markdown_headings_label_what_follows(self):
        blocks = blocks_from_text("# Lease\n\nRent is monthly.\n\n## Deposit\nIt is 60,000.\n")
        self.assertEqual([b.text for b in blocks][0], "# Lease")          # not markdown: kept as text
        blocks = blocks_from_text("# Lease\n\nRent is monthly.\n\n## Deposit\nIt is 60,000.\n", markdown=True)
        self.assertEqual([(b.heading, b.text) for b in blocks],
                         [("Lease", "Rent is monthly."), ("Deposit", "It is 60,000.")])


if __name__ == "__main__":
    unittest.main()
