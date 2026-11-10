"""brain.attachments: reading a shared file into passages, and summarizing it."""

import json
import unittest
from datetime import UTC, datetime

from adapters.stores.in_memory import InMemoryStore
from brain.attachments import read_attachment, sample_passages, summarize_attachment
from brain.types import Chunk, ExtractedDocument, MemoryKind, Role, TextBlock
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


class FakeExtractor:
    def __init__(self, document=None, error=None):
        self.document, self.error = document, error

    def supports(self, media_type: str) -> bool:
        return media_type == "text/plain"

    def extract(self, path: str, media_type: str) -> ExtractedDocument:
        if self.error:
            raise self.error
        assert self.document is not None
        return self.document


LEASE = ExtractedDocument((
    TextBlock("The monthly rent is 38,000 pesos.", page=1, heading="Rent"),
    TextBlock("Mr. Peña returns the deposit within 30 days.", page=2, heading="Deposit"),
), page_count=2)


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        episode = self.store.open_episode(T0)
        message = self.store.add_message(episode.id, Role.USER, "the lease", T0)
        found = self.store.add_attachment("lease.txt", "text/plain", 100, "h", "h.txt", T0)
        assert found is not None
        self.attachment = found[0]
        self.store.link_attachment(message.id, self.attachment.id)

    def read(self, extractor, **kwargs):
        return read_attachment(self.attachment.id, "unused", store=self.store, embedder=self.embedder,
                               extractors=[extractor], max_chars=300, overlap=40, **kwargs)

    def test_passages_are_embedded_and_known_names_recorded(self):
        pena = self.store.upsert_entity("Mr. Peña", "person", ["Peña"])
        assert pena is not None
        report = self.read(FakeExtractor(LEASE))
        assert report is not None
        self.assertEqual((report.status, report.chunks, report.mentions, report.pages), ("done", 2, 1, 2))
        [hit] = self.store.mention_candidates(MemoryKind.ATTACHMENT, [pena.id], 5)
        self.assertIn("deposit", hit.text)
        self.assertEqual(hit.label, "lease.txt, p.2")
        self.assertIsNone(self.read(FakeExtractor(LEASE)))               # already read
        self.assertIsNotNone(self.read(FakeExtractor(LEASE), force=True))

    def test_unsupported_failed_and_empty_files_are_recorded(self):
        other = self.store.add_attachment("x.bin", "application/octet-stream", 1, "b", "b.bin", T0)
        assert other is not None
        report = read_attachment(other[0].id, "unused", store=self.store, embedder=self.embedder,
                                 extractors=[FakeExtractor(LEASE)], max_chars=300, overlap=40)
        assert report is not None
        self.assertEqual(report.status, "unsupported")
        failed = self.read(FakeExtractor(error=ValueError("broken xref")))
        assert failed is not None
        self.assertEqual(failed.status, "failed")
        stored = self.store.get_attachment(self.attachment.id)
        assert stored is not None
        self.assertIn("broken xref", stored.error)
        empty = self.read(FakeExtractor(ExtractedDocument((), page_count=3)), force=True)
        assert empty is not None
        self.assertIn("scanned PDF", empty.error)


class SummaryTests(ReadTests):
    def test_one_call_writes_title_summaries_and_tags(self):
        self.read(FakeExtractor(LEASE))
        llm = FakeLLM(json.dumps({"language": "English", "title": "Naco apartment lease",
                                  "summary_short": "A lease for an apartment in Naco.",
                                  "summary_long": "The document says the rent is 38,000 pesos.",
                                  "english_tags": ["Housing", "legal"]}))
        report = summarize_attachment(self.attachment.id, store=self.store, llm=llm, profile=PROFILE,
                                      embedder=self.embedder)
        assert report is not None
        self.assertEqual((report.title, report.tags), ("Naco apartment lease", ["housing", "legal"]))
        stored = self.store.get_attachment(self.attachment.id)
        assert stored is not None
        self.assertEqual(stored.summary_short, "A lease for an apartment in Naco.")
        prompt = llm.prompts[0][-1].content
        self.assertIn('"lease.txt" (2 pages)', prompt)
        self.assertIn("Headings: Rent; Deposit", prompt)
        self.assertIn("never to the user", " ".join(prompt.split()))       # attributed to the document
        self.assertIsNone(summarize_attachment(self.attachment.id, store=self.store, llm=llm, profile=PROFILE,
                                               embedder=self.embedder))           # already summarized

    def test_an_unreadable_reply_still_leaves_a_title(self):
        self.read(FakeExtractor(LEASE))
        report = summarize_attachment(self.attachment.id, store=self.store, llm=FakeLLM("not json"),
                                      profile=PROFILE, embedder=self.embedder)
        assert report is not None
        self.assertEqual(report.title, "lease.txt")

    def test_nothing_to_summarize_before_reading(self):
        self.assertIsNone(summarize_attachment(self.attachment.id, store=self.store, llm=FakeLLM("{}"),
                                               profile=PROFILE, embedder=self.embedder))


class SampleTests(unittest.TestCase):
    def test_the_start_then_an_even_sample_within_the_budget(self):
        chunks = [Chunk(i, f"passage {i} " + "x" * 90, heading="H" if i < 5 else "") for i in range(50)]
        text = sample_passages(chunks, 1500)
        self.assertLessEqual(len(text), 1500 + 200)
        self.assertTrue(text.startswith("Headings: H"))
        self.assertIn("passage 0 ", text)
        self.assertIn("[...]", text)
        later = [int(p.split()[1]) for p in text.split("\n\n") if p.startswith("passage")]
        self.assertGreater(max(later), 25)                                # reaches the second half


if __name__ == "__main__":
    unittest.main()
