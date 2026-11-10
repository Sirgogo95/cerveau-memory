"""Shared files in recall: passages as candidates (at most two per file), prompt lines,
the files marked in the prompt and the consolidation transcript, the "still reading"
note, and the search and read tools."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from adapters.stores.in_memory import InMemoryStore
from brain.attachments import embedding_texts, mark_shared_files, reading_notes
from brain.config import EngineConfig, SourceConfig
from brain.embedding import embed_cached
from brain.prompt import memory_line
from brain.recall import recall
from brain.retrieval import ScoredCandidate, retrieve
from brain.tools import ToolContext, run_tool
from brain.types import Candidate, Chunk, MemoryKind, Role, ToolCall
from tests.fakes import FakeEmbedder
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
CONFIG = EngineConfig(timezone="UTC", recent_episodes=0, attachments=SourceConfig(min_score=0.2, max_items=5))
PASSAGES = ["the deposit is 76,000 pesos returned within thirty days",
            "the deposit cannot pay the last month of rent",
            "the deposit for a pet is 10,000 pesos",
            "the tenant pays for electricity"]


class FileTestCase(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.episode = self.store.open_episode(T0)
        [vector] = embed_cached(["here is the lease"], self.embedder, self.store)
        self.message = self.store.add_message(self.episode.id, Role.USER, "here is the lease", T0, vector)
        found = self.store.add_attachment("lease.md", "text/markdown", 100, "h", "h.md", T0)
        assert found is not None
        self.attachment = found[0]
        self.store.link_attachment(self.message.id, self.attachment.id)

    def read(self, texts=PASSAGES):
        chunks = [Chunk(i, t, page_start=i + 1) for i, t in enumerate(texts)]
        vectors = embed_cached(texts, self.embedder, self.store)
        self.store.save_chunks(self.attachment.id, list(zip(chunks, vectors)), [], len(texts))


class RetrievalTests(FileTestCase):
    def test_passages_are_recalled_two_per_file_even_from_this_conversation(self):
        self.read()
        [query] = self.embedder.embed(["how much is the deposit"])
        found = retrieve(query, self.store, CONFIG, current_episode_id=self.episode.id)
        passages = [r.candidate for r in found if r.candidate.kind is MemoryKind.ATTACHMENT]
        self.assertEqual(len(passages), 2)                      # three mention the deposit
        self.assertTrue(all(p.parent_id == self.attachment.id for p in passages))
        self.assertTrue(all("deposit" in p.text for p in passages))

    def test_a_passage_line_names_the_file_and_where(self):
        line = memory_line(ScoredCandidate(Candidate(
            MemoryKind.ATTACHMENT, 9, "the deposit is 76,000", at=T0, parent_id=3, label="lease.md, p.4"), 0.7))
        self.assertEqual(line, "- [A3 · Thu 2026-10-01 · lease.md, p.4] the deposit is 76,000")

    def test_the_vector_gets_the_documents_context(self):
        [text] = embedding_texts([Chunk(0, "The tenant pays for electricity.", heading="Utilities")],
                                 "Residential Lease", "lease.md", "Here's the lease for Bella Vista")
        self.assertEqual(text, "Residential Lease > Utilities (lease.md; shared with: Here's the lease for "
                               "Bella Vista)\nThe tenant pays for electricity.")


class ConversationTests(FileTestCase):
    def test_messages_that_shared_a_file_say_so(self):
        [marked] = mark_shared_files([self.message], self.store)
        self.assertEqual(marked.content, 'here is the lease\n[shared a file: A1 "lease.md"]')
        self.assertEqual(self.store.get_messages(self.episode.id)[0].content, "here is the lease")  # not stored

    def test_a_file_not_read_yet_gets_a_note_in_recall(self):
        [query] = self.embedder.embed(["here is the lease"])
        notes = recall("here is the lease", query, self.store, CONFIG, embedder=self.embedder,
                       episode_id=self.episode.id, now=T0).notes
        self.assertEqual(len(notes), 1)
        self.assertIn("still being read", notes[0])
        self.read()
        done = recall("here is the lease", query, self.store, CONFIG, embedder=self.embedder,
                      episode_id=self.episode.id, now=T0).notes
        self.assertEqual(done, [])
        failed = replace(self.attachment, extraction_status="failed", error="No text found.")
        self.assertIn("No text found.", reading_notes([failed])[0])


class ToolTests(FileTestCase):
    def context(self):
        return ToolContext(store=self.store, embedder=self.embedder,
                           config=replace(CONFIG, search_min_score=0.2), profile=PROFILE, now=T0, tz=UTC)

    def call(self, name, **arguments):
        return json.loads(run_tool(ToolCall(name, arguments), self.context()))

    def test_search_finds_passages_with_the_files_id(self):
        self.read()
        result = self.call("search_memory", query="tenant pays electricity", kinds=["attachment"])
        [first, *_] = result["results"]
        self.assertEqual((first["id"], first["kind"]), (f"A{self.attachment.id}", "attachment"))
        self.assertEqual(first["file"], "lease.md, p.4")

    def test_read_memory_opens_a_file_summary_or_passages(self):
        pending = self.call("read_memory", id=f"A{self.attachment.id}")
        self.assertEqual(pending["note"], "Still being read.")
        self.read()
        self.store.save_attachment_summary(self.attachment.id, "Naco lease", "A lease.", "A lease, at length.")
        self.assertEqual(self.call("read_memory", id=f"A{self.attachment.id}", detail="short")["summary"], "A lease.")
        passages = self.call("read_memory", id=f"A{self.attachment.id}", detail="passages", message_range=[2, 3])
        self.assertEqual([p["passage"] for p in passages["passages"]], [2, 3])
        self.assertEqual(passages["total_passages"], 4)
        self.assertIn("error", self.call("read_memory", id="A99"))


if __name__ == "__main__":
    unittest.main()
