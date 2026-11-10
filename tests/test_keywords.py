"""brain.keywords: which words of a message are exact terms, and their use in recall
and the search tool."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig, SourceConfig
from brain.embedding import embed_cached
from brain.keywords import MAX_TERMS, keyword_terms
from brain.retrieval import retrieve
from brain.types import MemoryKind, Role
from tests.fakes import FakeEmbedder

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


class KeywordTermsTests(unittest.TestCase):
    def test_identifiers_are_terms(self):
        self.assertEqual(keyword_terms("What happened with INV-2026-00417?"), ["INV-2026-00417"])
        self.assertEqual(keyword_terms("Where do I call invoice_number_seq?"), ["invoice_number_seq"])
        self.assertEqual(keyword_terms("What's in migration 0004?"), ["0004"])
        self.assertEqual(keyword_terms("Check settings.py"), ["settings.py"])

    def test_quotes_are_terms(self):
        self.assertEqual(keyword_terms('Why did I switch to "Royal Canin"?'), ["Royal Canin"])

    def test_ordinary_words_years_and_hyphens_are_not(self):
        self.assertEqual(keyword_terms("My knee is bothering me again."), [])
        self.assertEqual(keyword_terms("We met in 2026 for a follow-up by e-mail."), [])
        self.assertEqual(keyword_terms("Is logrotate set up?"), [])     # meaning finds it

    def test_repeats_are_dropped_and_the_count_is_capped(self):
        self.assertEqual(keyword_terms("INV-1000 inv-1000 Inv-1000"), ["INV-1000"])
        many = " ".join(f"ID{i:04d}" for i in range(20))
        self.assertEqual(len(keyword_terms(many)), MAX_TERMS)


class KeywordRecallTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        episode = self.store.open_episode(T0)
        trace = "Traceback: IntegrityError duplicate key INV-2026-00417 in worker 3 " + "frame " * 40
        [vector] = embed_cached([trace], self.embedder, self.store)
        self.trace = self.store.add_message(episode.id, Role.USER, trace, T0, vector)
        self.store.close_episode(episode.id, T0)
        self.config = EngineConfig(messages=SourceConfig(min_score=0.5))

    def test_a_long_message_holding_the_term_is_found_by_it(self):
        [query] = self.embedder.embed(["What happened with INV-2026-00417?"])
        terms = keyword_terms("What happened with INV-2026-00417?")
        self.assertEqual(retrieve(query, self.store, self.config, terms=terms), [])   # no weight
        on = replace(self.config, messages=replace(self.config.messages, w_keyword=0.6))
        [hit] = retrieve(query, self.store, on, terms=terms)
        self.assertEqual((hit.candidate.id, hit.candidate.keyword), (self.trace.id, 1.0))

    def test_the_search_tool_looks_terms_up_too(self):
        from brain.tools import ToolContext, run_tool
        from brain.types import ToolCall
        from tests.test_engine import PROFILE

        config = replace(EngineConfig(), search_min_score=0.5,
                         messages=replace(EngineConfig().messages, w_keyword=0.6))
        ctx = ToolContext(store=self.store, embedder=self.embedder, config=config, profile=PROFILE,
                          now=T0, tz=UTC)
        result = json.loads(run_tool(ToolCall("search_memory", {"query": "INV-2026-00417"}), ctx))
        self.assertEqual([r["id"] for r in result["results"]], [f"M{self.trace.id}"])


if __name__ == "__main__":
    unittest.main()
