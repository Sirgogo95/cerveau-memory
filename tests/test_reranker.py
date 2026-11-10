"""The reranker: recall reads a wider pool and keeps what the reranker puts first
(brain/retrieval.py `rerank`), off without one; and the cross-encoder adapter, with a
fake library (no model is downloaded)."""

import sys
import types
import unittest
from dataclasses import replace

from adapters.rerankers.cross_encoder import CrossEncoderReranker
from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig
from brain.hashing import text_hash
from brain.ports import Reranker
from brain.retrieval import ScoredCandidate, rerank, retrieve
from brain.types import Candidate, Embedding, MemoryKind, Role
from tests.fakes import FakeEmbedder
from tests.test_store_contract import at


class WordReranker:
    """Scores a text by the share of the query's words it contains; records calls."""

    model = "word-reranker"

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        words = set(query.lower().split())
        return [len(words & set(t.lower().split())) / len(words) for t in texts]


def scored(i: int, text: str, score: float) -> ScoredCandidate:
    return ScoredCandidate(Candidate(kind=MemoryKind.MESSAGE, id=i, text=text), score)


class RerankTests(unittest.TestCase):
    def test_the_reranker_orders_and_scores_stay(self):
        passing = [scored(1, "we talked about the support group", 0.8),
                   scored(2, "she went to the support group on 7 may", 0.7)]
        out = rerank(passing, "when did she go to the support group", WordReranker())
        self.assertEqual([s.candidate.id for s in out], [2, 1])
        self.assertEqual([s.score for s in out], [0.7, 0.8])
        self.assertIsInstance(WordReranker(), Reranker)

    def test_retrieve_reads_a_wider_pool_and_keeps_what_the_reranker_prefers(self):
        store, embedder = InMemoryStore(), FakeEmbedder()
        episode = store.open_episode(at(0))
        texts = ["the support group was great", "i love the support group", "the support group helps",
                 "i went to the support group on 7 may"]
        for i, text in enumerate(texts):
            store.add_message(episode.id, Role.USER, text, at(i),
                              Embedding(embedder.vector(text), embedder.model, text_hash(text)))
        store.close_episode(episode.id, at(9))
        question = "when did i go to the support group"
        query = Embedding(embedder.vector("the support group"), embedder.model, "q")
        config = replace(EngineConfig(), messages=replace(EngineConfig().messages, min_score=0.0, max_items=1))
        plain = retrieve(query, store, config, text=question)
        reranker = WordReranker()
        reranked = retrieve(query, store, config, reranker=reranker, text=question)
        self.assertNotEqual(plain[0].candidate.text, texts[3])
        self.assertEqual(reranked[0].candidate.text, texts[3])
        self.assertEqual(len(reranker.calls[0][1]), 4)            # the pool: up to 4 x max_items

    def test_off_without_a_reranker_or_without_the_message(self):
        reranker = WordReranker()
        store = InMemoryStore()
        query = Embedding(FakeEmbedder().vector("x"), "fake-embedder", "q")
        retrieve(query, store, EngineConfig(), reranker=reranker)  # no text: nothing to rerank against
        self.assertEqual(reranker.calls, [])

    def test_pool_and_sources_are_validated(self):
        with self.assertRaises(ConfigError):
            replace(EngineConfig(), rerank_pool=0)
        with self.assertRaises(ConfigError):
            replace(EngineConfig(), rerank_sources="message,gossip")
        self.assertEqual(replace(EngineConfig(), rerank_sources="message,fact").rerank_sources, "message,fact")

    def test_only_the_chosen_sources_are_reranked(self):
        store, embedder = InMemoryStore(), FakeEmbedder()
        episode = store.open_episode(at(0))
        for i, text in enumerate(["the support group was great", "i went to the support group on 7 may"]):
            store.add_message(episode.id, Role.USER, text, at(i),
                              Embedding(embedder.vector(text), embedder.model, text_hash(text)))
        store.close_episode(episode.id, at(9))
        query = Embedding(embedder.vector("the support group"), embedder.model, "q")
        config = replace(EngineConfig(), messages=replace(EngineConfig().messages, min_score=0.0, max_items=1))
        reranker = WordReranker()
        retrieve(query, store, replace(config, rerank_sources="fact"), reranker=reranker, text="when did i go")
        self.assertEqual(reranker.calls, [])                      # messages left in their order
        retrieve(query, store, replace(config, rerank_sources="message"), reranker=reranker, text="when did i go")
        self.assertEqual(len(reranker.calls), 1)


class CrossEncoderTests(unittest.TestCase):
    def setUp(self):
        self.loads = []
        loads = self.loads

        class FakeCrossEncoder:
            def __init__(self, model, device, max_length):
                loads.append((model, device, max_length))

            def predict(self, pairs, batch_size, show_progress_bar):
                return [len(text) for _, text in pairs]

        module = types.ModuleType("sentence_transformers")
        module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
        self._saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = module
        self.addCleanup(self.restore)

    def restore(self):
        if self._saved is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = self._saved

    def test_loads_once_on_first_use_and_scores_pairs(self):
        reranker = CrossEncoderReranker("some/model")
        self.assertEqual(self.loads, [])                           # nothing loaded yet
        self.assertEqual(reranker.score("q", ["ab", "abcd"]), [2.0, 4.0])
        self.assertEqual(reranker.score("q", []), [])
        reranker.score("q", ["a"])
        self.assertEqual(self.loads, [("some/model", "cpu", 512)])


if __name__ == "__main__":
    unittest.main()
