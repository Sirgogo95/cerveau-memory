"""Lateral inhibition (brain/retrieval.py `inhibit`): near-duplicates give way to
different memories, scores stay as they were, and it is off by default."""

import unittest
from dataclasses import replace

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig
from brain.hashing import text_hash
from brain.retrieval import ScoredCandidate, inhibit, retrieve
from brain.types import Candidate, Embedding, MemoryKind, Role
from tests.fakes import FakeEmbedder
from tests.test_store_contract import at


def scored(i: int, score: float) -> ScoredCandidate:
    return ScoredCandidate(Candidate(kind=MemoryKind.MESSAGE, id=i, text=str(i)), score)


class InhibitTests(unittest.TestCase):
    def test_a_near_duplicate_gives_way_to_a_different_memory(self):
        # 1 and 2 say the same thing; 3 is a little weaker but different.
        vectors = {1: [1.0, 0.0], 2: [0.99, 0.14], 3: [0.0, 1.0]}
        passing = [scored(1, 0.80), scored(2, 0.79), scored(3, 0.70)]
        self.assertEqual([s.candidate.id for s in inhibit(passing, 2, 0.5, vectors)], [1, 3])
        self.assertEqual([s.candidate.id for s in inhibit(passing, 2, 0.05, vectors)], [1, 2])  # weak: order kept
        self.assertEqual([s.score for s in inhibit(passing, 3, 0.5, vectors)], [0.80, 0.70, 0.79])

    def test_a_candidate_without_a_vector_is_never_lowered(self):
        passing = [scored(1, 0.80), scored(2, 0.79), scored(3, 0.70)]
        self.assertEqual([s.candidate.id for s in inhibit(passing, 2, 0.5, {1: [1.0, 0.0], 3: [1.0, 0.0]})], [1, 2])

    def test_settings_are_validated_and_off_by_default(self):
        self.assertEqual(EngineConfig().inhibition, 0.0)
        for bad in (dict(inhibition=1.5), dict(inhibition=-0.1), dict(inhibition_pool=0)):
            with self.assertRaises(ConfigError):
                replace(EngineConfig(), **bad)


class RetrieveTests(unittest.TestCase):
    def test_retrieve_keeps_a_different_memory_over_a_repeat(self):
        store, embedder = InMemoryStore(), FakeEmbedder()
        episode = store.open_episode(at(0))
        texts = ["we hiked the mountain trail with the kids",
                 "we hiked the mountain trail with the kids again",
                 "the kids loved roasting marshmallows after we hiked"]
        for i, text in enumerate(texts):
            store.add_message(episode.id, Role.USER, text, at(i), Embedding(embedder.vector(text), embedder.model, text_hash(text)))
        store.close_episode(episode.id, at(5))
        query = Embedding(embedder.vector("we hiked with the kids"), embedder.model, "q")
        config = replace(EngineConfig(), messages=replace(EngineConfig().messages, min_score=0.0, max_items=2))
        plain = {s.candidate.text for s in retrieve(query, store, config)}
        inhibited = {s.candidate.text for s in retrieve(query, store, replace(config, inhibition=0.5))}
        self.assertEqual(plain, set(texts[:2]))
        self.assertIn(texts[2], inhibited)


if __name__ == "__main__":
    unittest.main()
