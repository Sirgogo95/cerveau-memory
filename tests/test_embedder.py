"""SentenceTransformerEmbedder with a fake sentence_transformers module: no PyTorch
import, no model download."""

import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
from brain.hashing import text_hash


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeModel:
    instances = 0

    def __init__(self, name, device):
        time.sleep(0.05)  # widen the window for a double load
        FakeModel.instances += 1
        self.name, self.device = name, device

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        assert normalize_embeddings
        self.seen = list(texts)
        return [FakeVector([float(len(t)), 0.0, 0.0]) for t in texts]


def fake_library():
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"sentence_transformers": module})


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def setUp(self):
        FakeModel.instances = 0
        patcher = fake_library()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nothing_loads_until_needed(self):
        SentenceTransformerEmbedder("m")
        self.assertEqual(FakeModel.instances, 0)

    def test_concurrent_loads_build_the_model_once(self):
        embedder = SentenceTransformerEmbedder("m")
        threads = [threading.Thread(target=embedder.load) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(FakeModel.instances, 1)

    def test_embed_after_load_reuses_the_model(self):
        embedder = SentenceTransformerEmbedder("m")
        embedder.load()
        embedder.embed(["hello"])
        self.assertEqual(FakeModel.instances, 1)

    def test_embeddings_carry_model_and_hash(self):
        [embedding] = SentenceTransformerEmbedder("m").embed(["hello"])
        self.assertEqual((embedding.model, embedding.text_hash), ("m", text_hash("hello")))
        self.assertEqual(embedding.vector, [5.0, 0.0, 0.0])

    def test_prefix_goes_before_every_text_but_not_into_the_hash(self):
        embedder = SentenceTransformerEmbedder("m", prefix="query: ")
        [embedding] = embedder.embed(["hello"])
        self.assertEqual(embedder._model.seen, ["query: hello"])
        self.assertEqual(embedding.text_hash, text_hash("hello"))

    def test_runs_on_the_cpu_by_default(self):
        embedder = SentenceTransformerEmbedder("m")
        embedder.load()
        self.assertEqual(embedder._model.device, "cpu")

    def test_dimensions_come_from_the_model(self):
        self.assertEqual(SentenceTransformerEmbedder("m").dimensions, 3)

    def test_empty_input_needs_no_model(self):
        self.assertEqual(SentenceTransformerEmbedder("m").embed([]), [])
        self.assertEqual(FakeModel.instances, 0)


if __name__ == "__main__":
    unittest.main()
