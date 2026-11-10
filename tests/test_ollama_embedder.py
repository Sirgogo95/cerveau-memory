"""OllamaEmbedder and the embedder factory, against a fake HTTP session: no Ollama
server, no model download."""

import math
import unittest

import requests

from adapters.embeddings.factory import embedder_from_profile
from adapters.embeddings.ollama import OllamaEmbedder
from adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
from brain.config import EmbeddingProfile
from brain.hashing import text_hash
from brain.ports import Embedder, EmbedderError
from tests.test_ollama_provider import FakeResponse, FakeSession


def embedder(response=None, error=None, **kwargs):
    session = FakeSession(response, error)
    return OllamaEmbedder("bge-m3", session=session, **kwargs), session


def vectors(*rows):
    return FakeResponse(body={"model": "bge-m3", "embeddings": [list(r) for r in rows]})


class OllamaEmbedderTests(unittest.TestCase):
    def test_satisfies_the_protocol(self):
        e, _ = embedder()
        self.assertIsInstance(e, Embedder)

    def test_posts_every_text_in_one_request_with_the_context_and_prefix(self):
        e, session = embedder(vectors([1, 0], [0, 1]), max_tokens=8192, prefix="query: ",
                              base_url="http://gpu:11434/")
        e.embed(["a", "b"])
        [(url, kwargs)] = session.calls
        self.assertEqual(url, "http://gpu:11434/api/embed")
        self.assertEqual(kwargs["json"], {
            "model": "bge-m3", "input": ["query: a", "query: b"], "options": {"num_ctx": 8192},
        })

    def test_returns_unit_vectors_tagged_with_model_and_hash_of_the_original_text(self):
        e, _ = embedder(vectors([3, 4]), prefix="query: ")
        [embedding] = e.embed(["hello"])
        self.assertEqual(embedding.vector, [0.6, 0.8])
        self.assertTrue(math.isclose(sum(x * x for x in embedding.vector), 1.0))
        self.assertEqual(embedding.model, "bge-m3")
        self.assertEqual(embedding.text_hash, text_hash("hello"))

    def test_empty_input_makes_no_request(self):
        e, session = embedder()
        self.assertEqual(e.embed([]), [])
        self.assertEqual(session.calls, [])

    def test_dimensions_come_from_a_real_vector(self):
        e, session = embedder(vectors([1, 2, 3]))
        self.assertEqual(e.dimensions, 3)
        self.assertEqual(e.dimensions, 3)
        self.assertEqual(len(session.calls), 1)  # asked once, then remembered

    def test_load_sends_a_request_with_the_real_options(self):
        e, session = embedder(vectors([1]), max_tokens=512)
        e.load()
        self.assertEqual(session.calls[0][1]["json"]["options"], {"num_ctx": 512})

    def test_failures_raise_embedder_error(self):
        unreachable, _ = embedder(error=requests.ConnectionError("refused"))
        with self.assertRaisesRegex(EmbedderError, "unreachable"):
            unreachable.embed(["a"])
        unknown, _ = embedder(FakeResponse(404, {"error": "model 'bge-m3' not found"}))
        with self.assertRaisesRegex(EmbedderError, "not found"):
            unknown.embed(["a"])
        short, _ = embedder(vectors([1, 0]))
        with self.assertRaisesRegex(EmbedderError, "1 vectors for 2 texts"):
            short.embed(["a", "b"])


class FactoryTests(unittest.TestCase):
    def test_ollama_profile(self):
        profile = EmbeddingProfile("bge-m3", "ollama", "bge-m3", 0.53, 8192, prefix="p: ")
        e = embedder_from_profile(profile, base_url="http://gpu:11434")
        assert isinstance(e, OllamaEmbedder)
        self.assertEqual((e.model, e.max_tokens, e.prefix, e.base_url),
                         ("bge-m3", 8192, "p: ", "http://gpu:11434"))

    def test_sentence_transformers_profile(self):
        profile = EmbeddingProfile("mini", "sentence-transformers", "all-MiniLM-L6-v2", 0.35, 256)
        e = embedder_from_profile(profile)
        assert isinstance(e, SentenceTransformerEmbedder)
        self.assertEqual(e.model, "all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
