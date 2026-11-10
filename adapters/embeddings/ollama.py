"""Embedder served by a local Ollama server (/api/embed).

One copy of the model, on the GPU, shared by every process that embeds (the web
server and the worker), instead of each loading its own copy into RAM.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import requests

from adapters.llm.ollama import DEFAULT_BASE_URL
from brain.hashing import text_hash
from brain.ports import EmbedderError
from brain.types import Embedding


class OllamaEmbedder:
    """`max_tokens` becomes Ollama's num_ctx: the model's own limit, so long text is
    read whole (Ollama's default context would cut it). Requests share one
    connection (`session`), which matters on Windows: a new connection to localhost
    can cost about 2 s."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int | None = None,
        prefix: str = "",
        keep_alive: str | int | None = None,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.prefix = prefix
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.session = session or requests.Session()
        self._dimensions: int | None = None

    def load(self) -> None:
        """Load the model now (e.g. at server start), with the options real requests
        use, so Ollama doesn't reload it for the first message."""
        self.embed(["warm up"])

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self.embed(["dimensions"])
        assert self._dimensions is not None
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        payload: dict = {"model": self.model, "input": [self.prefix + t for t in texts]}
        if self.max_tokens:
            payload["options"] = {"num_ctx": self.max_tokens}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            response = self.session.post(
                f"{self.base_url}/api/embed", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise EmbedderError(f"Ollama is unreachable at {self.base_url}: {exc}") from exc
        try:
            data = response.json()
        except ValueError:
            data = {}
        if not response.ok or "error" in data:
            detail = data.get("error", response.text)
            raise EmbedderError(f"Ollama returned {response.status_code}: {detail}")
        vectors = data.get("embeddings", [])
        if len(vectors) != len(texts):
            raise EmbedderError(f"Ollama returned {len(vectors)} vectors for {len(texts)} texts.")
        self._dimensions = len(vectors[0])
        return [
            Embedding(vector=_normalized(v), model=self.model, text_hash=text_hash(t))
            for t, v in zip(texts, vectors)
        ]


def _normalized(vector: list[float]) -> list[float]:
    # Unit length, like the sentence-transformers embedder: cosine similarity is then
    # a plain dot product.
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]
