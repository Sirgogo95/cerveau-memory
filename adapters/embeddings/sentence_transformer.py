"""Embedder backed by a local sentence-transformers model.

Runs on the CPU by default so the GPU stays free for the chat model. The model is
loaded on the first call, not at import, so processes that never embed (or Django's
startup) don't pay for loading PyTorch and the weights. The first use downloads the
model from Hugging Face once; later runs load it from the local cache.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from brain.hashing import text_hash
from brain.types import Embedding


class SentenceTransformerEmbedder:
    """`prefix` is text the model was trained to see before every input (the e5
    family expects "query: "); it comes from the EmbeddingProfile."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", *, device: str = "cpu", prefix: str = ""):
        self.model = model
        self.device = device
        self.prefix = prefix
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model now instead of on first use (e.g. at server start)."""
        self._load()

    def _load(self) -> Any:
        # Locked so a warm-up thread and an early request never load it twice.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
                    except ModuleNotFoundError as error:
                        raise ModuleNotFoundError(
                            "This needs PyTorch and sentence-transformers: pip install -r requirements-optional.txt"
                        ) from error

                    self._model = SentenceTransformer(self.model, device=self.device)
        return self._model

    @property
    def dimensions(self) -> int:
        size = self._load().get_embedding_dimension()
        if size is None:
            raise RuntimeError(f"{self.model} does not report its embedding size.")
        return int(size)

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        # Normalized vectors: cosine similarity is then a plain dot product, and
        # pgvector's cosine distance stays well defined.
        vectors = self._load().encode(
            [self.prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True
        )
        return [
            Embedding(vector=v.tolist(), model=self.model, text_hash=text_hash(t))
            for t, v in zip(texts, vectors)
        ]
