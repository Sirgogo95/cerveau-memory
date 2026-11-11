"""Reranker backed by a local sentence-transformers cross-encoder.

A cross-encoder reads the question and a memory together and scores how well the
memory answers it, which an embedding (each text encoded alone, ahead of time)
can't. It's slower, so recall uses it only on the shortlist the embedder found.

Runs on the CPU by default, so the GPU stays free for the chat model and the
embedder. The model is loaded on the first call, not at import: processes that
never rerank (Django's startup, the tests) don't load PyTorch. The first use
downloads the model from Hugging Face once; later runs load it from the local cache.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """`max_length` caps the tokens of each question-and-memory pair: memories longer
    than that are judged by their beginning (and it keeps scoring fast on a CPU)."""

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str = "cpu", max_length: int = 512,
                 batch_size: int = 16):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
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
                        from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
                    except ModuleNotFoundError as error:
                        raise ModuleNotFoundError(
                            "This needs PyTorch and sentence-transformers: pip install -r requirements-optional.txt"
                        ) from error

                    self._model = CrossEncoder(self.model, device=self.device, max_length=self.max_length)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(query, t) for t in texts], batch_size=self.batch_size,
                                      show_progress_bar=False)
        return [float(s) for s in scores]
