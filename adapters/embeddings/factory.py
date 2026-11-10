"""Build an Embedder from an EmbeddingProfile.

Used by every caller that wires the engine (the Django app, embed_memory, the eval
runner), so each runs the model exactly as its profile describes.
"""

from __future__ import annotations

from brain.config import EmbeddingProfile
from brain.ports import Embedder


def embedder_from_profile(profile: EmbeddingProfile, *, base_url: str | None = None) -> Embedder:
    """`base_url` is the Ollama server, for Ollama profiles (default: local)."""
    if profile.provider == "ollama":
        from adapters.llm.ollama import DEFAULT_BASE_URL

        from .ollama import OllamaEmbedder

        return OllamaEmbedder(
            profile.model,
            base_url=base_url or DEFAULT_BASE_URL,
            max_tokens=profile.max_tokens,
            prefix=profile.prefix,
        )
    # sentence-transformers (the profile already rejected anything else). Imported
    # here so Ollama setups never import PyTorch.
    from .sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(profile.model, prefix=profile.prefix)
