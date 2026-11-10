"""Wiring: builds the engine's dependencies from settings, once per process.

The only place in the Django app that knows which concrete store and provider are in
use. Views call get_engine() and pass its parts to brain.turn.run_turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from django.conf import settings

from adapters.embeddings.factory import embedder_from_profile
from adapters.llm.factory import provider_from_profile
from brain.config import (
    EmbeddingProfile,
    EngineConfig,
    ModelProfile,
    load_embedding_profile,
    load_engine_config,
    load_model_profile,
    with_embedding,
    with_thinking,
)
from brain.ports import ChatProvider, Embedder, Reranker

from .store import DjangoMemoryStore


@dataclass(frozen=True)
class Engine:
    store: DjangoMemoryStore
    llm: ChatProvider
    embedder: Embedder
    profile: ModelProfile
    config: EngineConfig
    # The model for background work (consolidation); the chat model by default.
    background_llm: ChatProvider
    background_profile: ModelProfile
    embedding_profile: EmbeddingProfile | None = None
    # Re-reads recall's shortlist against the message; None unless CERVEAU_RERANKER.
    reranker: Reranker | None = None
    # The chat model with thinking on, for turns sent with the page's Think switch
    # (brain.config.with_thinking); None: the switch has no effect.
    thinking_llm: ChatProvider | None = None
    thinking_profile: ModelProfile | None = None


def _profile(name: str) -> ModelProfile:
    return load_model_profile(settings.CERVEAU_CONFIG_DIR / "models" / f"{name}.toml")


def embedding_profile(name: str) -> EmbeddingProfile:
    return load_embedding_profile(settings.CERVEAU_CONFIG_DIR / "embeddings" / f"{name}.toml")


@cache
def get_engine() -> Engine:
    profile = _profile(settings.CERVEAU_CHAT_MODEL)
    llm = provider_from_profile(profile, base_url=settings.CERVEAU_OLLAMA_URL)
    background_path = settings.CERVEAU_CONFIG_DIR / "models" / f"{settings.CERVEAU_BACKGROUND_MODEL}.toml"
    if settings.CERVEAU_BACKGROUND_MODEL == settings.CERVEAU_CHAT_MODEL or not background_path.exists():
        # No background profile for this model: background work uses the chat one.
        background_profile, background_llm = profile, llm
    else:
        background_profile = _profile(settings.CERVEAU_BACKGROUND_MODEL)
        background_llm = provider_from_profile(
            background_profile, base_url=settings.CERVEAU_OLLAMA_URL
        )
    embedding = embedding_profile(settings.CERVEAU_EMBEDDING_MODEL)
    # The same model with thinking on, for the Think switch (thinking is a request
    # option: Ollama keeps one copy of the model loaded).
    thinking_profile = with_thinking(profile)
    return Engine(
        store=DjangoMemoryStore(),
        llm=llm,
        embedder=embedder_from_profile(embedding, base_url=settings.CERVEAU_OLLAMA_URL),
        profile=profile,
        # The retrieval threshold comes with the embedding model it was calibrated on.
        config=with_embedding(load_engine_config(settings.CERVEAU_ENGINE_CONFIG), embedding),
        background_llm=background_llm,
        background_profile=background_profile,
        embedding_profile=embedding,
        reranker=reranker_from_settings(),
        thinking_llm=provider_from_profile(thinking_profile, base_url=settings.CERVEAU_OLLAMA_URL),
        thinking_profile=thinking_profile,
    )


def reranker_from_settings() -> Reranker | None:
    if not settings.CERVEAU_RERANKER:
        return None
    from adapters.rerankers.cross_encoder import CrossEncoderReranker  # PyTorch only when asked for

    return CrossEncoderReranker(settings.CERVEAU_RERANKER, device=settings.CERVEAU_RERANKER_DEVICE)
