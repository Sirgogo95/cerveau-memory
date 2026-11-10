"""Build a ChatProvider from a ModelProfile.

Used by every caller that wires the engine (the Django app, the eval runner), so the
provider always runs with exactly the window and reply cap the prompt was sized for.
"""

from __future__ import annotations

import os

from brain.config import ModelProfile
from brain.ports import ChatProvider

from .ollama import OllamaProvider


def provider_from_profile(profile: ModelProfile, **kwargs) -> ChatProvider:
    """`kwargs` go to the provider's constructor (e.g. base_url, keep_alive)."""
    if profile.provider == "ollama":
        if profile.base_url:
            # A remote model (Ollama Cloud): its own server, and the key from the
            # environment (.env: OLLAMA_API_KEY). Local servers never get the key.
            kwargs = {**kwargs, "base_url": profile.base_url, "schema_in_prompt": True,
                      "api_key": kwargs.get("api_key") or os.environ.get("OLLAMA_API_KEY", "").strip()}
        return OllamaProvider(
            profile.name,
            options={
                **profile.options,
                # The window the budget assumes; Ollama otherwise silently truncates.
                "num_ctx": profile.context_window,
                # Caps the reply at its reserve.
                "num_predict": profile.output_reserve,
            },
            think=profile.think,
            **kwargs,
        )
    raise ValueError(f"Unknown provider {profile.provider!r} in profile {profile.name!r}")
