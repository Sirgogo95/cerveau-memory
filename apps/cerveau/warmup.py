"""Warm start: load the embedding model and the chat model when the web server starts,
so the first message is as fast as every other one, and sweep for episodes waiting
for consolidation.

Called from core/wsgi.py and core/asgi.py, which only web servers import (runserver's
serving process, gunicorn, uvicorn), never manage.py commands or tests. Each model
loads in its own background thread, in parallel (the embedder uses the CPU, Ollama the
GPU, or both on Ollama): the server accepts requests immediately, and a message that
arrives before loading finishes just waits for it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from django.conf import settings

from .engine import Engine, get_engine

logger = logging.getLogger("cerveau")


def _loadable_parts(engine: Engine) -> list[tuple[str, Any]]:
    parts = (("embedding model", engine.embedder), ("chat model", engine.llm))
    return [(label, part) for label, part in parts if callable(getattr(part, "load", None))]


def load_part(label: str, part: Any) -> None:
    """Load one model. Failures are logged, never raised: a model that can't load now
    (say, Ollama isn't running yet) will load on first use."""
    started = time.monotonic()
    try:
        part.load()
    except Exception as exc:  # noqa: BLE001  # never take the server down
        logger.warning(
            "Cerveau: %s didn't load at startup (%s); it will load on first use.", label, exc
        )
    else:
        logger.info("Cerveau: %s %s ready (%.1fs).", label, part.model, time.monotonic() - started)


def warm_up(engine: Engine) -> None:
    """Load every model that supports it, one after the other (used directly in tests;
    the server uses start_warm_up)."""
    for label, part in _loadable_parts(engine):
        load_part(label, part)


def sweep_consolidation() -> None:
    """Queue consolidation for episodes that missed it (see tasks.sweep). Logged,
    never raised: the database may not be up yet."""
    from .tasks import sweep  # imported late: tasks imports the engine wiring

    try:
        queued = sweep()
    except Exception as exc:  # noqa: BLE001  # never take the server down
        logger.warning("Cerveau: consolidation sweep failed (%s).", exc)
        return
    if queued:
        logger.info(
            "Cerveau: queued consolidation for %d episode(s); "
            "run `python manage.py db_worker` to process them.", queued,
        )


def start_warm_up() -> list[threading.Thread]:
    """Load each model and run the consolidation sweep, each in its own background
    thread, unless CERVEAU_WARM_START is off."""
    if not settings.CERVEAU_WARM_START:
        return []
    jobs: list[tuple[Callable[..., None], tuple[Any, ...], str]] = [
        (load_part, (label, part), f"cerveau-warm-{label}")
        for label, part in _loadable_parts(get_engine())
    ]
    jobs.append((sweep_consolidation, (), "cerveau-sweep"))
    threads = [
        threading.Thread(target=target, args=args, name=name, daemon=True)
        for target, args, name in jobs
    ]
    for thread in threads:
        thread.start()
    return threads
