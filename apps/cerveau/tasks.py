"""Background jobs, through Django's task API (backend: django_tasks_db; run them with
`python manage.py db_worker`): consolidating closed episodes, and folding a long
episode's oldest messages into its running summary.

Shared files are read (read_attachment) and then summarized (summarize_attachment)
in the background too; the sweep picks up any that were missed.

Consolidation is queued when a message closes an episode, and by a sweep that catches
anything missed (the worker wasn't running, a job failed, the episode closed while
the server was down). A failed job leaves its episode unconsolidated, so the next
sweep retries it.
"""

from __future__ import annotations

import logging

from django.tasks import task  # pyright: ignore[reportMissingImports]  # Django 6 module, newer than django-types

from brain.consolidation import ConsolidationReport
from brain.consolidation import consolidate_episode as consolidate
from brain.facts import fact_settings
from brain.segmentation import close_stale_episode
from brain.summary import fold_running_summary
from brain.turn import utcnow

from . import models
from .engine import get_engine

logger = logging.getLogger("cerveau")

# How many pending episodes one sweep queues.
SWEEP_LIMIT = 100


def mark_facts_learned(episode_id: int) -> None:
    """Record that facts were learned from the episode (bookkeeping: update())."""
    models.Episode.objects.filter(pk=episode_id).update(facts_learned_at=utcnow())


def fact_summary(report: ConsolidationReport) -> dict | None:
    """What fact learning did, for the task result (visible in the admin)."""
    if report.fact_error:
        return {"error": report.fact_error}
    if report.facts is None:
        return None
    f = report.facts
    return {"added": f.added, "confirmed": f.confirmed, "updated": f.updated,
            "skipped": f.skipped, "unsupported": f.unsupported}


@task()
def consolidate_episode(episode_id: int, force: bool = False) -> dict | None:
    """Consolidate one closed episode with the background model. Returns what was
    stored (kept with the task result, visible in the admin)."""
    engine = get_engine()
    report = consolidate(
        episode_id,
        store=engine.store,
        llm=engine.background_llm,
        embedder=engine.embedder,
        profile=engine.background_profile,
        now=utcnow(),
        force=force,
        facts=fact_settings(engine.config),
        tag_same_score=engine.config.tag_same_score,
        tz=engine.config.tz,
    )
    if report is None:
        logger.info("Cerveau: episode %s needs no consolidation.", episode_id)
        return None
    if report.facts is not None:
        mark_facts_learned(episode_id)
    logger.info(
        "Cerveau: consolidated episode %s: %r, tags %s, entities %s, %d mention(s).",
        episode_id, report.title, report.tags, report.entities, report.mentions,
    )
    return {
        "title": report.title,
        "tags": report.tags,
        "entities": report.entities,
        "mentions": report.mentions,
        "skipped": report.skipped,
        "unsupported": report.unsupported,
        "merged_tags": report.merged,
        "facts": fact_summary(report),
    }


def queue_consolidation(episode_id: int) -> None:
    """Queue consolidation for an episode; the run_turn on_episode_closed callback."""
    consolidate_episode.enqueue(episode_id)


@task()
def update_running_summary(episode_id: int) -> int:
    """Fold the episode's oldest messages into its running summary, as many batches
    as are due. Returns how many messages were folded. Harmless if queued twice: it
    folds only what is due when it runs."""
    engine = get_engine()
    folded = fold_running_summary(
        episode_id, store=engine.store, llm=engine.background_llm, config=engine.config
    )
    if folded:
        logger.info(
            "Cerveau: folded %d message(s) of episode %s into its running summary.",
            folded, episode_id,
        )
    return folded


def queue_running_summary(episode_id: int) -> None:
    """Queue a running-summary fold; the run_turn on_summary_due callback."""
    update_running_summary.enqueue(episode_id)


@task()
def read_attachment(attachment_id: int, force: bool = False) -> dict | None:
    """Read a shared file into passages (no model call), then queue its summary."""
    from adapters.extractors import all_extractors
    from brain.attachments import read_attachment as read
    from brain.config import attachment_chunk_size

    from .attachments import attachment_path

    engine = get_engine()
    attachment = engine.store.get_attachment(attachment_id)
    if attachment is None:
        return None
    assert engine.embedding_profile is not None
    size, overlap = attachment_chunk_size(engine.config, engine.embedding_profile)
    report = read(attachment_id, str(attachment_path(attachment)), store=engine.store, embedder=engine.embedder,
                  extractors=all_extractors(), max_chars=size, overlap=overlap, force=force)
    if report is None:
        return None
    logger.info("Cerveau: read attachment %s (%s): %d passage(s), %d mention(s).",
                attachment_id, report.status, report.chunks, report.mentions)
    if report.status == "done":
        summarize_attachment.enqueue(attachment_id)
    return {"status": report.status, "chunks": report.chunks, "mentions": report.mentions,
            "pages": report.pages, "error": report.error}


@task()
def summarize_attachment(attachment_id: int, force: bool = False) -> dict | None:
    """Title, summaries, and tags for a read file, with the background model."""
    from brain.attachments import summarize_attachment as summarize

    engine = get_engine()
    report = summarize(attachment_id, store=engine.store, llm=engine.background_llm,
                       profile=engine.background_profile, embedder=engine.embedder,
                       tag_same_score=engine.config.tag_same_score, force=force)
    if report is None:
        return None
    logger.info("Cerveau: summarized attachment %s: %r, tags %s.", attachment_id, report.title, report.tags)
    return {"title": report.title, "tags": report.tags, "merged_tags": report.merged,
            "skipped": report.skipped}


def queue_attachment(attachment_id: int) -> None:
    read_attachment.enqueue(attachment_id)


def sweep() -> int:
    """Close the open episode if it has gone stale, then queue every closed episode
    that isn't consolidated yet. Returns how many were queued.

    Run at web-server start, so yesterday's conversation is consolidated before the
    first message of the day rather than competing with it.
    """
    engine = get_engine()
    close_stale_episode(engine.store, utcnow(), engine.config)
    pending = engine.store.pending_consolidation(SWEEP_LIMIT)
    for episode in pending:
        queue_consolidation(episode.id)
    # Files not read or not summarized yet (the worker was down, a job failed).
    unread = engine.store.pending_attachments(SWEEP_LIMIT)
    for attachment in unread:
        queue_attachment(attachment.id)
    unsummarized = engine.store.unsummarized_attachments(SWEEP_LIMIT)
    for attachment in unsummarized:
        summarize_attachment.enqueue(attachment.id)
    return len(pending) + len(unread) + len(unsummarized)
