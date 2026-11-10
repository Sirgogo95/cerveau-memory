"""Consolidate episodes now, in this process (no worker needed).

    python manage.py consolidate                 # every closed episode not consolidated yet
    python manage.py consolidate --episode 12    # one episode
    python manage.py consolidate --episode 12 --force   # redo it
    python manage.py consolidate --queue         # hand them to the worker instead
"""

from django.core.management.base import BaseCommand

from apps.cerveau.engine import get_engine
from apps.cerveau.tasks import SWEEP_LIMIT, mark_facts_learned, queue_consolidation
from brain.consolidation import ConsolidationError, consolidate_episode
from brain.facts import fact_settings
from brain.segmentation import close_stale_episode
from brain.turn import utcnow


def write_fact_report(out, facts) -> None:
    """The facts lines of a report (shared with extract_facts)."""
    for label, items in (("new facts", facts.added), ("confirmed", facts.confirmed),
                         ("skipped (forgotten)", facts.skipped),
                         ("dropped (no user line states it)", facts.unsupported)):
        if items:
            out.write(f"  {label}: " + "; ".join(items))
    for old, new in facts.updated:
        out.write(f"  updated: {old!r} -> {new!r}")
    if not (facts.added or facts.confirmed or facts.updated or facts.skipped or facts.unsupported):
        out.write("  facts: none")


class Command(BaseCommand):
    help = "Consolidate closed episodes: summaries, tags, entities."

    def add_arguments(self, parser):
        parser.add_argument("--episode", type=int, help="Only this episode.")
        parser.add_argument("--force", action="store_true", help="Redo already consolidated episodes.")
        parser.add_argument("--queue", action="store_true", help="Queue for the worker instead of running here.")

    def handle(self, *args, episode, force, queue, **options):
        engine = get_engine()
        if episode is not None:
            ids = [episode]
        else:
            close_stale_episode(engine.store, utcnow(), engine.config)
            ids = [e.id for e in engine.store.pending_consolidation(SWEEP_LIMIT)]
        if not ids:
            self.stdout.write("Nothing to consolidate.")
            return

        if queue:
            for episode_id in ids:
                queue_consolidation(episode_id)
            self.stdout.write(f"Queued {len(ids)} episode(s) for the worker.")
            return

        for episode_id in ids:
            try:
                report = consolidate_episode(
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
            except ConsolidationError as exc:
                self.stderr.write(f"Episode {episode_id}: failed ({exc}).")
                continue
            if report is None:
                self.stdout.write(f"Episode {episode_id}: nothing to do.")
                continue
            self.stdout.write(self.style.SUCCESS(f"Episode {episode_id}: {report.title!r}"))
            self.stdout.write(f"  tags: {', '.join(report.tags) or '-'}")
            for new, existing in report.merged:
                self.stdout.write(f"  tag {new!r} is {existing!r} (same topic)")
            self.stdout.write(f"  entities: {', '.join(report.entities) or '-'} ({report.mentions} mention(s))")
            if report.skipped:
                self.stdout.write(f"  skipped (forgotten): {', '.join(report.skipped)}")
            if report.unsupported:
                self.stdout.write(f"  dropped (never named by the user): {', '.join(report.unsupported)}")
            if report.fact_error:
                self.stderr.write(f"  facts: failed ({report.fact_error}); run extract_facts later")
            elif report.facts is not None:
                mark_facts_learned(episode_id)
                write_fact_report(self.stdout, report.facts)
