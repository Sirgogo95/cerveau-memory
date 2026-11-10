"""Learn facts from conversations that are already consolidated (docs/FACTS.md).

Consolidation learns facts itself; this backfills episodes consolidated before facts
existed, or whose fact step failed. Safe to rerun: episodes whose facts were already
learned are skipped unless --force (which would count their facts as confirmed again).

    python manage.py extract_facts                 # every episode not done yet
    python manage.py extract_facts --episode 12    # one episode
    python manage.py extract_facts --episode 12 --force
"""

from django.core.management.base import BaseCommand

from apps.cerveau import models
from apps.cerveau.engine import get_engine
from apps.cerveau.management.commands.consolidate import write_fact_report
from apps.cerveau.tasks import mark_facts_learned
from brain.consolidation import ConsolidationError
from brain.facts import FactSettings, learn_facts
from brain.ports import ChatProviderError
from brain.turn import utcnow


class Command(BaseCommand):
    help = "Learn facts from consolidated episodes that don't have them yet."

    def add_arguments(self, parser):
        parser.add_argument("--episode", type=int, help="Only this episode.")
        parser.add_argument("--force", action="store_true", help="Redo episodes already done.")
        parser.add_argument("--limit", type=int, default=50, help="At most this many episodes.")

    def handle(self, *args, episode, force, limit, **options):
        engine = get_engine()
        episodes = models.Episode.active_objects.filter(consolidated_at__isnull=False)
        if episode is not None:
            episodes = episodes.filter(pk=episode)
        if not force:
            episodes = episodes.filter(facts_learned_at__isnull=True)
        ids = list(episodes.order_by("started_at").values_list("pk", flat=True)[:limit])
        if not ids:
            self.stdout.write("No episode needs facts.")
            return
        settings = FactSettings(
            same_score=engine.config.fact_same_score,
            related_score=engine.config.fact_related_score,
            max_facts=engine.config.max_facts,
        )
        for episode_id in ids:
            messages = engine.store.get_messages(episode_id)
            try:
                report = learn_facts(
                    episode_id, messages, store=engine.store, llm=engine.background_llm,
                    embedder=engine.embedder, profile=engine.background_profile,
                    settings=settings, now=utcnow(), tz=engine.config.tz,
                )
            except (ConsolidationError, ChatProviderError) as exc:
                self.stderr.write(f"Episode {episode_id}: failed ({exc}).")
                continue
            mark_facts_learned(episode_id)
            self.stdout.write(self.style.SUCCESS(f"Episode {episode_id}"))
            write_fact_report(self.stdout, report)
