"""Link existing facts to the entities they're about, for entity blocks
(docs/features/ENTITY_PROFILES.md).

Consolidation links new facts itself; this backfills facts learned before entity
blocks existed. Safe to rerun: a link made twice is kept once.

    python manage.py link_fact_entities --dry-run   # count what would be linked
    python manage.py link_fact_entities
"""

from django.core.management.base import BaseCommand

from apps.cerveau import models
from apps.cerveau.engine import get_engine
from brain.entity_links import link_fact
from brain.text import phrase_keys


class Command(BaseCommand):
    help = "Link existing facts to the entities their statements name."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Count the links without writing them.")

    def handle(self, *args, dry_run, **options):
        store = get_engine().store
        facts = store.get_facts(list(models.Fact.objects.filter(status="A").order_by("id").values_list("pk", flat=True)))
        linked = links = 0
        for fact in facts:
            entities = store.match_entities(phrase_keys(fact.statement))
            if not entities:
                continue
            if not dry_run:
                link_fact(store, fact)
            linked += 1
            links += len(entities)
        verb = "Would link" if dry_run else "Linked"
        self.stdout.write(self.style.SUCCESS(f"{verb} {linked} of {len(facts)} fact(s), {links} link(s)."))
