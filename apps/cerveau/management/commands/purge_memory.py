"""Erase forgotten memory for good: python manage.py purge_memory.

    python manage.py purge_memory --dry-run                  # what would go
    python manage.py purge_memory                            # forgotten (Z) for 30+ days
    python manage.py purge_memory --status I Z --older-than 0
    python manage.py purge_memory --episode 12 --older-than 0

Only memory you forgot (Z) or set aside (I) can be purged; active memory never is.
Asks for confirmation unless --no-input. See apps/cerveau/purge.py for what goes.
"""

from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.cerveau.purge import KINDS, execute_purge, plan_purge


class Command(BaseCommand):
    help = "Erase forgotten memory for good, with its audit copies and derived memory."

    def add_arguments(self, parser):
        parser.add_argument("--status", nargs="+", default=["Z"], choices=["I", "Z"],
                            help="Which statuses to purge (default: Z, forgotten).")
        parser.add_argument("--older-than", type=float, default=30,
                            help="Grace period in days since the status was set (default 30).")
        parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
        parser.add_argument("--episode", type=int, action="append", dest="episodes",
                            help="Only this episode (repeatable); it must be forgotten or set aside.")
        parser.add_argument("--dry-run", action="store_true", help="List what would be erased.")
        parser.add_argument("--no-input", action="store_false", dest="interactive",
                            help="Don't ask for confirmation.")

    def handle(self, *args, **options):
        if options["older_than"] < 0:
            raise CommandError("--older-than can't be negative.")
        plan = plan_purge(
            options["status"], timedelta(days=options["older_than"]), options["kinds"],
            now=timezone.now(), episode_ids=options["episodes"],
        )
        if plan.empty:
            self.stdout.write("Nothing to purge.")
            return
        self.stdout.write(f"Would erase {plan.summary()}.")
        for label, ids in (("episodes", plan.episodes), ("facts", plan.facts),
                           ("facts going with them (only source, or history)", plan.derived_facts),
                           ("entities", plan.entities), ("tags", plan.tags),
                           ("files", plan.attachments)):
            if ids:
                self.stdout.write(f"  {label}: {', '.join(map(str, ids))}")
        if options["dry_run"]:
            return
        if options["interactive"]:
            answer = input("This can't be undone. Type 'purge' to erase: ")
            if answer.strip() != "purge":
                self.stdout.write("Cancelled.")
                return
        execute_purge(plan)
        self.stdout.write(self.style.SUCCESS(f"Erased {plan.summary()}."))
