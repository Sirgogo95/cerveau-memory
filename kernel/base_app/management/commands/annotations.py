"""Inspect or clear pending UI annotations (the dev-only annotation store).

    python manage.py annotations show    # print what is pending
    python manage.py annotations clear    # truncate the store to {}

`clear` is a human-only escape hatch for resetting a wedged store. Claude should
never blanket-clear; it removes only the specific keys it has addressed.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Inspect or clear pending UI annotations."

    def add_arguments(self, parser):
        parser.add_argument(
            "action", choices=["show", "clear"], nargs="?", default="show"
        )

    def handle(self, *args, **opts):
        store = Path(settings.BASE_DIR) / ".annotations" / "pending.json"
        if opts["action"] == "clear":
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text("{}")
            self.stdout.write("Cleared pending annotations.")
            return
        data = json.loads(store.read_text() or "{}") if store.exists() else {}
        self.stdout.write(json.dumps(data, indent=2))
