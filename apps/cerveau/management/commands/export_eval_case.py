"""Turn real episodes into an eval scenario, then add the questions by hand.

    python manage.py export_eval_case --episode 12 --episode 15 --name family

Writes evals/cases/personal/family.json (git-ignored: it holds your real
conversations) with the episodes' messages, summaries, and entities, and an empty
`queries` list. Add questions that should recall these memories, and a few that
should recall nothing, then run `python -m evals`.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.cerveau.engine import get_engine
from apps.cerveau.models import Entity, EpisodeTag

PERSONAL = Path(settings.BASE_DIR) / "evals" / "cases" / "personal"


def episode_case(store, episode) -> dict:
    messages = store.get_messages(episode.id)
    case = {
        "key": f"e{episode.id}",
        "messages": [[m.role.value, m.content] for m in messages],
    }
    if episode.summary_short:
        case["summary"] = episode.summary_short
        case["summary_long"] = episode.summary_long
    # Active tags only: a forgotten tag no longer counts in recall.
    tags = EpisodeTag.objects.filter(episode_id=episode.id, tag__status="A").order_by("tag__name")
    if tags:
        case["tags"] = [t.tag.name for t in tags.select_related("tag")]
    entities = Entity.objects.filter(mentions__message__episode_id=episode.id).distinct()
    case["entities"] = [
        {"name": e.name, "type": e.type, "aliases": list(e.aliases)} for e in entities
    ]
    return case


class Command(BaseCommand):
    help = "Export episodes as an eval scenario (evals/cases/personal/<name>.json)."

    def add_arguments(self, parser):
        parser.add_argument("--episode", type=int, action="append", required=True,
                            help="An episode to include (repeat for several).")
        parser.add_argument("--name", required=True, help="Scenario name and file name.")
        parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")

    def handle(self, *args, episode, name, force, **options):
        store = get_engine().store
        episodes = store.get_episodes(episode)
        missing = set(episode) - {e.id for e in episodes}
        if missing:
            raise CommandError(f"No such episode(s): {', '.join(map(str, sorted(missing)))}")

        path = PERSONAL / f"{name}.json"
        if path.exists() and not force:
            raise CommandError(f"{path} exists; use --force to overwrite it.")
        scenario = {
            "name": name,
            "episodes": [episode_case(store, e) for e in episodes],
            "queries": [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Wrote {path}"))
        self.stdout.write("Now add queries, for example:")
        key = scenario["episodes"][0]["key"]
        self.stdout.write(f'  {{"text": "a question about it", "expect": ["{key}:0", "{key}"]}}')
        self.stdout.write('  {"text": "a question it should not answer", "expect": []}')
        self.stdout.write(f'"{key}:0" is its first message; "{key}" its summary.')
