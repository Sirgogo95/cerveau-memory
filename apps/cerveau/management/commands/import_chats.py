"""Bring an exported chat history into memory (docs/PERSONAL_CHATS_EVAL.md, 4.3).

    python manage.py import_chats --source chatgpt "C:\\exports\\chatgpt" --dry-run
    python manage.py import_chats --source chatgpt "C:\\exports\\chatgpt" --select selected-chatgpt.json

Each chat is replayed as a live conversation (brain/importing.py): a new episode per
chat, split further by the time gap and the change of subject, every message at its
original time. Chats already imported are skipped; chats marked "do not remember" in
the source app are skipped unless --include-do-not-remember. The closed episodes are
then queued for consolidation, oldest first (the worker, or `consolidate`, runs it).
Importing closes the episode that is open now: run it on the database you mean.
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from adapters.chats import read_chatgpt
from apps.cerveau.engine import get_engine
from apps.cerveau.tasks import queue_consolidation
from brain.importing import import_chat

READERS = {"chatgpt": read_chatgpt}


def selected_ids(path: Path) -> set[str]:
    """The chats to import: the selection page's file ({"in": [...]}) or one id per line."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return set(json.loads(text).get("in", []))
    return {line.strip() for line in text.splitlines() if line.strip()}


class Command(BaseCommand):
    help = "Import an exported chat history (ChatGPT) into memory."

    def add_arguments(self, parser):
        parser.add_argument("path", help="The export's folder.")
        parser.add_argument("--source", choices=sorted(READERS), required=True)
        parser.add_argument("--select", type=Path, help="Only these chats (the selection page's file, or ids).")
        parser.add_argument("--include-do-not-remember", action="store_true",
                            help="Also import chats marked \"do not remember\" in the source app.")
        parser.add_argument("--dry-run", action="store_true", help="Show what would be imported; store nothing.")
        parser.add_argument("--no-queue", action="store_true", help="Don't queue consolidation.")

    def handle(self, *args, path, source, select, include_do_not_remember, dry_run, no_queue, **options):
        if not Path(path).is_dir():
            raise CommandError(f"{path} is not a folder.")
        chats = READERS[source](path)
        if select is not None:
            wanted = selected_ids(select)
            missing = wanted - {c.conversation_id for c in chats}
            if missing:
                self.stdout.write(f"{len(missing)} selected chat(s) not in the export.")
            chats = [c for c in chats if c.conversation_id in wanted]
        engine = get_engine()
        done = engine.store.imported_conversations(source)
        skipped_flag = [c for c in chats if c.do_not_remember and not include_do_not_remember]
        skipped_done = [c for c in chats if c.conversation_id in done]
        chats = [c for c in chats if c not in skipped_flag and c not in skipped_done]
        self.stdout.write(f"{len(chats)} chat(s) to import ({sum(len(c.lines) for c in chats)} messages); "
                          f"skipped: {len(skipped_done)} already imported, {len(skipped_flag)} marked do not remember.")
        if dry_run or not chats:
            for chat in chats:
                self.stdout.write(f"  {chat.started_at:%Y-%m-%d}  {len(chat.lines):5} messages  {chat.title}")
            return

        closed: list[int] = []
        for number, chat in enumerate(chats, 1):
            before = len(closed)
            with transaction.atomic():         # a chat goes in whole or not at all
                import_chat(chat, store=engine.store, embedder=engine.embedder, config=engine.config,
                            on_close=closed.append)
            self.stdout.write(f"[{number}/{len(chats)}] {chat.started_at:%Y-%m-%d}  {len(chat.lines)} messages, "
                              f"{len(closed) - before} episode(s)")
        if no_queue:
            self.stdout.write(f"{len(closed)} episode(s) closed, not queued: run `consolidate` when ready.")
            return
        episodes = sorted(engine.store.get_episodes(closed), key=lambda e: (e.ended_at or e.started_at, e.id))
        for episode in episodes:
            queue_consolidation(episode.id)
        self.stdout.write(f"{len(episodes)} episode(s) queued for consolidation (oldest first).")
