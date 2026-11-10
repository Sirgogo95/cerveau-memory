"""Answer the personal chats eval's questions as the app would (docs/PERSONAL_CHATS_EVAL.md, 6).

    python manage.py personal_chats_eval --questions questions-chatgpt.json --run P1 --out p1.jsonl
    python manage.py personal_chats_eval --questions questions-chatgpt.json --run P0 --out p0.jsonl

P1 answers from the memory in the database (the imported chats), P0 from an empty
memory (the baseline: what the model says without Cerveau). Each question is one fresh
turn through run_turn, recall and memory tools included, and its writes are rolled
back, so asking leaves the memory as imported. "Today" is the day after the last stored
message, so "last time" means the imported history. Answers at temperature 0, saved as
they come; a rerun skips the questions already in --out. Grading is by hand.
"""

import json
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Max

from adapters.llm.factory import provider_from_profile
from adapters.stores.in_memory import InMemoryStore
from apps.cerveau import models
from apps.cerveau.engine import get_engine
from brain.config import with_min_score, with_thinking
from brain.turn import run_turn
from evals.turns import with_settings

EMPTY_TRIES = 3   # an empty reply is a failed call (as in the LoCoMo runner), asked again


def ask(text, now, *, store, engine, llm, profile) -> tuple[str, int]:
    """One turn: the reply and how many memory searches it made."""
    reply = ""
    for _ in range(EMPTY_TRIES):
        reply = "".join(c.content for c in run_turn(
            text, now, store=store, llm=llm, embedder=engine.embedder, profile=profile,
            config=engine.config, clock=lambda: now, reranker=engine.reranker))
        if reply.strip():
            break
    current = store.current_episode()
    return reply, len(store.get_tool_calls(current.id)) if current else 0


class Command(BaseCommand):
    help = "Answer the personal chats eval's questions with the memory (P1) or without it (P0)."

    def add_arguments(self, parser):
        parser.add_argument("--questions", type=Path, required=True)
        parser.add_argument("--run", choices=["P1", "P0"], required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--think", action="store_true",
                            help="Let the model reason before answering (its profile's think, on), with room for it.")
        parser.add_argument("--min-score", type=float, default=None,
                            help="Automatic recall's threshold for every source (default: the embedding profile's).")
        parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                            help="Override a config value, e.g. messages.max_items=6 (repeatable).")

    def handle(self, *args, questions, run, out, temperature, think, min_score, overrides, **options):
        items = json.loads(questions.read_text(encoding="utf-8"))["questions"]
        last = models.Message.objects.aggregate(last=Max("created_at"))["last"]
        if last is None:
            raise CommandError("No messages in this database: import the chats first (import_chats).")
        now = last + timedelta(days=1)
        engine = get_engine()
        config = engine.config if min_score is None else with_min_score(engine.config, min_score)
        engine = replace(engine, config=with_settings(config, overrides))
        profile = replace(engine.profile, options={**engine.profile.options, "temperature": temperature})
        if think:   # reasoning tokens count against the reply's reserve: give it room
            profile = with_thinking(profile)
        llm = provider_from_profile(profile, base_url=settings.CERVEAU_OLLAMA_URL)
        done = set()
        if out.exists():
            done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
        todo = [q for q in items if q["id"] not in done]
        self.stdout.write(f"{run}: {len(todo)} question(s) to answer ({len(done)} already in {out}); today is {now:%Y-%m-%d}.")
        out.parent.mkdir(parents=True, exist_ok=True)
        for number, q in enumerate(todo, 1):
            started = time.monotonic()
            if run == "P1":
                with transaction.atomic():       # the question and its reply never stay in memory
                    reply, searches = ask(q["question"], now, store=engine.store, engine=engine, llm=llm, profile=profile)
                    transaction.set_rollback(True)
            else:
                reply, searches = ask(q["question"], now, store=InMemoryStore(), engine=engine, llm=llm, profile=profile)
            row = {**{k: q.get(k) for k in ("id", "category", "language", "question", "gold", "stale")},
                   "run": run, "answer": reply, "searches": searches, "seconds": round(time.monotonic() - started, 1),
                   "model": profile.name, "temperature": temperature, "think": bool(think), "today": f"{now:%Y-%m-%d}",
                   "min_score": engine.config.messages.min_score, "settings": overrides}
            with out.open("a", encoding="utf-8") as saved:
                saved.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.stdout.write(f"[{number}/{len(todo)}] {q['id']} {row['seconds']}s, {searches} search(es)")
