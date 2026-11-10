"""Calibrate automatic recall's threshold on real memory (docs/PERSONAL_CHATS_EVAL.md, 6g).

    python manage.py recall_sweep --questions questions-chatgpt.json --questions questions-chatgpt-2.json
    python manage.py recall_sweep --questions q.json --grid 0.45,0.61,0.02 --out sweep.jsonl

`python -m evals --sweep` calibrates `min_score` on written cases; this does the same on
the database (an imported chat history), with the personal chats eval's questions and
their evidence. Each question is one fresh recall (brain/recall.py), as the app runs it
before the model sees anything, at each threshold, its writes rolled back. A memory is
counted as found when it belongs to an episode holding the question's evidence (a
message, that episode's summary, or a fact learned there). Never-said questions should
recall nothing: what they get is noise. No model is called; only the embedder.
"""

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Max

from apps.cerveau import models
from apps.cerveau.engine import get_engine
from brain.config import with_min_score
from brain.embedding import embed_cached
from brain.recall import recall


@dataclass
class Row:
    """One question at one threshold."""

    question: str
    negative: bool
    kept: list[tuple[str, int | None, float]]   # kind, episode id, score; best first
    wanted: set[int]                     # the evidence's episodes

    @property
    def found(self) -> int:
        return sum(1 for _, episode, _ in self.kept if episode in self.wanted)


def parse_grid(text: str) -> list[float]:
    low, high, step = (float(x) for x in text.split(","))
    count = int(round((high - low) / step)) + 1
    return [round(low + i * step, 3) for i in range(count)]


def evidence_episodes(question: dict) -> set[int]:
    ids = [e["message"] for e in question.get("evidence", [])]
    return set(models.ImportedMessage.objects.filter(external_id__in=ids)
               .values_list("message__episode_id", flat=True))


def summarize(rows: list[Row]) -> dict:
    answered = [r for r in rows if not r.negative and r.wanted]
    negatives = [r for r in rows if r.negative]
    kept = sum(len(r.kept) for r in answered)
    return {
        "hit_rate": sum(r.found > 0 for r in answered) / len(answered) if answered else 1.0,
        "precision": sum(r.found for r in answered) / kept if kept else 1.0,
        "kept": kept / len(answered) if answered else 0.0,
        "noise": sum(len(r.kept) for r in negatives) / len(negatives) if negatives else 0.0,
        "silent": sum(not r.kept for r in negatives),
    }


class Command(BaseCommand):
    help = "Sweep automatic recall's min_score on the imported memory, against the eval questions' evidence."

    def add_arguments(self, parser):
        parser.add_argument("--questions", type=Path, action="append", required=True,
                            help="A question file (repeatable).")
        parser.add_argument("--grid", default=None, metavar="LOW,HIGH,STEP",
                            help="Thresholds to try (default: the app's min_score +/- 0.10, step 0.02).")
        parser.add_argument("--out", type=Path, default=None, help="Save every row (JSON lines).")

    def handle(self, *args, questions, grid: str | None, out, **options):
        items = [q for f in questions for q in json.loads(f.read_text(encoding="utf-8"))["questions"]]
        last = models.Message.objects.aggregate(last=Max("created_at"))["last"]
        if last is None:
            raise CommandError("No messages in this database: import the chats first (import_chats).")
        now = last + timedelta(days=1)
        engine = get_engine()
        current = engine.config.messages.min_score
        thresholds = parse_grid(grid) if grid else [round(current + d / 100, 2) for d in range(-10, 11, 2)]
        cases = [(q, q["category"] == "never_said", evidence_episodes(q)) for q in items]
        missing = [q["id"] for q, negative, wanted in cases if not negative and not wanted]
        if missing:
            self.stderr.write(f"No stored evidence for {', '.join(missing)}: left out of the hit rate.")
        self.stdout.write(f"{len(items)} question(s), {sum(n for _, n, _ in cases)} never said; "
                          f"today is {now:%Y-%m-%d}; the app's min_score is {current}.")
        self.stdout.write(f"{'min_score':>9}  {'hit rate':>8}  {'precision':>9}  {'kept':>5}  "
                          f"{'noise':>5}  {'silent':>6}")
        saved = out.open("w", encoding="utf-8") if out else None
        try:
            for threshold in thresholds:
                config = with_min_score(engine.config, threshold)
                rows = []
                for q, negative, wanted in cases:
                    with transaction.atomic():           # the question's episode and activation never stay
                        episode = engine.store.open_episode(now)
                        [vector] = embed_cached([q["question"]], engine.embedder, engine.store)
                        found = recall(q["question"], vector, engine.store, config, embedder=engine.embedder,
                                       episode_id=episode.id, now=now, reranker=engine.reranker)
                        transaction.set_rollback(True)
                    kept = [(m.candidate.kind.value, m.candidate.episode_id, round(m.score, 3))
                            for m in found.memories]
                    rows.append(Row(q["id"], negative, kept, wanted))
                    if saved:
                        saved.write(json.dumps({"min_score": threshold, "id": q["id"], "category": q["category"],
                                                "wanted": sorted(wanted), "kept": kept}) + "\n")
                m = summarize(rows)
                mark = "  <- now" if abs(threshold - current) < 1e-9 else ""
                self.stdout.write(f"{threshold:>9.2f}  {m['hit_rate']:>8.0%}  {m['precision']:>9.0%}  "
                                  f"{m['kept']:>5.1f}  {m['noise']:>5.1f}  {m['silent']:>6}{mark}")
        finally:
            if saved:
                saved.close()
