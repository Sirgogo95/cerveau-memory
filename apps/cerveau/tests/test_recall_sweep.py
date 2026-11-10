"""recall_sweep: automatic recall at each threshold, scored against the questions'
evidence (by episode), never-said questions counted as noise, the memory left as it was."""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import T0, at

QUESTIONS = {"questions": [
    {"id": "pc-001", "category": "single", "question": "What bike do I ride?",
     "evidence": [{"source": "chatgpt", "conversation": "c1", "message": "m1"}]},
    {"id": "pc-002", "category": "never_said", "question": "What is my chess rating?", "evidence": []},
]}


class RecallSweepTests(TestCase):
    def setUp(self):
        store = DjangoMemoryStore()
        embedder = FakeEmbedder()
        episode = store.open_episode(T0)
        [vector] = embedder.embed(["My bike is a Brompton."])
        message = store.add_message(episode.id, Role.USER, "My bike is a Brompton.", at(1), vector)
        store.close_episode(episode.id, at(1))
        models.ImportedMessage.objects.create(source="chatgpt", conversation_id="c1", external_id="m1",
                                              message_id=message.id)
        self.episode = episode.id
        engine = Engine(store=store, llm=FakeLLM(), embedder=embedder, profile=PROFILE, config=EngineConfig(),
                        background_llm=FakeLLM(), background_profile=PROFILE)
        patcher = patch("apps.cerveau.management.commands.recall_sweep.get_engine", return_value=engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        (self.folder / "q.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")

    def test_each_threshold_is_scored_against_the_evidence_and_nothing_stays(self):
        before = models.Episode.objects.count(), models.Message.objects.count()
        out = StringIO()
        call_command("recall_sweep", "--questions", str(self.folder / "q.json"), "--grid", "0.0,1.0,0.5",
                     "--out", str(self.folder / "rows.jsonl"), stdout=out)
        rows = [json.loads(line) for line in (self.folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["min_score"], r["id"]) for r in rows],
                         [(t, q) for t in (0.0, 0.5, 1.0) for q in ("pc-001", "pc-002")])
        self.assertEqual(rows[0]["wanted"], [self.episode])
        self.assertIn(self.episode, [episode for _, episode, _ in rows[0]["kept"]])   # found at 0
        self.assertEqual(rows[-2]["kept"], [])                                          # nothing passes 1.0
        self.assertIn("  0.00      100%", out.getvalue())
        self.assertEqual((models.Episode.objects.count(), models.Message.objects.count()), before)
