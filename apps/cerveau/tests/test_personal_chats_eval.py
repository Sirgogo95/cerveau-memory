"""personal_chats_eval: questions answered from the memory (P1) or without it (P0), the
memory left as it was, "today" after the last message, and a rerun that skips what's saved."""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import T0, at

COMMAND = "apps.cerveau.management.commands.personal_chats_eval"
QUESTIONS = {"questions": [
    {"id": "pc-001", "category": "single", "language": "en", "question": "What bike do I have?", "gold": "a Brompton", "stale": []},
    {"id": "pc-002", "category": "never_said", "language": "en", "question": "My chess rating?", "gold": "not mentioned", "stale": []},
]}


class PersonalChatsEvalTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        episode = self.store.open_episode(T0)
        embedder = FakeEmbedder()
        [vector] = embedder.embed(["My bike is a Brompton."])
        self.store.add_message(episode.id, Role.USER, "My bike is a Brompton.", at(1), vector)
        self.store.close_episode(episode.id, at(1))
        self.llm = FakeLLM("A Brompton.")
        engine = Engine(store=self.store, llm=self.llm, embedder=embedder, profile=PROFILE, config=EngineConfig(),
                        background_llm=FakeLLM(), background_profile=PROFILE)
        for patcher in (patch(f"{COMMAND}.get_engine", return_value=engine),
                        patch(f"{COMMAND}.provider_from_profile", return_value=self.llm)):
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        (self.folder / "q.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")

    def run_command(self, run, out="out.jsonl") -> list[dict]:
        call_command("personal_chats_eval", "--questions", str(self.folder / "q.json"), "--run", run,
                     "--out", str(self.folder / out), stdout=StringIO())
        return [json.loads(line) for line in (self.folder / out).read_text(encoding="utf-8").splitlines()]

    def test_p1_answers_from_memory_and_leaves_it_as_it_was(self):
        before = models.Message.objects.count(), models.Episode.objects.count()
        rows = self.run_command("P1")
        self.assertEqual([r["id"] for r in rows], ["pc-001", "pc-002"])
        self.assertEqual(rows[0]["answer"], "A Brompton.")
        self.assertEqual((rows[0]["run"], rows[0]["gold"], rows[0]["today"]), ("P1", "a Brompton", "2026-10-15"))
        self.assertIn("My bike is a Brompton.", "".join(m.content for m in self.llm.prompts[0]))   # the memory was recalled
        self.assertEqual((models.Message.objects.count(), models.Episode.objects.count()), before)

    def test_p0_answers_without_the_memory(self):
        rows = self.run_command("P0", "p0.jsonl")
        self.assertEqual(rows[0]["run"], "P0")
        self.assertNotIn("My bike is a Brompton.", "".join(m.content for m in self.llm.prompts[0]))

    def test_think_turns_reasoning_on_with_room_for_it(self):
        out = StringIO()
        with patch(f"{COMMAND}.provider_from_profile", return_value=self.llm) as factory:
            call_command("personal_chats_eval", "--questions", str(self.folder / "q.json"), "--run", "P0", "--think",
                         "--out", str(self.folder / "think.jsonl"), stdout=out)
        profile = factory.call_args.args[0]
        self.assertTrue(profile.think)
        self.assertGreaterEqual(profile.output_reserve, PROFILE.output_reserve)   # raised as far as the context allows
        row = json.loads((self.folder / "think.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(row["think"])

    def test_the_threshold_and_settings_can_be_overridden_and_are_recorded(self):
        call_command("personal_chats_eval", "--questions", str(self.folder / "q.json"), "--run", "P1",
                     "--min-score", "1.0", "--set", "messages.max_items=6",
                     "--out", str(self.folder / "strict.jsonl"), stdout=StringIO())
        row = json.loads((self.folder / "strict.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((row["min_score"], row["settings"]), (1.0, ["messages.max_items=6"]))
        self.assertFalse(self.llm.prompts[0][-1].content.startswith("<memory>"))   # nothing recalled

    def test_a_rerun_skips_saved_answers(self):
        self.run_command("P1")
        self.assertEqual(len(self.run_command("P1")), 2)

    def test_an_empty_database_is_refused(self):
        models.Message.objects.all().delete()
        with self.assertRaises(CommandError):
            self.run_command("P1")
