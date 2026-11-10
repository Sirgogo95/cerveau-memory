"""export_eval_case: real episodes become an eval scenario the harness can load."""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import CommandError, call_command
from django.test import TestCase

from apps.cerveau.engine import Engine
from apps.cerveau.models import Tag
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig
from brain.types import Role
from evals.cases import load_scenarios
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import T0, at


class ExportEvalCaseTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        embedder = FakeEmbedder()
        engine = Engine(
            store=self.store, llm=FakeLLM(), embedder=embedder, profile=PROFILE,
            config=EngineConfig(), background_llm=FakeLLM(), background_profile=PROFILE,
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        command = "apps.cerveau.management.commands.export_eval_case"
        for patcher in (patch(f"{command}.get_engine", return_value=engine),
                        patch(f"{command}.PERSONAL", self.folder)):
            patcher.start()
            self.addCleanup(patcher.stop)

        self.episode = self.store.open_episode(T0)
        first = self.store.add_message(self.episode.id, Role.USER, "My sister Lucía moved.", at(1))
        self.store.add_message(self.episode.id, Role.ASSISTANT, "Big move!", at(2))
        self.store.close_episode(self.episode.id, at(3))
        [vector] = embedder.embed(["s"])
        self.store.save_summary(
            self.episode.id, title="Sister", summary_short="Lucía moved.",
            summary_long="The user's sister Lucía moved.", short_embedding=vector, long_embedding=vector,
        )
        lucia = self.store.upsert_entity("Lucía", "person", ["my sister"])
        assert lucia is not None
        self.store.add_mention(lucia.id, first.id)
        for name in ("moving", "family", "old"):
            tag = self.store.upsert_tag(name, vector)
            assert tag is not None
            self.store.tag_episode(self.episode.id, tag.id, 1.0)
        Tag.objects.filter(name="old").update(status="Z")      # forgotten: left out

    def test_writes_a_loadable_scenario(self):
        call_command("export_eval_case", episode=[self.episode.id], name="family", stdout=StringIO())
        path = self.folder / "family.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        [case] = data["episodes"]
        key = f"e{self.episode.id}"
        self.assertEqual(case["key"], key)
        self.assertEqual(case["messages"], [["user", "My sister Lucía moved."], ["assistant", "Big move!"]])
        self.assertEqual(case["summary"], "Lucía moved.")
        self.assertEqual(case["entities"], [{"name": "Lucía", "type": "person", "aliases": ["my sister"]}])
        self.assertEqual(case["tags"], ["family", "moving"])
        self.assertEqual(data["queries"], [])
        [scenario] = load_scenarios(path)
        self.assertEqual(scenario.refs(), {f"{key}:0", f"{key}:1", key})

    def test_refuses_missing_episodes_and_overwrites(self):
        with self.assertRaisesRegex(CommandError, "No such episode"):
            call_command("export_eval_case", episode=[999], name="x", stdout=StringIO())
        call_command("export_eval_case", episode=[self.episode.id], name="x", stdout=StringIO())
        with self.assertRaisesRegex(CommandError, "exists"):
            call_command("export_eval_case", episode=[self.episode.id], name="x", stdout=StringIO())
        call_command("export_eval_case", episode=[self.episode.id], name="x", force=True, stdout=StringIO())
