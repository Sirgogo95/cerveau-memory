"""import_chats: a made-up ChatGPT export into the database, once, with the selection,
the "do not remember" flag, and the mapping from the export's ids."""

import json
import tempfile
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_importing import CHAT, conversation, node

COMMAND = "apps.cerveau.management.commands.import_chats"


class ImportChatsTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        engine = Engine(store=self.store, llm=FakeLLM(), embedder=FakeEmbedder(), profile=PROFILE,
                        config=replace(EngineConfig(), drift_threshold=0),
                        background_llm=FakeLLM(), background_profile=PROFILE)
        self.queued: list[int] = []
        for patcher in (patch(f"{COMMAND}.get_engine", return_value=engine),
                        patch(f"{COMMAND}.queue_consolidation", side_effect=self.queued.append)):
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        private = conversation("c2", [{"id": "r", "parent": None, "message": None},
                                      node("x", "r", "user", ["Private chat"], 600)],
                               current="x", is_do_not_remember=True)
        (self.folder / "conversations.json").write_text(json.dumps([CHAT, private]), encoding="utf-8")

    def run_command(self, *args) -> str:
        out = StringIO()
        call_command("import_chats", str(self.folder), "--source", "chatgpt", *args, stdout=out)
        return out.getvalue()

    def test_imports_once_and_queues_consolidation(self):
        output = self.run_command()
        self.assertIn("1 chat(s) to import (5 messages); skipped: 0 already imported, 1 marked do not remember", output)
        self.assertEqual(models.Message.objects.count(), 5)
        self.assertEqual(len(self.queued), 1)
        mapping = self.store.imported_messages("chatgpt", "c1")
        self.assertEqual(set(mapping), {"u1", "a1", "a1b", "u2", "u3", "a3", "u4"})
        self.assertIn("skipped: 1 already imported", self.run_command())     # a second run adds nothing
        self.assertEqual(models.Message.objects.count(), 5)

    def test_dry_run_stores_nothing(self):
        self.assertIn("5 messages", self.run_command("--dry-run"))
        self.assertEqual(models.Message.objects.count(), 0)

    def test_selection_and_the_do_not_remember_flag(self):
        selection = self.folder / "selected.json"
        selection.write_text(json.dumps({"in": ["c2", "nope"], "out": ["c1"]}), encoding="utf-8")
        output = self.run_command("--select", str(selection), "--include-do-not-remember", "--no-queue")
        self.assertIn("1 selected chat(s) not in the export", output)
        self.assertEqual(list(models.Message.objects.values_list("content", flat=True)), ["Private chat"])
        self.assertEqual(self.queued, [])
