"""Consolidation as a background task: queuing, running, and the sweep, on Postgres
with a scripted LLM."""

import json
from unittest.mock import patch

from django.tasks import default_task_backend  # pyright: ignore[reportMissingImports]  # Django 6 module, newer than django-types
from django.test import TestCase, override_settings

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from apps.cerveau.tasks import (
    consolidate_episode,
    queue_consolidation,
    queue_running_summary,
    sweep,
    update_running_summary,
)
from brain.config import EngineConfig
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_consolidation import EXTRACTION
from tests.test_engine import PROFILE
from tests.test_store_contract import T0, at

DUMMY = {"default": {"BACKEND": "django.tasks.backends.dummy.DummyBackend"}}
IMMEDIATE = {"default": {"BACKEND": "django.tasks.backends.immediate.ImmediateBackend"}}


class TaskTestCase(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        self.llm = FakeLLM(json.dumps(EXTRACTION))
        engine = Engine(
            store=self.store, llm=self.llm, embedder=FakeEmbedder(), profile=PROFILE,
            config=EngineConfig(), background_llm=self.llm, background_profile=PROFILE,
        )
        patcher = patch("apps.cerveau.tasks.get_engine", return_value=engine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def closed_episode(self, minutes: int = 0) -> int:
        episode = self.store.open_episode(at(minutes))
        self.store.add_message(episode.id, Role.USER, "My sister Lucía moved to Montreal.", at(minutes))
        self.store.close_episode(episode.id, at(minutes))
        return episode.id


@override_settings(TASKS=IMMEDIATE)
class ConsolidationTaskTests(TaskTestCase):
    def test_the_task_consolidates_and_reports(self):
        episode_id = self.closed_episode()
        with self.assertLogs("cerveau", "INFO"):
            result = consolidate_episode.enqueue(episode_id)
        self.assertEqual(result.return_value["tags"], ["family", "relocation"])
        episode = models.Episode.objects.get(pk=episode_id)
        self.assertEqual(episode.title, "Lucia moves to Montreal")
        self.assertIsNotNone(episode.consolidated_at)
        self.assertEqual(models.EntityMention.objects.count(), 2)  # Lucía, Montreal

    def test_the_running_summary_task_folds_the_oldest_messages(self):
        episode = self.store.open_episode(T0)
        for i in range(25):
            self.store.add_message(episode.id, Role.USER, f"m{i}", at(i))
        self.llm.reply = "The user counted to twenty-five."
        with self.assertLogs("cerveau", "INFO"):
            result = update_running_summary.enqueue(episode.id)
        self.assertEqual(result.return_value, 10)
        row = models.Episode.objects.get(pk=episode.id)
        self.assertEqual(row.running_summary, "The user counted to twenty-five.")

    def test_nothing_to_do_returns_none(self):
        episode = self.store.open_episode(T0)  # still open
        with self.assertLogs("cerveau", "INFO"):
            self.assertIsNone(consolidate_episode.enqueue(episode.id).return_value)


@override_settings(TASKS=DUMMY)
class QueueTests(TaskTestCase):
    def setUp(self):
        super().setUp()
        default_task_backend.clear()  # type: ignore[attr-defined]

    def queued_ids(self) -> list[int]:
        return [r.args[0] for r in default_task_backend.results]  # type: ignore[attr-defined]

    def test_queue_consolidation_enqueues_the_task(self):
        queue_consolidation(42)
        self.assertEqual(self.queued_ids(), [42])

    def test_queue_running_summary_enqueues_its_task(self):
        queue_running_summary(7)
        [result] = default_task_backend.results  # type: ignore[attr-defined]
        self.assertEqual((result.task.name, result.args), ("update_running_summary", [7]))

    def test_sweep_queues_every_pending_episode(self):
        first, second = self.closed_episode(0), self.closed_episode(10)
        done = self.closed_episode(20)
        self.store.mark_consolidated(done, at(30))
        self.assertEqual(sweep(), 2)
        self.assertEqual(self.queued_ids(), [first, second])

    def test_sweep_closes_a_stale_open_episode_first(self):
        stale = self.store.open_episode(T0)  # T0 is long past
        self.store.add_message(stale.id, Role.USER, "hello", T0)
        self.assertEqual(sweep(), 1)
        self.assertIsNone(self.store.current_episode())
        self.assertEqual(self.queued_ids(), [stale.id])
