"""Facts on Postgres: the consolidation task learns them and records it, and
extract_facts backfills consolidated episodes once."""

import json
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase, override_settings

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from apps.cerveau.tasks import consolidate_episode
from brain.config import EngineConfig
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_consolidation import EXTRACTION
from tests.test_engine import PROFILE
from tests.test_store_contract import T0, at

IMMEDIATE = {"default": {"BACKEND": "django.tasks.backends.immediate.ImmediateBackend"}}
FACTS = json.dumps({"language": "English", "facts": [
    {"statement": "The user's sister Lucía lives in Montreal.", "type": "person", "lines": [1]},
]})


class FactsOnPostgresTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        self.llm = FakeLLM(replies=[json.dumps(EXTRACTION), FACTS])
        engine = Engine(
            store=self.store, llm=self.llm, embedder=FakeEmbedder(), profile=PROFILE,
            config=EngineConfig(), background_llm=self.llm, background_profile=PROFILE,
        )
        for target in ("apps.cerveau.tasks.get_engine",
                       "apps.cerveau.management.commands.extract_facts.get_engine"):
            patcher = patch(target, return_value=engine)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.episode = self.store.open_episode(T0)
        self.store.add_message(self.episode.id, Role.USER, "My sister Lucía just moved to Montreal.", at(1))
        self.store.close_episode(self.episode.id, at(2))

    @override_settings(TASKS=IMMEDIATE)
    def test_consolidation_learns_facts_and_records_it(self):
        result = consolidate_episode.enqueue(self.episode.id).return_value
        self.assertEqual(result["facts"]["added"], ["The user's sister Lucía lives in Montreal."])
        fact = models.Fact.objects.get()
        self.assertEqual(fact.episode_id, self.episode.id)
        self.assertEqual(fact.sources.count(), 1)
        self.assertIsNotNone(models.Episode.objects.get(pk=self.episode.id).facts_learned_at)

    def test_extract_facts_backfills_once(self):
        models.Episode.objects.filter(pk=self.episode.id).update(consolidated_at=at(3))
        self.llm.replies = [FACTS]
        out = StringIO()
        call_command("extract_facts", stdout=out)
        self.assertIn("new facts: The user's sister Lucía lives in Montreal.", out.getvalue())
        self.assertEqual(models.Fact.objects.count(), 1)
        again = StringIO()
        call_command("extract_facts", stdout=again)
        self.assertIn("No episode needs facts.", again.getvalue())
