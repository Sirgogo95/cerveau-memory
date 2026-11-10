"""link_fact_entities: existing facts linked to the entities their statements name."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig
from brain.hashing import text_hash
from brain.types import Embedding
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import at


class LinkFactEntitiesTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        engine = Engine(store=self.store, llm=FakeLLM(), embedder=FakeEmbedder(), profile=PROFILE,
                        config=EngineConfig(), background_llm=FakeLLM(), background_profile=PROFILE)
        patcher = patch("apps.cerveau.management.commands.link_fact_entities.get_engine", return_value=engine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_fact(self, statement):
        vector = Embedding(FakeEmbedder().vector(statement), "fake-embedder", text_hash(statement))
        return self.store.add_fact(statement, "person", episode_id=None, message_ids=[], at=at(0), embedding=vector)

    def test_links_facts_once_and_counts_on_a_dry_run(self):
        lena = self.store.upsert_entity("Lena", "person", ["my wife"])
        assert lena is not None
        named = self.add_fact("The user's wife is named Lena.")
        self.add_fact("The user likes short replies.")
        out = StringIO()
        call_command("link_fact_entities", dry_run=True, stdout=out)
        self.assertIn("Would link 1 of 2", out.getvalue())
        self.assertFalse(models.FactEntity.objects.exists())
        call_command("link_fact_entities", stdout=StringIO())
        call_command("link_fact_entities", stdout=StringIO())        # twice: still one link
        self.assertEqual(list(models.FactEntity.objects.values_list("fact_id", "entity_id")),
                         [(named.id, lena.id)])
