"""purge_memory: forgotten memory erased for good, with its audit copies, derived
facts, quoting tool calls, cached vectors, and activation; nothing active touched."""

from datetime import timedelta
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from apps.cerveau import models
from apps.cerveau.purge import execute_purge, plan_purge
from apps.cerveau.store import DjangoMemoryStore
from brain.embedding import embed_cached
from brain.types import MemoryKind, Role, ToolRun
from kernel.base_app.models import ChangeLog
from tests.fakes import FakeEmbedder
from tests.test_store_contract import at

OLD = timedelta(days=40)


class PurgeTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        self.embedder = FakeEmbedder()
        # The episode to forget: a message, a fact learned only from it, a tag.
        self.gone = self.store.open_episode(at(0))
        self.secret = self.say(self.gone.id, "my secret plan is to move to Lisbon", 1)
        self.store.close_episode(self.gone.id, at(2))
        self.fact = self.add_fact("The user plans to move to Lisbon.", self.gone.id, [self.secret.id])
        # An episode that stays, with a fact confirmed in both, and a search that quoted
        # the secret message.
        self.kept = self.store.open_episode(at(500))
        both = self.say(self.kept.id, "Lisbon is lovely in spring", 501)
        self.shared = self.add_fact("The user likes Lisbon.", self.gone.id, [self.secret.id, both.id])
        self.store.add_tool_calls(self.kept.id, None, [ToolRun(
            "search_memory", {"query": "plan"}, f'{{"results": [{{"id": "M{self.secret.id}"}}]}}', 5, at(502))])
        self.store.add_tool_calls(self.kept.id, None, [ToolRun(
            "search_memory", {"query": "spring"}, f'{{"results": [{{"id": "M{both.id}"}}]}}', 5, at(503))])
        self.store.close_episode(self.kept.id, at(504))
        self.forget(models.Episode, self.gone.id)
        # A search query's vector: cached, but its text is stored nowhere else.
        embed_cached(["where am I moving?"], self.embedder, self.store)

    def say(self, episode_id, text, minutes):
        [vector] = embed_cached([text], self.embedder, self.store)
        return self.store.add_message(episode_id, Role.USER, text, at(minutes), vector)

    def add_fact(self, statement, episode_id, message_ids):
        [vector] = embed_cached([statement], self.embedder, self.store)
        return self.store.add_fact(statement, "other", episode_id=episode_id, message_ids=message_ids,
                                   at=at(3), embedding=vector)

    def forget(self, model, pk, ago=OLD):
        row = model.objects.get(pk=pk)
        row.set_archived()                      # audited, like the viewer's Forget
        model.objects.filter(pk=pk).update(updated_at=timezone.now() - ago)

    def plan(self, **kwargs):
        return plan_purge(now=timezone.now(), **kwargs)

    def test_a_forgotten_episode_goes_with_everything_derived_from_it(self):
        plan = self.plan()
        self.assertEqual(plan.episodes, [self.gone.id])
        self.assertEqual(plan.derived_facts, [self.fact.id])      # its only source was there
        self.assertEqual(len(plan.tool_calls), 1)                 # the search quoting it
        self.assertTrue(ChangeLog.objects.filter(model_name="Episode", instance_id=self.gone.id).exists())
        execute_purge(plan)
        self.assertFalse(models.Episode.objects.filter(pk=self.gone.id).exists())
        self.assertFalse(models.Message.objects.filter(pk=self.secret.id).exists())
        self.assertFalse(models.Fact.objects.filter(pk=self.fact.id).exists())
        self.assertFalse(ChangeLog.objects.filter(model_name="Episode", instance_id=self.gone.id).exists())
        self.assertFalse(models.EmbeddingCache.objects.exists())     # all of it, queries too
        # What was also said elsewhere stays.
        self.assertTrue(models.Episode.objects.filter(pk=self.kept.id).exists())
        self.assertTrue(models.Fact.objects.filter(pk=self.shared.id).exists())
        self.assertEqual(models.ToolCall.objects.filter(episode_id=self.kept.id).count(), 1)

    def test_the_grace_period_and_status_are_respected(self):
        self.forget(models.Episode, self.gone.id, ago=timedelta(days=2))
        self.assertTrue(self.plan().empty)                                  # 30 days by default
        self.assertEqual(self.plan(older_than=timedelta(0)).episodes, [self.gone.id])
        models.Episode.objects.filter(pk=self.gone.id).update(status="I")
        self.assertTrue(self.plan(older_than=timedelta(0)).empty)          # Z only by default
        self.assertEqual(self.plan(statuses=["I"], older_than=timedelta(0)).episodes, [self.gone.id])
        self.assertTrue(self.plan(statuses=["I"], older_than=timedelta(0), episode_ids=[self.kept.id]).empty)

    def test_a_forgotten_fact_takes_its_history_and_tags_and_entities_go_too(self):
        newer = self.add_fact("The user moved to Porto instead.", self.kept.id, [])
        self.store.supersede_fact(self.shared.id, newer.id)
        self.forget(models.Fact, newer.id)
        tag = self.store.upsert_tag("travel")
        person = self.store.upsert_entity("Ana", "person")
        assert tag is not None and person is not None
        self.store.save_activation({(MemoryKind.TAG, tag.id): (0.5, at(0)),
                                    (MemoryKind.ENTITY, person.id): (0.4, at(0))})
        self.forget(models.Tag, tag.id)
        self.forget(models.Entity, person.id)
        plan = self.plan(kinds=["fact", "tag", "entity"])
        self.assertEqual(plan.facts, [newer.id])
        self.assertEqual(plan.derived_facts, [self.shared.id])    # what it replaced
        execute_purge(plan)
        self.assertFalse(models.Fact.objects.filter(pk__in=[newer.id, self.shared.id]).exists())
        self.assertFalse(models.Tag.objects.filter(pk=tag.id).exists())
        self.assertFalse(models.Entity.objects.filter(pk=person.id).exists())
        self.assertEqual(self.store.load_activation(), {})

    def test_the_command_lists_asks_and_erases(self):
        out = StringIO()
        call_command("purge_memory", "--dry-run", stdout=out)
        self.assertIn("Would erase 1 episode(s)", out.getvalue())
        self.assertTrue(models.Episode.objects.filter(pk=self.gone.id).exists())
        with patch("builtins.input", return_value="no"):
            call_command("purge_memory", stdout=StringIO())
        self.assertTrue(models.Episode.objects.filter(pk=self.gone.id).exists())
        with patch("builtins.input", return_value="purge"):
            call_command("purge_memory", stdout=StringIO())
        self.assertFalse(models.Episode.objects.filter(pk=self.gone.id).exists())
        done = StringIO()
        call_command("purge_memory", "--no-input", stdout=done)
        self.assertIn("Nothing to purge", done.getvalue())
