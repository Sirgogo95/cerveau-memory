"""embed_memory: messages, episode summaries, facts, and tags get vectors for a new model, once."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.store import DjangoMemoryStore
from brain.types import Role
from tests.fakes import FakeEmbedder
from tests.test_store_contract import T0, at

COMMAND = "apps.cerveau.management.commands.embed_memory"


class EmbedMemoryTests(TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.embedder.model = "all-MiniLM-L6-v2"  # the profile's model
        patcher = patch(f"{COMMAND}.embedder_from_profile", return_value=self.embedder)
        patcher.start()
        self.addCleanup(patcher.stop)

        store = DjangoMemoryStore()
        old = FakeEmbedder()  # model "fake-embedder"
        episode = store.open_episode(T0)
        for i, text in enumerate(["first message", "second message"]):
            [vector] = old.embed([text])
            store.add_message(episode.id, Role.USER, text, at(i), vector)
        store.close_episode(episode.id, at(5))
        short, long = old.embed(["short summary", "long summary"])
        store.save_summary(episode.id, title="T", summary_short="short summary",
                           summary_long="long summary", short_embedding=short, long_embedding=long)
        store.open_episode(at(500))  # no summary: nothing to embed for it
        self.episode = episode

    def run_command(self, *args) -> str:
        out = StringIO()
        call_command("embed_memory", "--profile", "all-MiniLM-L6-v2", *args, stdout=out)
        return out.getvalue()

    def test_embeds_messages_and_summaries_for_the_new_model_and_keeps_the_old(self):
        self.assertIn("2 message(s), 1 episode summary pair(s)", self.run_command())
        self.assertEqual(models.MessageEmbedding.objects.filter(model="all-MiniLM-L6-v2").count(), 2)
        self.assertEqual(models.MessageEmbedding.objects.filter(model="fake-embedder").count(), 2)
        row = models.EpisodeEmbedding.objects.get(episode_id=self.episode.id, model="all-MiniLM-L6-v2")
        [short] = self.embedder.embed(["short summary"])
        for stored, expected in zip(row.summary_short, short.vector):  # float32 in Postgres
            self.assertAlmostEqual(stored, expected, places=6)
        self.assertEqual(row.summary_long_hash, self.embedder.embed(["long summary"])[0].text_hash)

    def test_facts_are_embedded_for_the_new_model(self):
        store = DjangoMemoryStore()
        [old] = FakeEmbedder().embed(["Lucía lives in Montreal."])
        fact = store.add_fact("Lucía lives in Montreal.", "person", episode_id=self.episode.id,
                              message_ids=[], at=at(9), embedding=old)
        self.assertIn("1 fact(s)", self.run_command())
        self.assertTrue(models.FactEmbedding.objects.filter(fact_id=fact.id, model="all-MiniLM-L6-v2").exists())

    def test_tags_are_embedded_for_the_new_model(self):
        tag = DjangoMemoryStore().upsert_tag("family")
        assert tag is not None
        self.assertIn("1 tag(s)", self.run_command())
        row = models.TagEmbedding.objects.get(tag_id=tag.id, model="all-MiniLM-L6-v2")
        self.assertEqual(row.text_hash, self.embedder.embed(["family"])[0].text_hash)

    def test_second_run_finds_nothing_to_do(self):
        self.run_command()
        self.assertIn("0 message(s), 0 episode", self.run_command())

    def test_dry_run_only_counts(self):
        self.assertIn("2 message(s)", self.run_command("--dry-run"))
        self.assertFalse(models.MessageEmbedding.objects.filter(model="all-MiniLM-L6-v2").exists())
