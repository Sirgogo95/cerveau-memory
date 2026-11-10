"""DjangoMemoryStore against Postgres: the shared contract, plus what only the real
database can show (visibility of inactive and forgotten memory, audit behavior,
vector storage, cascade on purge)."""

from django.test import TestCase

from apps.cerveau import models
from apps.cerveau.store import DjangoMemoryStore
from brain.types import Embedding, MemoryKind, Role
from kernel.base_app.models import ChangeLog
from tests.test_store_contract import T0, Contracts, at


class DjangoConversationContractTests(Contracts.Conversation, TestCase):
    def make_store(self):
        return DjangoMemoryStore()


class DjangoSearchContractTests(Contracts.Search, TestCase):
    def make_store(self):
        return DjangoMemoryStore()


class DjangoKnowledgeContractTests(Contracts.Knowledge, TestCase):
    def make_store(self):
        return DjangoMemoryStore()

    def set_status(self, kind: MemoryKind, item_id: int, status: str) -> None:
        model = {MemoryKind.TAG: models.Tag, MemoryKind.ENTITY: models.Entity,
                 MemoryKind.FACT: models.Fact, MemoryKind.ATTACHMENT: models.Attachment}[kind]
        row = model.objects.get(pk=item_id)
        {"Z": row.set_archived, "I": row.set_inactive, "A": row.reactivate}[status]()


class DjangoSearchVisibilityTests(TestCase):
    def test_messages_of_forgotten_episodes_are_never_found(self):
        store = DjangoMemoryStore()
        episode = store.open_episode(T0)
        vector = Embedding([1.0, 0.0], "m", "h")
        store.add_message(episode.id, Role.USER, "secret", at(1), vector)
        models.Episode.objects.get(pk=episode.id).set_archived()
        self.assertEqual(store.semantic_candidates(MemoryKind.MESSAGE, vector, 5), [])


class DjangoVisibilityTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()

    def test_inactive_and_forgotten_episodes_are_never_returned(self):
        kept, inactive, forgotten = (self.store.open_episode(at(i)) for i in range(3))
        models.Episode.objects.get(pk=inactive.id).set_inactive()
        models.Episode.objects.get(pk=forgotten.id).set_archived()
        found = self.store.get_episodes([kept.id, inactive.id, forgotten.id])
        self.assertEqual([e.id for e in found], [kept.id])

    def test_forgotten_episodes_are_not_recent(self):
        episode = self.store.open_episode(T0)
        self.store.close_episode(episode.id, at(1))
        models.Episode.objects.get(pk=episode.id).set_archived()
        self.assertEqual(self.store.recent_episodes(5), [])

    def test_forgotten_open_episode_is_not_current(self):
        episode = self.store.open_episode(T0)
        models.Episode.objects.get(pk=episode.id).set_archived()
        self.assertIsNone(self.store.current_episode())


class DjangoAuditTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()
        self.vector = Embedding(vector=[0.1, 0.2, 0.3], model="m", text_hash="h1")

    def episode_logs(self, episode_id):
        return ChangeLog.objects.filter(model_name="Episode", instance_id=episode_id)

    def test_messages_and_closing_write_no_audit_snapshots(self):
        episode = self.store.open_episode(T0)
        created = self.episode_logs(episode.id).count()
        self.store.add_message(episode.id, Role.USER, "a", at(1))
        self.store.add_message(episode.id, Role.ASSISTANT, "b", at(2))
        self.store.close_episode(episode.id, at(2))
        self.assertEqual(self.episode_logs(episode.id).count(), created)

    def test_the_running_summary_writes_no_audit_snapshot(self):
        episode = self.store.open_episode(T0)
        message = self.store.add_message(episode.id, Role.USER, "a", at(1))
        before = self.episode_logs(episode.id).count()
        self.store.save_running_summary(episode.id, "summary", message.id)
        self.assertEqual(self.episode_logs(episode.id).count(), before)

    def test_saving_a_summary_is_audited(self):
        episode = self.store.open_episode(T0)
        before = self.episode_logs(episode.id).count()
        self.store.save_summary(
            episode.id, title="T", summary_short="s", summary_long="l",
            short_embedding=self.vector, long_embedding=self.vector,
        )
        self.assertEqual(self.episode_logs(episode.id).count(), before + 1)


class DjangoEmbeddingTests(TestCase):
    def setUp(self):
        self.store = DjangoMemoryStore()

    def test_message_embedding_is_stored_with_model_and_hash(self):
        episode = self.store.open_episode(T0)
        vector = Embedding(vector=[0.1, 0.2, 0.3], model="m", text_hash="h1")
        message = self.store.add_message(episode.id, Role.USER, "a", at(1), vector)
        row = models.MessageEmbedding.objects.get(message_id=message.id)
        self.assertEqual((row.model, row.text_hash), ("m", "h1"))
        self.assertEqual([round(x, 3) for x in row.embedding], [0.1, 0.2, 0.3])

    def test_vectors_of_different_sizes_share_the_table(self):
        episode = self.store.open_episode(T0)
        self.store.add_message(episode.id, Role.USER, "a", at(1), Embedding([0.1] * 3, "small", "h"))
        self.store.add_message(episode.id, Role.USER, "b", at(2), Embedding([0.1] * 768, "big", "h"))
        self.assertEqual(models.MessageEmbedding.objects.count(), 2)

    def test_summary_embeddings_go_to_one_row_per_model(self):
        episode = self.store.open_episode(T0)
        self.store.save_summary(
            episode.id, title="T", summary_short="s", summary_long="l",
            short_embedding=Embedding([0.1, 0.2], "m", "hs"),
            long_embedding=Embedding([0.3, 0.4], "m", "hl"),
        )
        row = models.EpisodeEmbedding.objects.get(episode_id=episode.id)
        self.assertEqual((row.summary_short_hash, row.summary_long_hash), ("hs", "hl"))

    def test_re_embedding_updates_instead_of_duplicating(self):
        episode = self.store.open_episode(T0)
        for text_hash in ("old", "new"):
            self.store.save_summary(
                episode.id, title="T", summary_short="s", summary_long="l",
                short_embedding=Embedding([0.1], "m", text_hash),
                long_embedding=Embedding([0.1], "m", text_hash),
            )
        row = models.EpisodeEmbedding.objects.get(episode_id=episode.id)
        self.assertEqual(row.summary_short_hash, "new")

    def test_purging_an_episode_removes_its_messages_and_vectors(self):
        episode = self.store.open_episode(T0)
        self.store.add_message(episode.id, Role.USER, "a", at(1), Embedding([0.1], "m", "h"))
        models.Episode.objects.filter(pk=episode.id).delete()  # hard delete, as purge does
        self.assertEqual(models.Message.objects.count(), 0)
        self.assertEqual(models.MessageEmbedding.objects.count(), 0)
