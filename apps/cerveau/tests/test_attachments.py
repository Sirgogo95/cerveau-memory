"""Shared files on Postgres: storing an upload by hash, reading and summarizing it in
background tasks, and purging it, file included."""

import json
import tempfile
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from apps.cerveau import models
from apps.cerveau.attachments import UploadRefused, attachment_path, store_upload
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from apps.cerveau.tasks import read_attachment, sweep
from brain.config import EmbeddingProfile, EngineConfig
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE
from tests.test_store_contract import at

IMMEDIATE = {"default": {"BACKEND": "django.tasks.backends.immediate.ImmediateBackend"}}
LEASE = b"# Rent\n\nThe monthly rent is 38,000 pesos.\n\n# Deposit\n\nThe deposit is 60,000 pesos.\n"
SUMMARY = {"language": "English", "title": "Naco lease", "summary_short": "A lease.",
           "summary_long": "The document says the rent is 38,000 pesos.", "english_tags": ["housing"]}


class AttachmentTestCase(TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        settings = override_settings(CERVEAU_ATTACHMENTS_DIR=self.folder, CERVEAU_MAX_ATTACHMENT_MB=1)
        settings.enable()
        self.addCleanup(settings.disable)
        self.store = DjangoMemoryStore()
        self.episode = self.store.open_episode(at(0))
        self.message = self.store.add_message(self.episode.id, Role.USER, "here's my lease", at(0))

    def upload(self, name="lease.md", data=LEASE):
        stored = store_upload(name, data, self.store, timezone.now())
        if stored.attachment is not None:
            self.store.link_attachment(self.message.id, stored.attachment.id)
        return stored


class StoreUploadTests(AttachmentTestCase):
    def test_a_file_is_written_once_by_hash(self):
        first = self.upload()
        again = self.upload("copy-of-lease.md")
        assert first.attachment is not None and again.attachment is not None
        self.assertEqual((first.created, again.created), (True, False))
        self.assertEqual(first.attachment.id, again.attachment.id)
        path = attachment_path(first.attachment)
        self.assertEqual(path.read_bytes(), LEASE)
        self.assertTrue(str(path.relative_to(self.folder)).replace("\\", "/").endswith(".md"))
        self.assertEqual(len(list(self.folder.rglob("*.md"))), 1)
        self.assertEqual(first.attachment.media_type, "text/markdown")

    def test_too_large_empty_and_unreadable_files_are_refused(self):
        with self.assertRaisesRegex(UploadRefused, "larger than 1 MB"):
            self.upload("big.txt", b"x" * (1024 * 1024 + 1))
        with self.assertRaisesRegex(UploadRefused, "empty"):
            self.upload("empty.txt", b"")
        with self.assertRaisesRegex(UploadRefused, "can't read photo.png"):
            self.upload("photo.png", b"\x89PNG\r\n")

    def test_a_forgotten_file_isnt_relearned_and_names_cant_escape_the_folder(self):
        first = self.upload("../../evil.md")
        assert first.attachment is not None
        self.assertEqual(first.filename, "evil.md")
        models.Attachment.objects.get(pk=first.attachment.id).set_archived()
        self.assertIsNone(self.upload("lease-again.md").attachment)


class TaskTests(AttachmentTestCase):
    def setUp(self):
        super().setUp()
        self.llm = FakeLLM(json.dumps(SUMMARY))
        engine = Engine(store=self.store, llm=self.llm, embedder=FakeEmbedder(), profile=PROFILE,
                        config=EngineConfig(), background_llm=self.llm, background_profile=PROFILE,
                        embedding_profile=EmbeddingProfile("fake", "ollama", "fake-embedder", 0.5, 512))
        patcher = patch("apps.cerveau.tasks.get_engine", return_value=engine)
        patcher.start()
        self.addCleanup(patcher.stop)

    @override_settings(TASKS=IMMEDIATE)
    def test_reading_then_summarizing_in_the_background(self):
        stored = self.upload()
        assert stored.attachment is not None
        read_attachment.enqueue(stored.attachment.id)
        row = models.Attachment.objects.get(pk=stored.attachment.id)
        self.assertEqual((row.extraction_status, row.title), ("done", "Naco lease"))
        self.assertEqual(models.AttachmentChunk.objects.filter(attachment=row).count(), 2)
        self.assertEqual(list(models.AttachmentTag.objects.filter(attachment=row).values_list("tag__name", flat=True)),
                         ["housing"])
        self.assertIsNotNone(row.summarized_at)

    @override_settings(TASKS=IMMEDIATE)
    def test_the_sweep_reads_what_was_missed(self):
        stored = self.upload()
        assert stored.attachment is not None
        self.assertGreaterEqual(sweep(), 1)
        self.assertEqual(models.Attachment.objects.get(pk=stored.attachment.id).extraction_status, "done")


class PurgeAttachmentTests(TaskTests):
    @override_settings(TASKS=IMMEDIATE)
    def test_a_forgotten_file_is_erased_from_disk_too(self):
        stored = self.upload()
        assert stored.attachment is not None
        read_attachment.enqueue(stored.attachment.id)
        path = attachment_path(stored.attachment)
        row = models.Attachment.objects.get(pk=stored.attachment.id)
        row.set_archived()
        models.Attachment.objects.filter(pk=row.pk).update(updated_at=timezone.now() - timedelta(days=40))
        out = StringIO()
        with self.captureOnCommitCallbacks(execute=True):  # type: ignore[attr-defined]  # newer than django-types
            call_command("purge_memory", "--no-input", stdout=out)
        self.assertIn("1 file(s)", out.getvalue())
        self.assertFalse(models.Attachment.objects.filter(pk=row.pk).exists())
        self.assertFalse(models.AttachmentChunk.objects.filter(attachment_id=row.pk).exists())
        self.assertFalse(path.exists())

    def test_a_file_shared_only_in_a_purged_episode_goes_with_it(self):
        stored = self.upload()
        assert stored.attachment is not None
        self.store.close_episode(self.episode.id, at(1))
        episode = models.Episode.objects.get(pk=self.episode.id)
        episode.set_archived()
        models.Episode.objects.filter(pk=episode.pk).update(updated_at=timezone.now() - timedelta(days=40))
        from apps.cerveau.purge import plan_purge

        self.assertEqual(plan_purge(now=timezone.now()).attachments, [stored.attachment.id])
