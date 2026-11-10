"""Files in the chat and the memory viewer: sharing a file with a message, refusals, a
forgotten file, the reading state, and the Files pages."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EmbeddingProfile, EngineConfig
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

NOTES = b"# Deposit\n\nThe deposit is 76,000 pesos, returned within thirty days.\n"


class FileViewTests(TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        settings = override_settings(CERVEAU_ATTACHMENTS_DIR=Path(folder.name), CERVEAU_MAX_ATTACHMENT_MB=1)
        settings.enable()
        self.addCleanup(settings.disable)
        self.store = DjangoMemoryStore()
        self.engine = Engine(
            store=self.store, llm=FakeLLM("Noted."), embedder=FakeEmbedder(), profile=PROFILE,
            config=EngineConfig(), background_llm=FakeLLM("{}"), background_profile=PROFILE,
            embedding_profile=EmbeddingProfile("fake", "ollama", "fake-embedder", 0.5, 512),
        )
        for target in ("apps.cerveau.views.get_engine", "apps.cerveau.engine.get_engine",
                       "apps.cerveau.memory_views.get_engine"):
            patcher = patch(target, return_value=self.engine)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.mocks = {}
        for target in ("queue_consolidation", "queue_running_summary", "queue_attachment", "summarize_attachment"):
            patcher = patch(f"apps.cerveau.views.{target}")
            self.mocks[target] = patcher.start()
            self.addCleanup(patcher.stop)

    def post(self, text, *files):
        return self.client.post(reverse("cerveau:send"), {"text": text, "files": list(files)})

    def send(self, text, *files) -> tuple[int, list[dict]]:
        response = self.post(text, *files)
        body = b"".join(response.streaming_content).decode()  # type: ignore[attr-defined]
        return response.status_code, [json.loads(line) for line in body.splitlines()]

    def test_a_small_text_file_is_read_before_the_reply(self):
        status, events = self.send("here are my notes", SimpleUploadedFile("notes.md", NOTES))
        self.assertEqual(status, 200)
        [shared] = [e["files"] for e in events if "files" in e]
        self.assertEqual((shared[0]["filename"], shared[0]["status"]), ("notes.md", "done"))
        message = models.Message.objects.get(content="here are my notes")
        self.assertTrue(models.MessageAttachment.objects.filter(message=message).exists())
        self.mocks["summarize_attachment"].enqueue.assert_called_once()
        self.mocks["queue_attachment"].assert_not_called()
        # The reply could already quote it: the prompt held a passage.
        prompt = self.engine.llm.prompts[0][-1].content  # type: ignore[attr-defined]
        self.assertIn("76,000 pesos", prompt)
        self.assertIn('[shared a file: A', prompt)

    def test_a_pdf_is_read_in_the_background_and_a_file_alone_can_be_sent(self):
        status, events = self.send("", SimpleUploadedFile("lease.pdf", b"%PDF-1.4 not really"))
        self.assertEqual(status, 200)
        self.mocks["queue_attachment"].assert_called_once()
        self.assertTrue(models.Message.objects.filter(content="(shared lease.pdf)").exists())
        prompt = self.engine.llm.prompts[0][-1].content  # type: ignore[attr-defined]
        self.assertIn("still being read", prompt)

    def test_a_refused_file_stops_the_message(self):
        response = self.post("look", SimpleUploadedFile("photo.png", b"\x89PNG\r\n"))
        self.assertEqual(response.status_code, 400)
        self.assertIn("can't read photo.png", response.json()["error"])
        self.assertFalse(models.Message.objects.exists())

    def test_a_forgotten_file_isnt_kept_and_the_page_is_told(self):
        self.send("first", SimpleUploadedFile("notes.md", NOTES))
        models.Attachment.objects.get().set_archived()
        status, events = self.send("again", SimpleUploadedFile("notes-copy.md", NOTES))
        self.assertIn("forget notes-copy.md before", [e for e in events if "notice" in e][0]["notice"])
        self.assertFalse(any("files" in e for e in events))

    def test_the_reading_state_and_the_chat_page_show_files(self):
        self.send("notes", SimpleUploadedFile("notes.md", NOTES))
        attachment = models.Attachment.objects.get()
        state = self.client.get(reverse("cerveau:file_status", args=[attachment.pk])).json()
        self.assertEqual((state["status"], state["filename"]), ("done", "notes.md"))
        self.assertEqual(self.client.get(reverse("cerveau:file_status", args=[999])).status_code, 404)
        page = self.client.get(reverse("cerveau:chat"))
        self.assertContains(page, f'data-file-id="{attachment.pk}"')

    def test_the_files_pages(self):
        self.send("notes", SimpleUploadedFile("notes.md", NOTES))
        attachment = models.Attachment.objects.get()
        listing = self.client.get(reverse("cerveau:files"))
        self.assertContains(listing, "notes.md")
        detail = self.client.get(reverse("cerveau:file", args=[attachment.pk]))
        self.assertContains(detail, "76,000 pesos")                  # its passages
        self.assertContains(detail, "Open the original")
        original = self.client.get(reverse("cerveau:file_original", args=[attachment.pk]))
        self.assertEqual(b"".join(original.streaming_content), NOTES)  # type: ignore[attr-defined]
        episode = self.client.get(reverse("cerveau:episode", args=[attachment.episode_id]))
        self.assertContains(episode, reverse("cerveau:file", args=[attachment.pk]))

    def test_a_file_can_be_forgotten_and_is_then_not_served(self):
        self.send("notes", SimpleUploadedFile("notes.md", NOTES))
        attachment = models.Attachment.objects.get()
        self.client.post(reverse("cerveau:status", args=["attachment", attachment.pk]),
                         {"action": "forget", "next": "/memory/files/"})
        attachment.refresh_from_db()
        self.assertEqual(attachment.status, "Z")
        self.assertEqual(self.client.get(reverse("cerveau:file_original", args=[attachment.pk])).status_code, 404)
