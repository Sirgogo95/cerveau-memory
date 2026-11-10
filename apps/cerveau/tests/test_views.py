"""The chat page and the streaming endpoint, with the real Django store and a scripted
fake LLM (no Ollama)."""

import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from apps.cerveau import models
from apps.cerveau.engine import Engine
from apps.cerveau.store import DjangoMemoryStore
from brain.config import EngineConfig, with_thinking
from brain.ports import EmbedderError
from brain.types import ToolCall
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE


class ChatViewTests(TestCase):
    def setUp(self):
        self.llm = FakeLLM("Hello there.")
        self.embedder = FakeEmbedder()
        engine = self.engine = Engine(
            store=DjangoMemoryStore(), llm=self.llm, embedder=self.embedder,
            profile=PROFILE, config=EngineConfig(),
            background_llm=self.llm, background_profile=PROFILE,
        )
        patcher = patch("apps.cerveau.views.get_engine", return_value=engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        queue = patch("apps.cerveau.views.queue_consolidation")
        self.queued = queue.start()
        self.addCleanup(queue.stop)
        summary = patch("apps.cerveau.views.queue_running_summary")
        self.summary_queued = summary.start()
        self.addCleanup(summary.stop)

    def send(self, text: str) -> list[dict]:
        response = self.client.post(reverse("cerveau:send"), {"text": text})
        self.assertEqual(response["Content-Type"], "application/x-ndjson")
        body = b"".join(response.streaming_content).decode()  # type: ignore[attr-defined]
        return [json.loads(line) for line in body.splitlines()]

    def test_the_think_switch_answers_that_turn_with_the_thinking_model(self):
        thinking = FakeLLM("Thought it through.")
        with patch("apps.cerveau.views.get_engine", return_value=replace(
                self.engine, thinking_llm=thinking, thinking_profile=with_thinking(PROFILE))):
            response = self.client.post(reverse("cerveau:send"), {"text": "Did I decide that?", "think": "1"})
            body = b"".join(response.streaming_content).decode()  # type: ignore[attr-defined]
            said = "".join(json.loads(line).get("delta", "") for line in body.splitlines())
            self.assertEqual(said, "Thought it through.")
            self.assertEqual(len(self.llm.prompts), 0)
            self.send("And without it?")                  # off by default: the usual model
            self.assertEqual(len(thinking.prompts), 1)
            self.assertEqual(len(self.llm.prompts), 1)

    def test_the_page_has_the_think_switch_off(self):
        page = self.client.get(reverse("cerveau:chat")).content.decode()
        self.assertIn('id="think"', page)
        self.assertNotIn('id="think" name="think" value="1" role="switch" autocomplete="off" checked', page)

    def test_the_stream_gives_the_message_ids_and_a_message_can_be_pinned(self):
        events = self.send("Keep this in mind: the launch is December 5.")
        [ids] = [e["ids"] for e in events if "ids" in e]
        user = models.Message.objects.get(pk=ids["user"])
        self.assertEqual(user.content, "Keep this in mind: the launch is December 5.")
        self.assertEqual(models.Message.objects.get(pk=ids["reply"]).role, "assistant")
        response = self.client.post(reverse("cerveau:pin", args=[user.pk]), {"pinned": "1"})
        self.assertEqual(response.json(), {"id": user.pk, "pinned": True})
        user.refresh_from_db()
        self.assertTrue(user.pinned)
        self.assertContains(self.client.get(reverse("cerveau:chat")), 'aria-pressed="true"')
        self.client.post(reverse("cerveau:pin", args=[user.pk]), {"pinned": "0"})
        user.refresh_from_db()
        self.assertFalse(user.pinned)
        self.assertEqual(self.client.post(reverse("cerveau:pin", args=[999]), {"pinned": "1"}).status_code, 404)

    def test_empty_page_shows_the_empty_state(self):
        response = self.client.get(reverse("cerveau:chat"))
        self.assertContains(response, "on your mind?")
        self.assertContains(response, "No episode yet")

    def test_empty_message_is_rejected(self):
        response = self.client.post(reverse("cerveau:send"), {"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_send_streams_episode_deltas_and_done(self):
        events = self.send("Hi")
        self.assertEqual(events[0], {"episode": events[0]["episode"], "new": True})
        self.assertEqual("".join(e.get("delta", "") for e in events), "Hello there.")
        self.assertEqual(events[-1], {"done": True})

    def test_both_messages_are_stored(self):
        self.send("Hi")
        self.assertEqual(
            list(models.Message.objects.values_list("role", "content")),
            [("user", "Hi"), ("assistant", "Hello there.")],
        )

    def test_second_message_continues_the_episode(self):
        first = self.send("Hi")
        second = self.send("Again")
        self.assertFalse(second[0]["new"])
        self.assertEqual(second[0]["episode"], first[0]["episode"])

    def test_a_message_that_closes_an_episode_queues_its_consolidation(self):
        self.send("Hi")
        old = models.Episode.objects.get()
        # Pretend the conversation went quiet three hours ago.
        models.Message.objects.update(created_at=old.started_at - timedelta(hours=3))
        events = self.send("Back again")
        self.assertTrue(events[0]["new"])
        self.queued.assert_called_once_with(old.pk)

    def test_a_long_episode_queues_its_running_summary(self):
        self.send("Hi")
        episode = models.Episode.objects.get()
        for i in range(19):  # 2 + 19 = 21; the next turn makes 23
            models.Message.objects.create(
                episode=episode, role="user", content=f"m{i}", created_at=episode.started_at
            )
        self.send("And another")
        self.summary_queued.assert_called_once_with(episode.pk)

    def test_model_failure_streams_an_error_and_keeps_the_user_message(self):
        self.llm.reply, self.llm.fail_after = "one two three", 1
        events = self.send("Hi")
        self.assertIn("error", events[-1])
        self.assertNotIn({"done": True}, events)
        self.assertEqual(list(models.Message.objects.values_list("role", flat=True)), ["user"])

    def test_a_memory_search_is_announced_before_the_reply(self):
        self.llm.tool_calls = [(ToolCall("search_memory", {"query": "portal"}),)]
        capable = replace(self.engine, profile=replace(PROFILE, supports_tools=True))
        with patch("apps.cerveau.views.get_engine", return_value=capable):
            events = self.send("What did we decide about the portal?")
        self.assertIn({"tool": "search_memory", "arguments": {"query": "portal"}}, events)
        tool_at = events.index({"tool": "search_memory", "arguments": {"query": "portal"}})
        self.assertLess(tool_at, next(i for i, e in enumerate(events) if "delta" in e))
        self.assertEqual(events[-1], {"done": True})

    def test_past_replies_carry_the_searches_that_led_to_them(self):
        self.llm.tool_calls = [(ToolCall("search_memory", {"query": "portal"}),)]
        capable = replace(self.engine, profile=replace(PROFILE, supports_tools=True))
        with patch("apps.cerveau.views.get_engine", return_value=capable):
            self.send("What did we decide about the portal?")
            page = self.client.get(reverse("cerveau:chat"))
        self.assertEqual(models.ToolCall.objects.count(), 1)
        self.assertContains(page, 'data-tools="[{&quot;tool&quot;: &quot;search_memory&quot;')
        self.assertContains(page, "data-tools=", count=1)  # only on the reply, not on your message

    def test_embedder_failure_streams_an_error(self):
        with patch.object(self.embedder, "embed", side_effect=EmbedderError("Ollama is unreachable")):
            events = self.send("Hi")
        self.assertEqual(events, [{"error": "Ollama is unreachable"}])

    def test_page_shows_the_stored_conversation(self):
        self.send("My dog is Rex")
        response = self.client.get(reverse("cerveau:chat"))
        self.assertContains(response, "My dog is Rex")
        self.assertContains(response, "Hello there.")
        self.assertNotContains(response, "on your mind?")
