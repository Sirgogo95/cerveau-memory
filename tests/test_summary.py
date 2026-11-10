"""The running summary: when a fold is due, what the model is asked, how folding
advances, and how the summary reaches the prompt. InMemoryStore and a scripted LLM."""

import unittest
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig
from brain.prompt import SO_FAR_HEADER, system_message
from brain.summary import SummaryError, fold, fold_running_summary, summary_due
from brain.turn import run_turn
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)
CONFIG = EngineConfig(recent_window=20, summary_chunk=10, timezone="UTC")


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.episode = self.store.open_episode(T0)

    def talk(self, count: int, start: int = 0) -> list[int]:
        ids = []
        for i in range(start, start + count):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            ids.append(self.store.add_message(self.episode.id, role, f"message {i}", at(i)).id)
        return ids

    def run_fold(self, llm: FakeLLM) -> int:
        return fold_running_summary(self.episode.id, store=self.store, llm=llm, config=CONFIG)

    def test_due_only_past_the_window(self):
        self.assertFalse(summary_due(20, CONFIG))
        self.assertTrue(summary_due(21, CONFIG))

    def test_nothing_is_folded_within_the_window(self):
        self.talk(20)
        llm = FakeLLM("summary")
        self.assertEqual(self.run_fold(llm), 0)
        self.assertEqual(llm.prompts, [])

    def test_the_oldest_chunk_is_folded(self):
        ids = self.talk(25)
        self.assertEqual(self.run_fold(FakeLLM("The user counted messages.")), 10)
        [episode] = self.store.get_episodes([self.episode.id])
        self.assertEqual(episode.running_summary, "The user counted messages.")
        self.assertEqual(episode.summarized_until, ids[9])
        uncovered = self.store.get_messages(self.episode.id, after=episode.summarized_until)
        self.assertEqual(len(uncovered), 15)

    def test_folds_repeat_until_within_the_window(self):
        self.talk(41)
        llm = FakeLLM(replies=["first summary", "second summary", "third summary"])
        self.assertEqual(self.run_fold(llm), 30)  # 41 -> 31 -> 21 -> 11
        self.assertEqual(len(llm.prompts), 3)
        [episode] = self.store.get_episodes([self.episode.id])
        self.assertEqual(episode.running_summary, "third summary")

    def test_each_fold_builds_on_the_previous_summary(self):
        self.talk(31)
        llm = FakeLLM(replies=["first summary", "second summary"])
        self.run_fold(llm)
        first, second = (p[-1].content for p in llm.prompts)
        self.assertIn("nothing yet", first)
        self.assertIn("User: message 0", first)
        self.assertIn("Summary so far:\nfirst summary", second)
        self.assertIn("message 10", second)
        self.assertNotIn("message 0\n", second)

    def test_a_late_or_repeated_run_does_nothing_more(self):
        self.talk(25)
        self.run_fold(FakeLLM("summary"))
        llm = FakeLLM("other")
        self.assertEqual(self.run_fold(llm), 0)
        self.assertEqual(llm.prompts, [])

    def test_an_empty_reply_raises_and_keeps_earlier_progress(self):
        self.talk(31)

        class EmptySecond(FakeLLM):
            def chat(self, messages, *, schema=None):
                response = super().chat(messages, schema=schema)
                return response if len(self.prompts) == 1 else type(response)("", response.model)

        with self.assertRaises(SummaryError):
            self.run_fold(EmptySecond("first summary"))
        [episode] = self.store.get_episodes([self.episode.id])
        self.assertEqual(episode.running_summary, "first summary")

    def test_unknown_episode_is_a_no_op(self):
        self.assertEqual(
            fold_running_summary(999, store=self.store, llm=FakeLLM(), config=CONFIG), 0
        )

    def test_fold_attributes_speakers(self):
        self.talk(2)
        llm = FakeLLM("s")
        fold("", self.store.get_messages(self.episode.id), llm)
        prompt = llm.prompts[0][-1].content
        self.assertIn("User: message 0", prompt)
        self.assertIn("Cerveau: message 1", prompt)


class PromptTests(unittest.TestCase):
    def test_the_running_summary_goes_into_the_system_message(self):
        text = system_message(T0, running_summary="The user is planning a trip.", tz=UTC)
        self.assertIn(SO_FAR_HEADER, text)
        self.assertIn("The user is planning a trip.", text)

    def test_no_summary_means_no_block(self):
        self.assertNotIn(SO_FAR_HEADER, system_message(T0, tz=UTC))


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.due: list[int] = []

    def turn(self, text: str, minutes: float, llm: FakeLLM) -> None:
        list(run_turn(
            text, at(minutes), store=self.store, llm=llm, embedder=self.embedder,
            profile=PROFILE, config=CONFIG, clock=lambda: at(minutes) + timedelta(seconds=5),
            on_summary_due=self.due.append,
        ))

    def test_the_fold_is_announced_once_the_window_is_outgrown(self):
        llm = FakeLLM("ok")
        for i in range(10):  # 10 turns = 20 messages: within the window
            self.turn(f"question {i}", i, llm)
        self.assertEqual(self.due, [])
        self.turn("question 10", 10, llm)  # 22 messages
        self.assertEqual(self.due, [1])

    def test_the_prompt_shows_the_summary_and_only_uncovered_messages(self):
        llm = FakeLLM("ok")
        for i in range(11):
            self.turn(f"question {i}", i, llm)
        fold_running_summary(1, store=self.store, llm=FakeLLM("They asked ten questions."), config=CONFIG)
        self.turn("question 11", 11, llm)
        system, *chat = llm.prompts[-1]
        self.assertIn("They asked ten questions.", system.content)
        contents = [m.content for m in chat]
        self.assertNotIn("question 0", contents)  # folded into the summary
        self.assertIn("question 5", contents)
        self.assertEqual(contents[-1], "question 11")


if __name__ == "__main__":
    unittest.main()
