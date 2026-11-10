"""Pinned messages (they stay in the prompt after the window moves past them) and
carry-over (after a split by subject, the previous episode's last messages lead the
prompt for a few turns)."""

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig
from brain.prompt import pinned_lines
from brain.turn import run_turn
from brain.types import Message, Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_drift import KNEE, ON, TRIP
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
PLAIN = EngineConfig(timezone="UTC", memory_tools=False)


class Conversation(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.minutes = 0
        self.llm = FakeLLM("Noted, thanks.")

    def say(self, text, config=PLAIN, minutes=2):
        self.minutes += minutes
        at = T0 + timedelta(minutes=self.minutes)
        list(run_turn(text, at, store=self.store, llm=self.llm, embedder=self.embedder, profile=PROFILE,
                      config=config, clock=lambda: at + timedelta(seconds=5)))
        return self.llm.prompts[-1]


class PinTests(Conversation):
    def test_a_pinned_message_stays_after_the_window_moves_past_it(self):
        self.say("the database password policy is twelve characters minimum")
        episode = self.store.current_episode()
        assert episode is not None
        [first, _] = self.store.get_messages(episode.id)
        self.store.set_pinned(first.id, True)
        self.store.save_running_summary(episode.id, "They talked about passwords.", first.id + 1)
        system = self.say("what else should I configure")[0].content
        self.assertIn("<pinned>", system)
        self.assertIn(f"[M{first.id} · Thu 2026-10-01 · user] the database password policy", system)
        self.store.set_pinned(first.id, False)
        self.assertNotIn("<pinned>", self.say("and the backups")[0].content)

    def test_a_pin_still_in_the_window_isnt_repeated(self):
        self.say("remember the launch is on december fifth")
        episode = self.store.current_episode()
        assert episode is not None
        self.store.set_pinned(self.store.get_messages(episode.id)[0].id, True)
        self.assertNotIn("<pinned>", self.say("ok")[0].content)

    def test_pins_that_dont_fit_keep_the_newest(self):
        pins = [Message(i, 1, Role.USER, f"pin {i} " + "x" * 200, T0) for i in range(1, 6)]
        lines = pinned_lines(pins, 160, 4.0)
        self.assertTrue(lines)
        self.assertLess(len(lines), 5)
        self.assertTrue(lines[-1].startswith("- [M5 "))         # the newest kept, shown oldest first

    def test_settings_are_validated(self):
        for bad in (dict(carryover_messages=-1), dict(pinned_share=1.0)):
            with self.assertRaises(ConfigError):
                replace(EngineConfig(), **bad)


class CarryOverTests(Conversation):
    def test_after_a_split_the_previous_episodes_last_messages_lead_the_prompt(self):
        config = replace(ON, carryover_messages=3)
        for text in KNEE + TRIP:
            prompt = self.say(text, config)
        # The split happened on the second trip message: the prompt holds the end of
        # the knee conversation before the new episode's messages.
        texts = [m.content for m in prompt[1:]]
        self.assertIn(KNEE[-1], texts)
        self.assertLess(texts.index(KNEE[-1]), texts.index(TRIP[0]))
        # 3 knee messages on the new episode's first turn, then 2, then 1, then none.
        self.assertEqual(texts.index(TRIP[0]), 3)
        counts = []
        for text in ("what about a boat tour in samana for the whale trip",
                     "and which beach near las terrenas for the whale trip",
                     "how early should we book the whale trip boats"):
            later = [m.content for m in self.say(text, config)[1:]]
            counts.append(later.index(TRIP[0]))
        self.assertEqual(counts, [2, 1, 0])

    def test_not_after_a_silence_and_off_at_zero(self):
        config = replace(ON, carryover_messages=3, drift_threshold=0)
        self.say(KNEE[0], config)
        prompt = self.say(TRIP[0], config, minutes=200)          # a new episode after a silence
        self.assertNotIn(KNEE[0], [m.content for m in prompt[1:]])
        off = replace(ON, carryover_messages=0)
        self.setUp()
        for text in KNEE + TRIP:
            prompt = self.say(text, off)
        self.assertNotIn(KNEE[-1], [m.content for m in prompt[1:]])


if __name__ == "__main__":
    unittest.main()
