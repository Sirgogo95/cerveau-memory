"""Episodes split by change of subject (brain/segmentation.py, PIPELINE.md step 3): the
two-strike rule, the split in a running conversation, and the segmentation eval."""

import contextlib
import io
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig
from brain.segmentation import drift_similarity, drift_step
from brain.turn import run_turn
from brain.types import Role
from evals.segments import main as segments_main
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
ON = EngineConfig(timezone="UTC", drift_threshold=0.35, drift_window=2, memory_tools=False)  # fake vectors: shared words
KNEE = ["my knee pain is swelling again after basketball",
        "the knee swelling hurts on the stairs",
        "should I put ice on the knee swelling",
        "how long does knee swelling take to heal"]
TRIP = ["planning a whale trip to samana in february",
        "which february week is best for the whale trip to samana"]


class StepTests(unittest.TestCase):
    def test_two_strikes_in_a_row_split_and_a_close_message_resets(self):
        config = replace(ON, drift_threshold=0.5)
        long = "a message long enough to be checked"
        self.assertEqual(drift_step(long, 0.7, 1, config), ("same", 0))
        self.assertEqual(drift_step(long, 0.3, 0, config), ("strike", 1))
        self.assertEqual(drift_step(long, 0.3, 1, config), ("split", 0))
        self.assertEqual(drift_step("ok thanks", 0.1, 1, config), ("skip", 1))       # too short
        self.assertEqual(drift_step(long, None, 1, config), ("skip", 1))            # episode too young

    def test_similarity_is_to_the_average_of_the_recent_messages(self):
        self.assertAlmostEqual(drift_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]), 0.7071, places=3)

    def test_settings_are_validated(self):
        for bad in (dict(drift_threshold=1.5), dict(drift_strikes=0), dict(drift_window=0)):
            with self.assertRaises(ConfigError):
                replace(EngineConfig(), **bad)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.closed: list[int] = []
        self.minutes = 0

    def say(self, text, config=ON):
        self.minutes += 2
        list(run_turn(text, T0 + timedelta(minutes=self.minutes), store=self.store, llm=FakeLLM("Noted, thanks."),
                      embedder=self.embedder, profile=PROFILE, config=config,
                      on_episode_closed=self.closed.append))

    def episode_texts(self):
        episodes = sorted({m.episode_id for m in self.store._messages.values()})
        return [[m.content for m in self.store.get_messages(e) if m.role is Role.USER] for e in episodes]

    def test_a_change_of_subject_starts_a_new_episode_at_its_first_message(self):
        for text in KNEE + TRIP:
            self.say(text)
        self.assertEqual(self.episode_texts(), [KNEE, TRIP])
        [old] = self.closed
        [first] = self.store.get_episodes([old])
        self.assertFalse(first.is_open)
        new = self.store.current_episode()
        assert new is not None
        # The first trip message moved with Cerveau's reply to it.
        roles = [m.role for m in self.store.get_messages(new.id)]
        self.assertEqual(roles, [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT])
        self.assertEqual(first.ended_at, self.store.get_messages(old)[-1].created_at)

    def test_one_drifting_message_is_a_tangent_not_a_split(self):
        for text in KNEE + [TRIP[0], KNEE[0]]:
            self.say(text)
        self.assertEqual(len(self.episode_texts()), 1)
        current = self.store.current_episode()
        assert current is not None
        self.assertEqual((current.drift_strikes, current.drift_pending_id), (0, None))

    def test_off_by_default(self):
        for text in KNEE + TRIP:
            self.say(text, replace(ON, drift_threshold=0))
        self.assertEqual(len(self.episode_texts()), 1)
        self.assertEqual(self.closed, [])


class EvalTests(unittest.TestCase):
    def test_the_segmentation_eval_runs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(segments_main(["--fake-embedder", "--details"]), 0)
        self.assertIn("topic changes", out.getvalue())


if __name__ == "__main__":
    unittest.main()
