"""evals.turns: conversation files, running turns through brain.recall, the report."""

import contextlib
import io
import unittest
from pathlib import Path

from brain.config import EngineConfig
from evals.cases import CaseError, parse_scenario
from evals.turns import RememberingEmbedder, main, parse_conversations, report, run, with_settings
from tests.fakes import FakeEmbedder
from tests.test_evals import SCENARIO

ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = {
    "scenario": "t",
    "conversations": [
        {"name": "sister, then montreal",
         "turns": [
             {"text": "sister moved montreal", "expect": ["sister:0"]},
             {"text": "montreal nursing", "expect": ["sister:0"]},
         ]},
        {"name": "sister, then coffee",
         "turns": [
             {"text": "sister moved montreal"},
             {"text": "favorite coffee place", "switch": True, "expect": ["coffee:0"], "reject": ["sister:0"]},
         ]},
    ],
}


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = {"t": parse_scenario(SCENARIO)}

    def test_turn_kinds(self):
        first, second = parse_conversations(CONVERSATIONS, self.scenarios)
        self.assertEqual([t.kind for t in first.turns], ["opener", "follow-up"])
        self.assertEqual([t.kind for t in second.turns], ["opener", "switch"])

    def test_unknown_scenario_and_reference_are_errors(self):
        with self.assertRaises(CaseError):
            parse_conversations({**CONVERSATIONS, "scenario": "nope"}, self.scenarios)
        bad = {"scenario": "t", "conversations": [{"name": "x", "turns": [{"text": "a", "expect": ["nope:0"]}]}]}
        with self.assertRaisesRegex(CaseError, "unknown 'nope:0'"):
            parse_conversations(bad, self.scenarios)

    def test_turns_run_through_recall_in_an_open_episode(self):
        conversations = parse_conversations(CONVERSATIONS, self.scenarios)
        results = run(conversations, RememberingEmbedder(FakeEmbedder()), EngineConfig(recent_episodes=0))
        self.assertEqual([(r.conversation, r.index, r.kind) for r in results][:2],
                         [("sister, then montreal", 0, "opener"), ("sister, then montreal", 1, "follow-up")])
        self.assertTrue(results[0].result.hit)
        # The conversation's own messages are never recalled: only the scenario's.
        self.assertTrue(all(ref.split(":")[0] in ("sister", "coffee") for r in results for ref in r.result.kept))
        self.assertIn("follow-up", report(results))

    def test_remembering_embedder_embeds_each_text_once(self):
        inner = FakeEmbedder()
        embedder = RememberingEmbedder(inner)
        embedder.embed(["a", "b"])
        embedder.embed(["b", "a", "c"])
        self.assertEqual(inner.calls, [["a", "b"], ["c"]])

    def test_settings_override_config_and_sources(self):
        config = with_settings(EngineConfig(), ["context_weight=0.4", "context_messages=3", "episodes.w_tag=0.1"])
        self.assertEqual((config.context_weight, config.context_messages, config.episodes.w_tag), (0.4, 3, 0.1))
        self.assertIsInstance(config.context_messages, int)

    def test_shipped_conversations_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--fake-embedder"]), 0)
        self.assertIn("follow-up", out.getvalue())
        self.assertIn("switch", out.getvalue())


if __name__ == "__main__":
    unittest.main()
