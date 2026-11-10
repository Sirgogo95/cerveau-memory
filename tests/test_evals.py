"""The eval harness: case files, filling a store, scoring queries, the sweep, and the
command line. Uses the bag-of-words FakeEmbedder, so no model is downloaded."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from brain.config import EngineConfig
from evals.__main__ import main
from evals.cases import CaseError, QueryCase, load_scenarios, parse_scenario
from evals.runner import QueryResult, prepare, run, summarize, sweep
from brain.types import MemoryKind
from tests.fakes import FakeEmbedder

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases"

SCENARIO = {
    "name": "t",
    "episodes": [
        {
            "key": "sister",
            "messages": [
                ["user", "my sister moved to montreal for nursing"],
                ["assistant", "big move"],
            ],
            "summary": "sister moved to montreal nursing",
            "tags": ["family", "moving"],
            "entities": [{"name": "Lucía", "type": "person", "aliases": ["my sister"]}],
        },
        {
            "key": "coffee",
            "messages": [["user", "la cafetera is my favorite coffee place"]],
        },
    ],
    "queries": [
        {"text": "sister moved montreal", "expect": ["sister:0", "sister"]},
        {"text": "favorite coffee place", "expect": ["coffee:0"], "reject": ["sister:0"]},
        {"text": "banana bread recipe", "expect": []},
    ],
}


class CaseTests(unittest.TestCase):
    def test_parses_episodes_entities_and_queries(self):
        scenario = parse_scenario(SCENARIO, "t.json")
        self.assertEqual([e.key for e in scenario.episodes], ["sister", "coffee"])
        self.assertEqual(scenario.episodes[0].entities[0].aliases, ("my sister",))
        self.assertEqual(scenario.queries[1].reject, ("sister:0",))

    def test_refs_cover_messages_and_summarized_episodes_only(self):
        refs = parse_scenario(SCENARIO).refs()
        self.assertEqual(refs, {"sister:0", "sister:1", "sister", "coffee:0"})

    def test_rejects_unknown_reference(self):
        bad = {**SCENARIO, "queries": [{"text": "q", "expect": ["coffee"]}]}  # no summary
        with self.assertRaisesRegex(CaseError, "unknown 'coffee'"):
            parse_scenario(bad)

    def test_rejects_duplicate_keys_unknown_roles_and_missing_fields(self):
        episode = SCENARIO["episodes"][1]
        with self.assertRaisesRegex(CaseError, "unique"):
            parse_scenario({"episodes": [episode, episode]})
        with self.assertRaisesRegex(CaseError, "role"):
            parse_scenario({"episodes": [{"key": "a", "messages": [["system", "hi"]]}]})
        with self.assertRaises(CaseError):
            parse_scenario({"episodes": [{"messages": []}]})

    def test_attachment_references_name_a_phrase_in_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "files").mkdir()
            (Path(folder) / "files" / "lease.md").write_text("# Deposit\n\nThe deposit is 76,000 pesos.", encoding="utf-8")
            scenario = {"name": "d", "episodes": [{"key": "e", "messages": [["user", "the lease"]],
                                                   "attachments": [{"key": "lease", "file": "files/lease.md"}]}],
                        "queries": [{"text": "deposit?", "expect": ["lease/deposit is 76,000"]}]}
            source = str(Path(folder) / "d.json")
            [episode] = parse_scenario(scenario, source).episodes
            self.assertEqual(episode.attachments[0].filename, "lease.md")
            scenario["queries"] = [{"text": "deposit?", "expect": ["lease/a gym"]}]
            with self.assertRaisesRegex(CaseError, "isn't in that file"):
                parse_scenario(scenario, source)

    def test_loads_a_folder_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "sub" / "a.json").write_text(json.dumps(SCENARIO), encoding="utf-8")
            (Path(tmp) / "b.json").write_text(json.dumps(SCENARIO), encoding="utf-8")
            (Path(tmp) / "sub" / "chats").mkdir()          # the personal chats eval's own files
            (Path(tmp) / "sub" / "chats" / "questions.json").write_text('{"questions": []}', encoding="utf-8")
            self.assertEqual(len(load_scenarios(tmp)), 2)
            (Path(tmp) / "c.json").write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(CaseError, "cannot read"):
                load_scenarios(tmp)

    def test_starter_cases_are_valid(self):
        scenarios = load_scenarios(CASES)
        names = {s.name for s in scenarios}
        self.assertTrue({"life", "work"} <= names)
        self.assertTrue(all(s.queries for s in scenarios if s.name in {"life", "work"}))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.prepared = prepare(parse_scenario(SCENARIO), self.embedder)

    def test_prepare_fills_store_like_the_app(self):
        store = self.prepared.store
        self.assertEqual(sorted(self.prepared.refs.values()), ["coffee:0", "sister", "sister:0", "sister:1"])
        self.assertTrue(all(not e.is_open for e in store.get_episodes([1, 2])))
        [lucia] = store.match_entities(["lucia"])
        self.assertEqual(lucia.name, "Lucía")
        # The alias "my sister" is found in message 0, so it is a mention.
        mentions = store.mention_candidates(MemoryKind.MESSAGE, [lucia.id], 5)
        self.assertEqual([self.prepared.refs[(c.kind, c.id)] for c in mentions], ["sister:0"])
        # Tags are stored with their vectors and attached to their episode.
        [sister_id] = [i for (k, i), ref in self.prepared.refs.items() if ref == "sister"]
        self.assertEqual(len(store.episode_tags([sister_id])[sister_id]), 2)
        self.assertEqual(store.similar_tags(self.embedder.embed(["moving"])[0], 1)[0][0].name, "moving")

    def test_queries_find_expected_memories_and_nothing_for_unrelated(self):
        results = run([self.prepared], self.embedder, EngineConfig())
        by_text = {r.query.text: r for r in results}
        self.assertTrue(by_text["sister moved montreal"].hit)
        self.assertIn("coffee:0", by_text["favorite coffee place"].kept)
        self.assertEqual(by_text["banana bread recipe"].kept, ())

    def test_metrics(self):
        results = [
            QueryResult("s", QueryCase("a", ("x", "y")), ("x", "z")),     # hit, recall 1/2
            QueryResult("s", QueryCase("b", ("w",)), ()),                  # miss
            QueryResult("s", QueryCase("c", (), ("z",)), ("z", "q")),     # 2 noise, 1 rejected
        ]
        metrics = summarize(results)
        self.assertEqual(metrics.queries, 3)
        self.assertAlmostEqual(metrics.hit_rate, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.25)
        self.assertAlmostEqual(metrics.precision, 0.5)   # 1 expected of 2 kept
        self.assertAlmostEqual(metrics.noise, 2.0)
        self.assertEqual(metrics.rejected, 1)

    def test_sweep_recall_never_rises_with_the_threshold(self):
        rows = sweep([self.prepared], self.embedder, EngineConfig(), [0.0, 0.3, 0.6, 0.99])
        recalls = [m.recall for _, m in rows]
        self.assertEqual(recalls, sorted(recalls, reverse=True))


class CommandLineTests(unittest.TestCase):
    def run_main(self, *args: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--fake-embedder", "--embedding", "all-MiniLM-L6-v2",
                                   "--cases", str(CASES / "work.json"), *args]), 0)
        return out.getvalue()

    def test_report(self):
        output = self.run_main("--misses")
        self.assertIn("1 scenario(s), 57 queries", output)
        self.assertIn("min_score 0.35: hit rate", output)  # the profile's threshold

    def test_sweep(self):
        output = self.run_main("--sweep")
        rows = [line.split()[0] for line in output.splitlines() if line.startswith("  0.")]
        self.assertEqual((rows[0], rows[-1], len(rows)), ("0.25", "0.45", 11))  # +/- 0.10
        self.assertIn("0.35", output.split("<- current")[0].splitlines()[-1])

    def test_sweep_grid(self):
        output = self.run_main("--sweep", "--grid", "0.30,0.40,0.05")
        rows = [line.split()[0] for line in output.splitlines() if line.startswith("  0.")]
        self.assertEqual(rows, ["0.30", "0.35", "0.40"])


if __name__ == "__main__":
    unittest.main()
