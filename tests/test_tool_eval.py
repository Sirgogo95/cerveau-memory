"""The tool-use evaluation's own pieces (no model): loading cases, grading answers,
and the remembering embedder."""

import unittest
from pathlib import Path

from evals.tool_eval import KINDS, Question, RememberingEmbedder, is_right, load
from tests.fakes import FakeEmbedder

CASES = Path(__file__).resolve().parent.parent / "evals" / "tool_cases" / "january.json"


class ToolEvalTests(unittest.TestCase):
    def test_the_shipped_cases_load(self):
        scenario, questions, now = load(CASES)
        self.assertEqual(len(scenario.episodes), 8)
        self.assertEqual({q.kind for q in questions}, set(KINDS))
        self.assertEqual(now.year, 2026)

    def test_answers_are_graded_by_words_accents_and_case_ignored(self):
        wedding = Question("When is Pedro's wedding?", "memory", ("14 de febrero",))
        self.assertTrue(is_right(wedding, "La boda es el **14 de Febrero**."))
        self.assertFalse(is_right(wedding, "En marzo."))
        self.assertIsNone(is_right(Question("Write a poem.", "general"), "Rain falls."))

    def test_near_misses_want_an_i_dont_know_and_no_invention(self):
        doctor = Question("Who looked at my knee?", "near_miss", wrong=("almonte",))
        self.assertTrue(is_right(doctor, "The doctor's name isn't mentioned in my notes."))
        self.assertTrue(is_right(doctor, "Je ne connais pas son nom."))
        self.assertFalse(is_right(doctor, "I don't know, maybe Dr. Almonte?"))      # an invention
        self.assertFalse(is_right(Question("Who is Pedro marrying?", "near_miss"), "Pedro is marrying María."))
        self.assertIn("near_miss", KINDS)

    def test_absent_questions_want_an_i_dont_remember(self):
        absent = Question("What's my wife's name?", "absent")
        self.assertTrue(is_right(absent, "I don't remember you telling me."))
        self.assertTrue(is_right(absent, "No recuerdo que me lo hayas dicho."))
        self.assertTrue(is_right(absent, "Je ne me souviens pas."))
        self.assertFalse(is_right(absent, "Her name is Ana."))

    def test_the_embedder_computes_each_text_once(self):
        inner = FakeEmbedder()
        embedder = RememberingEmbedder(inner)
        first = embedder.embed(["a b", "c d"])
        again = embedder.embed(["c d", "a b", "e f"])
        self.assertEqual((again[0], again[1]), (first[1], first[0]))
        self.assertEqual(inner.calls, [["a b", "c d"], ["e f"]])
        self.assertEqual(embedder.model, inner.model)


if __name__ == "__main__":
    unittest.main()
