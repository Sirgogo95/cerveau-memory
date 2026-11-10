"""Relative dates made absolute (brain/dating.py), in facts and summaries."""

import json
import unittest
from datetime import UTC, date, datetime

from adapters.stores.in_memory import InMemoryStore
from brain.consolidation import extract
from brain.dating import absolute_dates, annotate_dates
from brain.facts import parse_facts
from brain.prompt import REPLIES_HEADER, fit_memory, memory_line
from brain.retrieval import ScoredCandidate
from brain.types import Candidate, MemoryKind, Role
from tests.fakes import FakeLLM
from tests.test_consolidation import EXTRACTION
from tests.test_engine import PROFILE

SUNDAY = date(2023, 10, 22)


class RewriteTests(unittest.TestCase):
    def check(self, cases, day=SUNDAY):
        for before, after in cases:
            with self.subTest(before=before):
                self.assertEqual(absolute_dates(before, day), after)

    def test_exact_days(self):
        self.check([
            ("Caroline passed the interviews last Friday.", "Caroline passed the interviews on Friday 20 October 2023."),
            ("Yesterday Melanie bought figurines.", "On 21 October 2023 Melanie bought figurines."),
            ("John has been sick since yesterday.", "John has been sick since 21 October 2023."),
            ("Caroline will move next Friday.", "Caroline will move on Friday 27 October 2023."),
            ("Maria hiked three days ago.", "Maria hiked on 19 October 2023."),
            ("Jon starts tomorrow morning.", "Jon starts on the morning of 23 October 2023."),
        ])

    def test_last_friday_said_on_a_friday_is_a_week_back(self):
        self.assertEqual(absolute_dates("Jon met Gina last Friday.", date(2023, 10, 20)),
                         "Jon met Gina on Friday 13 October 2023.")

    def test_spans_are_anchored_as_the_answer_key_words_them(self):
        self.check([
            ("Caroline joined a program last weekend.", "Caroline joined a program the weekend before 22 October 2023."),
            ("Over the past week Jon practiced.", "Over the week before 22 October 2023 Jon practiced."),
            ("Tim went to London two weeks ago.", "Tim went to London two weeks before 22 October 2023."),
            ("Gina opened her store last month.", "Gina opened her store in September 2023."),
            ("Jon moved last year.", "Jon moved in 2022."),
            ("Jon met Gina a few days ago.", "Jon met Gina a few days before 22 October 2023."),
            ("Tim visited London years ago.", "Tim visited London years before 22 October 2023."),
            ("Caroline is moving today.", "Caroline is moving on 22 October 2023."),
        ])

    def test_vague_or_ambiguous_words_are_left_alone(self):
        for text in ("Caroline went to a concert on Friday.", "Jon recently started a gym routine.",
                     "The last week of June was hot.", "Last year's trip was fun.", "It was 7 May 2023."):
            self.assertEqual(absolute_dates(text, SUNDAY), text)

    def test_spanish_and_french_in_their_own_words(self):
        self.check([
            ("El usuario fue diagnosticado el mes pasado.", "El usuario fue diagnosticado en septiembre de 2023."),
            ("El usuario empieza mañana.", "El usuario empieza el 23 de octubre de 2023."),
            ("El usuario corre cada mañana.", "El usuario corre cada mañana."),   # "morning", not "tomorrow"
            ("L'utilisateur est allé à Paris vendredi dernier.", "L'utilisateur est allé à Paris le vendredi 20 octobre 2023."),
        ])


class AnnotateTests(unittest.TestCase):
    def test_a_quoted_message_keeps_its_words_with_the_date_beside_them(self):
        self.assertEqual(annotate_dates("Here's a pic from last Friday. Yesterday I went out.", date(2023, 5, 20)),
                         "Here's a pic from last Friday [Friday 19 May 2023]. Yesterday [19 May 2023] I went out.")
        self.assertEqual(annotate_dates("We met on Friday, recently.", SUNDAY), "We met on Friday, recently.")

    def test_message_notes_are_annotated_facts_are_not(self):
        at = datetime(2023, 5, 20, 12, 0, tzinfo=UTC)
        message = Candidate(MemoryKind.MESSAGE, 7, "John: a pic from last Friday", at=at, role=Role.USER)
        fact = Candidate(MemoryKind.FACT, 8, "John met veterans last Friday", at=at)
        self.assertEqual(memory_line(ScoredCandidate(message, 0.9), UTC),
                         "- [M7 · Sat 2023-05-20 · user] John: a pic from last Friday [Friday 19 May 2023]")
        self.assertTrue(memory_line(ScoredCandidate(fact, 0.9), UTC).endswith("John met veterans last Friday"))

    def test_an_earlier_reply_is_one_clipped_line_and_the_users_words_are_whole(self):
        at = datetime(2023, 5, 20, 12, 0, tzinfo=UTC)
        long = "## Pricing\n- License RD$15,000\n" + "detail " * 200
        reply = memory_line(ScoredCandidate(Candidate(MemoryKind.MESSAGE, 9, long, at=at, role=Role.ASSISTANT), 0.9), UTC)
        self.assertTrue(reply.startswith("- [M9 · Sat 2023-05-20 · Cerveau] ## Pricing - License RD$15,000 detail"))
        self.assertTrue(reply.endswith(" [...]"))
        self.assertNotIn("\n", reply)
        mine = memory_line(ScoredCandidate(Candidate(MemoryKind.MESSAGE, 10, long, at=at, role=Role.USER), 0.9), UTC)
        self.assertIn("\n", mine)            # the user's own words are never clipped here

    def test_earlier_replies_come_last_under_their_own_heading(self):
        at = datetime(2023, 5, 20, 12, 0, tzinfo=UTC)
        reply = ScoredCandidate(Candidate(MemoryKind.MESSAGE, 1, "# Moonlight Rally design", at=at, role=Role.ASSISTANT), 0.95)
        mine = ScoredCandidate(Candidate(MemoryKind.MESSAGE, 2, "apply this to a racing game?", at=at, role=Role.USER), 0.8)
        fact = ScoredCandidate(Candidate(MemoryKind.FACT, 3, "The user asked for a game document.", at=at), 0.7)
        lines = fit_memory([reply, mine, fact], 2000, 4.0, UTC)
        self.assertEqual([l[:6] for l in lines], ["- [M2 ", "- [F3 ", REPLIES_HEADER[:6], "- [M1 "])
        self.assertEqual(lines[2], REPLIES_HEADER)
        self.assertNotIn(REPLIES_HEADER, fit_memory([mine, fact], 2000, 4.0, UTC))   # only when there are replies


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        at = datetime(2023, 10, 22, 13, 0, tzinfo=UTC)
        episode = self.store.open_episode(at)
        self.store.add_message(episode.id, Role.USER, "Caroline: I passed the interviews last Friday!", at, None)
        self.messages = self.store.get_messages(episode.id)

    def test_a_fact_is_dated_from_the_line_it_cites(self):
        reply = json.dumps({"language": "English", "facts": [
            {"statement": "Caroline passed the interviews last Friday.", "type": "other", "lines": [1]}]})
        [fact], _ = parse_facts(reply, self.messages, tz=UTC)
        self.assertEqual(fact.statement, "Caroline passed the interviews on Friday 20 October 2023.")
        [undated], _ = parse_facts(reply, self.messages, tz=UTC, dates=False)   # the undated runs, as recorded
        self.assertEqual(undated.statement, "Caroline passed the interviews last Friday.")

    def test_summaries_are_dated_from_the_conversation(self):
        reply = {**EXTRACTION, "summary_long": "Caroline passed the interviews last Friday."}
        extraction = extract(self.messages, FakeLLM(json.dumps(reply)), PROFILE, UTC)
        self.assertEqual(extraction.summary_long, "Caroline passed the interviews on Friday 20 October 2023.")


if __name__ == "__main__":
    unittest.main()
