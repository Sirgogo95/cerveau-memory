"""Time expressions: finding the period a message asks about, and what automatic
recall does with it (a filter on memory and on the recent conversations)."""

import unittest
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig, SourceConfig
from brain.periods import find_period, period_note
from brain.retrieval import retrieve
from brain.turn import run_turn
from brain.types import Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

NOW = datetime(2026, 10, 17, 14, 0, tzinfo=UTC)   # Saturday
SD = ZoneInfo("America/Santo_Domingo")             # UTC-4


def label(text: str) -> str | None:
    period = find_period(text, NOW, SD)
    return period.label if period else None


class FindPeriodTests(unittest.TestCase):
    def test_months_in_three_languages(self):
        for text in ("What did we talk about in August?", "¿De qué hablamos en agosto?",
                     "On a parlé de quoi en août ?", "au mois d'août", "durante agosto"):
            self.assertEqual(label(text), "August 2026 (Aug 1 to Aug 31)", text)

    def test_a_month_starts_and_ends_at_local_midnight(self):
        period = find_period("in August", NOW, SD)
        assert period is not None
        self.assertEqual(period.start, datetime(2026, 8, 1, tzinfo=SD))
        self.assertEqual(period.end, datetime(2026, 9, 1, tzinfo=SD))
        self.assertTrue(period.contains(datetime(2026, 8, 31, 23, 59, tzinfo=SD)))
        self.assertFalse(period.contains(datetime(2026, 9, 1, tzinfo=SD)))

    def test_an_explicit_year_wins(self):
        self.assertEqual(label("August 2025 was busy"), "August 2025 (Aug 1 to Aug 31)")
        self.assertEqual(label("during March 2026"), "March 2026 (Mar 1 to Mar 31)")

    def test_a_month_still_ahead_is_a_plan_not_a_period(self):
        self.assertIsNone(label("What am I doing in December?"))
        self.assertEqual(label("in December 2025"), "December 2025 (Dec 1 to Dec 31)")

    def test_month_words_that_are_not_months(self):
        self.assertIsNone(label("May I ask something?"))
        self.assertIsNone(label("march on"))
        self.assertEqual(label("in May"), "May 2026 (May 1 to May 31)")

    def test_relative_periods(self):
        self.assertEqual(label("yesterday"), "yesterday (Oct 16)")
        self.assertEqual(label("la semana pasada"), "last week (Oct 5 to Oct 11)")
        self.assertEqual(label("la semaine dernière"), "last week (Oct 5 to Oct 11)")
        self.assertEqual(label("this week"), "this week (Oct 12 to Oct 17)")
        self.assertEqual(label("What was I up to last month?"),
                         "last month, September 2026 (Sep 1 to Sep 30)")
        self.assertEqual(label("este mes"), "this month (Oct 1 to Oct 17)")
        self.assertEqual(label("el año pasado"), "last year, 2025 (Jan 1 to Dec 31)")

    def test_the_users_timezone_decides_the_day(self):
        late = datetime(2026, 10, 17, 2, 0, tzinfo=UTC)   # still Friday the 16th in Santo Domingo
        period = find_period("yesterday", late, SD)
        assert period is not None
        self.assertEqual(period.label, "yesterday (Oct 15)")

    def test_no_period(self):
        self.assertIsNone(label("Explain list comprehensions."))
        self.assertIsNone(label(""))

    def test_the_note_says_whether_anything_is_from_then(self):
        period = find_period("in July", NOW, SD)
        assert period is not None
        self.assertIn("Notes dated then are included", period_note(period, True))
        self.assertIn("No conversation took place then", period_note(period, False))
        self.assertIn("July 2026", period_note(period, False))


class PeriodRecallTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.august = self.episode(datetime(2026, 8, 12, 15, tzinfo=UTC), "booked the dentist for the 28th")
        self.september = self.episode(datetime(2026, 9, 7, 15, tzinfo=UTC),
                                      "what did we talk about the wedding speech")

    def episode(self, start, text):
        episode = self.store.open_episode(start)
        [vector] = self.embedder.embed([text])
        self.store.add_message(episode.id, Role.USER, text, start, vector)
        self.store.close_episode(episode.id, start + timedelta(minutes=5))
        return episode.id

    def test_a_period_keeps_only_memory_from_then_whatever_its_score(self):
        [query] = self.embedder.embed(["what did we talk about in august"])
        strict = EngineConfig(messages=SourceConfig(min_score=0.99))
        plain = retrieve(query, self.store, strict)
        self.assertEqual(plain, [])  # nothing is close enough on meaning
        august = find_period("in August", NOW, SD)
        dated = retrieve(query, self.store, strict, period=august)
        self.assertEqual({m.candidate.episode_id for m in dated}, {self.august})  # not September's

    def turn(self, text, llm):
        config = EngineConfig(timezone="America/Santo_Domingo", recent_episodes=3)
        return list(run_turn(text, NOW, store=self.store, llm=llm, embedder=self.embedder,
                             profile=PROFILE, config=config))

    def test_the_turn_filters_recent_conversations_and_notes_the_period(self):
        llm = FakeLLM("Ok.")
        self.turn("What did we talk about in August?", llm)
        system, last = llm.prompts[0][0].content, llm.prompts[0][-1].content
        self.assertNotIn(f"E{self.september} ", system)       # September left out of the recent list
        self.assertIn("The user asked about August 2026", last)
        self.assertIn("Notes dated then are included", last)
        self.assertIn("booked the dentist", last)

    def test_a_period_with_nothing_in_it_says_so(self):
        llm = FakeLLM("Ok.")
        self.turn("What did we talk about in July?", llm)
        last = llm.prompts[0][-1].content
        self.assertIn("No conversation took place then", last)
        self.assertNotIn("booked the dentist", last)


if __name__ == "__main__":
    unittest.main()
