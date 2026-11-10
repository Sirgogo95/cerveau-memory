"""Names typed with a typo (brain/names.py): the matching rules, and the turn and the
search tool using them."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig
from brain.names import edit_distance, fuzzy_names, name_words, named_entities
from brain.tools import ToolContext, run_tool
from brain.turn import run_turn
from brain.types import Entity, MemoryKind, Role, ToolCall
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

PEOPLE = [
    Entity(1, "Lucía", "person", ("my sister",)),
    Entity(2, "Dr. Almonte", "person"),
    Entity(3, "Ramón", "person"),
    Entity(4, "Puerto Plata", "place"),
    Entity(5, "Pedro", "person"),
]


def found(text, **kwargs):
    return [e.name for e in fuzzy_names(text, PEOPLE, **kwargs)]


class EditDistanceTests(unittest.TestCase):
    def test_edits_and_swaps(self):
        self.assertEqual(edit_distance("lucia", "lucia"), 0)
        self.assertEqual(edit_distance("lucai", "lucia"), 1)     # two letters swapped
        self.assertEqual(edit_distance("almonti", "almonte"), 1)
        self.assertEqual(edit_distance("sopie", "sophie"), 1)    # one missing
        self.assertEqual(edit_distance("pedir", "pedro"), 2)


class FuzzyNamesTests(unittest.TestCase):
    def test_typos_in_capitalized_names_are_found(self):
        self.assertEqual(found("How is Lucai doing?"), ["Lucía"])
        self.assertEqual(found("Did Almonti call back?"), ["Dr. Almonte"])
        self.assertEqual(found("Where is Pedri now?"), ["Pedro"])

    def test_two_word_names(self):
        self.assertEqual(found("We went to Puerto Pltaa last year."), ["Puerto Plata"])

    def test_lowercase_and_sentence_starts_are_left_alone(self):
        self.assertEqual(found("how is lucai doing?"), [])
        self.assertEqual(found("Ramen was great. Lucai too."), [])   # both start a sentence
        self.assertEqual(found("Pedir algo"), [])

    def test_short_words_and_other_first_letters_dont_count(self):
        self.assertEqual(found("I saw Pdr and Bucia."), [])          # 3 letters; another first letter
        self.assertEqual(found("I saw Pdro."), ["Pedro"])            # 4 letters: a plausible typo

    def test_exact_names_and_excluded_entities_are_skipped(self):
        self.assertEqual(found("I asked Lucía."), [])                # exact: found elsewhere
        self.assertEqual(found("How is Lucai?", exclude=[1]), [])

    def test_a_name_asked_for_on_purpose_needs_no_capital(self):
        self.assertEqual(found("lucai", require_capital=False), ["Lucía"])

    def test_name_words(self):
        self.assertEqual(name_words("I met Ana Lopez in Puerto Plata."),
                         ["ana", "lopez", "puerto", "plata", "ana lopez", "puerto plata"])


class InUseTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        lucia = self.store.upsert_entity("Lucía", "person", ["my sister"])
        assert lucia is not None
        self.lucia = lucia
        episode = self.store.open_episode(datetime(2026, 10, 1, tzinfo=UTC))
        [vector] = self.embedder.embed(["Lucía moved to Montreal"])
        message = self.store.add_message(episode.id, Role.USER, "Lucía moved to Montreal",
                                         datetime(2026, 10, 1, tzinfo=UTC), vector)
        self.store.add_mention(lucia.id, message.id)
        self.store.close_episode(episode.id, datetime(2026, 10, 1, 1, tzinfo=UTC))

    def test_named_entities_adds_typos_unless_switched_off(self):
        self.assertEqual([e.id for e in named_entities("How is Lucai?", self.store)], [self.lucia.id])
        self.assertEqual(named_entities("How is Lucai?", self.store, fuzzy=False), [])

    def test_forgotten_entities_are_not_fuzzy_matched_either(self):
        self.store.set_status(MemoryKind.ENTITY, self.lucia.id, "Z")
        self.assertEqual(named_entities("How is Lucai?", self.store), [])

    def test_the_turn_recalls_what_mentions_a_misspelled_name(self):
        llm = FakeLLM("Ok.")
        list(run_turn("How is Lucai settling in?", datetime(2026, 10, 10, tzinfo=UTC),
                      store=self.store, llm=llm, embedder=self.embedder, profile=PROFILE,
                      config=EngineConfig(timezone="UTC")))
        prompt = llm.prompts[0][-1].content
        self.assertIn("Lucía moved to Montreal", prompt)    # found by name, despite the typo
        self.assertIn(f"[N{self.lucia.id}]", prompt)        # and its card

    def test_the_search_tool_resolves_a_misspelled_entity(self):
        context = ToolContext(store=self.store, embedder=self.embedder,
                              config=replace(EngineConfig(), search_min_score=0.2),
                              profile=PROFILE, now=datetime(2026, 10, 10, tzinfo=UTC), tz=UTC)
        result = json.loads(run_tool(ToolCall("search_memory", {"query": "news", "entities": ["lucai"]}), context))
        self.assertNotIn("unknown_names", result)
        self.assertEqual([r["text"] for r in result["results"]], ["Lucía moved to Montreal"])


if __name__ == "__main__":
    unittest.main()
