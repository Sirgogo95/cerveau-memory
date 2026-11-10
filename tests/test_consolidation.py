"""Consolidation on InMemoryStore with a scripted LLM: parsing and cleaning the
model's reply, condensing long episodes, and applying the result."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from adapters.stores.in_memory import InMemoryStore
from brain.config import ModelProfile
from brain.consolidation import (
    EXTRACTION_SCHEMA,
    ConsolidationError,
    clean_entities,
    clean_tags,
    consolidate_episode,
    parse_extraction,
    when,
    with_language,
)
from brain.text import (
    contains_phrase,
    identity_keys,
    normalize_key,
    phrase_keys,
    search_keys,
)
from brain.facts import FactSettings
from brain.types import MemoryKind, Message, Role
from tests.fakes import FakeEmbedder, FakeLLM

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
PROFILE = ModelProfile(
    name="fake", provider="fake", context_window=4096, output_reserve=512, memory_budget=2000
)

EXTRACTION = {
    "title": "Lucia moves to Montreal",
    "summary_short": "The user's sister Lucia moved to Montreal to work as a nurse.",
    "summary_long": "The user said their sister Lucia moved to Montreal, where she works as a nurse.",
    "tags": ["Family", "relocation"],
    "entities": [
        {"name": "Lucía", "type": "person", "aliases": ["my sister"]},
        {"name": "Montreal", "type": "place", "aliases": []},
    ],
}


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


class SynonymEmbedder(FakeEmbedder):
    """FakeEmbedder that reads "relatives" as "family": two tag names, one meaning."""

    def vector(self, text: str) -> list[float]:
        return super().vector(text.replace("relatives", "family"))


class TextTests(unittest.TestCase):
    def test_normalize_key_ignores_accents_case_and_spacing(self):
        self.assertEqual(normalize_key("  José  María "), "jose maria")
        self.assertEqual(normalize_key("MONTRÉAL"), normalize_key("montreal"))

    def test_keys_drop_punctuation(self):
        self.assertEqual(normalize_key("Dr. Marta Peña"), "dr marta pena")
        self.assertEqual(normalize_key("José-María, Jr."), "jose maria jr")

    def test_phrase_keys_are_every_run_of_up_to_four_words(self):
        keys = phrase_keys("Did Dr. Peña call?")
        self.assertIn("dr pena", keys)
        self.assertIn("did dr pena call", keys)
        self.assertIn("pena", keys)
        self.assertNotIn("did call", keys)  # not consecutive

    def test_identity_keys_are_the_name_and_aliases(self):
        self.assertEqual(
            identity_keys("Lucía", ["my sister", "LUCIA", ""]), ["lucia", "my sister"]
        )

    def test_people_are_also_found_by_name_parts(self):
        self.assertEqual(
            search_keys("Dr. Marta Peña", ["my colleague"], "person"),
            ["dr marta pena", "my colleague", "marta", "pena"],
        )

    def test_titles_and_short_parts_never_find_a_person(self):
        keys = search_keys("Dr. Ana de la Cruz", type="person")
        self.assertNotIn("dr", keys)
        self.assertNotIn("de", keys)
        self.assertNotIn("la", keys)
        self.assertIn("cruz", keys)

    def test_places_are_not_found_by_their_parts(self):
        self.assertEqual(search_keys("Santo Domingo", type="place"), ["santo domingo"])

    def test_contains_phrase_matches_whole_words_only(self):
        self.assertTrue(contains_phrase("I met Ana today", "ana"))
        self.assertFalse(contains_phrase("a banana", "ana"))
        self.assertTrue(contains_phrase("My sister lives there", "my sister"))
        self.assertTrue(contains_phrase("Lucia called", "Lucía"))


class ParsingTests(unittest.TestCase):
    def test_tags_are_read_from_english_tags(self):
        reply = {k: v for k, v in EXTRACTION.items() if k != "tags"} | {"english_tags": ["travel"]}
        self.assertEqual(parse_extraction(json.dumps(reply)).tags, ("travel",))

    def test_parses_a_complete_reply(self):
        result = parse_extraction(json.dumps(EXTRACTION))
        self.assertEqual(result.title, "Lucia moves to Montreal")
        self.assertEqual(result.tags, ("family", "relocation"))
        self.assertEqual([e.name for e in result.entities], ["Lucía", "Montreal"])

    def test_rejects_non_json(self):
        with self.assertRaises(ConsolidationError):
            parse_extraction("Sure! Here is the summary:")

    def test_rejects_a_reply_without_a_short_summary(self):
        with self.assertRaises(ConsolidationError):
            parse_extraction(json.dumps({**EXTRACTION, "summary_short": " "}))

    def test_missing_long_summary_falls_back_to_the_short_one(self):
        result = parse_extraction(json.dumps({**EXTRACTION, "summary_long": ""}))
        self.assertEqual(result.summary_long, result.summary_short)

    def test_tags_are_lowercased_deduplicated_and_capped(self):
        self.assertEqual(
            clean_tags(["Health", "health ", "#Travel", "", "a", "b", "c", "d"]),
            ("health", "travel", "a", "b", "c"),
        )

    def test_entities_are_merged_and_cleaned(self):
        entities = clean_entities([
            {"name": "Lucía", "type": "PERSON", "aliases": ["my sister", "lucia"]},
            {"name": "lucia", "type": "person", "aliases": ["Lu"]},
            {"name": "", "type": "place"},
            "not an object",
            {"name": "Acme", "type": "spaceship"},
        ])
        self.assertEqual(len(entities), 2)
        lucia, acme = entities
        self.assertEqual((lucia.name, lucia.type), ("Lucía", "person"))
        self.assertEqual(lucia.aliases, ("my sister", "Lu"))  # "lucia" repeats the name
        self.assertEqual(acme.type, "other")


class ConsolidateEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def episode(self, *texts: str, close: bool = True) -> int:
        episode = self.store.open_episode(T0)
        for i, text in enumerate(texts):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            self.store.add_message(episode.id, role, text, at(i))
        if close:
            self.store.close_episode(episode.id, at(len(texts)))
        return episode.id

    def run_it(self, episode_id: int, llm: FakeLLM, **kwargs):
        return consolidate_episode(
            episode_id, store=self.store, llm=llm, embedder=self.embedder,
            profile=PROFILE, now=at(500), **kwargs,
        )

    def lucia_episode(self) -> int:
        return self.episode(
            "My sister Lucía just moved to Montreal.",
            "Noted: Lucia moved to Montreal.",
            "She works as a nurse there.",
        )

    def test_the_model_is_told_the_conversations_date_in_the_users_timezone(self):
        # T0 is 09:00 UTC on Wed 2026-10-14: 05:00 the same day in Santo Domingo.
        episode_id = self.lucia_episode()
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(episode_id, llm, tz=ZoneInfo("America/Santo_Domingo"),
                    facts=FactSettings())
        extraction, facts = llm.prompts[0][1].content, llm.prompts[1][1].content
        self.assertIn("Conversation of Wed 2026-10-14 05:00:", extraction)
        self.assertIn("write the date it means", extraction)
        self.assertIn("Conversation of Wed 2026-10-14 05:00:", facts)
        self.assertIn("write the date it means", facts)

    def test_undated_prompts_are_the_ones_from_before_the_date_fix(self):
        episode_id = self.lucia_episode()
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(episode_id, llm, facts=FactSettings(), dated=False)
        extraction, facts = llm.prompts[0][1].content, llm.prompts[1][1].content
        for prompt in (extraction, facts):
            self.assertIn("Conversation:\n", prompt)
            self.assertNotIn("Conversation of", prompt)
            self.assertNotIn("write the date it means", prompt)
            self.assertNotIn("{dates}", prompt)

    def test_a_conversation_across_midnight_gives_both_days(self):
        late = datetime(2026, 10, 14, 23, 30, tzinfo=UTC)
        messages = [Message(1, 1, Role.USER, "hi", late), Message(2, 1, Role.USER, "still up", late + timedelta(hours=1))]
        self.assertEqual(when(messages, UTC), "Wed 2026-10-14 23:30 to Thu 2026-10-15 00:30")
        self.assertEqual(when(messages, ZoneInfo("America/Santo_Domingo")), "Wed 2026-10-14 19:30")

    def test_summaries_are_saved_and_the_episode_marked(self):
        episode_id = self.lucia_episode()
        self.run_it(episode_id, FakeLLM(json.dumps(EXTRACTION)))
        [episode] = self.store.get_episodes([episode_id])
        self.assertEqual(episode.title, "Lucia moves to Montreal")
        self.assertTrue(episode.summary_short.startswith("The user's sister"))
        self.assertEqual(episode.consolidated_at, at(500))

    def test_the_episode_becomes_findable_by_its_summary(self):
        episode_id = self.lucia_episode()
        self.run_it(episode_id, FakeLLM(json.dumps(EXTRACTION)))
        query = self.embedder.embed(["sister lucia nurse montreal"])[0]
        hits = self.store.semantic_candidates(MemoryKind.EPISODE, query, 5)
        self.assertEqual([h.id for h in hits], [episode_id])

    def test_the_language_is_worked_out_in_code_and_fixed_in_the_schema(self):
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(self.episode("¿Dónde vive mi hermana Lucía ahora?", "En Montreal.",
                                 "Ella trabaja como enfermera en el hospital."), llm)
        [schema] = llm.schemas
        assert schema is not None
        self.assertEqual(schema["properties"]["language"], {"type": "string", "enum": ["Spanish"]})
        self.assertTrue(llm.prompts[0][-1].content.startswith(
            "The user wrote in Spanish: write the title and summaries in Spanish."))

    def test_no_clear_language_leaves_it_to_the_model(self):
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(self.episode("ok", "Sure.", "thanks"), llm)
        self.assertIs(llm.schemas[0], EXTRACTION_SCHEMA)
        self.assertFalse(llm.prompts[0][-1].content.startswith("The user wrote in"))

    def test_the_model_gets_the_schema_and_transcript_but_not_known_tags(self):
        self.store.upsert_tag("pets")
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(self.lucia_episode(), llm)
        self.assertEqual(llm.schemas, [with_language(EXTRACTION_SCHEMA, "English")])
        prompt = llm.prompts[0][-1].content
        self.assertNotIn("pets", prompt)              # copied into unrelated episodes
        self.assertIn("english_tags", prompt)
        self.assertIn("User: My sister Lucía just moved to Montreal.", prompt)
        self.assertIn("Cerveau: Noted: Lucia moved to Montreal.", prompt)

    def test_summaries_follow_the_conversation_language_and_tags_stay_english(self):
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(self.lucia_episode(), llm)
        system, instructions = llm.prompts[0][0].content, llm.prompts[0][-1].content
        self.assertIn("summaries in the language the user wrote in", system)
        self.assertIn("tags in English", system)
        self.assertIn("in English and lowercase", instructions)

    def test_tags_entities_and_mentions_are_stored(self):
        episode_id = self.lucia_episode()
        report = self.run_it(episode_id, FakeLLM(json.dumps(EXTRACTION)))
        assert report is not None
        self.assertEqual(report.tags, ["family", "relocation"])
        self.assertEqual(report.entities, ["Lucía", "Montreal"])
        # Lucía: messages 1 ("My sister Lucía") and 2 ("Lucia"); Montreal: messages 1 and 2.
        self.assertEqual(report.mentions, 4)
        self.assertEqual([t.name for t in self.store.list_tags(10)], ["family", "relocation"])

    def test_forgotten_tags_and_entities_are_not_relearned(self):
        tag = self.store.upsert_tag("family")
        person = self.store.upsert_entity("Lucia", "person")
        assert tag is not None and person is not None
        self.store.set_status(MemoryKind.TAG, tag.id, "Z")
        self.store.set_status(MemoryKind.ENTITY, person.id, "Z")
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(EXTRACTION)))
        assert report is not None
        self.assertEqual(report.tags, ["relocation"])
        self.assertEqual(report.entities, ["Montreal"])
        self.assertEqual(report.skipped, ["family", "Lucía"])

    def test_a_tag_that_means_the_same_reuses_the_existing_one(self):
        self.embedder = SynonymEmbedder()
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(EXTRACTION)), tag_same_score=0.85)
        assert report is not None
        later = {**EXTRACTION, "tags": ["relatives", "relocation"]}
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(later)), tag_same_score=0.85)
        assert report is not None
        self.assertEqual(report.tags, ["family", "relocation"])
        self.assertEqual(report.merged, [("relatives", "family")])
        self.assertEqual(sorted(t.name for t in self.store.list_tags(10)), ["family", "relocation"])

    def test_a_forgotten_tag_is_not_relearned_under_another_name(self):
        self.embedder = SynonymEmbedder()
        family = self.store.upsert_tag("family", self.embedder.embed(["family"])[0])
        assert family is not None
        self.store.set_status(MemoryKind.TAG, family.id, "Z")
        later = {**EXTRACTION, "tags": ["relatives", "relocation"]}
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(later)), tag_same_score=0.85)
        assert report is not None
        self.assertEqual(report.tags, ["relocation"])
        self.assertEqual(report.skipped, ["relatives"])

    def test_different_tags_and_no_threshold_create_new_tags(self):
        self.embedder = SynonymEmbedder()
        self.run_it(self.lucia_episode(), FakeLLM(json.dumps(EXTRACTION)), tag_same_score=0.85)
        other = {**EXTRACTION, "tags": ["cooking"]}
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(other)), tag_same_score=0.85)
        assert report is not None
        self.assertEqual((report.tags, report.merged), (["cooking"], []))
        later = {**EXTRACTION, "tags": ["relatives"]}
        report = self.run_it(self.lucia_episode(), FakeLLM(json.dumps(later)))   # merging off
        assert report is not None
        self.assertEqual((report.tags, report.merged), (["relatives"], []))

    def test_already_consolidated_episodes_are_skipped(self):
        episode_id = self.lucia_episode()
        self.run_it(episode_id, FakeLLM(json.dumps(EXTRACTION)))
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.assertIsNone(self.run_it(episode_id, llm))
        self.assertEqual(llm.prompts, [])

    def test_force_redoes_an_episode(self):
        episode_id = self.lucia_episode()
        self.run_it(episode_id, FakeLLM(json.dumps(EXTRACTION)))
        redo = {**EXTRACTION, "title": "Better title"}
        self.run_it(episode_id, FakeLLM(json.dumps(redo)), force=True)
        [episode] = self.store.get_episodes([episode_id])
        self.assertEqual(episode.title, "Better title")

    def test_open_and_unknown_episodes_are_skipped(self):
        open_id = self.episode("still talking", close=False)
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.assertIsNone(self.run_it(open_id, llm))
        self.assertIsNone(self.run_it(999, llm))
        self.assertEqual(llm.prompts, [])

    def test_an_empty_episode_is_marked_without_calling_the_model(self):
        episode_id = self.episode()
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(episode_id, llm)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(self.store.pending_consolidation(10), [])

    def test_an_unusable_reply_leaves_the_episode_pending(self):
        episode_id = self.lucia_episode()
        with self.assertRaises(ConsolidationError):
            self.run_it(episode_id, FakeLLM("not json"))
        self.assertEqual([e.id for e in self.store.pending_consolidation(10)], [episode_id])

    def test_a_long_episode_is_condensed_part_by_part_first(self):
        class NotesThenJSON(FakeLLM):
            """Condensing calls (no schema) get numbered notes; extraction gets JSON."""

            def chat(self, messages, *, schema=None):
                response = super().chat(messages, schema=schema)
                if schema is None:
                    return replace(response, content=f"notes on part {len(self.prompts)}")
                return response

        long_texts = [f"message {i} " + "word " * 300 for i in range(12)]  # ~400 tokens each
        episode_id = self.episode(*long_texts)
        llm = NotesThenJSON(json.dumps(EXTRACTION))
        self.run_it(episode_id, llm)
        *condense_calls, final = llm.prompts
        self.assertGreaterEqual(len(condense_calls), 2)
        self.assertTrue(all(s is None for s in llm.schemas[:-1]))
        self.assertIs(llm.schemas[-1], EXTRACTION_SCHEMA)
        self.assertIn("Notes from consecutive parts", final[-1].content)
        self.assertIn("notes on part 1", final[-1].content)

    def test_entities_the_user_never_named_are_dropped(self):
        episode_id = self.episode(
            "I'm building an online bookshop in Santo Domingo.",
            "Make sure it follows DGII rules.",  # only Cerveau mentions DGII
        )
        extraction = {**EXTRACTION, "entities": [
            {"name": "Santo Domingo", "type": "place", "aliases": []},
            {"name": "DGII", "type": "organization", "aliases": []},
        ]}
        report = self.run_it(episode_id, FakeLLM(json.dumps(extraction)))
        assert report is not None
        self.assertEqual(report.entities, ["Santo Domingo"])
        self.assertEqual(report.unsupported, ["DGII"])

    def test_assistant_replies_are_shortened_in_the_transcript(self):
        episode_id = self.episode("short question", "x" * 5000)
        llm = FakeLLM(json.dumps(EXTRACTION))
        self.run_it(episode_id, llm)
        prompt = llm.prompts[0][-1].content
        self.assertIn("Cerveau: " + "x" * 600 + " [...]", prompt)
        self.assertNotIn("x" * 601, prompt)


if __name__ == "__main__":
    unittest.main()
