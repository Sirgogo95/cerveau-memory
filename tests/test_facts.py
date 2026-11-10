"""Learning facts: extraction (user lines only), deduplication (same, update,
distinct, forgotten), and consolidation's best-effort fact step. InMemoryStore, the
bag-of-words FakeEmbedder, and a scripted model."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EmbeddingProfile, EngineConfig, with_embedding
from brain.consolidation import ConsolidationError, consolidate_episode, with_language
from brain.facts import (
    FACT_SCHEMA,
    GROUP_FACT_SCHEMA,
    JUDGE_SCHEMA,
    FactSettings,
    extract_facts,
    fact_settings,
    judge,
    learn_facts,
    numbered,
    parse_facts,
)
from brain.types import Fact, MemoryKind, Role
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_consolidation import EXTRACTION
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
# The FakeEmbedder's similarities: identical texts score 1.0; texts sharing most
# words score high. Thresholds chosen to exercise each path.
SETTINGS = FactSettings(same_score=0.99, related_score=0.5)


def facts_reply(*facts):
    return json.dumps({"language": "English", "facts": [
        {"statement": s, "type": t, "lines": lines} for s, t, lines in facts
    ]})


def relation(name):
    return json.dumps({"relation": name})


class Memory:
    def __init__(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def episode(self, *messages, minutes=0):
        episode = self.store.open_episode(T0 + timedelta(minutes=minutes))
        stored = [
            self.store.add_message(episode.id, Role(role), text, T0 + timedelta(minutes=minutes + i))
            for i, (role, text) in enumerate(messages)
        ]
        self.store.close_episode(episode.id, T0 + timedelta(minutes=minutes + len(messages)))
        return episode.id, stored

    def learn(self, episode_id, llm, settings=SETTINGS):
        return learn_facts(
            episode_id, self.store.get_messages(episode_id), store=self.store, llm=llm,
            embedder=self.embedder, profile=PROFILE, settings=settings, now=T0,
        )

    def active(self):
        return [f for f in self.store.get_facts(range(1, 50)) if f.status == "A" and f.superseded_by is None]


class ParseFactsTests(unittest.TestCase):
    def setUp(self):
        memory = Memory()
        _, self.messages = memory.episode(
            ("user", "My sister Lucía lives in Montreal."),
            ("assistant", "You should visit her in winter."),
            ("user", "I'm allergic to penicillin."),
        )

    def test_lines_map_to_user_messages_only(self):
        reply = facts_reply(
            ("The user's sister Lucía lives in Montreal.", "person", [1]),
            ("The user should visit Lucía in winter.", "plan", [2]),       # Cerveau's line
            ("The user is allergic to penicillin.", "health", [3, 99]),  # 99 doesn't exist
        )
        facts, unsupported = parse_facts(reply, self.messages)
        self.assertEqual([(f.statement, f.message_ids) for f in facts], [
            ("The user's sister Lucía lives in Montreal.", (self.messages[0].id,)),
            ("The user is allergic to penicillin.", (self.messages[2].id,)),
        ])
        self.assertEqual(unsupported, ["The user should visit Lucía in winter."])

    def test_cleans_types_duplicates_and_limit(self):
        reply = facts_reply(("A fact.", "weird", [1]), ("a  FACT.", "person", [1]), ("B fact.", "health", [3]))
        facts, _ = parse_facts(reply, self.messages, limit=5)
        self.assertEqual([(f.statement, f.type) for f in facts], [("A fact.", "other"), ("B fact.", "health")])
        limited, _ = parse_facts(reply, self.messages, limit=1)
        self.assertEqual(len(limited), 1)

    def test_rejects_unusable_replies(self):
        from brain.consolidation import ConsolidationError
        with self.assertRaises(ConsolidationError):
            parse_facts("not json", self.messages)
        with self.assertRaises(ConsolidationError):
            parse_facts(json.dumps({"facts": "nope"}), self.messages)

    def test_the_user_becomes_the_speaker_when_several_people_talk(self):
        memory = Memory()
        _, shared = memory.episode(
            ("user", "Caroline: I went to a support group yesterday."),
            ("user", "Melanie: That's brave. I took my kids camping."),
            ("user", "Caroline: My dog loves the beach."),
        )
        reply = facts_reply(
            ("The user went to a support group.", "other", [1]),
            ("The user's dog loves the beach.", "other", [3]),
            ("The user and the user's friend talked.", "other", [1, 2]),   # two speakers: left alone
        )
        facts, _ = parse_facts(reply, shared)
        self.assertEqual([f.statement for f in facts], [
            "Caroline went to a support group.", "Caroline's dog loves the beach.",
            "The user and the user's friend talked."])
        # One person writing "Note: ..." is not a shared conversation.
        _, alone = memory.episode(("user", "Note: I'm allergic to penicillin."), ("user", "Note: really."))
        facts, _ = parse_facts(facts_reply(("The user is allergic to penicillin.", "health", [1])), alone)
        self.assertEqual(facts[0].statement, "The user is allergic to penicillin.")

    def test_numbered_lines_start_at_one(self):
        self.assertTrue(numbered(self.messages)[0].startswith("[1] User: My sister"))
        self.assertTrue(numbered(self.messages)[1].startswith("[2] Cerveau:"))


class ExtractTests(unittest.TestCase):
    def test_the_model_gets_numbered_lines_and_the_schema(self):
        memory = Memory()
        _, messages = memory.episode(("user", "My sister Lucía lives in Montreal."))
        llm = FakeLLM(facts_reply(("Lucía lives in Montreal.", "person", [1])))
        facts, _ = extract_facts(messages, llm, PROFILE)
        self.assertEqual(llm.schemas, [with_language(FACT_SCHEMA, "English")])
        self.assertIn("[1] User: My sister Lucía lives in Montreal.", llm.prompts[0][-1].content)
        self.assertEqual(len(facts), 1)

    def test_a_long_conversation_is_read_in_parts_with_global_line_numbers(self):
        memory = Memory()
        long = [("user", f"message {i} " + "words " * 300) for i in range(12)]
        _, messages = memory.episode(*long)
        small = replace(PROFILE, memory_budget=1500)
        llm = FakeLLM(facts_reply(("A fact.", "other", [12])))
        extract_facts(messages, llm, small)
        self.assertGreater(len(llm.prompts), 1)
        self.assertIn("[12] User: message 11", llm.prompts[-1][-1].content)

    def test_a_group_conversation_is_asked_speaker_by_speaker(self):
        memory = Memory()
        _, messages = memory.episode(("user", "Caroline: I adopted a dog named Max."),
                                     ("user", "Melanie: I start pottery classes in June."))
        reply = json.dumps({"people": [
            {"name": "Caroline", "facts": [{"statement": "Caroline adopted a dog named Max.", "lines": [1]}]},
            {"name": "Melanie", "facts": [{"statement": "Melanie starts pottery classes in June.", "lines": [2]}]}]})
        llm = FakeLLM(reply)
        facts, _ = extract_facts(messages, llm, PROFILE, group=True)
        self.assertEqual(llm.schemas, [GROUP_FACT_SCHEMA])
        self.assertIn("For each person", llm.prompts[0][-1].content)
        self.assertEqual([(f.statement, f.message_ids) for f in facts],
                         [("Caroline adopted a dog named Max.", (messages[0].id,)),
                          ("Melanie starts pottery classes in June.", (messages[1].id,))])
        self.assertEqual(extract_facts(messages, FakeLLM(facts_reply(("x.", "other", [1]))), PROFILE)[0][0].statement, "x.")
        with self.assertRaises(ConsolidationError):                       # unreadable, as before
            extract_facts(messages, FakeLLM("not json"), PROFILE, group=True)


class JudgeTests(unittest.TestCase):
    def test_relations_and_the_safe_default(self):
        old = Fact(1, "Lucía lives in Montreal.")
        for name in ("same", "update", "contradiction", "distinct"):
            llm = FakeLLM(relation(name))
            self.assertEqual(judge(old, "Lucía moved.", llm), name)
            self.assertEqual(llm.schemas, [JUDGE_SCHEMA])
        self.assertEqual(judge(old, "x", FakeLLM("not json")), "distinct")
        self.assertEqual(judge(old, "x", FakeLLM(relation("maybe"))), "distinct")


class LearnFactsTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.first, _ = self.memory.episode(("user", "lucia lives in montreal with her cat"))
        self.memory.learn(self.first, FakeLLM(facts_reply(("lucia lives in montreal", "person", [1]))))
        [self.old] = self.memory.active()

    def test_a_new_fact_is_added_with_its_sources(self):
        self.assertEqual(self.old.statement, "lucia lives in montreal")
        self.assertEqual(self.old.episode_id, self.first)
        self.assertEqual(len(self.old.source_message_ids), 1)

    def test_the_same_fact_is_confirmed_without_asking(self):
        later, _ = self.memory.episode(("user", "lucia lives in montreal"), minutes=100)
        llm = FakeLLM(facts_reply(("lucia lives in montreal", "person", [1])))
        report = self.memory.learn(later, llm)
        self.assertEqual(report.confirmed, ["lucia lives in montreal"])
        self.assertEqual(len(llm.prompts), 1)  # no judge call
        [fact] = self.memory.store.get_facts([self.old.id])
        self.assertEqual(fact.times_confirmed, 2)

    def test_a_close_fact_judged_an_update_replaces_the_old(self):
        later, _ = self.memory.episode(("user", "lucia lives in toronto now"), minutes=100)
        llm = FakeLLM(replies=[facts_reply(("lucia lives in toronto", "person", [1])), relation("update")])
        report = self.memory.learn(later, llm)
        self.assertEqual(report.updated, [("lucia lives in montreal", "lucia lives in toronto")])
        [current] = self.memory.active()
        self.assertEqual(current.statement, "lucia lives in toronto")
        self.assertEqual(self.memory.store.get_facts([self.old.id])[0].superseded_by, current.id)

    def test_a_close_fact_judged_distinct_is_added(self):
        later, _ = self.memory.episode(("user", "lucia lives near the river"), minutes=100)
        llm = FakeLLM(replies=[facts_reply(("lucia lives near the river", "place", [1])), relation("distinct")])
        report = self.memory.learn(later, llm)
        self.assertEqual(report.added, ["lucia lives near the river"])
        self.assertEqual(len(self.memory.active()), 2)

    def test_a_close_fact_judged_same_is_confirmed(self):
        later, _ = self.memory.episode(("user", "lucia montreal"), minutes=100)
        llm = FakeLLM(replies=[facts_reply(("lucia still lives in montreal", "person", [1])), relation("same")])
        report = self.memory.learn(later, llm)
        self.assertEqual(report.confirmed, ["lucia lives in montreal"])

    def test_a_forgotten_fact_is_never_relearned(self):
        self.memory.store.set_status(MemoryKind.FACT, self.old.id, "Z")
        later, _ = self.memory.episode(("user", "lucia lives in montreal"), minutes=100)
        report = self.memory.learn(later, FakeLLM(facts_reply(("lucia lives in montreal", "person", [1]))))
        self.assertEqual(report.skipped, ["lucia lives in montreal"])
        self.assertEqual(self.memory.active(), [])

    def test_unrelated_facts_are_added_without_asking(self):
        later, _ = self.memory.episode(("user", "penicillin allergy"), minutes=100)
        llm = FakeLLM(facts_reply(("the user is allergic to penicillin", "health", [1])))
        report = self.memory.learn(later, llm)
        self.assertEqual(report.added, ["the user is allergic to penicillin"])
        self.assertEqual(len(llm.prompts), 1)


class ConsolidationFactsTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.episode, _ = self.memory.episode(("user", "My sister Lucía just moved to Montreal."))

    def consolidate(self, llm, facts: FactSettings | None = SETTINGS):
        return consolidate_episode(
            self.episode, store=self.memory.store, llm=llm, embedder=self.memory.embedder,
            profile=PROFILE, now=T0, facts=facts,
        )

    def test_facts_are_learned_in_a_second_call(self):
        llm = FakeLLM(replies=[json.dumps(EXTRACTION), facts_reply(("Lucía lives in Montreal.", "person", [1]))])
        report = self.consolidate(llm)
        assert report is not None and report.facts is not None
        self.assertEqual(report.facts.added, ["Lucía lives in Montreal."])
        self.assertEqual(llm.schemas[1], with_language(FACT_SCHEMA, "English"))

    def test_a_failed_fact_step_keeps_the_rest(self):
        llm = FakeLLM(replies=[json.dumps(EXTRACTION), "not json"])
        report = self.consolidate(llm)
        assert report is not None
        self.assertIn("not JSON", report.fact_error)
        [episode] = self.memory.store.get_episodes([self.episode])
        self.assertTrue(episode.summary_short)
        self.assertIsNotNone(episode.consolidated_at)

    def test_without_settings_no_facts_are_learned(self):
        llm = FakeLLM(json.dumps(EXTRACTION))
        report = self.consolidate(llm, facts=None)
        assert report is not None
        self.assertIsNone(report.facts)
        self.assertEqual(len(llm.prompts), 1)


class SettingsTests(unittest.TestCase):
    def test_thresholds_come_from_the_embedding_profile(self):
        profile = EmbeddingProfile("m", "ollama", "m", 0.5, 512, fact_same_score=0.96, fact_related_score=0.6)
        config = with_embedding(EngineConfig(), profile)
        settings = fact_settings(config)
        assert settings is not None
        self.assertEqual((settings.same_score, settings.related_score), (0.96, 0.6))
        self.assertEqual(config.messages.min_score, 0.5)
        self.assertIsNone(fact_settings(replace(config, learn_facts=False)))
        more = fact_settings(replace(config, max_facts=16))
        assert more is not None
        self.assertEqual((settings.max_facts, more.max_facts), (8, 16))
        with self.assertRaises(ConfigError):
            EngineConfig(max_facts=0)
        tagged = with_embedding(EngineConfig(), EmbeddingProfile("m", "ollama", "m", 0.5, 512,
                                                                 tag_same_score=0.8, tag_match_score=0.45))
        self.assertEqual((tagged.tag_same_score, tagged.tag_match_score), (0.8, 0.45))



class FactsInUseTests(unittest.TestCase):
    """Facts in automatic recall, the prompt, and the memory tools."""

    def setUp(self):
        self.memory = Memory()
        self.episode, [self.said] = self.memory.episode(("user", "lucia lives in montreal now"))
        [vector] = self.memory.embedder.embed(["lucia lives in montreal"])
        self.old = self.memory.store.add_fact(
            "lucia lives in montreal", "person", episode_id=self.episode,
            message_ids=[self.said.id], at=T0, embedding=vector,
        )
        [vector] = self.memory.embedder.embed(["lucia lives in toronto"])
        self.new = self.memory.store.add_fact(
            "lucia lives in toronto", "person", episode_id=self.episode,
            message_ids=[self.said.id], at=T0 + timedelta(days=5), embedding=vector,
        )
        self.memory.store.supersede_fact(self.old.id, self.new.id)

    def context(self):
        from brain.tools import ToolContext
        config = replace(EngineConfig(timezone="UTC"), search_min_score=0.2)
        return ToolContext(store=self.memory.store, embedder=self.memory.embedder, config=config,
                           profile=PROFILE, now=T0 + timedelta(days=10), tz=UTC)

    def call(self, name, **arguments):
        from brain.tools import run_tool
        from brain.types import ToolCall
        return json.loads(run_tool(ToolCall(name, arguments), self.context()))

    def test_recall_brings_current_facts_into_the_prompt(self):
        from brain.prompt import memory_line
        from brain.retrieval import retrieve
        [query] = self.memory.embedder.embed(["where does lucia live"])
        config = replace(EngineConfig(), facts=replace(EngineConfig().facts, min_score=0.2))
        found = [m for m in retrieve(query, self.memory.store, config) if m.candidate.kind is MemoryKind.FACT]
        self.assertEqual([m.candidate.id for m in found], [self.new.id])   # not the replaced one
        self.assertIn(f"[F{self.new.id} · ", memory_line(found[0]))
        self.assertIn(" · fact] lucia lives in toronto", memory_line(found[0]))

    def test_search_finds_facts_and_their_history_on_request(self):
        plain = self.call("search_memory", query="lucia lives", kinds=["fact"])
        [hit] = plain["results"]
        self.assertEqual((hit["id"], hit["statement"]), (f"F{self.new.id}", "lucia lives in toronto"))
        self.assertNotIn("replaced", hit)
        history = self.call("search_memory", query="lucia lives", kinds=["fact"], include_history=True)
        self.assertEqual([r["statement"] for r in history["results"][0]["replaced"]], ["lucia lives in montreal"])

    def test_facts_are_searched_by_default(self):
        result = self.call("search_memory", query="lucia lives in toronto")
        self.assertIn("fact", {r["kind"] for r in result["results"]})

    def test_read_memory_opens_a_fact_with_sources_and_history(self):
        opened = self.call("read_memory", id=f"F{self.new.id}")
        self.assertEqual(opened["statement"], "lucia lives in toronto")
        self.assertEqual([s["text"] for s in opened["sources"]], ["lucia lives in montreal now"])
        self.assertEqual([h["statement"] for h in opened["history"]],
                         ["lucia lives in montreal", "lucia lives in toronto"])
        self.assertTrue(opened["history"][1]["current"])
        old = self.call("read_memory", id=f"f{self.old.id}")
        self.assertIn(f"replaced by F{self.new.id}", old["note"])

    def test_a_forgotten_fact_cannot_be_opened(self):
        self.memory.store.set_status(MemoryKind.FACT, self.new.id, "Z")
        self.assertIn("no memory", self.call("read_memory", id=f"F{self.new.id}")["error"])


if __name__ == "__main__":
    unittest.main()
