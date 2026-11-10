"""brain.recall: what a turn recalls, the conversation-aware search vector, and
carry-over between turns."""

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import ConfigError, EngineConfig, SourceConfig
from brain.embedding import embed_cached
from brain.recall import conversation_query, cosine, previous_messages, recall
from brain.types import CarriedMemory, MemoryKind, Role
from tests.fakes import FakeEmbedder

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
CONFIG = EngineConfig(timezone="UTC", recent_episodes=0, messages=SourceConfig(min_score=0.3))
KNEE = "my knee hurts after basketball, no running for six weeks"


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        past = self.store.open_episode(T0)
        self.knee = self.say(past.id, Role.USER, KNEE, 0)
        self.store.close_episode(past.id, T0)
        self.now = T0 + timedelta(days=30)
        self.episode = self.store.open_episode(self.now)
        self.minutes = 0

    def say(self, episode_id, role, text, minutes):
        [vector] = embed_cached([text], self.embedder, self.store)
        return self.store.add_message(episode_id, role, text, T0 + timedelta(minutes=minutes), vector)

    def turn(self, text, config):
        self.minutes += 2
        message = self.say(self.episode.id, Role.USER, text, 1000 + self.minutes)
        [query] = embed_cached([text], self.embedder, self.store)
        recalled = recall(text, query, self.store, config, embedder=self.embedder,
                          episode_id=self.episode.id, now=self.now)
        return message, [m.candidate.text for m in recalled.memories]


class ConversationQueryTests(RecallTests):
    def test_off_or_alone_the_message_is_searched_as_is(self):
        [query] = self.embedder.embed(["my knee hurts again"])
        self.assertIs(conversation_query(query, [], 0.5), query)
        self.assertIs(conversation_query(query, self.embedder.embed(["knee"]), 0), query)

    def test_previous_messages_are_the_users_before_the_newest(self):
        self.say(self.episode.id, Role.USER, "first", 1001)
        self.say(self.episode.id, Role.ASSISTANT, "a reply from cerveau", 1002)
        self.say(self.episode.id, Role.USER, "second", 1003)
        self.say(self.episode.id, Role.USER, "newest", 1004)
        self.assertEqual(previous_messages(self.store, self.episode.id, 5), ["first", "second"])
        self.assertEqual(previous_messages(self.store, self.episode.id, 1), ["second"])
        self.assertEqual(previous_messages(self.store, self.episode.id, 0), [])

    def test_a_follow_up_keeps_the_subject_of_the_message_before(self):
        on = replace(CONFIG, context_weight=0.5)
        self.turn("my knee hurts again", on)
        self.say(self.episode.id, Role.ASSISTANT, "sorry to hear that, rest and ice help a lot here", 1100)
        _, with_context = self.turn("can I still play on saturday", on)
        self.assertIn(KNEE, with_context)
        _, alone = self.turn("can I still play on saturday", CONFIG)
        self.assertEqual(alone, [])
        [query, knee] = self.embedder.embed(["can I still play on saturday", "my knee hurts again"])
        combined = conversation_query(query, [knee], 0.5)
        self.assertGreater(cosine(combined.vector, knee.vector), cosine(query.vector, knee.vector))


class CarryOverTests(RecallTests):
    CARRY = replace(CONFIG, carry_decay=0.9)

    def test_what_a_turn_recalled_is_saved_and_carried_into_a_vague_follow_up(self):
        self.turn("my knee hurts again after basketball", self.CARRY)
        [saved] = self.store.carried_memories(self.episode.id)
        self.assertEqual((saved.kind, saved.id), (MemoryKind.MESSAGE, self.knee.id))
        _, follow_up = self.turn("so what now", self.CARRY)
        self.assertEqual(follow_up, [KNEE])
        [faded] = self.store.carried_memories(self.episode.id)
        self.assertAlmostEqual(faded.score, saved.score * 0.9, places=6)

    def test_carried_memories_fade_out(self):
        self.store.save_carried_memories(self.episode.id, [CarriedMemory(MemoryKind.MESSAGE, self.knee.id, 0.35)])
        self.say(self.episode.id, Role.USER, "earlier", 1001)
        _, kept = self.turn("so what now", self.CARRY)        # 0.35 x 0.9 = 0.315, above 0.3
        self.assertEqual(kept, [KNEE])
        _, gone = self.turn("so what now", self.CARRY)        # 0.28: below the threshold
        self.assertEqual(gone, [])

    def test_a_message_with_its_own_subject_or_unlike_the_last_drops_what_was_carried(self):
        self.store.save_carried_memories(self.episode.id, [CarriedMemory(MemoryKind.MESSAGE, self.knee.id, 0.9)])
        self.say(self.episode.id, Role.USER, "so what now", 1001)
        _, unlike = self.turn("banana bread recipe", replace(self.CARRY, subject_similarity=0.5))
        self.assertEqual(unlike, [])
        self.store.save_carried_memories(self.episode.id, [CarriedMemory(MemoryKind.MESSAGE, self.knee.id, 0.9)])
        _, own = self.turn("banana bread recipe", replace(self.CARRY, carry_gate=0.0))  # found nothing, but gate 0
        self.assertEqual(own, [])

    def test_a_forgotten_memory_is_never_carried(self):
        [vector] = self.embedder.embed(["The user sprained a knee."])
        fact = self.store.add_fact("The user sprained a knee.", "health", episode_id=self.knee.episode_id,
                                   message_ids=[self.knee.id], at=T0, embedding=vector)
        carried = [CarriedMemory(MemoryKind.FACT, fact.id, 0.9)]
        self.say(self.episode.id, Role.USER, "earlier", 1001)
        self.store.save_carried_memories(self.episode.id, carried)
        _, kept = self.turn("so what now", self.CARRY)
        self.assertEqual(kept, ["The user sprained a knee."])
        self.store.save_carried_memories(self.episode.id, carried)
        self.store.set_status(MemoryKind.FACT, fact.id, "Z")
        _, forgotten = self.turn("so what now", self.CARRY)
        self.assertEqual(forgotten, [])

    def test_carry_off_saves_nothing(self):
        self.turn("my knee hurts again after basketball", CONFIG)
        self.assertEqual(self.store.carried_memories(self.episode.id), [])


class ActivationInRecallTests(RecallTests):
    def setUp(self):
        super().setUp()
        sports = self.store.upsert_tag("basketball", self.embedder.embed(["basketball"])[0])
        assert sports is not None
        self.sports = (MemoryKind.TAG, sports.id)
        self.store.tag_episode(self.knee.episode_id, sports.id, 1.0)
        self.on = replace(CONFIG, tag_match_score=0.5, subject_similarity=0.5,
                          messages=replace(CONFIG.messages, w_act=0.2))

    def test_a_matched_topic_comes_to_mind_and_boosts_its_memories(self):
        _, recalled = self.turn("basketball", self.on)
        self.assertGreater(self.store.load_activation()[self.sports][0], 0.5)
        self.assertIn(KNEE, recalled)

    def test_a_message_unlike_the_one_before_starts_fresh(self):
        self.turn("basketball", self.on)
        self.turn("banana bread recipe", self.on)
        self.assertNotIn(self.sports, self.store.load_activation())

    def test_no_weight_no_activation(self):
        self.turn("basketball", CONFIG)
        self.assertEqual(self.store.load_activation(), {})


class SettingsTests(unittest.TestCase):
    def test_settings_are_validated(self):
        for bad in (dict(context_weight=1.5), dict(context_messages=-1), dict(carry_decay=1.0),
                    dict(carry_gate=2.0), dict(subject_similarity=-0.1),
                    dict(activation_turn_decay=1.5), dict(activation_half_life_hours=0)):
            with self.assertRaises(ConfigError):
                replace(EngineConfig(), **bad)


if __name__ == "__main__":
    unittest.main()
