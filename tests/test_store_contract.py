"""The store contract: rules every MemoryStore implementation must follow.

The contract classes are written once and run against each implementation by
subclassing them with a make_store(). They are nested in `Contracts` so unittest
doesn't collect the abstract versions themselves. DjangoMemoryStore will run the
same classes (combined with django.test.TestCase for its database).
"""

import json
import unittest
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.hashing import text_hash
from brain.ports import ConversationStore, MemoryStore, TurnStore
from brain.types import CarriedMemory, Chunk, Embedding, MemoryKind, Role, ToolRun
from tests.fakes import FakeEmbedder

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


class Contracts:
    class Conversation(unittest.TestCase):
        """EpisodeStore and MessageStore."""

        def make_store(self) -> ConversationStore:
            raise NotImplementedError

        def setUp(self):
            self.store = self.make_store()

        # -- episodes ---------------------------------------------------------

        def test_no_current_episode_initially(self):
            self.assertIsNone(self.store.current_episode())

        def test_open_episode_returns_an_open_episode(self):
            episode = self.store.open_episode(T0)
            self.assertEqual(episode.started_at, T0)
            self.assertTrue(episode.is_open)
            self.assertIsNone(episode.last_message_at)

        def test_opened_episode_becomes_current(self):
            episode = self.store.open_episode(T0)
            self.assertEqual(self.store.current_episode(), episode)

        def test_episodes_get_distinct_ids(self):
            first = self.store.open_episode(T0)
            self.store.close_episode(first.id, T0)
            second = self.store.open_episode(at(5))
            self.assertNotEqual(first.id, second.id)

        def test_close_episode_sets_end_and_clears_current(self):
            episode = self.store.open_episode(T0)
            self.store.close_episode(episode.id, at(30))
            self.assertIsNone(self.store.current_episode())
            [closed] = self.store.get_episodes([episode.id])
            self.assertEqual(closed.ended_at, at(30))
            self.assertFalse(closed.is_open)

        def test_get_episodes_keeps_requested_order(self):
            ids = []
            for minutes in (0, 10, 20):
                episode = self.store.open_episode(at(minutes))
                self.store.close_episode(episode.id, at(minutes + 1))
                ids.append(episode.id)
            requested = [ids[2], ids[0], ids[1]]
            self.assertEqual([e.id for e in self.store.get_episodes(requested)], requested)

        def test_get_episodes_skips_unknown_ids(self):
            episode = self.store.open_episode(T0)
            self.assertEqual(
                [e.id for e in self.store.get_episodes([999_999, episode.id])], [episode.id]
            )

        def test_get_episodes_of_nothing_is_empty(self):
            self.assertEqual(self.store.get_episodes([]), [])

        def test_saved_summary_is_returned_with_the_episode(self):
            episode = self.store.open_episode(T0)
            vector = Embedding(vector=[0.1, 0.2], model="test-model", text_hash="h")
            self.store.save_summary(
                episode.id,
                title="Orders table",
                summary_short="Designed the table.",
                summary_long="Designed the orders table, linked to customers.",
                short_embedding=vector,
                long_embedding=vector,
            )
            [saved] = self.store.get_episodes([episode.id])
            self.assertEqual(saved.title, "Orders table")
            self.assertEqual(saved.summary_short, "Designed the table.")
            self.assertEqual(
                saved.summary_long, "Designed the orders table, linked to customers."
            )

        # -- messages ---------------------------------------------------------

        def test_add_message_returns_the_stored_message(self):
            episode = self.store.open_episode(T0)
            message = self.store.add_message(episode.id, Role.USER, "Hello", at(1))
            self.assertEqual(message.episode_id, episode.id)
            self.assertEqual(message.role, Role.USER)
            self.assertEqual(message.content, "Hello")
            self.assertEqual(message.created_at, at(1))

        def test_messages_get_distinct_ids(self):
            episode = self.store.open_episode(T0)
            first = self.store.add_message(episode.id, Role.USER, "a", at(1))
            second = self.store.add_message(episode.id, Role.ASSISTANT, "b", at(2))
            self.assertNotEqual(first.id, second.id)

        def test_add_message_advances_last_message_at(self):
            episode = self.store.open_episode(T0)
            self.store.add_message(episode.id, Role.USER, "a", at(1))
            self.store.add_message(episode.id, Role.ASSISTANT, "b", at(2))
            current = self.store.current_episode()
            assert current is not None
            self.assertEqual(current.last_message_at, at(2))
            [fetched] = self.store.get_episodes([episode.id])
            self.assertEqual(fetched.last_message_at, at(2))

        def test_add_message_accepts_an_embedding(self):
            episode = self.store.open_episode(T0)
            vector = Embedding(vector=[0.1, 0.2], model="test-model", text_hash="h")
            message = self.store.add_message(episode.id, Role.USER, "a", at(1), vector)
            self.assertEqual(message.content, "a")

        def test_get_messages_oldest_first(self):
            episode = self.store.open_episode(T0)
            for i, minutes in enumerate((1, 2, 3)):
                self.store.add_message(episode.id, Role.USER, f"m{i}", at(minutes))
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id)], ["m0", "m1", "m2"]
            )

        def test_get_messages_last_n_keeps_the_most_recent_oldest_first(self):
            episode = self.store.open_episode(T0)
            for i in range(5):
                self.store.add_message(episode.id, Role.USER, f"m{i}", at(i))
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id, last=2)],
                ["m3", "m4"],
            )

        def test_messages_keep_insertion_order_even_if_the_clock_goes_back(self):
            episode = self.store.open_episode(T0)
            self.store.add_message(episode.id, Role.USER, "first", at(10))
            self.store.add_message(episode.id, Role.ASSISTANT, "second", at(5))
            self.store.add_message(episode.id, Role.USER, "third", at(1))
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id)],
                ["first", "second", "third"],
            )
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id, last=2)],
                ["second", "third"],
            )

        def test_get_messages_after_an_id_keeps_only_later_ones(self):
            episode = self.store.open_episode(T0)
            ids = [self.store.add_message(episode.id, Role.USER, f"m{i}", at(i)).id
                   for i in range(5)]
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id, after=ids[1])],
                ["m2", "m3", "m4"],
            )
            self.assertEqual(
                [m.content for m in self.store.get_messages(episode.id, after=ids[1], last=2)],
                ["m3", "m4"],
            )

        def test_messages_can_be_pinned_and_unpinned(self):
            episode = self.store.open_episode(at(0))
            first = self.store.add_message(episode.id, Role.USER, "keep this", at(0))
            self.store.add_message(episode.id, Role.ASSISTANT, "ok", at(1))
            self.assertEqual(self.store.pinned_messages(episode.id), [])
            self.store.set_pinned(first.id, True)
            [pinned] = self.store.pinned_messages(episode.id)
            self.assertEqual((pinned.id, pinned.pinned), (first.id, True))
            self.assertTrue(self.store.get_messages(episode.id)[0].pinned)
            self.store.set_pinned(first.id, False)
            self.assertEqual(self.store.pinned_messages(episode.id), [])

        def test_drift_state_is_saved_with_the_episode(self):
            episode = self.store.open_episode(at(0))
            message = self.store.add_message(episode.id, Role.USER, "hello", at(0))
            self.store.save_drift(episode.id, 1, message.id)
            [saved] = self.store.get_episodes([episode.id])
            self.assertEqual((saved.drift_strikes, saved.drift_pending_id), (1, message.id))

        def test_messages_move_to_another_episode(self):
            old = self.store.open_episode(at(0))
            kept = self.store.add_message(old.id, Role.USER, "about the knee", at(0))
            moved = self.store.add_message(old.id, Role.USER, "about the trip", at(5))
            reply = self.store.add_message(old.id, Role.ASSISTANT, "nice trip", at(6))
            self.store.close_episode(old.id, at(0))
            new = self.store.open_episode(at(5))
            self.store.move_messages([moved.id, reply.id], new.id)
            self.assertEqual([m.id for m in self.store.get_messages(old.id)], [kept.id])
            self.assertEqual([m.id for m in self.store.get_messages(new.id)], [moved.id, reply.id])
            [after] = self.store.get_episodes([new.id])
            self.assertEqual(after.last_message_at, at(6))

        def test_saved_running_summary_comes_back_with_the_episode(self):
            episode = self.store.open_episode(T0)
            message = self.store.add_message(episode.id, Role.USER, "hello", at(1))
            self.store.save_running_summary(episode.id, "They said hello.", message.id)
            current = self.store.current_episode()
            assert current is not None
            self.assertEqual(
                (current.running_summary, current.summarized_until),
                ("They said hello.", message.id),
            )
            [fetched] = self.store.get_episodes([episode.id])
            self.assertEqual(fetched.running_summary, "They said hello.")

        def test_get_messages_last_zero_is_empty(self):
            episode = self.store.open_episode(T0)
            self.store.add_message(episode.id, Role.USER, "a", at(1))
            self.assertEqual(self.store.get_messages(episode.id, last=0), [])

        def test_get_messages_only_returns_that_episodes_messages(self):
            first = self.store.open_episode(T0)
            self.store.add_message(first.id, Role.USER, "first", at(1))
            self.store.close_episode(first.id, at(1))
            second = self.store.open_episode(at(200))
            self.store.add_message(second.id, Role.USER, "second", at(201))
            self.assertEqual([m.content for m in self.store.get_messages(first.id)], ["first"])
            self.assertEqual([m.content for m in self.store.get_messages(second.id)], ["second"])

        def test_get_message_by_id(self):
            episode = self.store.open_episode(at(0))
            message = self.store.add_message(episode.id, Role.USER, "hello", at(1))
            self.assertEqual(self.store.get_message(message.id), message)
            self.assertIsNone(self.store.get_message(message.id + 999))

    class Search(unittest.TestCase):
        """SearchStore (semantic) and EmbeddingCacheStore."""

        def make_store(self) -> TurnStore:
            raise NotImplementedError

        def setUp(self):
            self.store = self.make_store()
            self.embedder = FakeEmbedder()

        def vector(self, text: str, model: str = "fake-embedder") -> Embedding:
            return Embedding(self.embedder.vector(text), model, text_hash(text))

        def say(self, episode_id: int, text: str, minutes: int, role: Role = Role.USER):
            return self.store.add_message(episode_id, role, text, at(minutes), self.vector(text))

        def past_episode(self, minutes: int) -> int:
            episode = self.store.open_episode(at(minutes))
            self.store.close_episode(episode.id, at(minutes))
            return episode.id

        def search(self, text: str, limit: int = 10, **kwargs):
            return self.store.semantic_candidates(
                MemoryKind.MESSAGE, self.vector(text), limit, **kwargs
            )

        # -- messages ---------------------------------------------------------

        def test_stored_vectors_come_back_by_id_for_their_model(self):
            episode = self.past_episode(0)
            first = self.say(episode, "the build uses docker", 1)
            self.say(episode, "my sister lives in montreal", 2)
            vectors = self.store.stored_vectors(MemoryKind.MESSAGE, [first.id, 9999], "fake-embedder")
            self.assertEqual(list(vectors), [first.id])
            self.assertAlmostEqual(sum(a * b for a, b in zip(vectors[first.id], self.vector("the build uses docker").vector)),
                                   1.0, places=4)
            self.assertEqual(self.store.stored_vectors(MemoryKind.MESSAGE, [first.id], "another-model"), {})
            self.assertEqual(self.store.stored_vectors(MemoryKind.MESSAGE, [], "fake-embedder"), {})

        def test_closest_message_comes_first(self):
            episode = self.past_episode(0)
            self.say(episode, "the build uses docker", 1)
            close = self.say(episode, "my sister lives in montreal", 2)
            hits = self.search("my sister lives in montreal")
            self.assertEqual(hits[0].id, close.id)
            self.assertAlmostEqual(hits[0].semantic or 0.0, 1.0, places=4)
            self.assertGreater(hits[0].semantic or 0.0, hits[1].semantic or 0.0)

        def test_hits_carry_text_episode_date_and_role(self):
            episode = self.past_episode(0)
            self.say(episode, "noted, rex is a beagle", 3, Role.ASSISTANT)
            [hit] = self.search("rex is a beagle")
            self.assertEqual(hit.kind, MemoryKind.MESSAGE)
            self.assertEqual(hit.text, "noted, rex is a beagle")
            self.assertEqual(hit.episode_id, episode)
            self.assertEqual(hit.at, at(3))
            self.assertEqual(hit.role, Role.ASSISTANT)

        def test_time_window_keeps_messages_from_after_inclusive_to_before_exclusive(self):
            episode = self.past_episode(0)
            early = self.say(episode, "rex the beagle early", 10)
            middle = self.say(episode, "rex the beagle middle", 20)
            self.say(episode, "rex the beagle late", 30)
            hits = self.search("rex the beagle", after=at(10), before=at(30))
            self.assertEqual({h.id for h in hits}, {early.id, middle.id})
            self.assertEqual([h.id for h in self.search("rex the beagle", before=at(20))], [early.id])

        def test_time_window_applies_to_episodes_by_their_start(self):
            for minutes in (0, 100):
                episode = self.past_episode(minutes)
                vector = self.vector("trip to samana")
                self.store.save_summary(
                    episode, title="t", summary_short="trip to samana", summary_long="trip to samana",
                    short_embedding=vector, long_embedding=vector,
                )
            hits = self.store.semantic_candidates(
                MemoryKind.EPISODE, self.vector("trip to samana"), 10, after=at(50)
            )
            self.assertEqual([h.at for h in hits], [at(100)])

        def test_limit_caps_the_results(self):
            episode = self.past_episode(0)
            for i in range(5):
                self.say(episode, f"rex note {i}", i)
            self.assertEqual(len(self.search("rex note", limit=2)), 2)

        def test_excluded_episodes_are_skipped_before_the_limit(self):
            old, current = self.past_episode(0), self.past_episode(10)
            for i in range(3):
                self.say(current, f"rex the beagle {i}", 11 + i)
            kept = self.say(old, "rex the beagle again", 1)
            hits = self.search("rex the beagle", limit=1, exclude_episode_ids=[current])
            self.assertEqual([h.id for h in hits], [kept.id])

        def test_only_vectors_of_the_query_model_are_compared(self):
            episode = self.past_episode(0)
            self.store.add_message(
                episode, Role.USER, "rex", at(1), self.vector("rex", model="other-model")
            )
            self.assertEqual(self.search("rex"), [])

        def test_messages_without_embeddings_are_not_found(self):
            episode = self.past_episode(0)
            self.store.add_message(episode, Role.USER, "rex", at(1))
            self.assertEqual(self.search("rex"), [])

        # -- episodes ---------------------------------------------------------

        def test_episodes_match_on_the_closer_of_their_two_summaries(self):
            episode = self.past_episode(0)
            long_text = "designed the orders table for the bookshop"
            self.store.save_summary(
                episode, title="Bookshop", summary_short="Worked on the bookshop.",
                summary_long="Designed the orders table for the bookshop.",
                short_embedding=self.vector("worked on the bookshop"),
                long_embedding=self.vector(long_text),
            )
            [hit] = self.store.semantic_candidates(
                MemoryKind.EPISODE, self.vector("orders table design"), 5
            )
            self.assertEqual((hit.id, hit.text, hit.at), (episode, "Worked on the bookshop.", at(0)))
            expected = self.embedder.vector(long_text)
            query = self.embedder.vector("orders table design")
            closer = sum(a * b for a, b in zip(expected, query))
            self.assertAlmostEqual(hit.semantic or 0.0, closer, places=4)

        def test_episodes_without_summaries_are_not_found(self):
            self.past_episode(0)
            self.assertEqual(
                self.store.semantic_candidates(MemoryKind.EPISODE, self.vector("anything"), 5),
                [],
            )

        # -- embedding cache --------------------------------------------------

        def test_cache_returns_what_was_put(self):
            vector = self.vector("alpha")
            self.store.put_cached({"k1": vector})
            found = self.store.get_cached(["k1"], "fake-embedder")["k1"]
            self.assertEqual(
                [round(x, 5) for x in found.vector], [round(x, 5) for x in vector.vector]
            )

        def test_cache_misses_are_absent(self):
            self.store.put_cached({"k1": self.vector("alpha")})
            self.assertEqual(set(self.store.get_cached(["k1", "k2"], "fake-embedder")), {"k1"})

        def test_cache_is_keyed_by_model(self):
            self.store.put_cached({"k1": self.vector("alpha")})
            self.assertEqual(self.store.get_cached(["k1"], "other-model"), {})

        def test_putting_the_same_key_twice_is_harmless(self):
            self.store.put_cached({"k1": self.vector("alpha")})
            self.store.put_cached({"k1": self.vector("alpha")})
            self.assertEqual(set(self.store.get_cached(["k1"], "fake-embedder")), {"k1"})


    class Knowledge(unittest.TestCase):
        """TagStore, EntityStore, and the consolidation bookkeeping on EpisodeStore."""

        def make_store(self) -> MemoryStore:
            raise NotImplementedError

        def set_status(self, kind: MemoryKind, item_id: int, status: str) -> None:
            """Set a tag's or entity's status the way this store's admin would."""
            raise NotImplementedError

        def setUp(self):
            self.store = self.make_store()

        def closed_episode(self, minutes: int = 0) -> int:
            episode = self.store.open_episode(at(minutes))
            self.store.add_message(episode.id, Role.USER, "hello", at(minutes))
            self.store.close_episode(episode.id, at(minutes))
            return episode.id

        # -- tags -------------------------------------------------------------

        def test_upsert_tag_creates_then_returns_the_same_tag(self):
            first = self.store.upsert_tag("databases")
            again = self.store.upsert_tag("Databases ")
            assert first is not None and again is not None
            self.assertEqual(first.id, again.id)
            self.assertEqual(first.name, "databases")

        def test_list_tags_orders_by_use(self):
            rare, common = self.store.upsert_tag("travel"), self.store.upsert_tag("health")
            assert rare is not None and common is not None
            for minutes in (0, 10):
                self.store.tag_episode(self.closed_episode(minutes), common.id, 1.0)
            self.store.tag_episode(self.closed_episode(20), rare.id, 1.0)
            self.assertEqual([t.name for t in self.store.list_tags(10)], ["health", "travel"])
            self.assertEqual(len(self.store.list_tags(1)), 1)

        def test_tagging_twice_is_harmless(self):
            tag = self.store.upsert_tag("health")
            assert tag is not None
            episode = self.closed_episode()
            self.store.tag_episode(episode, tag.id, 1.0)
            self.store.tag_episode(episode, tag.id, 0.5)
            self.assertEqual([t.name for t in self.store.list_tags(10)], ["health"])

        def test_forgotten_tag_is_refused_and_hidden(self):
            tag = self.store.upsert_tag("health")
            assert tag is not None
            self.set_status(MemoryKind.TAG, tag.id, "Z")
            self.assertIsNone(self.store.upsert_tag("health"))
            self.assertEqual(self.store.list_tags(10), [])

        def test_set_aside_tag_comes_back(self):
            tag = self.store.upsert_tag("health")
            assert tag is not None
            self.set_status(MemoryKind.TAG, tag.id, "I")
            back = self.store.upsert_tag("health")
            assert back is not None
            self.assertEqual(back.id, tag.id)
            self.assertEqual([t.name for t in self.store.list_tags(10)], ["health"])

        def test_similar_tags_are_closest_first_with_their_status(self):
            def vec(*values: float) -> Embedding:
                return Embedding(list(values), "m", text_hash(str(values)))

            near = self.store.upsert_tag("programming", vec(1.0, 0.0, 0.0))
            far = self.store.upsert_tag("travel", vec(0.0, 1.0, 0.0))
            gone = self.store.upsert_tag("family", vec(0.6, 0.8, 0.0))
            self.store.upsert_tag("health")                      # no vector: never similar
            assert near is not None and far is not None and gone is not None
            self.set_status(MemoryKind.TAG, gone.id, "Z")
            hits = self.store.similar_tags(vec(0.9, 0.1, 0.0), 3)
            self.assertEqual([t.name for t, _ in hits], ["programming", "family", "travel"])
            self.assertEqual([t.status for t, _ in hits], ["A", "Z", "A"])
            self.assertGreater(hits[0][1], 0.99)
            self.assertEqual(len(self.store.similar_tags(vec(0.9, 0.1, 0.0), 1)), 1)
            other_model = Embedding([1.0, 0.0, 0.0], "other", "h")
            self.assertEqual(self.store.similar_tags(other_model, 3), [])

        def test_episode_tags_lists_active_tags_per_episode(self):
            kept, aside = self.store.upsert_tag("travel"), self.store.upsert_tag("health")
            assert kept is not None and aside is not None
            first, second, untagged = self.closed_episode(0), self.closed_episode(10), self.closed_episode(20)
            self.store.tag_episode(first, kept.id, 1.0)
            self.store.tag_episode(first, aside.id, 0.5)
            self.store.tag_episode(second, kept.id, 0.8)
            self.set_status(MemoryKind.TAG, aside.id, "I")
            self.assertEqual(self.store.episode_tags([first, second, untagged]),
                             {first: {kept.id: 1.0}, second: {kept.id: 0.8}})
            self.assertEqual(self.store.episode_tags([]), {})

        def test_set_aside_tags_are_not_similar(self):
            tag = self.store.upsert_tag("programming", Embedding([1.0, 0.0], "m", "h"))
            assert tag is not None
            self.set_status(MemoryKind.TAG, tag.id, "I")
            self.assertEqual(self.store.similar_tags(Embedding([1.0, 0.0], "m", "h"), 3), [])

        def test_moving_a_message_moves_its_tool_calls(self):
            old = self.store.open_episode(at(0))
            reply = self.store.add_message(old.id, Role.ASSISTANT, "found it", at(1))
            self.store.add_tool_calls(old.id, reply.id, [ToolRun("search_memory", {"query": "x"}, "{}", 3, at(1))])
            new = self.store.open_episode(at(1))
            self.store.move_messages([reply.id], new.id)
            self.assertEqual(self.store.get_tool_calls(old.id), [])
            self.assertEqual([r.message_id for r in self.store.get_tool_calls(new.id)], [reply.id])

        # -- the activation graph ---------------------------------------------

        def test_episode_nodes_and_node_episodes_list_tags_and_mentioned_entities(self):
            travel = self.store.upsert_tag("travel")
            lucia = self.store.upsert_entity("Lucía", "person")
            assert travel is not None and lucia is not None
            first, second = self.closed_episode(0), self.closed_episode(10)
            self.store.tag_episode(first, travel.id, 1.0)
            message = self.store.get_messages(second)[0]
            self.store.add_mention(lucia.id, message.id)
            t, e = (MemoryKind.TAG, travel.id), (MemoryKind.ENTITY, lucia.id)
            self.assertEqual(self.store.episode_nodes([first, second]), {first: {t}, second: {e}})
            self.assertEqual(self.store.node_episodes([t, e]), {t: {first}, e: {second}})
            self.set_status(MemoryKind.TAG, travel.id, "Z")
            self.assertEqual(self.store.episode_nodes([first]), {})
            self.assertEqual(self.store.node_episodes([t]), {})

        def test_activation_is_saved_and_replaced(self):
            self.assertEqual(self.store.load_activation(), {})
            node = (MemoryKind.TAG, 4)
            self.store.save_activation({node: (0.7, at(0)), (MemoryKind.ENTITY, 2): (0.3, at(0))})
            self.store.save_activation({node: (0.5, at(5))})
            [(saved, (value, when))] = self.store.load_activation().items()
            self.assertEqual((saved, when), (node, at(5)))
            self.assertAlmostEqual(value, 0.5)

        # -- attachments ----------------------------------------------------------

        def shared(self, name="lease.txt", digest="h1", minutes=0, text="the deposit is 60,000 pesos"):
            """A file shared in its own closed episode, read into two passages."""
            episode = self.store.open_episode(at(minutes))
            message = self.store.add_message(episode.id, Role.USER, "here is the lease", at(minutes))
            self.store.close_episode(episode.id, at(minutes))
            found = self.store.add_attachment(name, "text/plain", 120, digest, f"ab/{digest}.txt", at(minutes))
            assert found is not None
            attachment, _ = found
            self.store.link_attachment(message.id, attachment.id)
            return episode.id, message, attachment

        def read(self, attachment_id, texts, mentions=()):
            chunks = [(Chunk(i, t, page_start=i + 1, page_end=i + 1, heading="Terms"),
                       Embedding([1.0, float(i)], "m", text_hash(t))) for i, t in enumerate(texts)]
            self.store.save_chunks(attachment_id, chunks, list(mentions), page_count=len(texts))

        def test_the_same_file_is_stored_once_and_forgotten_files_stay_forgotten(self):
            first = self.store.add_attachment("a.txt", "text/plain", 10, "same", "x/same.txt", at(0))
            again = self.store.add_attachment("copy.txt", "text/plain", 10, "same", "x/same.txt", at(5))
            assert first is not None and again is not None
            self.assertEqual((first[1], again[1]), (True, False))
            self.assertEqual(first[0].id, again[0].id)
            self.assertEqual(first[0].extraction_status, "pending")
            self.set_status(MemoryKind.ATTACHMENT, first[0].id, "I")
            back = self.store.add_attachment("a.txt", "text/plain", 10, "same", "x/same.txt", at(9))
            assert back is not None
            self.assertEqual(back[0].status, "A")
            self.set_status(MemoryKind.ATTACHMENT, first[0].id, "Z")
            self.assertIsNone(self.store.add_attachment("a.txt", "text/plain", 10, "same", "x/same.txt", at(9)))
            self.assertIsNone(self.store.get_attachment(first[0].id))

        def test_a_link_records_when_and_where_a_file_was_first_shared(self):
            episode, message, attachment = self.shared(minutes=30)
            found = self.store.get_attachment(attachment.id)
            assert found is not None
            self.assertEqual((found.shared_at, found.episode_id), (at(30), episode))
            self.assertEqual(self.store.message_attachments([message.id, 999]), {message.id: [found]})
            self.assertEqual(self.store.pending_attachments(5), [found])

        def test_passages_are_searched_only_once_read_and_while_active(self):
            _, _, attachment = self.shared()
            query = Embedding([1.0, 0.0], "m", "q")
            self.assertEqual(self.store.semantic_candidates(MemoryKind.ATTACHMENT, query, 5), [])
            self.read(attachment.id, ["the deposit is 60,000 pesos", "notice: two months"])
            hits = self.store.semantic_candidates(MemoryKind.ATTACHMENT, query, 5)
            self.assertEqual([h.text for h in hits], ["the deposit is 60,000 pesos", "notice: two months"])
            self.assertEqual((hits[0].parent_id, hits[0].label), (attachment.id, "lease.txt, p.1"))
            self.assertEqual(hits[0].at, at(0))
            [kw] = self.store.keyword_candidates(MemoryKind.ATTACHMENT, ["60,000"], 5)
            self.assertEqual((kw.text, kw.keyword), ("the deposit is 60,000 pesos", 1.0))
            self.assertEqual(self.store.pending_attachments(5), [])
            [read] = self.store.unsummarized_attachments(5)
            self.assertEqual((read.extraction_status, read.page_count), ("done", 2))
            self.set_status(MemoryKind.ATTACHMENT, attachment.id, "I")
            self.assertEqual(self.store.semantic_candidates(MemoryKind.ATTACHMENT, query, 5), [])
            self.assertEqual(self.store.attachment_chunks(attachment.id), [])

        def test_reading_again_replaces_the_passages_and_mentions_find_their_passage(self):
            ana = self.store.upsert_entity("Ana", "person")
            assert ana is not None
            _, _, attachment = self.shared()
            self.read(attachment.id, ["old one", "old two", "old three"])
            self.read(attachment.id, ["Ana signs here", "nothing"], mentions=[(0, ana.id)])
            self.assertEqual([c.text for c in self.store.attachment_chunks(attachment.id)], ["Ana signs here", "nothing"])
            [hit] = self.store.mention_candidates(MemoryKind.ATTACHMENT, [ana.id], 5)
            self.assertEqual((hit.text, hit.lexical), ("Ana signs here", 1.0))

        def test_failures_summaries_and_tags_are_recorded(self):
            _, _, attachment = self.shared()
            self.store.mark_attachment(attachment.id, "failed", "No text found.")
            failed = self.store.get_attachment(attachment.id)
            assert failed is not None
            self.assertEqual((failed.extraction_status, failed.error), ("failed", "No text found."))
            self.read(attachment.id, ["text"])
            self.store.save_attachment_summary(attachment.id, "Lease for Naco", "A lease.", "A longer view.")
            tag = self.store.upsert_tag("housing")
            assert tag is not None
            self.store.tag_attachment(attachment.id, tag.id)
            self.store.tag_attachment(attachment.id, tag.id)          # twice is harmless
            done = self.store.get_attachment(attachment.id)
            assert done is not None
            self.assertEqual((done.title, done.summary_short), ("Lease for Naco", "A lease."))
            self.assertEqual(self.store.unsummarized_attachments(5), [])

        # -- carried memories -------------------------------------------------

        def test_carried_memories_are_replaced_per_episode(self):
            first, other = self.store.open_episode(at(0)), self.store.open_episode(at(500))
            self.assertEqual(self.store.carried_memories(first.id), [])
            self.store.save_carried_memories(first.id, [CarriedMemory(MemoryKind.MESSAGE, 7, 0.61),
                                                        CarriedMemory(MemoryKind.EPISODE, 3, 0.55)])
            self.store.save_carried_memories(first.id, [CarriedMemory(MemoryKind.FACT, 2, 0.58)])
            self.assertEqual(self.store.carried_memories(first.id), [CarriedMemory(MemoryKind.FACT, 2, 0.58)])
            self.assertEqual(self.store.carried_memories(other.id), [])

        # -- imported chats ---------------------------------------------------

        def test_imported_messages_are_recorded_by_the_exports_ids(self):
            episode = self.store.open_episode(at(0))
            first = self.store.add_message(episode.id, Role.USER, "hola", at(0), None)
            second = self.store.add_message(episode.id, Role.ASSISTANT, "hello", at(1), None)
            self.assertEqual(self.store.imported_conversations("chatgpt"), set())
            self.store.record_imports("chatgpt", "c1", {"m1": first.id})
            self.store.record_imports("chatgpt", "c1", {"m1": first.id, "m2": second.id})   # again is harmless
            self.assertEqual(self.store.imported_conversations("chatgpt"), {"c1"})
            self.assertEqual(self.store.imported_conversations("claude"), set())
            self.assertEqual(self.store.imported_messages("chatgpt", "c1"), {"m1": first.id, "m2": second.id})
            self.assertEqual(self.store.imported_messages("chatgpt", "other"), {})

        # -- entities ---------------------------------------------------------

        def test_upsert_entity_creates_with_aliases(self):
            entity = self.store.upsert_entity("Lucía", "person", ["my sister", "lucia"])
            assert entity is not None
            self.assertEqual((entity.name, entity.type), ("Lucía", "person"))
            self.assertEqual(entity.aliases, ("my sister",))  # "lucia" repeats the name

        def test_entity_matches_however_the_name_is_typed(self):
            first = self.store.upsert_entity("Lucía", "person")
            again = self.store.upsert_entity("LUCIA", "person")
            assert first is not None and again is not None
            self.assertEqual(first.id, again.id)

        def test_entity_matches_on_an_alias_and_learns_new_names(self):
            first = self.store.upsert_entity("Lucía", "person", ["my sister"])
            again = self.store.upsert_entity("Lu", "person", ["My Sister"])
            assert first is not None and again is not None
            self.assertEqual(first.id, again.id)
            self.assertEqual(again.aliases, ("my sister", "Lu"))

        def test_forgotten_entity_is_refused(self):
            entity = self.store.upsert_entity("Lucía", "person", ["my sister"])
            assert entity is not None
            self.set_status(MemoryKind.ENTITY, entity.id, "Z")
            self.assertIsNone(self.store.upsert_entity("my sister", "person"))

        def test_set_aside_entity_comes_back(self):
            entity = self.store.upsert_entity("Lucía", "person")
            assert entity is not None
            self.set_status(MemoryKind.ENTITY, entity.id, "I")
            back = self.store.upsert_entity("Lucia", "person")
            assert back is not None
            self.assertEqual(back.id, entity.id)

        def test_people_with_the_same_first_name_stay_separate(self):
            pena = self.store.upsert_entity("Marta Peña", "person")
            gomez = self.store.upsert_entity("Marta Gómez", "person")
            assert pena is not None and gomez is not None
            self.assertNotEqual(pena.id, gomez.id)

        def test_match_entities_by_name_alias_and_first_name(self):
            marta = self.store.upsert_entity("Dr. Marta Peña", "person", ["my colleague"])
            assert marta is not None
            for keys in ({"dr marta pena"}, {"my colleague"}, {"marta"}, {"pena"}):
                self.assertEqual([e.id for e in self.store.match_entities(keys)], [marta.id], keys)

        def test_places_are_not_matched_by_a_part(self):
            self.store.upsert_entity("Santo Domingo", "place")
            self.assertEqual(self.store.match_entities({"domingo"}), [])
            self.assertEqual(len(self.store.match_entities({"santo domingo"})), 1)

        def test_new_aliases_become_searchable(self):
            first = self.store.upsert_entity("Lucía", "person")
            self.store.upsert_entity("Lucia", "person", ["my sister"])
            assert first is not None
            self.assertEqual([e.id for e in self.store.match_entities({"my sister"})], [first.id])

        def test_forgotten_entities_are_never_matched(self):
            entity = self.store.upsert_entity("Lucía", "person")
            assert entity is not None
            self.set_status(MemoryKind.ENTITY, entity.id, "Z")
            self.assertEqual(self.store.match_entities({"lucia"}), [])

        def test_active_entities_lists_only_active_ones_oldest_first(self):
            ana = self.store.upsert_entity("Ana", "person")
            bob = self.store.upsert_entity("Bob", "person")
            gone = self.store.upsert_entity("Cleo", "person")
            assert ana is not None and bob is not None and gone is not None
            self.set_status(MemoryKind.ENTITY, gone.id, "Z")
            self.assertEqual([e.name for e in self.store.active_entities()], ["Ana", "Bob"])

        def test_match_entities_with_no_keys_is_empty(self):
            self.store.upsert_entity("Lucía", "person")
            self.assertEqual(self.store.match_entities(set()), [])

        def mentioned(self, minutes: int, text: str, *entity_ids: int, summary: str = ""):
            """A closed episode with one message mentioning the given entities."""
            episode = self.store.open_episode(at(minutes))
            message = self.store.add_message(episode.id, Role.USER, text, at(minutes))
            for entity_id in entity_ids:
                self.store.add_mention(entity_id, message.id)
            self.store.close_episode(episode.id, at(minutes))
            if summary:
                vector = Embedding([1.0, 0.0], "m", "h")
                self.store.save_summary(
                    episode.id, title="t", summary_short=summary, summary_long=summary,
                    short_embedding=vector, long_embedding=vector,
                )
            return episode.id, message

        def test_messages_mentioning_more_of_the_named_come_first(self):
            ana = self.store.upsert_entity("Ana", "person")
            bob = self.store.upsert_entity("Bob", "person")
            assert ana is not None and bob is not None
            _, one = self.mentioned(0, "Ana called", ana.id)
            _, both = self.mentioned(10, "Ana and Bob came", ana.id, bob.id)
            self.mentioned(20, "unrelated")
            hits = self.store.mention_candidates(MemoryKind.MESSAGE, [ana.id, bob.id], 10)
            self.assertEqual([h.id for h in hits], [both.id, one.id])
            self.assertEqual([h.lexical for h in hits], [1.0, 0.5])
            self.assertEqual((hits[0].text, hits[0].role), ("Ana and Bob came", Role.USER))

        def test_mention_candidates_respect_exclusions_and_limit(self):
            ana = self.store.upsert_entity("Ana", "person")
            assert ana is not None
            excluded, _ = self.mentioned(0, "Ana one", ana.id)
            _, kept = self.mentioned(10, "Ana two", ana.id)
            self.mentioned(20, "Ana three", ana.id)
            hits = self.store.mention_candidates(
                MemoryKind.MESSAGE, [ana.id], 5, exclude_episode_ids=[excluded]
            )
            self.assertNotIn(excluded, [h.episode_id for h in hits])
            self.assertEqual(len(self.store.mention_candidates(MemoryKind.MESSAGE, [ana.id], 1)), 1)
            self.assertIn(kept.id, [h.id for h in hits])

        def test_keyword_candidates_hold_the_terms_literally_case_ignored(self):
            first, a = self.mentioned(0, "The job failed on INV-2026-00417 again")
            _, b = self.mentioned(10, "inv-2026-00417 and invoice_number_seq are fine now")
            self.mentioned(20, "nothing to see here")
            hits = self.store.keyword_candidates(MemoryKind.MESSAGE, ["INV-2026-00417", "invoice_number_seq"], 5)
            self.assertEqual([(h.id, h.keyword) for h in hits], [(b.id, 1.0), (a.id, 0.5)])
            self.assertEqual(hits[1].role, Role.USER)
            excluded = self.store.keyword_candidates(MemoryKind.MESSAGE, ["INV-2026-00417"], 5,
                                                     exclude_episode_ids=[first])
            self.assertEqual([h.id for h in excluded], [b.id])
            self.assertEqual(self.store.keyword_candidates(MemoryKind.MESSAGE, [], 5), [])
            later = self.store.keyword_candidates(MemoryKind.MESSAGE, ["INV-2026-00417"], 5, after=at(5))
            self.assertEqual([h.id for h in later], [b.id])

        def test_keyword_candidates_search_summaries_and_facts(self):
            episode, message = self.mentioned(0, "the fix", summary="Fixed by moving the key to migration 0004.")
            [hit] = self.store.keyword_candidates(MemoryKind.EPISODE, ["0004"], 5)
            self.assertEqual((hit.id, hit.keyword), (episode, 1.0))
            vector = Embedding([1.0, 0.0], "m", "h")
            fact = self.store.add_fact("The fix lives in migration 0004.", "work", episode_id=episode,
                                       message_ids=[message.id], at=at(1), embedding=vector)
            [found] = self.store.keyword_candidates(MemoryKind.FACT, ["0004"], 5)
            self.assertEqual((found.id, found.keyword), (fact.id, 1.0))

        def test_mention_candidates_respect_the_time_window(self):
            ana = self.store.upsert_entity("Ana", "person")
            assert ana is not None
            self.mentioned(0, "Ana one", ana.id, summary="Ana one.")
            later, message = self.mentioned(100, "Ana two", ana.id, summary="Ana two.")
            messages = self.store.mention_candidates(MemoryKind.MESSAGE, [ana.id], 5, after=at(50))
            episodes = self.store.mention_candidates(MemoryKind.EPISODE, [ana.id], 5, before=at(50))
            self.assertEqual([h.id for h in messages], [message.id])
            self.assertNotIn(later, [h.id for h in episodes])
            self.assertEqual(len(episodes), 1)

        # -- facts ---------------------------------------------------------------

        def fact(self, statement: str, minutes: int = 0, episode: int | None = None, messages=()):
            vector = Embedding(FakeEmbedder().vector(statement), "fake-embedder", text_hash(statement))
            return self.store.add_fact(
                statement, "person", episode_id=episode, message_ids=list(messages),
                at=at(minutes), embedding=vector,
            )

        def fact_query(self, text: str) -> Embedding:
            return Embedding(FakeEmbedder().vector(text), "fake-embedder", text_hash(text))

        def test_a_fact_is_stored_with_its_sources(self):
            episode, message = self.mentioned(0, "my sister lucia lives in montreal")
            fact = self.fact(" Lucía lives in Montreal. ", 1, episode, [message.id, message.id])
            [stored] = self.store.get_facts([fact.id])
            self.assertEqual(stored.statement, "Lucía lives in Montreal.")
            self.assertEqual((stored.type, stored.episode_id, stored.status), ("person", episode, "A"))
            self.assertEqual(stored.source_message_ids, (message.id,))
            self.assertEqual((stored.times_confirmed, stored.learned_at), (1, at(1)))
            self.assertIsNone(stored.superseded_by)

        def test_confirming_counts_and_adds_sources(self):
            _, first = self.mentioned(0, "lucia montreal")
            _, again = self.mentioned(10, "lucia montreal again")
            fact = self.fact("Lucía lives in Montreal.", 1, messages=[first.id])
            self.store.confirm_fact(fact.id, at(11), [again.id])
            [stored] = self.store.get_facts([fact.id])
            self.assertEqual((stored.times_confirmed, stored.last_confirmed_at), (2, at(11)))
            self.assertEqual(set(stored.source_message_ids), {first.id, again.id})

        def test_a_superseded_fact_is_history_not_memory(self):
            old = self.fact("lucia lives in montreal", 0)
            new = self.fact("lucia lives in toronto", 10)
            self.store.supersede_fact(old.id, new.id)
            hits = self.store.semantic_candidates(MemoryKind.FACT, self.fact_query("lucia lives"), 10)
            self.assertEqual([h.id for h in hits], [new.id])
            self.assertEqual(hits[0].kind, MemoryKind.FACT)
            self.assertEqual([f.id for f in self.store.fact_history(old.id)], [old.id, new.id])
            self.assertEqual([f.id for f in self.store.fact_history(new.id)], [old.id, new.id])
            self.assertEqual(self.store.get_facts([old.id])[0].superseded_by, new.id)

        def test_similar_facts_include_forgotten_but_not_set_aside_or_superseded(self):
            kept = self.fact("lucia lives in montreal")
            forgotten = self.fact("lucia lives in montreal now")
            aside = self.fact("lucia lives in montreal still")
            old = self.fact("lucia lives in montreal before")
            self.store.supersede_fact(old.id, kept.id)
            self.set_status(MemoryKind.FACT, forgotten.id, "Z")
            self.set_status(MemoryKind.FACT, aside.id, "I")
            found = self.store.similar_facts(self.fact_query("lucia lives in montreal"), 10)
            self.assertEqual({f.id: f.status for f, _ in found}, {kept.id: "A", forgotten.id: "Z"})
            self.assertAlmostEqual(found[0][1], 1.0, places=4)

        def test_fact_search_sees_only_active_facts_in_the_window(self):
            episode = self.closed_episode(0)
            inside = self.fact("lucia montreal", 10, episode)
            self.fact("lucia montreal later", 100)
            gone = self.fact("lucia montreal gone", 10)
            self.set_status(MemoryKind.FACT, gone.id, "Z")
            query = self.fact_query("lucia montreal")
            window = self.store.semantic_candidates(MemoryKind.FACT, query, 10, after=at(5), before=at(50))
            self.assertEqual([h.id for h in window], [inside.id])
            self.assertEqual(window[0].episode_id, episode)
            excluded = self.store.semantic_candidates(MemoryKind.FACT, query, 10, exclude_episode_ids=[episode])
            self.assertNotIn(inside.id, [h.id for h in excluded])

        def test_facts_are_found_by_the_entities_their_sources_mention(self):
            lucia = self.store.upsert_entity("Lucía", "person")
            assert lucia is not None
            _, message = self.mentioned(0, "Lucía moved", lucia.id)
            fact = self.fact("Lucía lives in Montreal.", 1, messages=[message.id])
            self.fact("Unrelated fact.", 2)
            hits = self.store.mention_candidates(MemoryKind.FACT, [lucia.id], 10)
            self.assertEqual([(h.id, h.lexical) for h in hits], [(fact.id, 1.0)])

        def test_an_entitys_linked_facts_are_its_block(self):
            melanie = self.store.upsert_entity("Melanie", "person")
            caroline = self.store.upsert_entity("Caroline", "person")
            assert melanie is not None and caroline is not None
            camping = self.fact("Melanie went camping", 10)
            paints = self.fact("Melanie paints sunsets", 20)
            theirs = self.fact("Caroline paints too", 20)
            old = self.fact("Melanie paints portraits", 5)
            gone = self.fact("Melanie paints murals", 15)
            for fact in (camping, paints, old, gone):
                self.store.link_fact_entities(fact.id, [melanie.id])
            self.store.link_fact_entities(paints.id, [melanie.id, 999])      # twice, and an unknown entity
            self.store.link_fact_entities(theirs.id, [caroline.id])
            self.store.supersede_fact(old.id, paints.id)
            self.set_status(MemoryKind.FACT, gone.id, "Z")
            query = self.fact_query("melanie paints sunsets")
            hits = self.store.entity_fact_candidates([melanie.id], query, 10)
            self.assertEqual([h.id for h in hits], [paints.id, camping.id])   # closest first; active, current only
            self.assertEqual(hits[0].kind, MemoryKind.FACT)
            self.assertEqual([h.id for h in self.store.entity_fact_candidates([melanie.id], query, 1)], [paints.id])
            window = self.store.entity_fact_candidates([melanie.id], query, 10, after=at(0), before=at(15))
            self.assertEqual([h.id for h in window], [camping.id])
            both = self.store.entity_fact_candidates([melanie.id, caroline.id], query, 10)
            self.assertEqual(len(both), len({h.id for h in both}))             # a fact once, however many links
            self.set_status(MemoryKind.ENTITY, melanie.id, "Z")
            self.assertEqual(self.store.entity_fact_candidates([melanie.id], query, 10), [])

        def test_tool_calls_are_kept_per_episode_with_their_reply(self):
            episode = self.store.open_episode(at(0))
            other = self.store.open_episode(at(500))
            reply = self.store.add_message(episode.id, Role.ASSISTANT, "It moved.", at(2))
            search = ToolRun("search_memory", {"query": "portal"}, '{"results": []}', 120, at(1))
            read = ToolRun("read_memory", {"id": "E3"}, '{"id": "E3"}', 15, at(1))
            self.store.add_tool_calls(episode.id, reply.id, [search, read])
            self.store.add_tool_calls(other.id, None, [search])

            runs = self.store.get_tool_calls(episode.id)
            self.assertEqual([r.name for r in runs], ["search_memory", "read_memory"])
            self.assertEqual(runs[0].arguments, {"query": "portal"})
            self.assertEqual(json.loads(runs[0].result), {"results": []})
            self.assertEqual((runs[0].duration_ms, runs[0].at), (120, at(1)))
            self.assertEqual({r.message_id for r in runs}, {reply.id})
            [orphan] = self.store.get_tool_calls(other.id)
            self.assertIsNone(orphan.message_id)

        def test_episodes_mentioning_an_entity_need_a_summary(self):
            ana = self.store.upsert_entity("Ana", "person")
            assert ana is not None
            summarized, _ = self.mentioned(0, "Ana called", ana.id, summary="Ana called about the trip.")
            self.mentioned(10, "Ana again", ana.id)  # no summary yet
            hits = self.store.mention_candidates(MemoryKind.EPISODE, [ana.id], 5)
            self.assertEqual([(h.id, h.text, h.lexical) for h in hits],
                             [(summarized, "Ana called about the trip.", 1.0)])

        def test_no_entities_means_no_mention_candidates(self):
            self.assertEqual(self.store.mention_candidates(MemoryKind.MESSAGE, [], 5), [])

        def test_recording_a_mention_twice_is_harmless(self):
            entity = self.store.upsert_entity("Lucía", "person")
            assert entity is not None
            episode = self.store.open_episode(T0)
            message = self.store.add_message(episode.id, Role.USER, "Lucía called", at(1))
            self.store.add_mention(entity.id, message.id)
            self.store.add_mention(entity.id, message.id)  # no error

        # -- consolidation bookkeeping ---------------------------------------

        def test_pending_lists_closed_unconsolidated_episodes_oldest_first(self):
            later, earlier = self.closed_episode(30), self.closed_episode(10)
            self.store.open_episode(at(50))  # still open: not pending
            self.assertEqual([e.id for e in self.store.pending_consolidation(10)], [earlier, later])

        def test_recent_episodes_are_closed_ones_newest_first(self):
            oldest, middle, newest = (self.closed_episode(m) for m in (0, 10, 20))
            self.store.open_episode(at(30))  # open: never "recent"
            self.assertEqual(
                [e.id for e in self.store.recent_episodes(10)], [newest, middle, oldest]
            )
            self.assertEqual([e.id for e in self.store.recent_episodes(2)], [newest, middle])

        def test_recent_episodes_skip_excluded_ids(self):
            first, second = self.closed_episode(0), self.closed_episode(10)
            self.assertEqual(
                [e.id for e in self.store.recent_episodes(10, exclude_ids=[second])], [first]
            )

        def test_marking_consolidated_removes_it_from_pending(self):
            episode = self.closed_episode()
            self.store.mark_consolidated(episode, at(100))
            self.assertEqual(self.store.pending_consolidation(10), [])
            [found] = self.store.get_episodes([episode])
            self.assertEqual(found.consolidated_at, at(100))


class InMemoryConversationTests(Contracts.Conversation):
    def make_store(self) -> ConversationStore:
        return InMemoryStore()


class InMemorySearchTests(Contracts.Search):
    def make_store(self) -> TurnStore:
        return InMemoryStore()


if __name__ == "__main__":
    unittest.main()


class InMemoryKnowledgeTests(Contracts.Knowledge):
    def make_store(self) -> MemoryStore:
        return InMemoryStore()

    def set_status(self, kind: MemoryKind, item_id: int, status: str) -> None:
        store = self.store
        assert isinstance(store, InMemoryStore)
        store.set_status(kind, item_id, status)
