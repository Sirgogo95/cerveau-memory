"""Engine tests: segmentation, embedding through the cache, retrieval, prompt assembly,
and full turns, on InMemoryStore with fakes (no database, no Ollama, no model
downloads)."""

import sys
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig, ModelProfile, SourceConfig
from brain.embedding import embed_cached
from brain.hashing import text_hash
from brain.ports import ChatProvider, ChatProviderError
from brain.prompt import (
    MEMORY_HEADER,
    RECENT_HEADER,
    SYSTEM_PROMPT,
    assemble_prompt,
    estimate_tokens,
    fit_recent,
    memory_line,
    system_message,
)
from brain.retrieval import (
    RecentEpisode,
    ScoredCandidate,
    entity_card,
    matched_tags,
    merge,
    recent_context,
    retrieve,
)
from brain.segmentation import close_stale_episode, ensure_episode
from brain.turn import run_turn
from brain.types import Candidate, ChatMessage, Entity, Episode, MemoryKind, Message, Role
from tests.fakes import FakeEmbedder, FakeLLM

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
CONFIG = EngineConfig(episode_gap=timedelta(hours=2), recent_window=20, timezone="UTC")
SANTO_DOMINGO = ZoneInfo("America/Santo_Domingo")
PROFILE = ModelProfile(
    name="fake", provider="fake", context_window=4096, output_reserve=512, memory_budget=2000
)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def message(i: int, content: str, role: Role = Role.USER) -> Message:
    return Message(id=i, episode_id=1, role=role, content=content, created_at=at(i))


def scored(i: int, text: str, score: float, role: Role = Role.USER) -> ScoredCandidate:
    candidate = Candidate(
        kind=MemoryKind.MESSAGE, id=i, text=text, semantic=score, episode_id=9,
        at=T0, role=role,
    )
    return ScoredCandidate(candidate, score)


class EnsureEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()

    def test_first_message_opens_an_episode(self):
        episode = ensure_episode(self.store, T0, CONFIG)
        self.assertEqual(self.store.current_episode(), episode)

    def test_message_within_the_gap_stays_in_the_episode(self):
        first = ensure_episode(self.store, T0, CONFIG)
        self.store.add_message(first.id, Role.USER, "a", T0)
        self.assertEqual(ensure_episode(self.store, at(90), CONFIG).id, first.id)

    def test_message_after_the_gap_opens_a_new_episode(self):
        first = ensure_episode(self.store, T0, CONFIG)
        self.store.add_message(first.id, Role.USER, "a", T0)
        second = ensure_episode(self.store, at(121), CONFIG)
        self.assertNotEqual(second.id, first.id)

    def test_closed_episode_ends_at_its_last_message(self):
        first = ensure_episode(self.store, T0, CONFIG)
        self.store.add_message(first.id, Role.USER, "a", at(10))
        ensure_episode(self.store, at(300), CONFIG)
        [closed] = self.store.get_episodes([first.id])
        self.assertEqual(closed.ended_at, at(10))

    def test_gap_counts_from_the_last_message_not_the_start(self):
        first = ensure_episode(self.store, T0, CONFIG)
        self.store.add_message(first.id, Role.USER, "a", at(100))
        self.assertEqual(ensure_episode(self.store, at(200), CONFIG).id, first.id)

    def test_empty_episode_measures_the_gap_from_its_start(self):
        first = ensure_episode(self.store, T0, CONFIG)
        self.assertNotEqual(ensure_episode(self.store, at(121), CONFIG).id, first.id)

    def test_closing_is_announced_with_the_episode_id(self):
        closed: list[int] = []
        first = ensure_episode(self.store, T0, CONFIG, closed.append)
        ensure_episode(self.store, at(60), CONFIG, closed.append)
        self.assertEqual(closed, [])
        ensure_episode(self.store, at(300), CONFIG, closed.append)
        self.assertEqual(closed, [first.id])

    def test_close_stale_episode_closes_without_opening(self):
        first = ensure_episode(self.store, T0, CONFIG)
        closed: list[int] = []
        self.assertIsNone(close_stale_episode(self.store, at(300), CONFIG, closed.append))
        self.assertIsNone(self.store.current_episode())
        self.assertEqual(closed, [first.id])

    def test_close_stale_episode_keeps_a_recent_one(self):
        first = ensure_episode(self.store, T0, CONFIG)
        kept = close_stale_episode(self.store, at(30), CONFIG)
        self.assertEqual(kept, first)


class EmbedCachedTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def test_returns_one_embedding_per_text_in_order_with_hashes(self):
        result = embed_cached(["alpha", "beta"], self.embedder, self.store)
        self.assertEqual([e.text_hash for e in result], [text_hash("alpha"), text_hash("beta")])
        self.assertEqual(result[0].vector, self.embedder.vector("alpha"))

    def test_second_call_is_served_from_the_cache(self):
        embed_cached(["alpha"], self.embedder, self.store)
        embed_cached(["alpha"], self.embedder, self.store)
        self.assertEqual(self.embedder.calls, [["alpha"]])

    def test_duplicates_are_embedded_once(self):
        result = embed_cached(["alpha", "alpha", "beta"], self.embedder, self.store)
        self.assertEqual(self.embedder.calls, [["alpha", "beta"]])
        self.assertEqual(result[0], result[1])

    def test_only_misses_are_embedded(self):
        embed_cached(["alpha"], self.embedder, self.store)
        embed_cached(["alpha", "beta"], self.embedder, self.store)
        self.assertEqual(self.embedder.calls, [["alpha"], ["beta"]])

    def test_cache_is_per_model(self):
        embed_cached(["alpha"], self.embedder, self.store)
        other = FakeEmbedder(model="other-model")
        embed_cached(["alpha"], other, self.store)
        self.assertEqual(other.calls, [["alpha"]])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def remember(self, text: str, when: datetime) -> int:
        """Store a message in its own past episode; returns the episode id."""
        episode = self.store.open_episode(when)
        [vector] = embed_cached([text], self.embedder, self.store)
        self.store.add_message(episode.id, Role.USER, text, when, vector)
        self.store.close_episode(episode.id, when)
        return episode.id

    def query(self, text: str):
        return self.embedder.embed([text])[0]

    def test_finds_the_related_message_and_drops_unrelated_ones(self):
        self.remember("my sister lucia lives in montreal", at(0))
        self.remember("the build pipeline uses docker", at(10))
        results = retrieve(self.query("where does my sister lucia live"), self.store, CONFIG)
        self.assertEqual([r.candidate.text for r in results], ["my sister lucia lives in montreal"])

    def say(self, role: Role, text: str, when: datetime) -> None:
        episode = self.store.open_episode(when)
        [vector] = embed_cached([text], self.embedder, self.store)
        self.store.add_message(episode.id, role, text, when, vector)
        self.store.close_episode(episode.id, when)

    def test_short_replies_from_cerveau_are_never_recalled(self):
        self.say(Role.ASSISTANT, "lucia montreal, noted.", at(0))                          # short
        self.say(Role.ASSISTANT, "lucia montreal: " + "she moved there for a nurse job " * 2, at(10))
        self.say(Role.USER, "lucia montreal", at(20))                                    # short, but yours
        found = retrieve(self.query("lucia montreal"), self.store, CONFIG)
        texts = [r.candidate.text for r in found]
        self.assertNotIn("lucia montreal, noted.", texts)
        self.assertIn("lucia montreal", texts)
        self.assertEqual(len(texts), 2)
        everything = retrieve(self.query("lucia montreal"), self.store, replace(CONFIG, short_reply_chars=0))
        self.assertIn("lucia montreal, noted.", [r.candidate.text for r in everything])

    def test_short_replies_dont_use_up_the_slots(self):
        for i in range(4):
            self.say(Role.ASSISTANT, f"rex beagle {i}", at(i))
        kept = self.remember("rex the beagle loves the park", at(50))
        config = replace(CONFIG, messages=replace(CONFIG.messages, max_items=2, min_score=0.1))
        found = retrieve(self.query("rex beagle"), self.store, config)
        self.assertIn(kept, [r.candidate.episode_id for r in found])

    def test_current_episode_is_excluded(self):
        episode_id = self.remember("my sister lucia lives in montreal", at(0))
        results = retrieve(
            self.query("my sister lucia lives in montreal"), self.store, CONFIG,
            current_episode_id=episode_id,
        )
        self.assertEqual(results, [])

    def test_max_items_caps_each_source(self):
        for i in range(5):
            self.remember(f"rex the beagle note {i}", at(i))
        config = EngineConfig(messages=SourceConfig(max_items=2, min_score=0.1))
        self.assertEqual(len(retrieve(self.query("rex the beagle"), self.store, config)), 2)

    def test_disabled_source_is_skipped(self):
        self.remember("rex the beagle", at(0))
        config = EngineConfig(messages=SourceConfig(enabled=False))
        self.assertEqual(retrieve(self.query("rex the beagle"), self.store, config), [])

    def test_merge_keeps_one_candidate_with_both_signals(self):
        semantic = [Candidate(MemoryKind.MESSAGE, 1, "a", semantic=0.5),
                    Candidate(MemoryKind.MESSAGE, 2, "b", semantic=0.2)]
        named = [Candidate(MemoryKind.MESSAGE, 2, "b", lexical=1.0),
                 Candidate(MemoryKind.MESSAGE, 3, "c", lexical=0.5)]
        merged = {c.id: c for c in merge(semantic, named)}
        self.assertEqual(sorted(merged), [1, 2, 3])
        self.assertEqual((merged[2].semantic, merged[2].lexical), (0.2, 1.0))
        self.assertEqual((merged[3].semantic, merged[3].lexical), (None, 0.5))

    def test_a_named_entity_brings_memories_that_share_no_words(self):
        marta = self.store.upsert_entity("Dr. Marta Peña", "person")
        assert marta is not None
        episode = self.store.open_episode(T0)
        [vector] = embed_cached(["she approved the schema"], self.embedder, self.store)
        message = self.store.add_message(episode.id, Role.USER, "she approved the schema", T0, vector)
        self.store.add_mention(marta.id, message.id)
        self.store.close_episode(episode.id, T0)
        results = retrieve(self.query("any news from marta"), self.store, CONFIG, entities=[marta])
        kinds = [(r.candidate.kind, r.candidate.id) for r in results]
        self.assertIn((MemoryKind.MESSAGE, message.id), kinds)
        self.assertIn((MemoryKind.ENTITY, marta.id), kinds)

    def test_entity_cards_describe_type_and_aliases(self):
        card = entity_card(Entity(3, "Dr. Marta Peña", "person", ("my colleague",)))
        self.assertEqual(card.text, "Dr. Marta Peña (person; also: my colleague)")
        self.assertEqual(entity_card(Entity(4, "Rex")).text, "Rex")

    def test_summaries_already_shown_as_recent_are_skipped(self):
        episode = self.store.open_episode(T0)
        self.store.close_episode(episode.id, T0)
        [vector] = embed_cached(["rex the beagle"], self.embedder, self.store)
        self.store.save_summary(
            episode.id, title="Rex", summary_short="rex the beagle",
            summary_long="rex the beagle", short_embedding=vector, long_embedding=vector,
        )
        query = self.query("rex the beagle")
        self.assertEqual(len(retrieve(query, self.store, CONFIG)), 1)
        self.assertEqual(retrieve(query, self.store, CONFIG, shown_episode_ids={episode.id}), [])

    def summarized(self, summary: str, *tags: str) -> int:
        episode = self.store.open_episode(T0)
        self.store.close_episode(episode.id, T0)
        [vector] = embed_cached([summary], self.embedder, self.store)
        self.store.save_summary(episode.id, title=summary, summary_short=summary,
                                summary_long=summary, short_embedding=vector, long_embedding=vector)
        for name in tags:
            tag = self.store.upsert_tag(name, self.embedder.embed([name])[0])
            assert tag is not None
            self.store.tag_episode(episode.id, tag.id, 1.0)
        return episode.id

    def test_a_matching_tag_nudges_its_episode_up(self):
        plain = self.summarized("knee pain after running")
        tagged = self.summarized("knee pain after the match", "running")
        config = replace(CONFIG, tag_match_score=0.5,
                         episodes=SourceConfig(min_score=0.1, w_tag=0.5))
        query = self.query("running knee pain")
        tags = matched_tags(query, self.store, config)
        self.assertEqual([t.name for t, _ in tags], ["running"])
        by_id = {r.candidate.id: r for r in retrieve(query, self.store, config, tags=tags)}
        without = {r.candidate.id: r for r in retrieve(query, self.store, config)}
        self.assertIsNone(by_id[plain].candidate.topical)
        self.assertGreater(by_id[tagged].candidate.topical or 0, 0.5)
        self.assertGreater(by_id[tagged].score, without[tagged].score)
        self.assertEqual(by_id[plain].score, without[plain].score)

    def test_tags_need_a_weight_and_forgotten_tags_never_match(self):
        tagged = self.summarized("knee pain after the match", "running")
        query = self.query("running knee pain")
        config = replace(CONFIG, tag_match_score=0.5, episodes=SourceConfig(min_score=0.1))  # w_tag 0
        tags = matched_tags(query, self.store, config)
        [result] = [r for r in retrieve(query, self.store, config, tags=tags) if r.candidate.id == tagged]
        self.assertIsNone(result.candidate.topical)
        [tag] = [t for t, _ in tags]
        self.store.set_status(MemoryKind.TAG, tag.id, "Z")
        self.assertEqual(matched_tags(query, self.store, config), [])
        self.assertEqual(matched_tags(query, self.store, replace(config, tag_match_score=0.99)), [])

    def test_results_are_sorted_best_first(self):
        self.remember("rex", at(0))
        self.remember("rex the beagle loves walks", at(1))
        config = EngineConfig(messages=SourceConfig(min_score=0.1))
        scores = [r.score for r in retrieve(self.query("rex the beagle loves walks"), self.store, config)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class PromptTests(unittest.TestCase):
    def test_estimate_rounds_up_and_adds_overhead(self):
        self.assertEqual(estimate_tokens("abcde", chars_per_token=4.0), 2 + 4)

    def test_fit_recent_keeps_everything_that_fits(self):
        messages = [message(1, "a" * 40), message(2, "b" * 40)]
        self.assertEqual(fit_recent(messages, budget=100, chars_per_token=4.0), messages)

    def test_fit_recent_drops_the_oldest_first(self):
        messages = [message(1, "a" * 400), message(2, "b" * 40), message(3, "c" * 40)]
        kept = fit_recent(messages, budget=40, chars_per_token=4.0)
        self.assertEqual([m.id for m in kept], [2, 3])

    def test_fit_recent_always_keeps_the_newest_message(self):
        messages = [message(1, "short"), message(2, "x" * 10_000)]
        self.assertEqual([m.id for m in fit_recent(messages, 10, 4.0)], [2])

    def test_fit_recent_stops_at_the_first_message_that_does_not_fit(self):
        # m1 would fit on its own, but keeping it would leave a gap where m2 was.
        messages = [message(1, "a"), message(2, "b" * 400), message(3, "c")]
        self.assertEqual([m.id for m in fit_recent(messages, 20, 4.0)], [3])

    def test_assemble_prompt_starts_with_the_system_prompt(self):
        prompt = assemble_prompt([message(1, "Hi")], PROFILE)
        self.assertEqual(prompt[0], ChatMessage(Role.SYSTEM, SYSTEM_PROMPT))
        self.assertEqual(prompt[1:], [ChatMessage(Role.USER, "Hi")])

    def test_assemble_prompt_keeps_roles_and_order(self):
        recent = [message(1, "Q1"), message(2, "A1", Role.ASSISTANT), message(3, "Q2")]
        prompt = assemble_prompt(recent, PROFILE)
        self.assertEqual(
            [(m.role, m.content) for m in prompt[1:]],
            [(Role.USER, "Q1"), (Role.ASSISTANT, "A1"), (Role.USER, "Q2")],
        )

    def test_assemble_prompt_fits_the_memory_budget(self):
        small = ModelProfile(
            name="small", provider="fake", context_window=1000, output_reserve=100,
            memory_budget=500,
        )
        recent = [message(i, "x" * 200) for i in range(1, 11)]  # ~54 tokens each
        prompt = assemble_prompt(recent, small)
        total = sum(estimate_tokens(m.content, 4.0) for m in prompt)
        self.assertLessEqual(total, small.memory_budget)
        self.assertEqual(prompt[-1].content, recent[-1].content)

    def test_memory_line_shows_id_date_and_speaker(self):
        self.assertEqual(
            memory_line(scored(12, "Lucia lives in Montreal", 0.8)),
            "- [M12 · Wed 2026-10-14 · user] Lucia lives in Montreal",
        )
        self.assertIn("· Cerveau]", memory_line(scored(13, "Noted.", 0.8, Role.ASSISTANT)))
        card = ScoredCandidate(entity_card(Entity(3, "Rex", "pet")), 1.0)
        self.assertEqual(memory_line(card), "- [N3] Rex (pet)")

    def test_memory_goes_into_the_last_user_message_before_its_text(self):
        recent = [message(1, "Q1"), message(2, "A1", Role.ASSISTANT), message(3, "Where does she live?")]
        prompt = assemble_prompt(
            recent, PROFILE, memories=[scored(7, "Lucia lives in Montreal", 0.8)], memory_share=0.3
        )
        last = prompt[-1].content
        self.assertTrue(last.startswith(MEMORY_HEADER))
        self.assertIn("Lucia lives in Montreal", last)
        self.assertTrue(last.endswith("Where does she live?"))
        # Earlier messages are untouched, so the prompt prefix stays cacheable.
        self.assertEqual([m.content for m in prompt[1:3]], ["Q1", "A1"])

    def test_no_memory_means_no_block(self):
        prompt = assemble_prompt([message(1, "Hi")], PROFILE, memories=[], memory_share=0.3)
        self.assertEqual(prompt[-1].content, "Hi")

    def test_memory_respects_its_share_of_the_budget(self):
        memories = [scored(i, "x" * 400, 0.9 - i / 100) for i in range(20)]  # ~100 tokens each
        prompt = assemble_prompt([message(1, "Hi")], PROFILE, memories=memories, memory_share=0.3)
        block = prompt[-1].content
        self.assertLessEqual(len(block) / 4, (PROFILE.memory_budget) * 0.3)
        self.assertIn("[M0 ", block)  # the best one made it

    def test_zero_share_disables_memory(self):
        prompt = assemble_prompt(
            [message(1, "Hi")], PROFILE, memories=[scored(1, "fact", 0.9)], memory_share=0.0
        )
        self.assertEqual(prompt[-1].content, "Hi")


class RecentContextTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()

    def closed(self, minutes: int, *texts: str) -> int:
        episode = self.store.open_episode(at(minutes))
        for i, text in enumerate(texts):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            self.store.add_message(episode.id, role, text, at(minutes + i))
        self.store.close_episode(episode.id, at(minutes + len(texts)))
        return episode.id

    def test_summarized_episodes_come_without_preview(self):
        episode = self.closed(0, "hello")
        vector = FakeEmbedder().embed(["x"])[0]
        self.store.save_summary(
            episode, title="T", summary_short="s", summary_long="l",
            short_embedding=vector, long_embedding=vector,
        )
        [recent] = recent_context(self.store, CONFIG, current_episode_id=None)
        self.assertEqual((recent.episode.id, recent.preview), (episode, []))

    def test_unsummarized_episodes_bring_their_last_messages(self):
        self.closed(0, *(f"m{i}" for i in range(6)))
        [recent] = recent_context(self.store, CONFIG, current_episode_id=None)
        self.assertEqual([m.content for m in recent.preview], ["m2", "m3", "m4", "m5"])

    def test_the_current_episode_is_not_recent(self):
        episode = self.closed(0, "hello")
        self.assertEqual(recent_context(self.store, CONFIG, current_episode_id=episode), [])

    def test_zero_disables_it(self):
        self.closed(0, "hello")
        config = EngineConfig(recent_episodes=0)
        self.assertEqual(recent_context(self.store, config, current_episode_id=None), [])


class SystemMessageTests(unittest.TestCase):
    def episode(self, **fields) -> Episode:
        values = dict(id=3, started_at=T0, ended_at=T0 + timedelta(minutes=40))
        values.update(fields)
        return Episode(**values)  # type: ignore[arg-type]

    def test_states_today_in_the_users_timezone(self):
        late_evening = datetime(2026, 10, 16, 2, 30, tzinfo=UTC)  # 22:30 on the 15th, UTC-4
        text = system_message(late_evening, tz=SANTO_DOMINGO)
        self.assertTrue(text.startswith(SYSTEM_PROMPT))
        self.assertIn("Today is Thursday, 2026-10-15.", text)

    def test_lists_recent_episodes_with_dates_titles_and_summaries(self):
        summarized = self.episode(title="Lisbon trip", summary_short="The user planned a trip.")
        text = system_message(T0, [RecentEpisode(summarized)], tz=UTC)
        self.assertIn(RECENT_HEADER, text)
        self.assertIn(
            "- [E3 · Wed 2026-10-14 09:00 to 09:40] Lisbon trip: The user planned a trip.", text
        )

    def test_an_episode_past_midnight_shows_its_end_date(self):
        late = self.episode(
            started_at=datetime(2026, 10, 14, 23, 36, tzinfo=UTC),
            ended_at=datetime(2026, 10, 15, 0, 13, tzinfo=UTC),
            title="Late talk", summary_short="s",
        )
        text = system_message(T0, [RecentEpisode(late)], tz=UTC)
        self.assertIn("[E3 · Wed 2026-10-14 23:36 to Thu 2026-10-15 00:13]", text)

    def test_weekday_names_do_not_depend_on_the_system_locale(self):
        import locale
        previous = locale.setlocale(locale.LC_TIME)
        try:
            try:
                locale.setlocale(locale.LC_TIME, "es_ES.UTF-8" if "win" not in sys.platform else "Spanish_Spain.1252")
            except locale.Error:
                self.skipTest("Spanish locale not installed")
            self.assertIn("Today is Wednesday, 2026-10-14.", system_message(T0, tz=UTC))
        finally:
            locale.setlocale(locale.LC_TIME, previous)

    def test_an_unsummarized_episode_shows_how_it_ended(self):
        preview = [
            message(1, "Planning a trip to Lisbon"),
            message(2, "Lisbon in May is lovely.", Role.ASSISTANT),
        ]
        text = system_message(T0, [RecentEpisode(self.episode(), preview)], tz=UTC)
        self.assertIn(
            "(not summarized yet; it ended with) user: Planning a trip to Lisbon"
            " / Cerveau: Lisbon in May is lovely.",
            text,
        )

    def test_no_recent_episodes_means_no_section(self):
        self.assertNotIn(RECENT_HEADER, system_message(T0, [], tz=UTC))

    def test_the_rule_against_denying_memory_is_in_the_prompt(self):
        self.assertIn("Never say you can't access past conversations", SYSTEM_PROMPT)


class RunTurnTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def turn(self, text: str, now: datetime, llm: ChatProvider) -> str:
        chunks = run_turn(
            text, now, store=self.store, llm=llm, embedder=self.embedder, profile=PROFILE,
            config=CONFIG, clock=lambda: now + timedelta(seconds=5),
        )
        return "".join(c.content for c in chunks)

    def test_streams_the_reply_and_stores_both_messages(self):
        reply = self.turn("Hi", T0, FakeLLM("Hello there."))
        self.assertEqual(reply, "Hello there.")
        episode = self.store.current_episode()
        assert episode is not None
        stored = self.store.get_messages(episode.id)
        self.assertEqual(
            [(m.role, m.content) for m in stored],
            [(Role.USER, "Hi"), (Role.ASSISTANT, "Hello there.")],
        )

    def test_reply_is_timestamped_by_the_clock(self):
        self.turn("Hi", T0, FakeLLM())
        episode = self.store.current_episode()
        assert episode is not None
        self.assertEqual(episode.last_message_at, T0 + timedelta(seconds=5))

    def test_second_turn_sees_the_first_exchange(self):
        llm = FakeLLM("Nice to meet you, Rex's owner.")
        self.turn("My dog is called Rex.", T0, llm)
        self.turn("What is my dog called?", at(1), llm)
        second_prompt = llm.prompts[-1]
        self.assertEqual(
            [m.content for m in second_prompt[1:]],
            ["My dog is called Rex.", "Nice to meet you, Rex's owner.",
             "What is my dog called?\n\n(Reply in English.)"],  # hint: prompt only
        )

    def test_turn_after_the_gap_starts_a_fresh_episode(self):
        llm = FakeLLM("Ok.")
        self.turn("Old topic about gardening", T0, llm)
        self.turn("Completely new day", at(600), llm)
        self.assertEqual([m.content for m in llm.prompts[-1][1:]], ["Completely new day"])

    def test_a_past_episode_is_recalled_into_the_prompt(self):
        llm = FakeLLM("Noted.")
        self.turn("my sister lucia lives in montreal", T0, llm)
        self.turn("where does my sister lucia live", at(600), llm)  # new episode
        last = llm.prompts[-1][-1].content
        self.assertTrue(last.startswith(MEMORY_HEADER))
        self.assertIn("my sister lucia lives in montreal", last)
        self.assertTrue(last.endswith("where does my sister lucia live\n\n(Reply in English.)"))

    def test_both_messages_are_stored_with_embeddings(self):
        self.turn("rex the beagle", T0, FakeLLM("rex is a good dog"))
        query = self.embedder.embed(["rex the beagle good dog"])[0]
        hits = self.store.semantic_candidates(MemoryKind.MESSAGE, query, 10)
        self.assertEqual({h.role for h in hits}, {Role.USER, Role.ASSISTANT})

    def test_a_turn_that_closes_an_episode_reports_it(self):
        closed: list[int] = []
        llm = FakeLLM("Ok.")
        for minutes in (0, 600):
            list(run_turn(
                "hello", at(minutes), store=self.store, llm=llm, embedder=self.embedder,
                profile=PROFILE, config=CONFIG, on_episode_closed=closed.append,
                clock=lambda m=minutes: at(m) + timedelta(seconds=5),
            ))
        self.assertEqual(len(closed), 1)
        [old] = self.store.get_episodes(closed)
        self.assertFalse(old.is_open)

    def test_what_did_we_talk_about_last_gets_the_previous_episode(self):
        """The bug this fixes: a question about the past that resembles nothing said in
        it. Right after the gap, before consolidation, the previous episode shows up
        through its last messages."""
        llm = FakeLLM("Sure.")
        self.turn("I'm planning a trip to Lisbon in May", T0, llm)
        self.turn("what did we talk about last?", at(600), llm)  # new episode
        system = llm.prompts[-1][0].content
        self.assertIn(RECENT_HEADER, system)
        self.assertIn("user: I'm planning a trip to Lisbon in May", system)

    def test_naming_someone_recalls_what_was_said_about_them(self):
        llm = FakeLLM("Ok.")
        marta = self.store.upsert_entity("Dr. Marta Peña", "person")
        assert marta is not None
        self.turn("the schema review is on friday", T0, llm)
        [said] = [m for m in self.store.get_messages(1) if m.role is Role.USER]
        self.store.add_mention(marta.id, said.id)  # as consolidation would
        self.turn("did Marta say anything?", at(600), llm)  # new episode, no shared words
        last = llm.prompts[-1][-1].content
        self.assertIn("[N1] Dr. Marta Peña (person)", last)
        self.assertIn("the schema review is on friday", last)

    def test_empty_reply_is_not_stored(self):
        self.turn("Hi", T0, FakeLLM(""))
        episode = self.store.current_episode()
        assert episode is not None
        self.assertEqual([m.role for m in self.store.get_messages(episode.id)], [Role.USER])

    def test_failed_stream_keeps_the_user_message_and_stores_no_reply(self):
        llm = FakeLLM("one two three", fail_after=1)
        with self.assertRaises(ChatProviderError):
            self.turn("Hi", T0, llm)
        episode = self.store.current_episode()
        assert episode is not None
        self.assertEqual([m.role for m in self.store.get_messages(episode.id)], [Role.USER])


if __name__ == "__main__":
    unittest.main()
