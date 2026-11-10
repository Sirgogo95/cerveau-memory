"""The memory tools (search_memory, read_memory) and the turn loop that runs them.
InMemoryStore, the bag-of-words FakeEmbedder, and a scripted LLM."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig
from brain.prompt import TOOLS_PROMPT
from brain.tools import (
    LIMIT_REACHED,
    NO_RESULTS,
    READ_MEMORY,
    SEARCH_MEMORY,
    TOOLS,
    ToolContext,
    run_tool,
    shown_ids,
)
from brain.turn import run_turn
from brain.types import Role, ToolCall
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
TOOL_PROFILE = replace(PROFILE, supports_tools=True)
# The FakeEmbedder's similarities are coarse; a low bar keeps the tests about logic.
CONFIG = EngineConfig(timezone="UTC", search_min_score=0.2)


def day(n: int, minutes: int = 0) -> datetime:
    return T0 + timedelta(days=n, minutes=minutes)


class Memory:
    """A store with a few past episodes, built the way the app fills it."""

    def __init__(self):
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()

    def episode(self, n: int, messages, summary: str = "", title: str = "") -> int:
        episode = self.store.open_episode(day(n))
        for i, (role, text) in enumerate(messages):
            [vector] = self.embedder.embed([text])
            self.store.add_message(episode.id, Role(role), text, day(n, i), vector)
        self.store.close_episode(episode.id, day(n, len(messages)))
        if summary:
            [vector] = self.embedder.embed([summary])
            self.store.save_summary(
                episode.id, title=title, summary_short=summary, summary_long=summary + " More.",
                short_embedding=vector, long_embedding=vector,
            )
        return episode.id

    def context(self, **overrides) -> ToolContext:
        base = ToolContext(store=self.store, embedder=self.embedder, config=CONFIG,
                           profile=TOOL_PROFILE, now=day(30), tz=UTC)
        return replace(base, **overrides)


def call(ctx: ToolContext, name: str, **arguments) -> dict:
    return json.loads(run_tool(ToolCall(name, arguments), ctx))


class SearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.portal = self.memory.episode(0, [
            ("user", "the portal launch moves to december"),
            ("assistant", "noted the portal launch moves"),
        ], summary="portal launch moved to december", title="Portal launch")
        self.coffee = self.memory.episode(10, [("user", "my favorite coffee place is la cafetera")])

    def test_finds_episodes_and_messages_by_meaning_with_ids_and_dates(self):
        result = call(self.memory.context(), "search_memory", query="portal launch december")
        ids = [r["id"] for r in result["results"]]
        self.assertIn(f"E{self.portal}", ids)
        episode = next(r for r in result["results"] if r["kind"] == "episode")
        self.assertEqual((episode["title"], episode["date"]), ("Portal launch", "Thu 2026-10-01"))
        message = next(r for r in result["results"] if r["kind"] == "message")
        self.assertEqual((message["speaker"], message["episode"]), ("user", f"E{self.portal}"))
        self.assertNotIn(f"E{self.coffee}", ids)

    def test_short_replies_from_cerveau_are_left_out(self):
        result = call(self.memory.context(), "search_memory", query="noted the portal launch moves",
                      kinds=["message"])
        self.assertNotIn("Cerveau", [r.get("speaker") for r in result["results"]])  # "noted the..." is short

    def test_kinds_restrict_what_is_searched(self):
        result = call(self.memory.context(), "search_memory", query="portal launch", kinds=["episode"])
        self.assertEqual({r["kind"] for r in result["results"]}, {"episode"})

    def test_dates_bound_the_search(self):
        later = call(self.memory.context(), "search_memory", query="portal launch", after="2026-10-05")
        self.assertEqual({r["date"] for r in later["results"]}, {"Sun 2026-10-11"})  # coffee only
        earlier = call(self.memory.context(), "search_memory", query="portal launch", before="2026-10-05")
        self.assertEqual({r["date"] for r in earlier["results"]}, {"Thu 2026-10-01"})
        empty = call(self.memory.context(), "search_memory", query="portal launch", after="2026-12-01")
        self.assertEqual((empty["results"], empty["note"]), ([], NO_RESULTS))

    def test_messages_in_the_prompt_are_left_out_and_the_rest_searched(self):
        in_prompt, earlier = self.memory.store.get_messages(self.portal)[::-1]
        ctx = self.memory.context(skip_message_ids={in_prompt.id})
        result = call(ctx, "search_memory", query="portal launch moves", kinds=["message"])
        ids = [r["id"] for r in result["results"]]
        self.assertNotIn(f"M{in_prompt.id}", ids)
        self.assertIn(f"M{earlier.id}", ids)

    def test_a_date_window_replaces_the_threshold(self):
        strict = replace(CONFIG, search_min_score=0.99)
        ctx = self.memory.context(config=strict)
        self.assertEqual(call(ctx, "search_memory", query="what did we talk about")["results"], [])
        window = call(ctx, "search_memory", query="what did we talk about",
                      after="2026-10-09", before="2026-10-12")
        self.assertEqual([r["id"] for r in window["results"] if r["kind"] == "message"],
                         [f"M{self.memory.store.get_messages(self.coffee)[0].id}"])

    def test_memories_already_in_the_prompt_are_only_listed(self):
        ctx = self.memory.context(shown={f"E{self.portal}"})
        result = call(ctx, "search_memory", query="portal launch december", kinds=["episode"])
        [item] = result["results"]
        self.assertEqual(item["note"], "already in context")
        self.assertNotIn("summary", item)

    def test_entities_restrict_to_memories_mentioning_them(self):
        lucia = self.memory.store.upsert_entity("Lucía", "person")
        assert lucia is not None
        other = self.memory.episode(20, [("user", "Lucía moved to montreal")])
        [message] = self.memory.store.get_messages(other)
        self.memory.store.add_mention(lucia.id, message.id)
        result = call(self.memory.context(), "search_memory", query="anything new", entities=["Lucia", "Nobody"])
        self.assertEqual([r["id"] for r in result["results"]], [f"M{message.id}"])
        self.assertEqual(result["unknown_names"], ["Nobody"])

    def test_bad_arguments_come_back_as_errors(self):
        ctx = self.memory.context()
        self.assertIn("non-empty query", call(ctx, "search_memory", query=" ")["error"])
        self.assertIn("date like", call(ctx, "search_memory", query="x", after="last week")["error"])
        self.assertIn("kinds", call(ctx, "search_memory", query="x", kinds=["skill"])["error"])
        self.assertIn("Unknown arguments", call(ctx, "search_memory", query="x", tags=["a"])["error"])
        self.assertIn("Unknown tool", call(ctx, "delete_memory")["error"])

    def test_results_stay_within_the_tool_budget(self):
        for n in range(12):
            self.memory.episode(40 + n, [("user", f"portal launch note number {n} " * 12)])
        ctx = self.memory.context(config=replace(CONFIG, tool_result_share=0.02))
        result = call(ctx, "search_memory", query="portal launch note", kinds=["message"], limit=20)
        self.assertTrue(result["truncated"])
        self.assertLess(len(result["results"]), 12)


class ReadMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.episode = self.memory.episode(0, [
            ("user", f"message number {i}") for i in range(6)
        ], summary="six numbered messages", title="Numbers")
        self.messages = self.memory.store.get_messages(self.episode)

    def test_episode_summaries(self):
        ctx = self.memory.context()
        self.assertEqual(call(ctx, "read_memory", id=f"E{self.episode}", detail="short")["summary"],
                         "six numbered messages")
        long = call(ctx, "read_memory", id=f"e{self.episode}")
        self.assertEqual((long["summary"], long["title"]), ("six numbered messages More.", "Numbers"))

    def test_episode_messages_with_a_range(self):
        result = call(self.memory.context(), "read_memory", id=f"E{self.episode}",
                      detail="messages", message_range=[2, 3])
        self.assertEqual([m["n"] for m in result["messages"]], [2, 3])
        self.assertEqual(result["messages"][0]["text"], "message number 1")
        self.assertEqual(result["total_messages"], 6)

    def test_an_unsummarized_episode_returns_its_messages(self):
        bare = self.memory.episode(5, [("user", "just one")])
        result = call(self.memory.context(), "read_memory", id=f"E{bare}")
        self.assertEqual([m["text"] for m in result["messages"]], ["just one"])
        self.assertIn("Not summarized", result["note"])

    def test_a_message_comes_with_its_neighbours(self):
        target = self.messages[3]
        result = call(self.memory.context(), "read_memory", id=f"M{target.id}")
        self.assertEqual([m["text"] for m in result["messages"]],
                         [f"message number {i}" for i in range(1, 6)])
        self.assertEqual([m.get("opened", False) for m in result["messages"]],
                         [False, False, True, False, False])

    def test_relative_dates_in_messages_are_annotated_as_in_the_notes(self):
        bare = self.memory.episode(8, [("user", "I saw Pedro yesterday at the market")])   # Fri 2026-10-09
        [message] = self.memory.store.get_messages(bare)
        said = "I saw Pedro yesterday [8 October 2026] at the market"
        read = call(self.memory.context(), "read_memory", id=f"M{message.id}")
        self.assertEqual([m["text"] for m in read["messages"]], [said])
        found = call(self.memory.context(), "search_memory", query="Pedro market", kinds=["message"])
        self.assertIn(said, [r["text"] for r in found["results"]])

    def test_unknown_and_malformed_ids(self):
        ctx = self.memory.context()
        self.assertIn("no memory E999", call(ctx, "read_memory", id="E999")["error"])
        self.assertIn("no memory M999", call(ctx, "read_memory", id="M999")["error"])
        self.assertIn("look like", call(ctx, "read_memory", id="S12")["error"])
        self.assertIn("no memory F12", call(ctx, "read_memory", id="F12")["error"])


class ShownIdsTests(unittest.TestCase):
    def test_reads_ids_from_memory_and_recent_lines(self):
        texts = ["- [E3 · Wed 2026-10-14 09:02 to 10:40] Title", "<memory>\n- [M12 · Thu 2026-10-15 · user] hi"]
        self.assertEqual(shown_ids(texts), {"E3", "M12"})


class TurnWithToolsTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.portal = self.memory.episode(0, [("user", "the portal launch moves to december")],
                                          summary="portal launch moved to december")

    def turn(self, llm, profile=TOOL_PROFILE, config=CONFIG, text="what about the portal?"):
        return list(run_turn(text, day(30), store=self.memory.store, llm=llm,
                             embedder=self.memory.embedder, profile=profile, config=config))

    def test_tool_call_runs_and_the_model_answers_with_the_result(self):
        llm = FakeLLM("It moved to December.",
                      tool_calls=[[ToolCall("search_memory", {"query": "portal launch"})]])
        chunks = self.turn(llm)
        self.assertEqual(chunks[0].tool_calls[0].name, "search_memory")  # visible to the app
        self.assertEqual(len(llm.prompts), 2)
        followup = llm.prompts[1]
        self.assertEqual(followup[-2].tool_calls, (ToolCall("search_memory", {"query": "portal launch"}),))
        self.assertEqual((followup[-1].role, followup[-1].tool_name), (Role.TOOL, "search_memory"))
        self.assertIn(f"E{self.portal}", followup[-1].content)
        current = self.memory.store.current_episode()
        assert current is not None
        stored = self.memory.store.get_messages(current.id)
        self.assertEqual([m.content for m in stored], ["what about the portal?", "It moved to December."])

    def test_the_calls_are_stored_with_the_reply_they_led_to(self):
        llm = FakeLLM("It moved to December.",
                      tool_calls=[[ToolCall("search_memory", {"query": "portal launch"})]])
        self.turn(llm)
        current = self.memory.store.current_episode()
        assert current is not None
        [run] = self.memory.store.get_tool_calls(current.id)
        reply = self.memory.store.get_messages(current.id)[-1]
        self.assertEqual((run.name, dict(run.arguments)), ("search_memory", {"query": "portal launch"}))
        self.assertEqual(run.message_id, reply.id)
        self.assertIn(f"E{self.portal}", run.result)
        self.assertGreaterEqual(run.duration_ms, 0)

    def test_calls_are_kept_even_when_the_reply_is_empty(self):
        llm = FakeLLM("", tool_calls=[[ToolCall("search_memory", {"query": "portal"})]])
        self.turn(llm)
        current = self.memory.store.current_episode()
        assert current is not None
        [run] = self.memory.store.get_tool_calls(current.id)
        self.assertIsNone(run.message_id)

    def test_a_reply_that_declines_without_searching_becomes_a_search(self):
        llm = FakeLLM(stream_replies=["I don't remember anything about a portal, sorry.", "It moved to December."])
        chunks = self.turn(llm)
        self.assertNotIn("remember", "".join(c.content for c in chunks))           # the decline never shows
        self.assertEqual(chunks[0].tool_calls, (ToolCall("search_memory", {"query": "what about the portal?"}),))
        result = llm.prompts[1][-1]
        self.assertEqual((result.role, result.tool_name), (Role.TOOL, "search_memory"))
        self.assertIn('"query": "what about the portal?"', result.content)
        current = self.memory.store.current_episode()
        assert current is not None
        self.assertEqual(self.memory.store.get_messages(current.id)[-1].content, "It moved to December.")
        [run] = self.memory.store.get_tool_calls(current.id)
        self.assertEqual(run.name, "search_memory")

    def test_a_short_decline_counts_too_and_only_once(self):
        llm = FakeLLM(stream_replies=["I don't know.", "I don't know."])
        chunks = self.turn(llm)
        self.assertEqual(len(llm.prompts), 2)                       # one search, then the answer stands
        self.assertEqual("".join(c.content for c in chunks), "I don't know.")

    def test_a_hedge_that_goes_on_to_answer_is_kept(self):
        hedge = ("I don't have a single last time to point to, as several conversations are recent. However, "
                 "looking at the latest summaries, on 31 March we discussed the portal launch moving to December, "
                 "and before that the budget for the next quarter and who would present it to the board.")
        llm = FakeLLM(hedge)
        self.assertEqual("".join(c.content for c in self.turn(llm)), hedge)
        self.assertEqual(len(llm.prompts), 1)

    def test_an_answer_streams_as_before_and_the_fallback_can_be_off(self):
        answer = FakeLLM("The portal launch moved to December, as you said last month.")
        self.assertEqual("".join(c.content for c in self.turn(answer)), answer.reply)
        self.assertEqual(len(answer.prompts), 1)
        off = FakeLLM("I don't remember that.")
        self.turn(off, config=replace(CONFIG, fallback_search=False))
        self.assertEqual(len(off.prompts), 1)

    def test_tools_and_guidance_are_offered_only_to_tool_capable_models(self):
        capable = FakeLLM()
        self.turn(capable)
        self.assertEqual(capable.tools[0], TOOLS)
        self.assertIn(TOOLS_PROMPT, capable.prompts[0][0].content)
        plain = FakeLLM()
        self.turn(plain, profile=PROFILE)
        self.assertEqual(plain.tools[0], ())
        self.assertNotIn(TOOLS_PROMPT, plain.prompts[0][0].content)
        switched_off = FakeLLM()
        self.turn(switched_off, config=replace(CONFIG, memory_tools=False))
        self.assertEqual(switched_off.tools[0], ())

    def test_the_call_limit_stops_the_loop_and_forces_an_answer(self):
        search = [ToolCall("search_memory", {"query": "portal"})]
        llm = FakeLLM("Done.", tool_calls=[search, search, search])
        self.turn(llm, config=replace(CONFIG, max_tool_calls=2))
        self.assertEqual(len(llm.prompts), 3)       # two tool rounds, then the answer
        self.assertEqual(llm.tools[2], ())          # no tools offered on the last call
        self.assertEqual(sum(m.role is Role.TOOL for m in llm.prompts[2]), 2)

    def test_extra_calls_in_one_round_get_the_limit_note(self):
        many = [ToolCall("search_memory", {"query": f"q{i}"}) for i in range(3)]
        llm = FakeLLM("Done.", tool_calls=[many])
        self.turn(llm, config=replace(CONFIG, max_tool_calls=2))
        results = [m.content for m in llm.prompts[1] if m.role is Role.TOOL]
        self.assertEqual(len(results), 3)
        self.assertIn(LIMIT_REACHED, results[2])

    def test_tool_definitions_are_valid_schemas_with_required_arguments(self):
        self.assertEqual(SEARCH_MEMORY.parameters["required"], ["query"])
        self.assertEqual(READ_MEMORY.parameters["required"], ["id"])


if __name__ == "__main__":
    unittest.main()
