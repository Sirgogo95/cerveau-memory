"""The LoCoMo runner: reading the dataset, building a conversation's memory, sampling
questions, retrieval scoring, token F1, the grader, the vector file, and the command
line. FakeEmbedder and FakeLLM: no model, no download."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from brain.config import EngineConfig
from brain.prompt import MEMORY_FOOTER, MEMORY_HEADER, RECENT_FOOTER, RECENT_HEADER
from brain.types import ChatMessage, MemoryKind, Role
from evals.locomo import (
    NOT_MENTIONED,
    Answered,
    Consolidator,
    DiskEmbedder,
    DiskLLM,
    Question,
    answer_question,
    answer_report,
    build,
    build_all,
    compare,
    f1,
    full_context_answer,
    in_notes,
    in_utc,
    interval,
    load_rows,
    judge,
    main,
    notes_report,
    notes_seen,
    parse,
    pick,
    question_keys,
    recall_questions,
    regrade,
    save,
    sign_test,
    verdict,
)
from evals.turns import with_settings
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_engine import PROFILE

DATA = [{
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to an LGBTQ support group yesterday."},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "I painted a lake sunrise last year.",
             "blip_caption": "a painting of a sunrise over a lake"},
        ],
        "session_2_date_time": "10:00 am on 20 May, 2023",
        "session_2": [
            {"speaker": "Melanie", "dia_id": "D2:1", "text": "We made pots at the pottery workshop."},
        ],
    },
    "session_summary": {"session_1_summary": "Caroline went to a support group; Melanie painted."},
    "qa": [
        {"question": "When did Caroline go to the support group?", "answer": "7 May 2023",
         "evidence": ["D1:1"], "category": 2},
        {"question": "What did Melanie make at the pottery workshop?", "answer": "pots",
         "evidence": ["D2:1; D1:2"], "category": 4},
        {"question": "What did Caroline paint?", "evidence": ["D1:2"], "category": 5,
         "adversarial_answer": "a sunrise"},
        {"question": "No evidence given", "answer": "x", "evidence": [], "category": 4},
    ],
}]


class ParseTests(unittest.TestCase):
    def test_sessions_turns_and_questions(self):
        [conversation] = parse(DATA)
        first, second = conversation.sessions
        self.assertEqual(first.at, datetime(2023, 5, 8, 13, 56, tzinfo=UTC))
        self.assertEqual(first.turns[0].text, "Caroline: I went to an LGBTQ support group yesterday.")
        self.assertIn("[shares a photo: a painting of a sunrise over a lake]", first.turns[1].text)
        self.assertEqual(second.summary, "")
        q = conversation.questions
        self.assertEqual(q[1].evidence, ("D2:1", "D1:2"))       # "a; b" split
        self.assertEqual((q[2].answer, q[2].trap), (NOT_MENTIONED, "a sunrise"))   # the trap isn't the answer
        self.assertEqual(conversation.end, second.at)

    def test_pick_skips_questions_without_evidence_and_samples_in_order(self):
        [conversation] = parse(DATA)
        self.assertEqual([q.category for q in pick(conversation.questions, {2, 4}, None, 0)], [2, 4])
        self.assertEqual(len(pick(conversation.questions, {2, 4, 5}, 2, 0)), 2)


class MemoryTests(unittest.TestCase):
    def test_build_makes_one_closed_episode_per_session(self):
        [conversation] = parse(DATA)
        prepared = build(conversation, FakeEmbedder(), summaries=True)
        episodes = prepared.store.get_episodes([1, 2])
        self.assertTrue(all(not e.is_open for e in episodes))
        self.assertEqual(sorted(prepared.refs.values()), ["D1:1", "D1:2", "D2:1", "S1"])
        self.assertNotIn((MemoryKind.EPISODE, 2), prepared.refs)  # no summary for session 2
        self.assertNotIn("S1", build(conversation, FakeEmbedder()).refs.values())

    def test_recall_finds_the_evidence_turn(self):
        [conversation] = parse(DATA)
        embedder = FakeEmbedder()
        prepared = build(conversation, embedder)
        rows = recall_questions(conversation, prepared, pick(conversation.questions, {4}, None, 0), embedder,
                                EngineConfig())
        [row] = rows
        self.assertIn("D2:1", row.result.kept)
        self.assertTrue(row.result.hit and row.session_hit)


class ScoringTests(unittest.TestCase):
    def test_token_f1(self):
        self.assertEqual(f1("Pots.", "pots"), 1.0)
        self.assertAlmostEqual(f1("they made the pots", "pots"), 2 * (1 / 3) / (1 / 3 + 1))
        self.assertEqual(f1("I don't know", "pots"), 0.0)

    def test_judge_reads_the_label(self):
        question = Question("What did they make?", "pots", ("D2:1",), 4)
        self.assertTrue(judge(question, "Pots.", FakeLLM('{"label": "CORRECT"}')))
        self.assertFalse(judge(question, "Vases.", FakeLLM('{"label": "WRONG"}')))
        self.assertTrue(judge(question, "Pots.", FakeLLM("CORRECT")))   # not JSON: read the word
        self.assertFalse(verdict("It could be CORRECT in spirit, but the fact differs.\n\nWRONG"))  # the last one
        self.assertFalse(verdict("I am not sure."))
        llm = FakeLLM('{"label": "WRONG"}')
        judge(question, "", llm)
        self.assertIn("(no answer)", llm.prompts[0][0].content)


class AnswerTests(unittest.TestCase):
    def test_full_context_puts_the_whole_conversation_in_the_prompt(self):
        [conversation] = parse(DATA)
        llm = FakeLLM("Pots.")
        self.assertEqual(full_context_answer(conversation, conversation.questions[1], llm), "Pots.")
        system, question = llm.prompts[0]
        self.assertIn("Session 1 (Monday 2023-05-08 13:56):", system.content)
        self.assertIn("Melanie: We made pots at the pottery workshop.", system.content)
        self.assertIn("(Today is Sunday 2023-05-21.)", question.content)

    def test_every_grader_grades_and_answers_are_saved(self):
        [conversation] = parse(DATA)
        strict, kind = FakeLLM('{"label": "WRONG"}'), FakeLLM('{"label": "CORRECT"}')
        strict.model, kind.model = "strict", "kind"
        row = answer_question(conversation, conversation.questions[1], embedder=None, llm=FakeLLM("Pots."),
                              judges=[strict, kind], profile=None, config=EngineConfig(), full_context=True)
        self.assertEqual(row.judged, {"strict": False, "kind": True})
        report = answer_report([row])
        self.assertIn("strict", report)
        self.assertIn("100%", report)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "answers.jsonl"
            save([row], path)
            [line] = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(line)["judged"], {"strict": False, "kind": True})


    def test_dates_are_shown_in_utc_as_the_sessions_are_stored(self):
        # 12:37 am on 31 October stays 31 October whatever the machine's timezone.
        config = in_utc(EngineConfig(timezone="America/Santo_Domingo"))
        at = datetime(2022, 10, 31, 0, 37, tzinfo=UTC)
        tz = config.tz
        assert tz is not None
        self.assertEqual(at.astimezone(tz).date().isoformat(), "2022-10-31")

    def test_an_empty_reply_is_asked_again(self):
        [conversation] = parse(DATA)
        llm = FakeLLM(replies=["", "  ", "Pots."])
        row = answer_question(conversation, conversation.questions[1], embedder=None, llm=llm,
                              judges=[FakeLLM('{"label": "CORRECT"}')], profile=None, config=EngineConfig(),
                              full_context=True)
        self.assertEqual(row.answer, "Pots.")
        self.assertEqual(len(llm.prompts), 3)

    def test_an_answer_that_stays_empty_is_kept_after_three_tries(self):
        [conversation] = parse(DATA)
        llm = FakeLLM(replies=[""])
        row = answer_question(conversation, conversation.questions[1], embedder=None, llm=llm,
                              judges=[FakeLLM('{"label": "WRONG"}')], profile=None, config=EngineConfig(),
                              full_context=True)
        self.assertEqual(row.answer, "")
        self.assertEqual(len(llm.prompts), 3)


class StatisticsTests(unittest.TestCase):
    def test_sign_test_and_interval(self):
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertAlmostEqual(sign_test(10, 0), 2 / 1024)
        self.assertGreater(sign_test(25, 17), 0.2)               # A10 vs A8: not proven by itself
        self.assertAlmostEqual(interval(130, 200), 1.96 * (0.65 * 0.35 / 200) ** 0.5)

    def test_adversarial_questions_are_graded_on_declining_and_kept_out_of_all(self):
        [conversation] = parse(DATA)
        adversarial = conversation.questions[2]
        llm = FakeLLM('{"label": "CORRECT"}')
        judge(adversarial, "That isn't mentioned.", llm)
        prompt = llm.prompts[0][0].content
        self.assertIn("a sunrise", prompt)
        self.assertIn("isn't mentioned", prompt)
        rows = [answer_question(conversation, q, embedder=None, llm=FakeLLM("Pots."), judges=[llm], profile=None,
                                config=EngineConfig(), full_context=True) for q in conversation.questions[1:3]]
        report = answer_report(rows)
        self.assertIn("adversarial", report)
        all_row = next(line for line in report.splitlines() if "all (1-4)" in line)
        self.assertIn(" 1 ", all_row)                             # only the single-hop question

    def test_compare_two_saved_runs(self):
        [conversation] = parse(DATA)
        questions = conversation.questions[:2]
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, label in (("a", "WRONG"), ("b", "CORRECT")):
                grader = FakeLLM(f'{{"label": "{label}"}}')
                grader.model = "g"
                rows = [answer_question(conversation, q, embedder=None, llm=FakeLLM("x"), judges=[grader],
                                        profile=None, config=EngineConfig(), full_context=True) for q in questions]
                paths.append(Path(folder) / f"{name}.jsonl")
                save(rows, paths[-1])
            text = compare(*paths)
        self.assertIn("2 questions in both runs", text)
        self.assertIn("0% -> 100%, 2 turned right, 0 turned wrong", text)


class NotesTests(unittest.TestCase):
    def test_the_notes_are_the_memory_sections_and_search_results_only(self):
        system = ChatMessage(Role.SYSTEM, "Instructions mention the <memory> block and <recent_conversations>.\n\n"
                             + RECENT_HEADER + "\n- [E1] a trip\n" + RECENT_FOOTER)
        user = ChatMessage(Role.USER, MEMORY_HEADER + "\n- [F3] Melanie paints\n" + MEMORY_FOOTER + "\nWhat does Melanie do?")
        tool = ChatMessage(Role.TOOL, '{"results": ["Melanie swims"]}')
        notes = notes_seen([[system, user], [system, user, tool]])
        self.assertIn("- [E1] a trip", notes)
        self.assertIn("- [F3] Melanie paints", notes)
        self.assertIn("Melanie swims", notes)
        self.assertNotIn("Instructions mention", notes)
        self.assertNotIn("What does Melanie do?", notes)

    def test_how_much_of_the_answer_the_notes_hold(self):
        [conversation] = parse(DATA)
        question = conversation.questions[1]
        self.assertEqual(in_notes(question, "notes", FakeLLM("They hold some of it.\n\nPART")), "PART")
        self.assertEqual(in_notes(question, "", FakeLLM("ALL")), "NONE")            # nothing was shown
        self.assertEqual(in_notes(conversation.questions[2], "notes", FakeLLM("ALL")), "")   # adversarial: no gold

    def test_the_report_splits_wrong_answers_by_what_the_notes_held(self):
        [conversation] = parse(DATA)
        q = conversation.questions[1]
        rows = [Answered("c", q, "a", {"g": ok}, 0.0, 1.0, in_notes=label)
                for ok, label in ((True, "ALL"), (False, "ALL"), (False, "PART"), (False, "NONE"), (False, "NONE"))]
        report = notes_report(rows, "g")
        self.assertIn("of 4 wrong answers: 2 had none of the answer in the notes (memory), 1 part of it, 1 all of it", report)


class RegradeTests(unittest.TestCase):
    def test_saved_answers_are_graded_again_without_answering(self):
        [conversation] = parse(DATA)
        strict = FakeLLM('{"label": "WRONG"}')
        strict.model = "g"
        rows = [answer_question(conversation, q, embedder=None, llm=FakeLLM("Pots."), judges=[strict],
                                profile=None, config=EngineConfig(), full_context=True)
                for q in conversation.questions[:3]]
        kind = FakeLLM('{"label": "CORRECT"}')
        kind.model = "g"
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.jsonl", Path(folder) / "b.jsonl"
            save(rows, first)
            again = regrade(first, [kind], second)
            self.assertEqual([r.answer for r in again], ["Pots."] * 3)
            self.assertTrue(all(r.judged == {"g": True} for r in again))
            self.assertEqual(again[2].question.trap, "a sunrise")       # adversarial kept
            self.assertIn("100%", compare(first, second))
            regrade(first, [kind], Path(folder) / "c.jsonl", lenient=True)
            self.assertIn("Be generous", kind.prompts[-3][-1].content)      # a normal question
            self.assertNotIn("Be generous", kind.prompts[-1][-1].content)  # the adversarial one keeps its rule


class ResumeTests(unittest.TestCase):
    def test_saved_answers_load_back_and_a_crash_line_is_skipped(self):
        [conversation] = parse(DATA)
        grader = FakeLLM('{"label": "CORRECT"}')
        grader.model = "g"
        rows = [answer_question(conversation, q, embedder=None, llm=FakeLLM("Pots."), judges=[grader], profile=None,
                                config=EngineConfig(), full_context=True) for q in conversation.questions[:2]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.jsonl"
            save(rows, path)
            with path.open("a", encoding="utf-8") as out:
                out.write('{"conversation": "conv-1", "quest')          # cut off by a crash
            back = load_rows(path)
        self.assertEqual([(r.question.text, r.answer, r.judged) for r in back],
                         [(r.question.text, r.answer, r.judged) for r in rows])

    def test_a_question_asked_twice_has_two_keys(self):
        self.assertEqual(question_keys([("c", "q"), ("c", "q"), ("d", "q")]),
                         [("c", "q", 1), ("c", "q", 2), ("d", "q", 1)])


class ConsolidationTests(unittest.TestCase):
    def test_each_session_is_consolidated_and_replies_are_kept_on_disk(self):
        [conversation] = parse(DATA)
        reply = json.dumps({"title": "Support group and painting", "summary_short": "Caroline went to a support group.",
                            "summary_long": "Caroline went to a support group; Melanie painted a sunrise.",
                            "english_tags": ["community"], "entities": [{"name": "Caroline", "type": "person", "aliases": []}]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "llm.jsonl"
            inner = FakeLLM(reply)
            config = replace(EngineConfig(), learn_facts=False)
            prepared = build(conversation, FakeEmbedder(), consolidator=Consolidator(DiskLLM(inner, path), PROFILE, config))
            self.assertIn("S1", prepared.refs.values())
            [episode] = prepared.store.get_episodes([1])
            self.assertEqual(episode.title, "Support group and painting")
            self.assertEqual([e.name for e in prepared.store.match_entities(["caroline"])], ["Caroline"])
            calls = len(inner.prompts)
            build(conversation, FakeEmbedder(), consolidator=Consolidator(DiskLLM(inner, path), PROFILE, config))
            self.assertEqual(len(inner.prompts), calls)                  # replayed from the file
            build(conversation, FakeEmbedder(), consolidator=Consolidator(DiskLLM(inner, path, 0.0), PROFILE, config))
            self.assertEqual(len(inner.prompts), 2 * calls)              # another temperature: asked again


class BuildAllTests(unittest.TestCase):
    def test_side_by_side_builds_the_same_memory(self):
        conversations = parse(DATA + [{**DATA[0], "sample_id": "conv-2"}])
        serial = build_all(conversations, FakeEmbedder())
        parallel = build_all(conversations, FakeEmbedder(), workers=2)
        self.assertEqual(list(parallel), ["conv-1", "conv-2"])
        self.assertEqual({k: sorted(p.refs.values()) for k, p in serial.items()},
                         {k: sorted(p.refs.values()) for k, p in parallel.items()})


class DiskEmbedderTests(unittest.TestCase):
    def test_vectors_are_embedded_once_and_kept_in_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.pkl"
            inner = FakeEmbedder()
            first = DiskEmbedder(inner, path)
            [a, b] = first.embed(["pots", "pots and pans"])
            first.save()
            again = DiskEmbedder(inner, path)
            [a2] = again.embed(["pots"])
            self.assertEqual(sum(map(len, inner.calls)), 2)    # "pots" not embedded again
            self.assertEqual(a2.text_hash, a.text_hash)
            self.assertAlmostEqual(sum(x * y for x, y in zip(a.vector, a2.vector)),
                                   sum(x * x for x in a.vector), places=4)


class CommandLineTests(unittest.TestCase):
    def test_retrieval_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "locomo10.json"
            path.write_text(json.dumps(DATA), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["--fake-embedder", "--data", str(path), "--set", "messages.max_items=3"]), 0)
            text = out.getvalue()
            self.assertIn("1 conversation(s), 2 questions", text)
            self.assertIn("single-hop", text)
            self.assertIn("all", text)

    def test_settings_keep_their_type(self):
        config = with_settings(EngineConfig(), ["messages.max_items=10", "messages.min_score=0.5"])
        self.assertEqual(config.messages.max_items, 10)
        self.assertIsInstance(config.messages.max_items, int)
        self.assertEqual(config.messages.min_score, 0.5)
        switches = with_settings(EngineConfig(), ["entity_block.enabled=false", "entity_block.min_score_offset=-0.15"])
        self.assertIs(switches.entity_block.enabled, False)
        self.assertEqual(switches.entity_block.min_score_offset, -0.15)
        self.assertIs(with_settings(EngineConfig(), ["entity_block.enabled=true"]).entity_block.enabled, True)
        with self.assertRaises(ValueError):
            with_settings(EngineConfig(), ["entity_block.enabled=maybe"])


if __name__ == "__main__":
    unittest.main()
