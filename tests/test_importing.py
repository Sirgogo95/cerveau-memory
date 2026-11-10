"""Importing a chat history: the ChatGPT export reader (adapters/chats/chatgpt.py) on a
small made-up export, and the replay into memory (brain/importing.py)."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from adapters.chats import read_chatgpt
from adapters.chats.chatgpt import continued_branch, read_conversation
from adapters.stores.in_memory import InMemoryStore
from brain.config import EngineConfig
from brain.importing import ChatLine, ImportedChat, import_chat, merge_lines
from brain.types import Role
from tests.fakes import FakeEmbedder

T0 = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)


def ts(minutes: float) -> float:
    return (T0 + timedelta(minutes=minutes)).timestamp()


def node(id_, parent, role, parts, minutes, content_type="text", **extra):
    message = {"id": id_, "author": {"role": role, "name": None}, "create_time": minutes if minutes is None else ts(minutes),
               "content": {"content_type": content_type, "parts": parts}, "metadata": extra.pop("metadata", {})}
    message.update(extra)
    return {"id": id_, "parent": parent, "message": message}


def conversation(cid, nodes, current, **extra):
    return {"conversation_id": cid, "title": f"Chat {cid}", "create_time": ts(0), "update_time": ts(60),
            "current_node": current, "mapping": {n["id"]: n for n in nodes}, **extra}


# A made-up export: a regenerated reply (two branches), the model's reasoning, a reply
# split in two nodes, an image, a file, a voice message, and a millisecond timestamp.
CHAT = conversation("c1", [
    {"id": "root", "parent": None, "message": None},
    node("sys", "root", "system", [""], 0),
    node("u1", "sys", "user", ["Which database for the Naco app?"], 1),
    node("a1-old", "u1", "assistant", ["An abandoned answer."], 2),
    node("think", "u1", "assistant", ["thinking..."], 2, content_type="thoughts"),
    node("a1", "think", "assistant", ["Postgres,"], 3),
    node("a1b", "a1", "assistant", ["with pgvector."], 3.5),
    node("u2", "a1b", "user", ["Here is the schema", {"content_type": "image_asset_pointer"}], 5,
         content_type="multimodal_text",
         metadata={"attachments": [{"name": "shot.png", "mime_type": "image/png"},
                                   {"name": "schema.sql", "mime_type": "text/plain"}]}),
    node("u3", "u2", "user", [{"content_type": "audio_transcription", "text": "Also add tags."}], 6,
         content_type="multimodal_text"),
    node("a3", "u3", "assistant", ["Done."], None),
    node("u4", "a3", "user", ["Thanks"], 0),
], current="u4")
CHAT["mapping"]["u4"]["message"]["create_time"] = ts(8) * 1000      # milliseconds, seen once


class ReaderTests(unittest.TestCase):
    def test_only_the_continued_branch_is_kept(self):
        ids = [m["id"] for m in continued_branch(CHAT)]
        self.assertEqual(ids, ["sys", "u1", "think", "a1", "a1b", "u2", "u3", "a3", "u4"])

    def test_reasoning_and_system_messages_are_skipped_and_parts_merged(self):
        chat = read_conversation(CHAT)
        assert chat is not None
        roles = [(line.role, line.external_ids) for line in chat.lines]
        self.assertEqual(roles, [(Role.USER, ("u1",)), (Role.ASSISTANT, ("a1", "a1b")),
                                 (Role.USER, ("u2", "u3")), (Role.ASSISTANT, ("a3",)), (Role.USER, ("u4",))])
        self.assertEqual(chat.lines[1].text, "Postgres,\n\nwith pgvector.")

    def test_images_files_and_voice_become_text(self):
        chat = read_conversation(CHAT)
        assert chat is not None
        self.assertEqual(chat.lines[2].text,
                         "Here is the schema\n[shared an image]\n[shared a file: schema.sql]\n\nAlso add tags.")

    def test_times_in_seconds_or_milliseconds_and_a_missing_one(self):
        chat = read_conversation(CHAT)
        assert chat is not None
        self.assertEqual(chat.lines[0].at, T0 + timedelta(minutes=1))
        self.assertEqual(chat.lines[3].at, chat.lines[2].at + timedelta(minutes=1))   # missing: the one before
        self.assertEqual(chat.lines[4].at, T0 + timedelta(minutes=8))                 # milliseconds

    def test_the_export_folder_split_or_not_and_the_do_not_remember_flag(self):
        later = conversation("c2", [{"id": "r", "parent": None, "message": None},
                                    node("x", "r", "user", ["Later chat"], 600)], current="x", is_do_not_remember=True)
        empty = conversation("c3", [{"id": "r", "parent": None, "message": None}], current="r")
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "conversations-000.json").write_text(json.dumps([later, empty]), encoding="utf-8")
            Path(folder, "conversations-001.json").write_text(json.dumps([CHAT]), encoding="utf-8")
            chats = read_chatgpt(folder)
        self.assertEqual([c.conversation_id for c in chats], ["c1", "c2"])            # by first message; empty dropped
        self.assertEqual([c.do_not_remember for c in chats], [False, True])
        self.assertEqual(chats[0].source, "chatgpt")


def line(role, text, minutes, *ids):
    return ChatLine(role, text, T0 + timedelta(minutes=minutes), ids or (text,))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.store, self.embedder = InMemoryStore(), FakeEmbedder()
        self.config = replace(EngineConfig(), drift_threshold=0)      # drift tested on its own below

    def run_import(self, chat, closed=None):
        return import_chat(chat, store=self.store, embedder=self.embedder, config=self.config,
                           on_close=(closed.append if closed is not None else None))

    def test_a_chat_is_stored_with_its_times_and_closed(self):
        chat = ImportedChat("chatgpt", "c1", "t", (line(Role.USER, "hola", 0, "u1"),
                                                   line(Role.ASSISTANT, "hello", 1, "a1", "a1b")))
        closed: list[int] = []
        mapping = self.run_import(chat, closed)
        [episode] = self.store.get_episodes(closed)
        messages = self.store.get_messages(episode.id)
        self.assertEqual([(m.role, m.content, m.created_at) for m in messages],
                         [(Role.USER, "hola", T0), (Role.ASSISTANT, "hello", T0 + timedelta(minutes=1))])
        self.assertEqual(episode.ended_at, T0 + timedelta(minutes=1))
        self.assertIsNone(self.store.current_episode())
        self.assertEqual(mapping, {"u1": messages[0].id, "a1": messages[1].id, "a1b": messages[1].id})
        self.assertEqual(self.store.imported_messages("chatgpt", "c1"), mapping)
        self.assertEqual(self.store.imported_conversations("chatgpt"), {"c1"})

    def test_each_chat_starts_a_new_episode_and_a_long_gap_splits_one(self):
        first = ImportedChat("chatgpt", "c1", "t", (line(Role.USER, "a", 0), line(Role.ASSISTANT, "b", 1)))
        gap = self.config.episode_gap + timedelta(minutes=5)
        second = ImportedChat("chatgpt", "c2", "t", (
            line(Role.USER, "c", 2), line(Role.ASSISTANT, "d", 3),
            line(Role.USER, "e", 3 + gap.total_seconds() / 60), line(Role.ASSISTANT, "f", 4 + gap.total_seconds() / 60)))
        closed: list[int] = []
        self.run_import(first, closed)
        self.run_import(second, closed)
        contents = [[m.content for m in self.store.get_messages(e)] for e in closed]
        self.assertEqual(contents, [["a", "b"], ["c", "d"], ["e", "f"]])

    def test_a_change_of_subject_splits_as_it_would_live(self):
        config = EngineConfig()          # drift on, as in the app
        lines = [line(Role.USER, f"same subject {i}", i * 2) for i in range(config.drift_min_messages + 1)]
        chat = ImportedChat("chatgpt", "c1", "t", tuple(lines))
        closed: list[int] = []
        import_chat(chat, store=self.store, embedder=self.embedder, config=config, on_close=closed.append)
        self.assertEqual(sum(len(self.store.get_messages(e)) for e in closed), len(lines))   # nothing lost

    def test_consecutive_lines_of_one_role_merge(self):
        merged = merge_lines([line(Role.ASSISTANT, "one", 0, "x"), line(Role.ASSISTANT, "two", 1, "y"),
                              line(Role.USER, "three", 2, "z")])
        self.assertEqual([(m.text, m.external_ids) for m in merged], [("one\n\ntwo", ("x", "y")), ("three", ("z",))])


if __name__ == "__main__":
    unittest.main()
