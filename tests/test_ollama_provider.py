"""OllamaProvider against a fake HTTP session: no Ollama server or Django needed.

Run with: python -m unittest discover -s tests -t .
"""

import json
import unittest

import requests

from adapters.llm.ollama import OllamaProvider
from brain.ports import ChatProvider, ChatProviderError
from brain.types import ChatChunk, ChatMessage, Role, ToolCall, ToolDefinition, Usage


class FakeResponse:
    def __init__(self, status_code=200, body=None, lines=()):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body
        self._lines = lines
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode() if isinstance(line, dict) else line

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession(requests.Session):
    """Records each POST and returns a canned response instead of using the network."""

    def __init__(self, response=None, error=None):
        super().__init__()
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


MESSAGES = [
    ChatMessage(Role.SYSTEM, "You are Cerveau."),
    ChatMessage(Role.USER, "Hello"),
]


class OllamaProviderTests(unittest.TestCase):
    def test_satisfies_chat_provider_protocol(self):
        self.assertIsInstance(OllamaProvider("llama3.2"), ChatProvider)

    def test_chat_sends_payload_and_parses_reply(self):
        session = FakeSession(
            FakeResponse(
                body={
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Hi!"},
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                }
            )
        )
        provider = OllamaProvider(
            "llama3.2",
            base_url="http://ollama:11434/",
            options={"num_ctx": 8192},
            keep_alive="10m",
            session=session,
        )

        response = provider.chat(MESSAGES)

        self.assertEqual(response.content, "Hi!")
        self.assertEqual(response.model, "llama3.2")
        self.assertEqual(response.usage, Usage(input_tokens=12, output_tokens=3))
        url, kwargs = session.calls[0]
        self.assertEqual(url, "http://ollama:11434/api/chat")
        self.assertEqual(
            kwargs["json"],
            {
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": "You are Cerveau."},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
                "options": {"num_ctx": 8192},
                "keep_alive": "10m",
            },
        )

    def test_stream_yields_chunks_with_usage_on_last(self):
        lines = [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            b"",
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 12,
                "eval_count": 2,
            },
        ]
        session = FakeSession(FakeResponse(lines=lines))
        provider = OllamaProvider("llama3.2", session=session)

        chunks = list(provider.stream(MESSAGES))

        self.assertEqual(
            chunks,
            [
                ChatChunk("Hel"),
                ChatChunk("lo"),
                ChatChunk("", usage=Usage(input_tokens=12, output_tokens=2)),
            ],
        )
        self.assertTrue(session.calls[0][1]["stream"])
        self.assertNotIn("options", session.calls[0][1]["json"])

    def test_tools_are_sent_and_tool_calls_parsed_from_the_stream(self):
        lines = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "search_memory", "arguments": {"query": "portal"}}},
                {"function": {"name": "read_memory", "arguments": '{"id": "E3"}'}},
            ]}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        session = FakeSession(FakeResponse(lines=lines))
        tool = ToolDefinition("search_memory", "Search.", {"type": "object", "properties": {}})

        chunks = list(OllamaProvider("qwen", session=session).stream(MESSAGES, tools=[tool]))

        self.assertEqual(chunks[0].tool_calls, (
            ToolCall("search_memory", {"query": "portal"}),
            ToolCall("read_memory", {"id": "E3"}),  # arguments sent as a JSON string
        ))
        self.assertEqual(session.calls[0][1]["json"]["tools"], [{
            "type": "function",
            "function": {"name": "search_memory", "description": "Search.",
                         "parameters": {"type": "object", "properties": {}}},
        }])

    def test_tool_calls_and_results_are_sent_back_in_ollama_shape(self):
        session = FakeSession(FakeResponse(body={
            "message": {"role": "assistant", "content": "Done."}, "done": True,
        }))
        history = [
            *MESSAGES,
            ChatMessage(Role.ASSISTANT, "", tool_calls=(ToolCall("search_memory", {"query": "q"}),)),
            ChatMessage(Role.TOOL, '{"results": []}', tool_name="search_memory"),
        ]

        response = OllamaProvider("qwen", session=session).chat(history)

        self.assertEqual(response.tool_calls, ())
        sent = session.calls[0][1]["json"]["messages"]
        self.assertEqual(sent[2], {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_memory", "arguments": {"query": "q"}}},
        ]})
        self.assertEqual(sent[3], {"role": "tool", "content": '{"results": []}', "tool_name": "search_memory"})
        self.assertNotIn("tools", session.calls[0][1]["json"])

    def test_http_error_raises_with_ollama_message(self):
        session = FakeSession(
            FakeResponse(status_code=404, body={"error": "model 'nope' not found"})
        )
        provider = OllamaProvider("nope", session=session)

        with self.assertRaisesRegex(ChatProviderError, "model 'nope' not found"):
            provider.chat(MESSAGES)

    def test_think_is_sent_only_when_set(self):
        for think, expected in ((None, None), (False, False), (True, True)):
            session = FakeSession(FakeResponse(body={"message": {"content": "ok"}}))
            OllamaProvider("qwen3.5:4b", think=think, session=session).chat(MESSAGES)
            self.assertEqual(session.calls[0][1]["json"].get("think"), expected)

    def test_load_sends_no_messages_with_the_same_options(self):
        session = FakeSession(FakeResponse(body={"done": True, "done_reason": "load"}))
        OllamaProvider(
            "qwen3.5:4b", options={"num_ctx": 16384}, think=False, session=session
        ).load()
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["options"], {"num_ctx": 16384})
        self.assertIs(payload["think"], False)

    def test_load_failure_raises(self):
        session = FakeSession(FakeResponse(status_code=404, body={"error": "model not found"}))
        with self.assertRaisesRegex(ChatProviderError, "model not found"):
            OllamaProvider("nope", session=session).load()

    def test_error_line_mid_stream_raises(self):
        lines = [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"error": "out of memory"},
        ]
        provider = OllamaProvider("llama3.2", session=FakeSession(FakeResponse(lines=lines)))

        with self.assertRaisesRegex(ChatProviderError, "out of memory"):
            list(provider.stream(MESSAGES))

    def test_unreachable_server_raises(self):
        session = FakeSession(error=requests.ConnectionError("refused"))
        provider = OllamaProvider("llama3.2", session=session)

        with self.assertRaisesRegex(ChatProviderError, "unreachable"):
            provider.chat(MESSAGES)



class SchemaInPromptTests(unittest.TestCase):
    """Ollama Cloud ignores "format": the schema goes into the prompt, and the JSON is
    taken out of the reply (fenced or in prose), asked once more if there is none."""

    def reply(self, content):
        return FakeResponse(body={"model": "m", "message": {"role": "assistant", "content": content}, "done": True})

    def test_the_schema_is_in_the_prompt_and_the_json_comes_out_of_a_fence(self):
        session = FakeSession(self.reply('Sure:\n```json\n{"label": "WRONG"}\n```'))
        provider = OllamaProvider("m", session=session, schema_in_prompt=True)
        response = provider.chat(MESSAGES, schema={"type": "object"})
        self.assertEqual(response.content, '{"label": "WRONG"}')
        last = session.calls[0][1]["json"]["messages"][-1]["content"]
        self.assertIn("JSON Schema", last)
        self.assertIn('{"type": "object"}', last)

    def test_an_unreadable_reply_is_asked_once_more(self):
        replies = iter([self.reply(""), self.reply('{"ok": true}')])
        session = FakeSession()
        session.post = lambda url, **kwargs: (session.calls.append((url, kwargs)), next(replies))[1]
        provider = OllamaProvider("m", session=session, schema_in_prompt=True)
        self.assertEqual(provider.chat(MESSAGES, schema={"type": "object"}).content, '{"ok": true}')
        self.assertEqual(len(session.calls), 2)

    def test_local_models_keep_format_and_the_prompt_unchanged(self):
        session = FakeSession(self.reply('{"ok": true}'))
        OllamaProvider("m", session=session).chat(MESSAGES, schema={"type": "object"})
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["messages"][-1]["content"], "Hello")
        self.assertEqual(payload["format"], {"type": "object"})

if __name__ == "__main__":
    unittest.main()
