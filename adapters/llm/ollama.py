"""ChatProvider for a local Ollama server, over its HTTP API (/api/chat).

Uses `requests` rather than the `ollama` package: the API is two JSON shapes, and
this keeps the dependency list short.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import requests

from brain.ports import ChatProviderError
from brain.types import ChatChunk, ChatMessage, ChatResponse, ToolCall, ToolDefinition, Usage

# 127.0.0.1 rather than localhost: on Windows, a new connection to "localhost" tries
# IPv6 first and waits about 2 s before falling back.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


SCHEMA_PROMPT = "Reply with only a JSON object matching this JSON Schema, nothing else:"


def is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def json_in(text: str) -> str:
    """The JSON object in a reply that may wrap it in prose or a code fence, as
    compact JSON; the reply unchanged if there is none to read."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end < start:
        return text
    try:
        return json.dumps(json.loads(body[start:end + 1]), ensure_ascii=False)
    except json.JSONDecodeError:
        return text


class OllamaProvider:
    """Talks to one model on an Ollama server.

    `options` is passed through as Ollama's model options (temperature, num_ctx,
    ...). Set `num_ctx` explicitly: Ollama's default context window is small, and it
    silently drops the start of longer prompts instead of failing. It should be at
    least the ModelProfile's memory budget plus its output reserve.

    `keep_alive` controls how long the model stays loaded after a call ("5m", "1h",
    or -1 for forever); None leaves the server default.

    `think` turns a reasoning model's thinking on or off (None: the model's default,
    which is on for reasoning models). The reasoning text itself is not returned;
    only the answer is.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        timeout: float = 300.0,
        session: requests.Session | None = None,
        api_key: str | None = None,
        schema_in_prompt: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = dict(options or {})
        self.keep_alive = keep_alive
        self.think = think
        self.timeout = timeout
        self.session = session or requests.Session()
        if api_key:  # Ollama Cloud; never sent to a local server (see the factory)
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        # Ollama Cloud ignores "format" (measured 2026-10-18 on six models): the schema
        # then goes into the prompt, and the JSON is taken out of the reply.
        self.schema_in_prompt = schema_in_prompt

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatResponse:
        # Ollama constrains the output to a JSON Schema passed as "format".
        if schema is not None and self.schema_in_prompt and messages:
            last = messages[-1]
            messages = [*messages[:-1], ChatMessage(last.role, f"{last.content}\n\n{SCHEMA_PROMPT}\n{json.dumps(dict(schema))}")]
        data = self._post(messages, stream=False, schema=schema, tools=tools).json()
        self._raise_if_error(data)
        content = data["message"]["content"]
        if schema is not None and self.schema_in_prompt:
            content = json_in(content)
            if not is_json(content):   # an empty or unreadable reply: ask once more
                data = self._post(messages, stream=False, schema=schema, tools=tools).json()
                self._raise_if_error(data)
                content = json_in(data["message"]["content"])
        return ChatResponse(
            content=content,
            model=data.get("model", self.model),
            usage=self._usage(data),
            tool_calls=_tool_calls(data["message"]),
        )

    def load(self) -> None:
        """Load the model into memory without generating anything, so the first real
        request doesn't wait for it. Sends the same options as real requests: Ollama
        reloads a model when asked for a different context size."""
        self._raise_if_error(self._post([], stream=False).json())

    def stream(
        self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolDefinition] = ()
    ) -> Iterator[ChatChunk]:
        # Ollama streams newline-delimited JSON objects; the last one has done=true
        # and carries the token counts. Tool calls arrive whole, in one chunk.
        with self._post(messages, stream=True, tools=tools) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                self._raise_if_error(data)
                message = data.get("message", {})
                content = message.get("content", "")
                calls = _tool_calls(message)
                if data.get("done"):
                    yield ChatChunk(content=content, usage=self._usage(data), tool_calls=calls)
                    return
                if content or calls:
                    yield ChatChunk(content=content, tool_calls=calls)

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message(m) for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": dict(t.parameters),
                }}
                for t in tools
            ]
        if self.options:
            payload["options"] = self.options
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think
        if schema is not None:
            payload["format"] = dict(schema)
        return payload

    def _post(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> requests.Response:
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=stream, schema=schema, tools=tools),
                stream=stream,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ChatProviderError(
                f"Ollama is unreachable at {self.base_url}: {exc}"
            ) from exc
        if not response.ok:
            # Ollama reports failures (e.g. an unknown model) as {"error": "..."}.
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            response.close()
            raise ChatProviderError(f"Ollama returned {response.status_code}: {detail}")
        return response

    @staticmethod
    def _raise_if_error(data: dict) -> None:
        if "error" in data:
            raise ChatProviderError(f"Ollama error: {data['error']}")

    @staticmethod
    def _usage(data: dict) -> Usage | None:
        if "prompt_eval_count" not in data and "eval_count" not in data:
            return None
        return Usage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def _message(m: ChatMessage) -> dict[str, Any]:
    """One prompt message in Ollama's shape, with tool calls and tool results."""
    out: dict[str, Any] = {"role": str(m.role), "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in m.tool_calls
        ]
    if m.tool_name:
        out["tool_name"] = m.tool_name
    return out


def _tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    calls = []
    for call in message.get("tool_calls") or ():
        function = call.get("function", {})
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):  # some models send the arguments as a JSON string
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(ToolCall(name=str(function.get("name", "")), arguments=arguments))
    return tuple(calls)
