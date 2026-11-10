"""Test doubles for the engine's Protocols: a scripted LLM and a deterministic
embedder. No network, no model downloads."""

import math
import re
import zlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from brain.hashing import text_hash
from brain.ports import ChatProviderError
from brain.types import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    Embedding,
    ToolCall,
    ToolDefinition,
    Usage,
)


class FakeLLM:
    """Answers with a scripted reply and records every prompt (and schema) it gets.

    `replies` scripts successive chat() calls (the last one repeats); `reply` is the
    single answer otherwise. stream() streams `reply` word by word."""

    model = "fake"

    def __init__(
        self,
        reply: str = "Hello there.",
        fail_after: int | None = None,
        replies: Sequence[str] = (),
        tool_calls: Sequence[Sequence[ToolCall]] = (),
        stream_replies: Sequence[str] = (),
    ):
        self.reply = reply
        self.fail_after = fail_after
        self.replies = list(replies)
        # Scripted tool calls: each stream() call takes the next entry; a non-empty
        # one is yielded as a tool-call chunk instead of the text reply.
        self.tool_calls = [tuple(c) for c in tool_calls]
        # Scripted streamed replies, one per stream() call; then `reply`.
        self.stream_replies = list(stream_replies)
        self.prompts: list[list[ChatMessage]] = []
        self.schemas: list[Mapping[str, Any] | None] = []
        self.tools: list[tuple[ToolDefinition, ...]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatResponse:
        self.prompts.append(list(messages))
        self.schemas.append(schema)
        if self.replies:
            content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        else:
            content = self.reply
        return ChatResponse(content=content, model=self.model)

    def stream(
        self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolDefinition] = ()
    ) -> Iterator[ChatChunk]:
        self.prompts.append(list(messages))
        self.tools.append(tuple(tools))
        calls = self.tool_calls.pop(0) if self.tool_calls else ()
        if calls and tools:
            yield ChatChunk("", tool_calls=calls)
            yield ChatChunk("", usage=Usage(input_tokens=10, output_tokens=5))
            return
        words = (self.stream_replies.pop(0) if self.stream_replies else self.reply).split(" ")
        for i, word in enumerate(words):
            if self.fail_after is not None and i == self.fail_after:
                raise ChatProviderError("connection dropped")
            yield ChatChunk(word if i == 0 else " " + word)
        yield ChatChunk("", usage=Usage(input_tokens=10, output_tokens=len(words)))


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get similar vectors, so semantic
    search behaves sensibly in tests. Deterministic across runs (crc32, not hash())."""

    dimensions = 64

    def __init__(self, model: str = "fake-embedder"):
        self.model = model
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> list[float]:
        v = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        return [Embedding(self.vector(t), self.model, text_hash(t)) for t in texts]
