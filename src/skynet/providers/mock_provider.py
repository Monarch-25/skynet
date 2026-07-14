"""Deterministic mock provider for offline development and tests.

The Z.AI account had no balance (429 / error 1113) at build time, so this mock
lets us verify the *entire* agent loop — tool-calling, approval gating, KB
writes, streaming — without spending tokens. It is fully deterministic: the same
conversation produces the same tool calls and replies.

It is deliberately scriptable. A test sets ``MockProvider.script`` to a list of
"frames"; each frame is the canned response returned for the next ``chat()`` /
``stream()`` call. Frames are consumed in order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

from skynet.core import Message, ProviderResponse, ToolCall
from skynet.providers.base import LLMProvider


@dataclass
class Frame:
    """One canned provider response.

    Either ``text`` (a final reply, possibly streamed) or ``calls`` (a list of
    (tool_name, args_dict, why) tuples). If both are set, calls come first as an
    assistant turn and ``text`` is returned on the *next* call.
    """

    text: str | None = None
    calls: list[tuple[str, dict, str | None]] = field(default_factory=list)
    # ``role`` is unused for now but reserved for richer scenarios.
    role: str = "assistant"


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, context_length: int = 128_000, script: list[Frame] | None = None) -> None:
        self._ctx_len = context_length
        self.script: list[Frame] = list(script or [])
        self._cursor = 0
        # Every call recorded, for assertions in tests.
        self.call_log: list[list[Message]] = []

    # ----- API surface -------------------------------------------------
    def context_length(self) -> int:
        return self._ctx_len

    def chat(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.call_log.append(list(messages))
        frame = self._next_frame()
        if frame and frame.calls:
            calls = [
                ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=json.dumps(args),
                    why=why,
                )
                for i, (name, args, why) in enumerate(frame.calls)
            ]
            return ProviderResponse(
                message=Message(role="assistant", tool_calls=calls, content=""),
                finish_reason="tool_calls",
                wants_tools=True,
            )
        text = (frame.text if frame else None) or "(mock) I have nothing to say."
        return ProviderResponse(
            message=Message(role="assistant", content=text),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self.call_log.append(list(messages))
        frame = self._next_frame()
        text = (frame.text if frame else None) or "(mock streamed reply)"
        # Stream word by word so the TUI streaming code path is exercised.
        for word in text.split():
            yield word + " "

    # ----- internals ---------------------------------------------------
    def _next_frame(self) -> Frame | None:
        if self._cursor < len(self.script):
            f = self.script[self._cursor]
            self._cursor += 1
            return f
        return None


def default_demo_script() -> list[Frame]:
    """A script that exercises search -> read -> reply, for quick demos.

    Frame 1: model decides to search source docs for "attention".
    Frame 2: after seeing results, it answers using the retrieved chunk.
    """
    return [
        Frame(
            calls=[
                (
                    "source_search",
                    {"query": "attention mechanism transformer", "k": 4},
                    "checking the source folder before answering",
                )
            ],
        ),
        Frame(
            text=(
                "Based on the source folder, the Transformer computes attention "
                "via scaled dot-product attention: it takes queries and keys of "
                "dimension d_k, computes their dot products, scales by 1/sqrt(d_k), "
                "and applies softmax to get weights over the values.\n\n"
                "Source: knowledge_source/attention-is-all-you-need.md#3.2.1"
            )
        ),
    ]
