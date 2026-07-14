"""LLM provider abstraction.

Both ``ZaiProvider`` and ``MockProvider`` satisfy this interface. ``BedrockProvider``
is stubbed for M6. The turn loop depends only on ``LLMProvider`` so provider
identity never leaks into orchestration code (plan §12).
"""

from __future__ import annotations

import abc
from typing import Iterator

from skynet.core import Message, ProviderResponse, ToolCall


class LLMProvider(abc.ABC):
    """A chat model that can call tools and (optionally) stream."""

    name: str = "base"

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Run one model turn. Returns a normalized ProviderResponse."""

    @abc.abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield assistant content deltas as strings (no tool calls mid-stream).

        Tool-calling turns use ``chat``; only the final plain-text reply is
        streamed to the TUI. Providers that can't stream internally should still
        yield the full text in one chunk so callers don't need to branch.
        """

    @abc.abstractmethod
    def context_length(self) -> int:
        """The model's maximum context window, in tokens."""

    def supports_tools(self) -> bool:
        return True

    def supports_parallel_tools(self) -> bool:
        return True

    def parse_tool_calls(self, raw_calls: list[dict]) -> list[ToolCall]:
        """Normalize raw provider tool calls into our ToolCall shape.

        Shared here because OpenAI-compatible providers (Zai, Bedrock via the
        OpenAI-compatible surface) use the same JSON structure.
        """
        calls: list[ToolCall] = []
        for rc in raw_calls or []:
            fn = rc.get("function", {})
            args = fn.get("arguments", "{}") or "{}"
            # Extract the model-supplied "why" if present, then strip it from the
            # arguments so the downstream tool doesn't see it as a real arg.
            why: str | None = None
            import json as _json

            try:
                parsed = _json.loads(args) if isinstance(args, str) else dict(args)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and "why" in parsed:
                why = str(parsed.pop("why"))
                args = _json.dumps(parsed)
            calls.append(
                ToolCall(
                    id=rc.get("id") or f"call_{len(calls)}",
                    name=fn.get("name", ""),
                    arguments=args,
                    why=why,
                )
            )
        return calls
