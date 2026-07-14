"""Z.AI provider.

Z.AI is OpenAI-compatible, so this provider reuses the **gateway** we already
built (``gateway/`` at the project root) for its ChatOpenAI construction, and
adds the OpenAI SDK client on top for tool-calling + streaming. This keeps a
single source of truth for "how to talk to Z.AI" while exposing the richer
interface (tools, streaming) the agent loop needs.

Per plan §12: a drop-in transport that normalizes into our internal
Message/ToolCall/ProviderResponse shape.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from skynet.core import Message, ProviderResponse, ToolCall
from skynet.providers.base import LLMProvider


class ZaiProvider(LLMProvider):
    name = "zai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "glm-5.2",
        base_url: str = "https://api.z.ai/api/paas/v4/",
        context_length: int = 128_000,
    ) -> None:
        # Resolve the key the same way the gateway does, for consistency.
        self.api_key = api_key or os.environ.get("ZAI_API_KEY") or os.environ.get("ZAI_KEY")
        if not self.api_key:
            raise ValueError(
                "ZaiProvider needs an API key. Set ZAI_API_KEY or pass api_key=."
            )
        self.model = model
        self.base_url = base_url
        self._ctx_len = context_length
        # Lazy import so the rest of the package works without openai installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def context_length(self) -> int:
        return self._ctx_len

    def _to_openai_messages(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            d = m.to_dict()
            # OpenAI rejects empty assistant content when tool_calls is absent.
            if m.role == "assistant" and not m.content and not m.tool_calls:
                d["content"] = ""
            out.append(d)
        return out

    def chat(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        kwargs = dict(
            model=self.model,
            messages=self._to_openai_messages(messages),
        )
        if tools:
            kwargs["tools"] = tools
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        raw_calls = [c.model_dump() for c in (msg.tool_calls or [])]
        calls = self.parse_tool_calls(raw_calls)
        assistant = Message(
            role="assistant",
            content=msg.content or "",
            tool_calls=calls,
        )
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        finish = choice.finish_reason or "stop"
        return ProviderResponse(
            message=assistant,
            finish_reason=finish,
            input_tokens=in_tok,
            output_tokens=out_tok,
            wants_tools=bool(calls),
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        # Streaming is used only for the final plain-text reply, so tools are
        # omitted from the streaming call intentionally.
        kwargs = dict(
            model=self.model,
            messages=self._to_openai_messages(messages),
            stream=True,
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError):
                continue
            # glm reasoning content is exposed as reasoning_content; skip it for
            # the visible stream unless we explicitly enable thinking later.
            text = getattr(delta, "content", None)
            if text:
                yield text
