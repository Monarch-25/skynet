"""Shared message and tool-call types.

Normalized shapes that providers map into and the turn loop consumes. Keeping
these provider-agnostic is what lets ``core/loop.py`` never branch on provider
identity (plan §12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str | None = None
    # For assistant messages that requested tool calls.
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # For tool-result messages: which call this answers.
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as providers expect
    # Skynet-specific: one short sentence the model fills in explaining *why*.
    why: str | None = None


@dataclass
class ProviderResponse:
    """Provider output normalized to what the turn loop needs."""

    message: Message
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    # Whether the model wants to call tools this turn.
    wants_tools: bool = False


ToolKind = Literal["read", "write", "control"]


@dataclass
class ToolResult:
    """Result of executing one tool call."""

    call_id: str
    name: str
    ok: bool
    content: str
    # write tools surface a preview/diff the TUI uses in approval cards.
    preview: str | None = None
    # read tools execute immediately; write tools block on approval first.
    kind: ToolKind = "read"
    # A short label for the activity feed even when the model omitted ``why``.
    summary: str = ""
