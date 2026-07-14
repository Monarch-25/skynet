"""Session compaction (plan §8).

Adapted in spirit from the Hermes context-compressor pattern and reimplemented
fresh. v1 implements the cheap pre-pass (collapse old tool results to one-liners)
plus a heuristic summary builder. A real LLM-backed summarization call slots in
behind ``summarize()`` without changing the call sites.

Public surface:
    - ``should_compress(messages, threshold_tokens)`` -> bool
    - ``pre_compress(messages)`` -> messages with old tool results collapsed
    - ``summarize(messages, focus)`` -> str  (heuristic; LLM hook TODO)
"""

from __future__ import annotations

import re

from skynet.core.tokens import estimate_messages_tokens
from skynet.core.types import Message


def should_compress(messages: list[Message], threshold_tokens: int) -> bool:
    return estimate_messages_tokens(messages) > threshold_tokens


def pre_compress(messages: list[Message]) -> list[Message]:
    """Replace old tool results with one-line descriptions.

    Keeps the most recent tool result verbatim (the active one) and collapses
    everything older. This alone often defers the need for a real summary call.
    """
    out: list[Message] = []
    last_tool_idx = max(
        (i for i, m in enumerate(messages) if m.role == "tool"),
        default=-1,
    )
    for i, m in enumerate(messages):
        if m.role == "tool" and i < last_tool_idx:
            out.append(Message(
                role="tool",
                content=_summarize_tool_result(m.content or "", m.name or "tool"),
                tool_call_id=m.tool_call_id,
                name=m.name,
            ))
        else:
            out.append(m)
    return out


def summarize(messages: list[Message], focus: str | None = None) -> str:
    """Heuristic summary of the session so far.

    A real implementation makes one bounded LLM call with the structured template
    from plan §8 (Active Task / Goal / Completed Actions / etc.). For v1 we
    extract the verbatim last user message as the "Active Task" and list the
    distinct tools used — enough to keep the thread alive across compaction.
    """
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user" and m.content),
        "",
    )
    tools_used: list[str] = []
    for m in messages:
        if m.role == "assistant":
            for c in m.tool_calls:
                if c.name not in tools_used:
                    tools_used.append(c.name)
    parts = ["# Session summary (auto-compacted)"]
    if focus:
        parts.append(f"\nFocus: {focus}")
    parts.append(f"\n## Active task\n{last_user.strip() or '(none)'}")
    if tools_used:
        parts.append("\n## Tools used this session\n" + "\n".join(f"- {t}" for t in tools_used))
    return "\n".join(parts)


_TOOL_NAME_RE = re.compile(r"\[(?P<name>[a-z_]+)\]")


def _summarize_tool_result(content: str, fallback_name: str) -> str:
    """Reduce a verbose tool result to a one-liner.

    For ``source_search``-style results (which we render with [name] headers per
    chunk), we report how many chunks came back; otherwise we keep the first
    line and truncate.
    """
    name = fallback_name
    m = _TOOL_NAME_RE.search(content)
    if m:
        name = m.group("name")
    chunk_count = content.count("[source]") + content.count("[kb_note]")
    if chunk_count:
        return f"[{name}] returned {chunk_count} chunk(s); details omitted by compaction"
    first = content.strip().splitlines()[0] if content.strip() else "(empty)"
    if len(first) > 160:
        first = first[:157] + "..."
    return f"[{name}] {first}"
