"""Secret scrubbing.

Runs on every block of text before it is persisted to memory or the KB
(plan §8/§16). The goal is defense-in-depth, not a hardened redactor: catch the
common shapes (Z.AI keys, generic ``key=``/``password=``/``Bearer`` tokens) so a
slip in a tool result never silently lands in a durable file.
"""

from __future__ import annotations

import re

# Z.AI key shape: 32 hex . alphanumeric tail.
_ZAI_KEY = re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_]{6,}\b")
# sk-... style keys (OpenAI / compatible).
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
# key=..., password=..., api_key=..., bearer ...
_KV = re.compile(
    r"(?i)(api[_-]?key|password|secret|token|bearer)\s*[:=]\s*[^\s,;'\"]+"
)
# AWS access keys (AKIA...) and long AWS secrets.
_AWS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
# Long hex runs that look like tokens (>=32 hex chars).
_HEX = re.compile(r"\b[0-9a-f]{40,}\b")

_RULES = (_ZAI_KEY, _SK_KEY, _KV, _AWS, _HEX)


def scrub(text: str) -> str:
    """Replace likely-secret substrings with ``[REDACTED]``.

    Idempotent: running it again on its own output is a no-op.
    """
    if not text:
        return text
    out = text
    for rule in _RULES:
        out = rule.sub("[REDACTED]", out)
    return out


def contains_secret(text: str) -> bool:
    """True if any scrubbing rule matches ``text`` (used by approval cards)."""
    return any(rule.search(text) for rule in _RULES)
