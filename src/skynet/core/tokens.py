"""Rough token estimation.

v1 uses a char/4 heuristic. The estimate is deliberately conservative and
swappable — every consumer goes through ``estimate_tokens`` rather than dividing
by a constant, so a tiktoken-backed estimator can replace this without touching
anything else. (Plan §2: ``tokens.py``.)
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Return a cheap, slightly-generous token estimate for ``text``."""
    if not text:
        return 0
    # char/4 is the standard rough heuristic; round up so thresholds trip early.
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages) -> int:
    """Sum token estimates across a list of messages.

    Accepts both ``list[dict]`` (raw provider shape) and ``list[Message]``
    (our internal type) so callers don't need to convert. Adds a small
    per-message overhead to approximate the role/format tokens the provider
    adds server-side, so context-window accounting isn't optimistically low.
    """
    total = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content") or ""
        else:
            content = getattr(m, "content", "") or ""
        if isinstance(content, list):  # structured content blocks
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        total += estimate_tokens(str(content)) + 4  # role/format overhead
    return total
