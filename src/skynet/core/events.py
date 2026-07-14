"""Activity events.

A single lightweight event shape flows from the turn loop out to the TUI's
activity feed (and the structured log). Keeping it one type makes the TUI
binding trivial and makes the JSONL audit log self-describing.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

EventType = Literal[
    "turn_start", "tool_call", "tool_result", "approval_request",
    "approval_decision", "compaction", "stream_token", "turn_end", "error", "info",
]


@dataclass
class ActivityEvent:
    type: EventType
    summary: str                 # one line, for the feed
    detail: str = ""             # longer, for the log / expandable view
    why: str = ""                # the model's stated reason, if any
    ts: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ActivityLog:
    """Append-only structured log writer (plan §2: ``logs/activity.jsonl``)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: ActivityEvent) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
