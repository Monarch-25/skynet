"""Activity feed — the live "what / why" stream (plan §13)."""

from __future__ import annotations

from textual.widgets import RichLog


class ActivityFeed(RichLog):
    """A read-only scrolling log of activity events.

    Each event renders as one line: icon + summary, with the ``why`` in muted
    text on the same line so the model's stated reason is always visible (not
    buried, per the plan).
    """

    DEFAULT_CSS = """
    ActivityFeed {
        border: round $accent;
        background: $surface-darken-1;
        padding: 0 1;
        max-width: 42;
        min-width: 30;
    }
    """

    ICONS = {
        "turn_start": "▶",
        "tool_call": "🔍",
        "tool_result": "↳",
        "approval_request": "✋",
        "approval_decision": "✓",
        "compaction": "♻",
        "stream_token": "💬",
        "turn_end": "■",
        "error": "✗",
        "info": "ℹ",
    }

    def __init__(self, *args, **kwargs) -> None:
        # RichLog doesn't accept markup/highlight as constructor args in all
        # versions; set them via attributes where supported, and forward widget
        # kwargs (id, classes, etc.) to the base class.
        super().__init__(*args, **kwargs)
        self.markup = True
        self.auto_scroll = True

    def emit_event(self, type_: str, summary: str, why: str = "") -> None:
        icon = self.ICONS.get(type_, "•")
        line = f"{icon} [b]{summary}[/b]"
        if why:
            line += f"  [dim]({why})[/dim]"
        self.write(line)
