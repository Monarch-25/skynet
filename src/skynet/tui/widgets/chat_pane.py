"""Chat pane — the scrollable transcript of user/assistant turns."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ChatPane(VerticalScroll):
    """A scrolling transcript. ``add_turn`` appends a styled message."""

    DEFAULT_CSS = """
    ChatPane {
        background: $surface;
        border: round $primary;
        padding: 0 1;
    }
    ChatPane > .chat-user {
        color: $text;
        background: $boost;
        padding: 0 1;
        margin: 1 0 0 0;
        border-left: outer $accent;
    }
    ChatPane > .chat-assistant {
        color: $text;
        background: $panel;
        padding: 0 1;
        margin: 1 0 0 0;
        border-left: outer $success;
    }
    ChatPane > .chat-system {
        color: $text-muted;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def add_turn(self, role: str, text: str) -> Static:
        cls = {
            "user": "chat-user",
            "assistant": "chat-assistant",
            "system": "chat-system",
        }.get(role, "chat-system")
        label = {"user": "you", "assistant": "skynet", "system": "system"}.get(role, role)
        widget = Static(f"[b]{label}[/b]\n{text}", classes=cls)
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def append_to_last(self, widget: Static, extra: str) -> None:
        """Used while streaming: append tokens to the in-progress assistant bubble."""
        # Re-render with accumulated text stored on the widget.
        prev = getattr(widget, "_accumulated", "")
        new = prev + extra
        widget._accumulated = new  # type: ignore[attr-defined]
        label = "skynet"
        widget.update(f"[b]{label}[/b]\n{new}")
        self.scroll_end(animate=False)
