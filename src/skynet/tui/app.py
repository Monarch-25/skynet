"""Skynet TUI app — full Textual application (plan §13).

Layout:
    ┌──────────────── Header (title + status) ─────────────────┐
    │                              │                            │
    │   Chat pane (scrolling)      │   Activity feed (live)     │
    │                              │   ─────────────────────    │
    │                              │   KB tree (toggle ctrl+k)  │
    │                              │                            │
    ├──────────────── Input row ────────────────────────────────┤
    │   > type a message, Enter to send                          │
    └────────────────────────────────────────────────────────────┘

Plus a status bar with user_id / session / model / token usage, and an approval
modal that pops up only for write tool calls.

Commands (slash): /compact  /new  /model <p>  /user <id>  /help.
"""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static, TabbedContent, TabPane

from skynet.core.approval import Approver, ApprovalRequest, ApprovalDecision
from skynet.core.events import ActivityEvent
from skynet.runtime import Runtime, build_runtime
from skynet.config import load_config
from skynet.tui.widgets.activity_feed import ActivityFeed
from skynet.tui.widgets.chat_pane import ChatPane
from skynet.tui.widgets.kb_tree import KbTree


class SkynetApp(App):
    """The full Skynet terminal app."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-row { height: 1fr; }
    #right-col { width: 44; }
    #input-row {
        height: 3;
        border: round $primary;
        padding: 0 1;
    }
    #input-row Input { border: none; }
    #status { height: 1; background: $boost; color: $text-muted; padding: 0 1; }
    """

    TITLE = "skynet"
    SUB_TITLE = "self-evolving knowledge base"

    BINDINGS = [
        Binding("ctrl+k", "toggle_kb", "KB tree"),
        Binding("ctrl+r", "toggle_feed", "Activity"),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
    ]

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self.runtime = runtime
        self._wire_approver()

    # ----- construction -----------------------------------------------
    @classmethod
    def from_args(cls, user_id: str, provider_name: str | None = None) -> "SkynetApp":
        config = load_config()
        if provider_name:
            # Override active provider without rewriting config.yaml.
            from skynet.config import ProviderCfg
            config = config.__class__(
                home=config.home,
                provider=ProviderCfg(
                    active=provider_name, zai=config.provider.zai,
                    bedrock=config.provider.bedrock, mock=config.provider.mock,
                ),
                context=config.context, memory=config.memory,
                retrieval=config.retrieval, paths=config.paths,
            )
        runtime = build_runtime(config, user_id)
        return cls(runtime)

    # ----- layout ------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main-row"):
            yield ChatPane(id="chat")
            with Vertical(id="right-col"):
                yield ActivityFeed(id="feed")
                yield KbTree()
        yield Static(self._status_text(), id="status")
        with Horizontal(id="input-row"):
            yield Input(placeholder="type a message, or /help for commands", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_kb_tree()
        self._refresh_status()
        self.query_one("#prompt", Input).focus()
        self._system("Welcome to skynet. Point me at a folder and ask away.")

    # ----- event sink binding -----------------------------------------
    def _wire_approver(self) -> None:
        # Bind the loop's sink to our UI, and use this app as the approver.
        self.runtime.loop.sink = self._on_event
        self.runtime.loop.approver = self

    def _on_event(self, evt: ActivityEvent) -> None:
        # Forward to the feed widget; this runs on the worker thread, so use
        # call_from_thread to marshal back to the UI thread.
        try:
            feed = self.query_one("#feed", ActivityFeed)
        except Exception:
            return
        self.call_from_thread(feed.emit_event, evt.type, evt.summary, evt.why)
        if evt.type in {"turn_end", "tool_result", "approval_decision"}:
            self.call_from_thread(self._refresh_status)

    # ----- Approver protocol ------------------------------------------
    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Push the modal and block the calling worker until decided."""
        from skynet.tui.widgets.approval_modal import ApprovalModal

        modal = ApprovalModal(req)
        # push_screen runs on the UI thread; the worker that called us is
        # blocked inside ``request`` until dismiss fires.
        return self._push_blocking(modal)

    def _push_blocking(self, modal) -> ApprovalDecision:
        # Textual's push_screen is async-native; we drive it via run_worker and
        # an event so the calling (worker) thread blocks cleanly.
        import threading

        result: dict = {}
        done = threading.Event()

        def _show():
            def _cb(decision):
                result["d"] = decision
                done.set()
            self.push_screen(modal, _cb)

        self.call_from_thread(_show)
        done.wait(timeout=3600)
        return result.get("d", ApprovalDecision(approved=False, reason="timeout"))

    # ----- input handling ---------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.query_one("#prompt", Input).value = ""
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._send(text)

    @work(thread=True, exclusive=True, name="turn")
    def _send(self, text: str) -> None:
        chat = self.query_one("#chat", ChatPane)
        self.call_from_thread(chat.add_turn, "user", text)
        # Stream tokens into a fresh assistant bubble.
        bubble = self.call_from_thread(chat.add_turn, "assistant", "…")
        try:
            for tok in self.runtime.loop.handle(text):
                self.call_from_thread(chat.append_to_last, bubble, tok)
        except Exception as exc:
            self.call_from_thread(chat.append_to_last, bubble, f"\n[error: {exc}]")
        # If the KB changed, refresh the tree.
        self.call_from_thread(self._refresh_kb_tree)

    # ----- slash commands ---------------------------------------------
    def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        chat = self.query_one("#chat", ChatPane)

        if cmd == "/help":
            self._system(
                "Commands: /compact  /new  /model <mock|zai|bedrock>  "
                "/user <id>  /kb  /help"
            )
        elif cmd == "/compact":
            self.runtime.loop._on_compact_requested(arg or None)
            self._system("Compaction requested for the next turn.")
        elif cmd == "/new":
            from skynet.runtime import build_runtime
            cfg = self.runtime.config
            self.runtime = build_runtime(cfg, self.runtime.loop.ctx.user_id)
            self._wire_approver()
            self._refresh_kb_tree()
            self._system("Started a new session.")
        elif cmd == "/model":
            if arg not in {"mock", "zai", "bedrock"}:
                self._system("Usage: /model <mock|zai|bedrock>")
            else:
                self._system(
                    f"Switching provider requires a restart: relaunch with --model {arg}"
                )
        elif cmd == "/user":
            if not arg:
                self._system("Usage: /user <id>")
            else:
                self._system(f"Switching user requires a restart: relaunch with --user {arg}")
        elif cmd == "/kb":
            self.action_toggle_kb()
        else:
            self._system(f"Unknown command: {cmd}")

    def _system(self, text: str) -> None:
        chat = self.query_one("#chat", ChatPane)
        chat.add_turn("system", text)

    # ----- keybindings -------------------------------------------------
    def action_toggle_kb(self) -> None:
        tree = self.query_one(KbTree)
        tree.display = not tree.display

    def action_toggle_feed(self) -> None:
        feed = self.query_one("#feed", ActivityFeed)
        feed.display = not feed.display

    def action_cancel(self) -> None:
        self._system("⚠ cancel: stopping the current turn worker…")
        self.workers.cancel_all()

    # ----- refresh helpers --------------------------------------------
    def _refresh_kb_tree(self) -> None:
        notes = self.runtime.notes
        tree = self.query_one(KbTree)
        items = {}
        for nid in notes.list_ids():
            n = notes.read(nid)
            if n:
                items[nid] = {"title": n.title, "tags": n.tags or []}
        tree.populate(items)

    def _status_text(self) -> str:
        loop = self.runtime.loop
        provider = self.runtime.provider
        tok = f"in {loop.session.input_tokens_total} / out {loop.session.output_tokens_total}"
        cap = provider.context_length()
        return (
            f" user: {loop.ctx.user_id}   "
            f"provider: {provider.name} ({provider.__class__.__name__})   "
            f"model ctx: {cap:,}   tokens {tok}   "
            f"compactions: {loop.session.compaction_count}"
        )

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status", Static).update(self._status_text())
        except Exception:
            pass


def run() -> None:
    """CLI entrypoint: parse args and launch the app."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the skynet TUI.")
    parser.add_argument("--user", default="default", help="user_id (default: default)")
    parser.add_argument(
        "--model", default=None, choices=["mock", "zai", "bedrock"],
        help="override the active provider",
    )
    args = parser.parse_args()
    app = SkynetApp.from_args(args.user, args.model)
    app.run()


if __name__ == "__main__":
    run()
