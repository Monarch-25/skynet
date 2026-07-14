"""Approval modal (plan §11).

Shown only when the loop emits a write tool call. Three buttons: Approve, Edit,
Reject. Edit swaps the preview into an editable TextArea; the edited content is
returned via the modal's ``decision`` attribute.

The modal implements the ``Approver`` protocol: ``request(req)`` pushes the
request onto the screen and blocks the worker until the user picks a button.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Static, TextArea
from textual.screen import ModalScreen

from skynet.core.approval import ApprovalDecision, ApprovalRequest


class ApprovalModal(ModalScreen[ApprovalDecision]):
    """A blocking modal that returns an ApprovalDecision when dismissed."""

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    ApprovalModal > Vertical {
        width: 80;
        max-width: 100;
        height: 80;
        max-height: 100;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }
    ApprovalModal #ap-title { color: $warning; text-style: bold; margin-bottom: 1; }
    ApprovalModal #ap-summary { color: $text; margin-bottom: 1; }
    ApprovalModal #ap-why { color: $text-muted; margin-bottom: 1; }
    ApprovalModal TextArea {
        height: 1fr;
        border: round $primary;
        margin-bottom: 1;
    }
    ApprovalModal #ap-buttons {
        height: 3;
        align-horizontal: center;
        layout: horizontal;
    }
    ApprovalModal Button { margin: 0 1; }
    ApprovalModal #ap-reject { background: $error; }
    ApprovalModal #ap-edit { background: $accent; }
    ApprovalModal #ap-approve { background: $success; }
    """

    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("e", "edit", "Edit"),
        Binding("r", "reject", "Reject"),
        Binding("escape", "reject", "Reject", show=False),
    ]

    def __init__(self, req: ApprovalRequest) -> None:
        super().__init__()
        self.req = req
        self._editing = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"✋ Write approval required — {self.req.tool}", id="ap-title"
            )
            yield Static(f"[b]Target:[/b] {self.req.target}   [b]{self.req.summary}[/b]",
                         id="ap-summary")
            why = self.req.why or "(no reason given)"
            yield Static(f"[b]Why:[/b] {why}", id="ap-why")
            yield TextArea(self.req.preview, language="markdown", id="ap-body")
            with Vertical(id="ap-buttons"):
                yield Button("Approve [a]", id="ap-approve", variant="success")
                yield Button("Edit [e]", id="ap-edit", variant="default")
                yield Button("Reject [r]", id="ap-reject", variant="error")

    # ----- actions -----------------------------------------------------
    def action_approve(self) -> None:
        body = self.query_one("#ap-body", TextArea).text
        # If the user edited the text, treat as edited approval.
        edited = body if body != self.req.preview else None
        self.dismiss(ApprovalDecision(approved=True, edited_content=edited))

    def action_edit(self) -> None:
        # Focus the text area so the user can edit; pressing Approve then
        # carries the edited content.
        self.query_one("#ap-body", TextArea).focus()

    def action_reject(self) -> None:
        self.dismiss(ApprovalDecision(approved=False, reason="rejected by user"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ap-approve":
            self.action_approve()
        elif event.button.id == "ap-edit":
            self.action_edit()
        elif event.button.id == "ap-reject":
            self.action_reject()
