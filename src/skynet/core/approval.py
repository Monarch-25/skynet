"""Approval protocol.

Decouples the turn loop from the TUI. The loop calls ``approver.request(...)``;
the TUI (or a test) implements that however it likes — modal, CLI prompt, or an
auto-approve stub — and returns an ``ApprovalDecision``. The loop never imports
Textual.

Three decisions (plan §11): approve (apply as-is), edit (apply modified
content), reject (don't apply; the loop feeds a synthetic tool result back so the
model can adapt instead of blindly retrying).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skynet.core.types import ToolResult


@dataclass
class ApprovalRequest:
    call_id: str
    tool: str
    target: str               # short label, e.g. "note:kv-cache" or "memory:user"
    summary: str              # one-line "what"
    why: str                  # the model's stated reason
    preview: str              # the proposed content (diff/new file)


@dataclass
class ApprovalDecision:
    approved: bool
    edited_content: str | None = None  # set when the user edited the proposal
    reason: str = ""                  # why rejected, fed back to the model


class Approver(Protocol):
    """Pauses the loop until the user decides on a write."""

    def request(self, req: ApprovalRequest) -> ApprovalDecision: ...


class AutoApprover:
    """Test/headless approver: approves everything, no edits."""

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True)


def request_from_result(result: ToolResult, why: str) -> ApprovalRequest:
    return ApprovalRequest(
        call_id=result.call_id,
        tool=result.name,
        target=result.summary.split()[0] if result.summary else result.name,
        summary=result.summary,
        why=why,
        preview=result.preview or result.content,
    )
