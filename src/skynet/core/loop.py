"""The turn loop orchestrator (plan §7).

Per user turn:
    1. (session start) build system prompt once, as a stable prefix.
    2. optional compaction if context is large.
    3. bounded tool-calling loop (max_iterations): call provider, execute read
       tools immediately, gate write tools on approval, surface everything as
       activity events.
    4. stream the final text reply.
    5. append the raw turn to the session log.

The loop is UI-agnostic: it takes an ``Approver`` and an event callback. The
Textual app supplies both; tests supply an ``AutoApprover`` and a list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from skynet.core.approval import ApprovalDecision, Approver, request_from_result
from skynet.core.context_compressor import pre_compress, should_compress, summarize
from skynet.core.events import ActivityEvent, ActivityLog
from skynet.core.memory_manager import MemoryManager
from skynet.core.prompt_builder import build_system_prompt
from skynet.core.tokens import estimate_messages_tokens
from skynet.core.types import Message, ProviderResponse, ToolResult
from skynet.providers.base import LLMProvider
from skynet.tools.registry import ToolContext, build_registry, commit_write, dispatch


# Event callback: the UI subscribes by passing one of these.
EventSink = Callable[[ActivityEvent], None]


def _noop_sink(_: ActivityEvent) -> None:
    pass


@dataclass
class Session:
    """Mutable session state: the message history plus identity."""

    user_id: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    todo_items: list[str] = field(default_factory=list)
    # bookkeeping
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    compaction_count: int = 0

    def to_dicts(self) -> list[dict]:
        return [Message(role="system", content=self.system_prompt).to_dict()] + [
            m.to_dict() for m in self.messages
        ]


class TurnLoop:
    """Owns one session's turn loop."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryManager,
        soul_path: Path,
        user_id: str,
        knowledge_source_root: Path,
        retriever,
        notes,
        max_iterations: int = 8,
        compact_threshold_tokens: int | None = None,
        max_summary_tokens: int = 3000,
        approver: Approver | None = None,
        sink: EventSink = _noop_sink,
        activity_log: ActivityLog | None = None,
        session_log_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.approver = approver
        self.sink = sink
        self.activity_log = activity_log
        self.session_log_path = session_log_path
        self.max_iterations = max_iterations
        self.compact_threshold = compact_threshold_tokens
        self.max_summary_tokens = max_summary_tokens

        self.registry = build_registry()
        self.tool_schemas = [t.schema() for t in self.registry.values()]

        self.ctx = ToolContext(
            user_id=user_id,
            retriever=retriever,
            notes=notes,
            memory=memory,
            knowledge_source_root=knowledge_source_root,
        )
        # Wire the context hooks the tools can invoke.
        self.ctx.request_compaction = self._on_compact_requested
        self.ctx.request_user_input = self._on_ask_user

        system_prompt = build_system_prompt(soul_path, memory)
        self.session = Session(user_id=user_id, system_prompt=system_prompt)
        self._pending_compaction: str | None = None

    # ----- public API --------------------------------------------------
    def handle(self, user_text: str) -> Iterable[str]:
        """Process one user turn. Yields streamed reply tokens.

        Tool calls happen synchronously between yields; the final assistant
        text is yielded token-by-token at the end.
        """
        self._emit("turn_start", f"user: {user_text[:80]}")
        self.session.messages.append(Message(role="user", content=user_text))

        # Compaction pre-check (and honor any tool-requested compaction).
        self._maybe_compact()

        # ---- bounded tool-calling loop --------------------------------
        final_text = ""
        for i in range(self.max_iterations):
            resp = self.provider.chat(
                self.session.messages, self.tool_schemas
            )
            self.session.input_tokens_total += resp.input_tokens
            self.session.output_tokens_total += resp.output_tokens

            if not resp.wants_tools:
                final_text = resp.message.content or ""
                break

            # Record the assistant's tool-call turn, then execute each call.
            self.session.messages.append(resp.message)
            all_results_ok = self._execute_tool_calls(resp)
            if not all_results_ok:
                # let the model react (e.g. after a rejection)
                continue
        else:
            # Hit the iteration cap — surface honestly, don't silently truncate.
            self._emit(
                "info",
                f"reached tool-call cap ({self.max_iterations}); forcing a reply",
            )
            # One final non-tool call to get *some* answer out.
            resp = self.provider.chat(self.session.messages, [])
            final_text = resp.message.content or "(no reply)"

        # ---- stream the reply ----------------------------------------
        self.session.messages.append(Message(role="assistant", content=final_text))
        self._emit("stream_token", final_text, detail=final_text)
        yield final_text

        self._emit(
            "turn_end",
            f"tokens in/out: {self.session.input_tokens_total}/"
            f"{self.session.output_tokens_total}",
        )
        self._append_session_log(user_text, final_text)

    # ----- tool execution --------------------------------------------
    def _execute_tool_calls(self, resp: ProviderResponse) -> bool:
        """Run each tool call; return True if all were read/control, False if a
        write was rejected (so the loop lets the model react)."""
        any_rejected = False
        for call in resp.message.tool_calls:
            self._emit(
                "tool_call",
                f"{call.name}({call.arguments[:60]})",
                why=call.why or "",
            )
            args = self._parse_args(call.arguments)
            result = dispatch(self.registry, call.id, call.name, args, self.ctx)

            if result.kind == "write":
                decision = self._gate_write(result, call.why or "")
                if decision.approved:
                    if decision.edited_content is not None:
                        result.content = decision.edited_content
                        result.preview = decision.edited_content
                    outcome = commit_write(result, self.ctx)
                    result.content = f"{outcome}\n\nWrote:\n{result.preview}"
                    self._emit("approval_decision", f"approved {result.name}: {outcome}",
                               why=decision.reason)
                else:
                    any_rejected = True
                    result.ok = False
                    result.content = (
                        f"User REJECTED this write"
                        + (f": {decision.reason}" if decision.reason else "")
                        + ". Do not retry the same write unchanged; adapt."
                    )
                    self._emit("approval_decision", f"rejected {result.name}",
                               why=decision.reason)

            # Size-limit the result before it enters context (plan §16).
            if len(result.content) > 8000:
                result.content = result.content[:7980] + "\n…[truncated by loop]"

            self._emit("tool_result", result.summary, detail=result.content[:200])
            self.session.messages.append(Message(
                role="tool", content=result.content, tool_call_id=call.id, name=call.name,
            ))
        return not any_rejected

    def _gate_write(self, result: ToolResult, why: str) -> ApprovalDecision:
        if not self.approver:
            # No approver wired (e.g. bare test) — default to approve so the
            # loop is still exercisable headless.
            return ApprovalDecision(approved=True)
        req = request_from_result(result, why)
        return self.approver.request(req)

    # ----- compaction -------------------------------------------------
    def _maybe_compact(self) -> None:
        requested = self._pending_compaction is not None
        self._pending_compaction, requested_flag = None, requested
        size_ok = (
            self.compact_threshold is not None
            and should_compress(self.session.messages, self.compact_threshold)
        )
        if not (size_ok or requested):
            return
        before = estimate_messages_tokens(self.session.messages)
        self.session.messages = pre_compress(self.session.messages)
        summary = summarize(self.session.messages, focus=None)
        self.session.messages = [
            Message(role="system", content=f"<compaction>\n{summary}\n</compaction>"),
            self.session.messages[-1] if self.session.messages else Message(role="user", content=""),
        ]
        self.session.compaction_count += 1
        after = estimate_messages_tokens(self.session.messages)
        self._emit("compaction", f"{before} -> {after} tokens (heuristic)")

    def _on_compact_requested(self, focus: str | None) -> None:
        self._pending_compaction = focus

    def _on_ask_user(self, question: str, options: list[str] | None) -> str:
        self._emit("info", f"ask_user: {question}")
        return "(handled by control tool)"

    # ----- helpers ----------------------------------------------------
    @staticmethod
    def _parse_args(raw: str) -> dict:
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _emit(self, type_: str, summary: str, *, detail: str = "", why: str = "") -> None:
        evt = ActivityEvent(type=type_, summary=summary, detail=detail, why=why)  # type: ignore[arg-type]
        self.sink(evt)
        if self.activity_log:
            self.activity_log.append(evt)

    def _append_session_log(self, user_text: str, assistant_text: str) -> None:
        if not self.session_log_path:
            return
        self.session_log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"role": "user", "content": user_text}
        rec_a = {"role": "assistant", "content": assistant_text}
        with self.session_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write(json.dumps(rec_a, ensure_ascii=False) + "\n")
