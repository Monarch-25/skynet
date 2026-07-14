"""Memory manager — orchestrates USER.md / MEMORY.md.

A thin layer that the prompt builder and the memory tools both talk to. It
enforces the size caps from config and triggers consolidation when a file would
overflow. The *write* path returns a proposed diff (the actual landing is gated
by approval in the turn loop); consolidation itself is a real LLM call done
elsewhere — this module only *detects* overflow and offers a merge fallback that
keeps the most recent content within budget (plan §3, §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skynet.core.scrub import scrub

Target = str  # "user" | "memory"


@dataclass
class MemoryFile:
    path: Path
    cap: int

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def exists(self) -> bool:
        return self.path.exists()


class MemoryManager:
    """Per-user memory file orchestration."""

    def __init__(self, user_dir: Path, user_cap: int, memory_cap: int) -> None:
        self.user_dir = user_dir.resolve()
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, MemoryFile] = {
            "user": MemoryFile(self.user_dir / "USER.md", user_cap),
            "memory": MemoryFile(self.user_dir / "MEMORY.md", memory_cap),
        }

    def target(self, name: str) -> MemoryFile:
        if name not in self.files:
            raise ValueError(f"unknown memory target {name!r}; expected one of {list(self.files)}")
        return self.files[name]

    # ----- read --------------------------------------------------------
    def render(self, name: str) -> str:
        f = self.target(name)
        return f.read()

    def build_system_prompt_block(self) -> str:
        """Size-capped USER.md + MEMORY.md block for the system prompt."""
        user = self._cap(self.render("user"))
        mem = self._cap(self.render("memory"))
        return (
            "<user-context>\n"
            f"## USER.md\n{user}\n\n"
            f"## MEMORY.md\n{mem}\n"
            "[System note: recalled memory, NOT new user input]\n"
            "</user-context>"
        )

    # ----- write proposal + apply -------------------------------------
    def propose(
        self, name: str, action: str, content: str
    ) -> tuple[str, str, bool]:
        """Return (proposed_full_content, preview_label, needs_consolidation).

        action ∈ {append, replace} for v1. ``consolidate`` is triggered by the
        tool layer when ``needs_consolidation`` is True — here we just signal it.
        """
        f = self.target(name)
        current = f.read()
        scrubbed = scrub(content)
        if action == "append":
            new = (current.rstrip() + "\n\n" + scrubbed).strip() if current.strip() else scrubbed
        elif action == "replace":
            new = scrubbed
        else:
            raise ValueError(f"unknown action {action!r}")
        over = len(new) > f.cap
        label = "append" if action == "append" else "replace"
        return new, label, over

    def apply(self, name: str, content: str) -> None:
        f = self.target(name)
        f.path.parent.mkdir(parents=True, exist_ok=True)
        f.path.write_text(content, encoding="utf-8")

    def _cap(self, text: str) -> str:
        # For the *prompt* we hard-cap to avoid blowing the context. Actual
        # persistence uses consolidation, but the injected snapshot is bounded.
        # Keep the most recent ~cap chars (newline-aligned) as a safe fallback.
        if len(text) <= 4000:
            return text
        cut = text[-4000:]
        nl = cut.find("\n")
        return ("…[truncated for prompt]\n" + cut[nl + 1 :]) if nl > 0 else cut
