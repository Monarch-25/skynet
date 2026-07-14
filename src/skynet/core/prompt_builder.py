"""System-prompt assembly.

Built once per session and treated as a stable cache prefix (plan §4): SOUL.md
first, then platform hints, then the per-user USER.md+MEMORY.md block (bounded),
then KB-write guidance and tool-call conventions. Nothing here mutates
mid-session, which is what lets provider-side prefix caching actually work.
"""

from __future__ import annotations

from pathlib import Path

from skynet.core.memory_manager import MemoryManager


TOOL_CONVENTIONS = """\
## How to call tools
- Fill the ``why`` field on EVERY tool call with one short sentence saying why.
- Prefer ``source_search`` for ground truth from the source folder; ``kb_search`` \
for notes you've already distilled. Pick deliberately — don't call both blindly.
- Never invent a source path you didn't read from a tool result.
"""

KB_NUDGE = """\
## When to write a KB note
When you've synthesized something durable — a concept explained from multiple \
sources, a correction to something you got wrong before, a recurring pattern — \
write or update a note with ``kb_write_note``, including a ``sources`` field. \
Don't write a note every turn; only when future-you would benefit from not \
re-deriving this.
"""


def build_system_prompt(
    soul_path: Path,
    memory: MemoryManager,
    extra_instructions: str = "",
) -> str:
    soul = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""
    user_block = memory.build_system_prompt_block()
    parts = [
        soul.strip(),
        "",
        user_block,
        "",
        TOOL_CONVENTIONS,
        KB_NUDGE,
    ]
    if extra_instructions:
        parts.append(extra_instructions.strip())
    return "\n".join(parts).strip() + "\n"
