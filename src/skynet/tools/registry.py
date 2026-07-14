"""Tool registry and dispatch.

Each tool is a small dataclass pairing an OpenAI-style function schema with a
handler. Handlers receive the parsed args dict + a ``ToolContext`` carrying the
per-session collaborators (retriever, note store, memory manager, user id).

Tools come in three kinds (plan §10):
    - read     -> executes immediately, no approval
    - write    -> returns a proposal; the turn loop gates it on approval
    - control  -> ``ask_user``; the loop pauses for structured input

Writes never touch disk from inside the handler. They return a *proposal*
(``ToolResult.preview``); only ``ToolContext.commit_write(call_id, ...)`` —
called by the loop after approval — persists anything. This keeps the
"plan-before-mutating" rule literally enforced by the type system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from skynet.core.types import ToolKind, ToolResult
from skynet.kb.notes import Note, NoteStore
from skynet.core.memory_manager import MemoryManager
from skynet.retrieval.base import Retriever
from skynet.retrieval.bm25_retriever import Bm25Retriever


@dataclass
class ToolContext:
    """Per-session collaborators passed to every tool handler."""

    user_id: str
    retriever: Retriever
    notes: NoteStore
    memory: MemoryManager
    knowledge_source_root: Path
    # Populated by the loop for the ``todo`` tool so the TUI sidebar can see it.
    todo_items: list[str] = field(default_factory=list)
    # Loop hooks the tools can invoke (set by the loop, not by tests).
    request_compaction: Callable[[str | None], None] | None = None
    request_user_input: Callable[[str, list[str] | None], str] | None = None


@dataclass
class Tool:
    name: str
    kind: ToolKind
    description: str
    parameters: dict
    handler: Callable[[dict, ToolContext], ToolResult]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def _ok(call_id: str, name: str, content: str, *, summary: str = "", kind: ToolKind = "read",
        preview: str | None = None) -> ToolResult:
    return ToolResult(
        call_id=call_id, name=name, ok=True, content=content,
        summary=summary, kind=kind, preview=preview,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]"


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def _source_search(args: dict, ctx: ToolContext) -> ToolResult:
    query = str(args.get("query", ""))
    k = int(args.get("k", 6))
    chunks = ctx.retriever.query(query, k=k, user_id=ctx.user_id, kinds=["source"])
    if not chunks:
        body = "No source chunks matched."
    else:
        body = "\n\n".join(c.render() for c in chunks)
    return _ok(args.get("__call_id", ""), "source_search", _truncate(body, 6000),
               summary=f'"{query}" -> {len(chunks)} chunk(s)')


def _source_read(args: dict, ctx: ToolContext) -> ToolResult:
    rel = str(args.get("path", "")).lstrip("/")
    # Strict path containment: never allow escapes from the source root.
    target = (ctx.knowledge_source_root / rel).resolve()
    try:
        target.relative_to(ctx.knowledge_source_root)
    except ValueError:
        return ToolResult(
            call_id=args.get("__call_id", ""), name="source_read", ok=False,
            content=f"Path escapes source root: {rel}",
            summary="blocked path escape",
        )
    if not target.exists():
        return _ok(args.get("__call_id", ""), "source_read",
                   f"File not found: {rel}", summary=f"missing: {rel}")
    text = target.read_text(encoding="utf-8", errors="replace")
    locator = str(args.get("locator", ""))
    snippet = text
    return _ok(args.get("__call_id", ""), "source_read", _truncate(snippet, 8000),
               summary=f"{rel}#{locator}" if locator else rel)


def _kb_search(args: dict, ctx: ToolContext) -> ToolResult:
    query = str(args.get("query", ""))
    k = int(args.get("k", 6))
    chunks = ctx.retriever.query(query, k=k, user_id=ctx.user_id, kinds=["kb_note"])
    if not chunks:
        body = "No KB notes matched. (The KB may be empty — consider writing one.)"
    else:
        body = "\n\n".join(c.render() for c in chunks)
    return _ok(args.get("__call_id", ""), "kb_search", _truncate(body, 6000),
               summary=f'"{query}" -> {len(chunks)} note(s)')


def _kb_view(args: dict, ctx: ToolContext) -> ToolResult:
    note = ctx.notes.read(str(args.get("note_id", "")))
    if not note:
        return _ok(args.get("__call_id", ""), "kb_view", "Note not found.",
                   summary="missing note")
    backlinks = ctx.notes.backlinks(note.note_id)
    body = note.to_markdown()
    if backlinks:
        body += "\n\n## Backlinks\n" + "\n".join(f"- {b}" for b in backlinks)
    return _ok(args.get("__call_id", ""), "kb_view", body,
               summary=f"{note.note_id} (+{len(backlinks)} backlinks)")


def _memory_view(args: dict, ctx: ToolContext) -> ToolResult:
    target = str(args.get("target", "memory"))
    try:
        body = ctx.memory.render(target)
    except ValueError as exc:
        return _ok(args.get("__call_id", ""), "memory_view", str(exc), summary="bad target")
    return _ok(args.get("__call_id", ""), "memory_view", body or "(empty)",
               summary=target)


def _compress_context(args: dict, ctx: ToolContext) -> ToolResult:
    focus = args.get("focus_topic")
    if ctx.request_compaction:
        ctx.request_compaction(str(focus) if focus else None)
    return _ok(args.get("__call_id", ""), "compress_context",
               "Compaction requested for this turn.", summary="compact requested")


def _todo(args: dict, ctx: ToolContext) -> ToolResult:
    items = list(args.get("items", []) or [])
    ctx.todo_items.clear()
    ctx.todo_items.extend(str(i) for i in items)
    body = "Todo list updated:\n" + "\n".join(f"- [ ] {i}" for i in ctx.todo_items)
    return _ok(args.get("__call_id", ""), "todo", body,
               summary=f"{len(items)} item(s)")


# ---------------------------------------------------------------------------
# Write tools (return proposals; the loop commits after approval)
# ---------------------------------------------------------------------------

def _kb_write_note(args: dict, ctx: ToolContext) -> ToolResult:
    note = Note(
        note_id=str(args.get("note_id", "")),
        title=str(args.get("title", "")),
        body=str(args.get("body", "")),
        tags=list(args.get("tags", []) or []),
        sources=list(args.get("sources", []) or []),
        links=list(args.get("links", []) or []),
        derived_from_user=ctx.user_id,
        confidence=str(args.get("confidence", "medium")),
    )
    if not note.note_id:
        return ToolResult(args.get("__call_id", ""), "kb_write_note", ok=False,
                          kind="write", content="note_id is required",
                          summary="missing note_id")
    existing = ctx.notes.read(note.note_id)
    proposed = ctx.notes.preview_write(note)
    label = "update" if existing else "new"
    return ToolResult(
        call_id=args.get("__call_id", ""), name="kb_write_note", ok=True,
        kind="write", content=proposed, preview=proposed,
        summary=f"{note.note_id} ({label})",
    )


def _kb_link(args: dict, ctx: ToolContext) -> ToolResult:
    from_id = str(args.get("from_id", ""))
    to_id = str(args.get("to_id", ""))
    note = ctx.notes.read(from_id)
    if not note:
        return ToolResult(args.get("__call_id", ""), "kb_link", ok=False, kind="write",
                          content=f"Note not found: {from_id}", summary="missing from_id")
    if to_id in note.links:
        return _ok(args.get("__call_id", ""), "kb_link",
                   f"Link {from_id} -> {to_id} already exists.", summary="noop",
                   kind="write", preview=note.to_markdown())
    note.links.append(to_id)
    proposed = ctx.notes.preview_write(note)
    return ToolResult(
        call_id=args.get("__call_id", ""), name="kb_link", ok=True, kind="write",
        content=proposed, preview=proposed,
        summary=f"{from_id} -> {to_id}",
    )


def _memory_write(args: dict, ctx: ToolContext) -> ToolResult:
    target = str(args.get("target", "memory"))
    action = str(args.get("action", "append"))
    content = str(args.get("content", ""))
    try:
        proposed, label, over = ctx.memory.propose(target, action, content)
    except ValueError as exc:
        return ToolResult(args.get("__call_id", ""), "memory_write", ok=False,
                          kind="write", content=str(exc), summary="bad target")
    note = ("[over cap — will need consolidation]" if over else "")
    return ToolResult(
        call_id=args.get("__call_id", ""), name="memory_write", ok=True,
        kind="write", content=proposed, preview=proposed,
        summary=f"{target}:{label} {note}".strip(),
    )


# ---------------------------------------------------------------------------
# Control tool
# ---------------------------------------------------------------------------

def _ask_user(args: dict, ctx: ToolContext) -> ToolResult:
    question = str(args.get("question", ""))
    options = args.get("options")
    # The loop binds ``request_user_input`` to a real prompt; in tests it's None.
    if ctx.request_user_input:
        answer = ctx.request_user_input(question, list(options) if options else None)
    else:
        answer = "(no input handler bound)"
    return _ok(args.get("__call_id", ""), "ask_user", f"User answered: {answer}",
               summary="asked user", kind="control")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ``why`` is added to every schema so the model can explain itself; it's parsed
# out of the arguments in ``LLMProvider.parse_tool_calls`` and never reaches the
# handler as a real arg.
def _with_why(props: dict) -> dict:
    props = dict(props)
    props["why"] = {"type": "string", "description": "One short sentence: why this call."}
    props.setdefault("required", [])
    props["required"] = list(props["required"]) + ["why"]
    props["type"] = "object"
    return props


def build_registry() -> dict[str, Tool]:
    """Construct the v1 tool registry (plan §10 inventory)."""
    specs: list[tuple[str, ToolKind, str, dict, Callable]] = [
        ("source_search", "read", "BM25 search over the read-only source folder, ACL-filtered for this user.",
         _with_why({
             "properties": {
                 "query": {"type": "string"},
                 "k": {"type": "integer", "default": 6},
                 "scope": {"type": "string", "description": "reserved: 'source'"},
             },
             "required": ["query"],
         }), _source_search),

        ("source_read", "read", "Read a specific file from the source folder verbatim (path + optional locator).",
         _with_why({
             "properties": {
                 "path": {"type": "string", "description": "path relative to source root"},
                 "locator": {"type": "string", "description": "page or section, e.g. p12"},
             },
             "required": ["path"],
         }), _source_read),

        ("kb_search", "read", "BM25 search over Skynet's own internal knowledge base notes.",
         _with_why({
             "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 6}},
             "required": ["query"],
         }), _kb_search),

        ("kb_view", "read", "Read a full KB note by id, with its backlinks.",
         _with_why({
             "properties": {"note_id": {"type": "string"}},
             "required": ["note_id"],
         }), _kb_view),

        ("kb_write_note", "write", "Create or update a KB note. REQUIRES APPROVAL. Include sources.",
         _with_why({
             "properties": {
                 "note_id": {"type": "string"},
                 "title": {"type": "string"},
                 "body": {"type": "string"},
                 "tags": {"type": "array", "items": {"type": "string"}},
                 "sources": {"type": "array", "items": {"type": "string"},
                             "description": "provenance: source paths or session:..."},
                 "links": {"type": "array", "items": {"type": "string"}},
                 "confidence": {"type": "string", "enum": ["high", "medium", "speculative"]},
             },
             "required": ["note_id", "title", "body"],
         }), _kb_write_note),

        ("kb_link", "write", "Add a backlink from one note to another without rewriting the body. REQUIRES APPROVAL.",
         _with_why({
             "properties": {"from_id": {"type": "string"}, "to_id": {"type": "string"}},
             "required": ["from_id", "to_id"],
         }), _kb_link),

        ("memory_view", "read", "View USER.md ('user') or MEMORY.md ('memory') for this user.",
         _with_why({
             "properties": {"target": {"type": "string", "enum": ["user", "memory"]}},
             "required": ["target"],
         }), _memory_view),

        ("memory_write", "write", "Append/replace USER.md or MEMORY.md. REQUIRES APPROVAL. Size-capped.",
         _with_why({
             "properties": {
                 "target": {"type": "string", "enum": ["user", "memory"]},
                 "action": {"type": "string", "enum": ["append", "replace"]},
                 "content": {"type": "string"},
             },
             "required": ["target", "action", "content"],
         }), _memory_write),

        ("compress_context", "read", "Request context compaction this turn (e.g. when the thread is long).",
         _with_why({
             "properties": {"focus_topic": {"type": "string"}},
         }), _compress_context),

        ("todo", "read", "Set a lightweight task list for this turn (shown in the TUI sidebar).",
         _with_why({
             "properties": {"items": {"type": "array", "items": {"type": "string"}}},
         }), _todo),

        ("ask_user", "control", "Ask the user a clarifying question (distinct from a normal reply).",
         {
             "type": "object",
             "properties": {
                 "question": {"type": "string"},
                 "options": {"type": "array", "items": {"type": "string"}},
                 "why": {"type": "string", "description": "Why ask now."},
             },
             "required": ["question", "why"],
         }, _ask_user),
    ]
    return {
        name: Tool(name=name, kind=kind, description=desc, parameters=params, handler=handler)
        for name, kind, desc, params, handler in specs
    }


def dispatch(registry: dict[str, Tool], call_id: str, name: str, args: dict, ctx: ToolContext) -> ToolResult:
    """Run a tool by name. Injects ``__call_id`` so handlers can tag results."""
    tool = registry.get(name)
    if not tool:
        return ToolResult(call_id=call_id, name=name, ok=False, content=f"Unknown tool: {name}",
                          summary="unknown tool")
    args_with_id = dict(args)
    args_with_id["__call_id"] = call_id
    try:
        return tool.handler(args_with_id, ctx)
    except Exception as exc:  # never let a tool crash kill the loop
        return ToolResult(call_id=call_id, name=name, ok=False, content=f"Tool error: {exc}",
                          summary="raised")


# Write tools know how to land their proposal once approved.
def commit_write(result: ToolResult, ctx: ToolContext) -> str:
    """Persist an approved write proposal. Returns a short outcome string."""
    if result.name == "kb_write_note":
        # Re-parse the preview (which is the markdown body) into a Note.
        import frontmatter as _fm
        post = _fm.loads(result.preview or "")
        note = Note(
            note_id=str(post.metadata.get("id", "")),
            title=str(post.metadata.get("title", "")),
            body=post.content,
            tags=list(post.metadata.get("tags", []) or []),
            sources=list(post.metadata.get("sources", []) or []),
            links=list(post.metadata.get("links", []) or []),
            derived_from_user=ctx.user_id,
            confidence=str(post.metadata.get("confidence", "medium")),
        )
        _, was_new = ctx.notes.write(note)
        # Keep the BM25 KB index in sync.
        from skynet.retrieval.base import RawChunk
        ctx.retriever.upsert_kb(RawChunk(
            text=f"{note.title}\n\n{note.body}", source_path=note.note_id,
            locator=note.note_id, kind="kb_note", title=note.title, tags=note.tags,
        ))
        return f"wrote note {note.note_id} ({'new' if was_new else 'updated'})"

    if result.name == "kb_link":
        import frontmatter as _fm
        post = _fm.loads(result.preview or "")
        note = ctx.notes.read(str(post.metadata.get("id", "")))
        if note:
            note.links = list(post.metadata.get("links", []) or [])
            ctx.notes.write(note)
            return f"linked {note.note_id}"
        return "link target missing"

    if result.name == "memory_write":
        # ``content`` already holds the proposed full file content.
        target = _infer_memory_target(result)
        ctx.memory.apply(target, result.content)
        return f"updated {target}.md"

    return f"(no commit handler for {result.name})"


def _infer_memory_target(result: ToolResult) -> str:
    # The summary is "user:append" / "memory:replace"; fall back to memory.
    if result.summary.startswith("user"):
        return "user"
    if result.summary.startswith("memory"):
        return "memory"
    return "memory"
