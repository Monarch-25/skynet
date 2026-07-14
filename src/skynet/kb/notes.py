"""Internal KB notes — the self-evolving part.

Each note is a markdown file under ``internal_kb/notes/<slug>.md`` with a fixed
frontmatter schema (plan §3). The KB module owns read/parse/write and exposes
operations to the tool layer. Writes are *proposed* (returning a preview) — the
approval gate in the turn loop decides whether they land on disk.

The retriever is also notified on write/remove so the BM25 KB index stays in
sync without the tool layer reaching into retrieval internals.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from skynet.core.scrub import scrub


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Note:
    note_id: str
    title: str = ""
    body: str = ""
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    derived_from_user: str = ""
    confidence: str = "medium"   # high | medium | speculative | stale
    extra: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        meta = {
            "id": self.note_id,
            "title": self.title,
            "tags": self.tags,
            "created": self.created or _now_iso(),
            "updated": _now_iso(),
            "sources": self.sources,
            "derived_from_user": self.derived_from_user,
            "confidence": self.confidence,
            "links": self.links,
        }
        post = frontmatter.Post(self.body, **meta)
        return frontmatter.dumps(post)


class NoteStore:
    """Filesystem-backed store of KB notes."""

    def __init__(self, notes_dir: Path) -> None:
        self.dir = notes_dir.resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, note_id: str) -> Path:
        self._check_id(note_id)
        return self.dir / f"{note_id}.md"

    def exists(self, note_id: str) -> bool:
        return self.path_for(note_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.md"))

    def read(self, note_id: str) -> Note | None:
        path = self.path_for(note_id)
        if not path.exists():
            return None
        post = frontmatter.load(path)
        return Note(
            note_id=str(post.metadata.get("id", note_id)),
            title=str(post.metadata.get("title", "")),
            body=post.content,
            tags=list(post.metadata.get("tags", []) or []),
            sources=list(post.metadata.get("sources", []) or []),
            links=list(post.metadata.get("links", []) or []),
            created=str(post.metadata.get("created", "")),
            updated=str(post.metadata.get("updated", "")),
            derived_from_user=str(post.metadata.get("derived_from_user", "")),
            confidence=str(post.metadata.get("confidence", "medium")),
        )

    def preview_write(self, note: Note) -> str:
        """Return the *proposed* file content for the approval card.

        Secrets are scrubbed before persistence — defense in depth (plan §16).
        """
        scrubbed = Note(
            note_id=note.note_id,
            title=note.title,
            body=scrub(note.body),
            tags=note.tags,
            sources=note.sources,
            links=note.links,
            created=note.created,
            updated=note.updated,
            derived_from_user=note.derived_from_user,
            confidence=note.confidence,
        )
        return scrubbed.to_markdown()

    def write(self, note: Note) -> tuple[str, bool]:
        """Persist a note. Returns (content_written, was_new)."""
        path = self.path_for(note.note_id)
        was_new = not path.exists()
        content = self.preview_write(note)
        path.write_text(content, encoding="utf-8")
        return content, was_new

    def delete(self, note_id: str) -> bool:
        path = self.path_for(note_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def backlinks(self, note_id: str) -> list[str]:
        """Note ids that link *to* ``note_id``."""
        out: list[str] = []
        for nid in self.list_ids():
            if nid == note_id:
                continue
            n = self.read(nid)
            if n and note_id in n.links:
                out.append(nid)
        return out

    @staticmethod
    def _check_id(note_id: str) -> None:
        if not note_id or "/" in note_id or ".." in note_id or " " in note_id:
            raise ValueError(f"unsafe note id: {note_id!r}")
