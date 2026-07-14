"""Retriever interface.

Two indices live behind one interface: source-folder chunks and KB notes. They
are kept in *separate* BM25 corpora so a query can target "source only," "KB
only," or both — the agent decides via the tool's ``scope`` argument (plan §6).

The ``Retriever`` protocol is what lets embeddings land in M7 without touching
anything outside this package: ``EmbeddingRetriever`` implements the same
interface and can run side-by-side with BM25 under reciprocal rank fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

ChunkKind = Literal["source", "kb_note"]


@dataclass
class Chunk:
    text: str
    source_path: str
    locator: str = ""        # "p12-14" / "L40-55" / note_id
    score: float = 0.0
    kind: ChunkKind = "source"
    title: str = ""          # human label for TUI / citations
    tags: list[str] | None = None

    def render(self) -> str:
        """Compact, citation-friendly rendering for tool results."""
        head = f"[{self.kind}] {self.title or self.source_path}"
        if self.locator:
            head += f"#{self.locator}"
        return f"{head}\n{self.text}"


@dataclass
class RawChunk:
    """Input to ``Retriever.index`` — pre-tokenized, pre-located text."""

    text: str
    source_path: str
    locator: str = ""
    kind: ChunkKind = "source"
    title: str = ""
    tags: list[str] | None = None


class Retriever(Protocol):
    def index(self, chunks: Iterable[RawChunk]) -> int:
        """Add/replace chunks. Returns number indexed."""

    def query(
        self,
        text: str,
        k: int = 6,
        user_id: str | None = None,
        kinds: list[ChunkKind] | None = None,
    ) -> list[Chunk]:
        """Return up to ``k`` ranked chunks, ACL-filtered for ``user_id``."""

    def remove(self, source_path: str) -> int:
        """Remove all chunks for a source path. Returns number removed."""

    def size(self, kind: ChunkKind | None = None) -> int:
        """Number of indexed chunks (optionally of one kind)."""
