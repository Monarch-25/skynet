"""Source-folder scanner.

Walks ``knowledge_source/`` for ``.pdf``/``.txt``/``.md``, parses each, chunks
the text on token-budget boundaries with overlap, and feeds the chunks to the
retriever. Content hashes make re-scan cheap: unchanged files are skipped.

Chunking (plan §4): ~``chunk_tokens`` windows with ``chunk_overlap`` of overlap,
split on paragraph boundaries where possible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from skynet.core.tokens import estimate_tokens
from skynet.ingestion.parsers import parse_markdown, parse_pdf, parse_text
from skynet.retrieval.base import RawChunk

SUPPORTED = (".pdf", ".txt", ".md")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


class Scanner:
    """Folder walker + chunker. Stateless between calls; caches hashes."""

    def __init__(self, root: Path, chunk_tokens: int = 650, chunk_overlap: int = 100) -> None:
        self.root = root.resolve()
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        # path -> hash; survives across scans within one process.
        self._hashes: dict[str, str] = {}

    def changed_files(self) -> list[Path]:
        """List supported files whose hash differs from the last scan."""
        changed: list[Path] = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED:
                continue
            key = str(p)
            h = file_hash(p)
            if self._hashes.get(key) != h:
                self._hashes[key] = h
                changed.append(p)
        return changed

    def scan(self) -> list[RawChunk]:
        """Parse + chunk all changed files. Returns RawChunks ready to index."""
        chunks: list[RawChunk] = []
        for path in self.changed_files():
            try:
                title, located = self._parse(path)
            except Exception as exc:  # don't let one bad file kill the scan
                chunks.append(
                    RawChunk(
                        text=f"[parse error: {exc}]",
                        source_path=str(path.relative_to(self.root)),
                        locator="error",
                        kind="source",
                        title=path.stem,
                    )
                )
                continue
            for body, locator in located:
                for piece in self._chunk_text(body):
                    rel = str(path.relative_to(self.root))
                    chunks.append(
                        RawChunk(
                            text=piece,
                            source_path=rel,
                            locator=locator,
                            kind="source",
                            title=title,
                        )
                    )
        return chunks

    def _parse(self, path: Path) -> tuple[str, list[tuple[str, str]]]:
        """Return (title, [(text, locator), ...]) for a file.

        Normalizes the per-parser return shapes: markdown returns a 3-tuple
        ``(title, full_body, located)``, txt/pdf return ``(title, located)``.
        """
        suf = path.suffix.lower()
        if suf == ".md":
            title, _body, located = parse_markdown(path)
            return title, located
        if suf == ".txt":
            return parse_text(path)
        if suf == ".pdf":
            return parse_pdf(path)
        raise ValueError(f"unsupported extension: {suf}")

    def _chunk_text(self, text: str) -> list[str]:
        """Token-budget chunker that respects paragraph boundaries."""
        target = self.chunk_tokens
        overlap = self.chunk_overlap
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text] if text.strip() else []

        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for para in paragraphs:
            pt = estimate_tokens(para)
            if buf and buf_tokens + pt > target:
                chunks.append("\n\n".join(buf))
                # carry overlap forward as whole paragraphs
                tail_tokens = 0
                tail: list[str] = []
                for p in reversed(buf):
                    if tail_tokens >= overlap:
                        break
                    tail.insert(0, p)
                    tail_tokens += estimate_tokens(p)
                buf = list(tail)
                buf_tokens = tail_tokens
            buf.append(para)
            buf_tokens += pt
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks
