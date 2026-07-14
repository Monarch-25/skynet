"""Document parsers.

Each parser returns a list of (text, locator) tuples — the locator is a
human-readable position like ``"L40-55"`` or ``"p12"`` that flows into the
``sources:`` provenance field of KB notes (plan §4).

- ``.md``  parsed frontmatter-aware (title extracted when present).
- ``.txt`` chunked directly.
- ``.pdf`` text extracted via PyMuPDF (``fitz``), per page.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter


def parse_markdown(path: Path) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, body_text, [(text, locator), ...]).

    Frontmatter is parsed separately; we keep the heading line nearest each
    paragraph as the locator so citations read like ``attention.md#3.2.1``.
    """
    post = frontmatter.load(path)
    title = (post.metadata.get("title") or path.stem)
    body = post.content
    return str(title), body, _locate_paragraphs(body)


def parse_text(path: Path) -> tuple[str, list[tuple[str, str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return path.stem, _locate_paragraphs(text)


def parse_pdf(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Per-page text extraction. Returns (title, [(page_text, 'p<N>'), ...])."""
    import fitz  # PyMuPDF

    title = path.stem
    pages: list[tuple[str, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            t = page.get_text("text").strip()
            if t:
                pages.append((t, f"p{i}"))
    return title, pages


def _locate_paragraphs(text: str) -> list[tuple[str, str]]:
    """Split body into paragraphs, attaching the most recent markdown heading.

    Headings (lines starting with ``#``) and the starting line number are used
    as the locator, so a chunk carries something like ``L40-55`` or
    ``scaled-dot-product-attention`` rather than an opaque index.
    """
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    current_heading = ""
    buf: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal buf
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            locator = current_heading or f"L{start_line}-{end_line}"
            chunks.append((body, _slug(locator)))
        buf = []

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("#"):
            flush(idx - 1)
            current_heading = line.strip().lstrip("#").strip()
            start_line = idx + 1
            continue
        if not line.strip() and buf:
            flush(idx - 1)
            start_line = idx + 1
            continue
        buf.append(line)
    flush(len(lines))
    return chunks


def _slug(s: str) -> str:
    return "-".join(s.lower().split()) if s else ""
