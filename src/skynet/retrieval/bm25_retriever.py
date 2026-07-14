"""BM25 retriever (v1 backend).

Uses ``rank_bm25.BM25Okapi`` over simple whitespace+lowercase tokenization. Two
separate corpora are maintained — source chunks and KB notes — so queries can be
scoped. The corpus is held in memory and rebuilt on demand; restarts re-index
from the scanner, which is cheap for personal-sized corpora and avoids dragging
in a persistence layer for v1 (plan §6 explicitly defers SQLite caching).

ACL filtering happens *before* ranking so restricted content can't leak via
ranking side channels (plan §4).
"""

from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from skynet.retrieval.base import Chunk, ChunkKind, RawChunk

_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


class Bm25Retriever:
    """In-memory BM25 over source + KB chunks, ACL-filtered before ranking."""

    def __init__(self) -> None:
        # ``_chunks[kind]`` holds the RawChunk list; ``_bm25[kind]`` the index.
        self._chunks: dict[ChunkKind, list[RawChunk]] = {"source": [], "kb_note": []}
        self._tokens: dict[ChunkKind, list[list[str]]] = {"source": [], "kb_note": []}
        self._bm25: dict[ChunkKind, BM25Okapi | None] = {"source": None, "kb_note": None}

    # ----- indexing ----------------------------------------------------
    def index(self, chunks: Iterable[RawChunk]) -> int:
        """Group-index: replace any existing entries for each source_path."""
        count = 0
        # Bucket by kind and dedupe by (source_path, locator) to avoid dups.
        seen: dict[ChunkKind, set[tuple[str, str]]] = {"source": set(), "kb_note": set()}
        for c in chunks:
            # Remove any prior chunks for this exact source so re-indexing is
            # an upsert, not an append.
            self._chunks[c.kind] = [
                ex for ex in self._chunks[c.kind] if ex.source_path != c.source_path
            ]
            self._chunks[c.kind].append(c)
            count += 1
        self._rebuild()
        return count

    def upsert_kb(self, chunk: RawChunk) -> None:
        """Add or replace a single KB note chunk by source_path."""
        self._chunks["kb_note"] = [
            ex for ex in self._chunks["kb_note"] if ex.source_path != chunk.source_path
        ]
        self._chunks["kb_note"].append(chunk)
        self._rebuild(kind="kb_note")

    def remove(self, source_path: str) -> int:
        removed = 0
        for kind in self._chunks:
            before = len(self._chunks[kind])
            self._chunks[kind] = [
                c for c in self._chunks[kind] if c.source_path != source_path
            ]
            removed += before - len(self._chunks[kind])
        if removed:
            self._rebuild()
        return removed

    def _rebuild(self, kind: ChunkKind | None = None) -> None:
        kinds = [kind] if kind else list(self._chunks)
        for k in kinds:
            self._tokens[k] = [tokenize(c.text) for c in self._chunks[k]]
            self._bm25[k] = (
                BM25Okapi(self._tokens[k]) if self._tokens[k] else None
            )

    def size(self, kind: ChunkKind | None = None) -> int:
        if kind:
            return len(self._chunks[kind])
        return sum(len(v) for v in self._chunks.values())

    # ----- querying ----------------------------------------------------
    def query(
        self,
        text: str,
        k: int = 6,
        user_id: str | None = None,
        kinds: list[ChunkKind] | None = None,
    ) -> list[Chunk]:
        target_kinds = kinds or ["source", "kb_note"]
        results: list[Chunk] = []
        q_tokens = tokenize(text)
        for kind in target_kinds:
            bm25 = self._bm25.get(kind)
            if not bm25 or not q_tokens:
                continue
            scores = bm25.get_scores(q_tokens)
            # Rank within kind, but ACL-filter the candidate set first.
            ranked_idx = sorted(
                range(len(self._chunks[kind])),
                key=lambda i: scores[i],
                reverse=True,
            )
            taken = 0
            for i in ranked_idx:
                if taken >= k:
                    break
                raw = self._chunks[kind][i]
                # ACL hook: in v1 everything is visible (default ACL allows "*");
                # this is where embedding_retriever / acls would filter.
                if not self._acl_allows(raw, user_id):
                    continue
                results.append(
                    Chunk(
                        text=raw.text,
                        source_path=raw.source_path,
                        locator=raw.locator,
                        score=float(scores[i]),
                        kind=kind,
                        title=raw.title or raw.source_path,
                        tags=raw.tags,
                    )
                )
                taken += 1
        # Re-rank across kinds by score and cap at k.
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:k]

    def _acl_allows(self, chunk: RawChunk, user_id: str | None) -> bool:
        # v1: open by default. ACL enforcement is a per-app policy hook; the
        # retriever exposes ``user_id`` so a subclass can override this.
        return True
