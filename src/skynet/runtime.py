"""Runtime wiring — construct a ready-to-run TurnLoop from a Config.

This is the single place that knows how to instantiate providers, the
retriever, scanner, note store, and memory manager and stitch them into a
``TurnLoop``. Both the CLI and the TUI (and tests) go through ``build_runtime``
so configuration changes propagate everywhere.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from skynet.config import Config
from skynet.core.events import ActivityLog
from skynet.core.memory_manager import MemoryManager
from skynet.core.loop import TurnLoop
from skynet.ingestion.scanner import Scanner
from skynet.kb.notes import NoteStore
from skynet.providers.base import LLMProvider
from skynet.retrieval.bm25_retriever import Bm25Retriever


@dataclass
class Runtime:
    config: Config
    provider: LLMProvider
    retriever: Bm25Retriever
    notes: NoteStore
    memory: MemoryManager
    loop: TurnLoop
    scanner: Scanner


def build_provider(config: Config) -> LLMProvider:
    active = config.provider.active
    if active == "mock":
        from skynet.providers.mock_provider import MockProvider, default_demo_script
        return MockProvider(script=default_demo_script())
    if active == "zai":
        from skynet.providers.zai_provider import ZaiProvider
        zai = config.provider.zai
        return ZaiProvider(
            api_key=config.zai_api_key(),
            model=zai.get("model", "glm-5.2"),
            base_url=zai.get("base_url", "https://api.z.ai/api/paas/v4/"),
        )
    if active == "bedrock":
        try:
            from skynet.providers.bedrock_provider import BedrockProvider  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Bedrock provider not implemented yet (M6). Use 'mock' or 'zai'."
            ) from e
        return BedrockProvider(**config.provider.bedrock)
    raise ValueError(f"unknown provider: {active}")


def build_runtime(
    config: Config,
    user_id: str,
    *,
    provider: LLMProvider | None = None,
    ingest: bool = True,
) -> Runtime:
    """Wire up a Runtime for ``user_id`` and return it.

    ``ingest`` runs the initial source-folder scan + index; set False in tests
    that supply their own retriever contents.
    """
    user_dir = config.user_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "sessions").mkdir(parents=True, exist_ok=True)

    provider = provider or build_provider(config)
    retriever = Bm25Retriever()
    notes = NoteStore(config.notes_dir)
    memory = MemoryManager(
        user_dir, config.memory.user_md_char_cap, config.memory.memory_md_char_cap
    )
    scanner = Scanner(
        config.knowledge_source_dir,
        chunk_tokens=config.retrieval.chunk_tokens,
        chunk_overlap=config.retrieval.chunk_overlap,
    )
    if ingest:
        chunks = scanner.scan()
        if chunks:
            retriever.index(chunks)

    # Index existing KB notes too so kb_search works from session 1.
    from skynet.retrieval.base import RawChunk
    kb_chunks: list[RawChunk] = []
    for nid in notes.list_ids():
        n = notes.read(nid)
        if n:
            kb_chunks.append(RawChunk(
                text=f"{n.title}\n\n{n.body}", source_path=nid, locator=nid,
                kind="kb_note", title=n.title, tags=n.tags,
            ))
    if kb_chunks:
        retriever.index(kb_chunks)

    session_id = uuid.uuid4().hex[:12]
    activity_log = ActivityLog(config.home / "logs" / "activity.jsonl")
    session_log = user_dir / "sessions" / f"{session_id}.jsonl"

    loop = TurnLoop(
        provider=provider,
        memory=memory,
        soul_path=config.home / "SOUL.md",
        user_id=user_id,
        knowledge_source_root=config.knowledge_source_dir,
        retriever=retriever,
        notes=notes,
        max_iterations=config.context.max_iterations,
        compact_threshold_tokens=int(
            provider.context_length() * config.context.threshold_percent
        ),
        max_summary_tokens=config.context.max_summary_tokens,
        activity_log=activity_log,
        session_log_path=session_log,
    )
    return Runtime(config, provider, retriever, notes, memory, loop, scanner)
