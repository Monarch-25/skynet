"""Headless verification of the full skynet agent loop.

Runs without a TTY and without spending Z.AI tokens by using the MockProvider.
Verifies, in order:

    1. Config loads and paths resolve.
    2. Ingestion parses + chunks the sample source docs.
    3. BM25 retrieval returns the right chunk for a real query.
    4. The full turn loop (search -> read -> reply) runs end-to-end via mock,
       with activity events emitted at every step.
    5. The approval gate works: a KB-write tool call is *proposed* but does NOT
       land on disk until the approver approves; rejecting feeds a synthetic
       result back so the model adapts.
    6. Approved writes persist to internal_kb/notes/ AND update the BM25 KB
       index so the note is immediately retrievable.
    7. Secrets are scrubbed before persistence.
    8. The Textual app can mount headless (pip-free of a real terminal).

Run:  python verify.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# --- test plumbing ----------------------------------------------------
_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {msg}")
    if not cond:
        _failures.append(msg)


# --- 1. config --------------------------------------------------------
print("\n[1] Config loads")
from skynet.config import load_config, ProviderCfg

cfg = load_config()
# Verification always uses the mock provider so it runs offline and without
# spending Z.AI tokens, regardless of what config.yaml currently says.
cfg = cfg.__class__(
    home=cfg.home,
    provider=ProviderCfg(
        active="mock", zai=cfg.provider.zai, bedrock=cfg.provider.bedrock, mock={"model": "mock"},
    ),
    context=cfg.context, memory=cfg.memory, retrieval=cfg.retrieval, paths=cfg.paths,
)
check(cfg.home.exists(), f"home exists: {cfg.home}")
check(cfg.provider.active == "mock" or cfg.provider.active == "zai",
      f"provider active = {cfg.provider.active}")
check(cfg.knowledge_source_dir.exists(), "knowledge_source dir exists")
check(cfg.notes_dir.exists(), "notes dir exists")

# --- 2. ingestion -----------------------------------------------------
print("\n[2] Ingestion parses + chunks source docs")
from skynet.ingestion.scanner import Scanner

scanner = Scanner(cfg.knowledge_source_dir, cfg.retrieval.chunk_tokens, cfg.retrieval.chunk_overlap)
chunks = scanner.scan()
check(len(chunks) > 0, f"scanner produced {len(chunks)} chunks (>0)")
sample_text = " ".join(c.text for c in chunks[:3])
check("attention" in sample_text.lower() or "retrieval" in sample_text.lower(),
      "chunk text looks like the source docs")

# --- 3. retrieval -----------------------------------------------------
print("\n[3] BM25 retrieval returns relevant chunks")
from skynet.retrieval.bm25_retriever import Bm25Retriever

retr = Bm25Retriever()
n = retr.index(chunks)
check(n > 0, f"indexed {n} chunks")
hits = retr.query("how does scaled dot product attention work", k=3, kinds=["source"])
check(len(hits) > 0, f"query returned {len(hits)} hits")
check(
    any("attention" in (h.title + h.text).lower() for h in hits),
    "top hit is about attention",
)
print(f"        top hit: {hits[0].title}#{hits[0].locator}  score={hits[0].score:.3f}")

# --- 4. full turn loop via mock provider ------------------------------
print("\n[4] Full turn loop end-to-end (mock provider)")
from skynet.providers.mock_provider import MockProvider, Frame, default_demo_script
from skynet.runtime import build_runtime

# Force mock provider regardless of config so this always runs offline.
events: list = []
runtime = build_runtime(cfg, "default", ingest=True)
runtime.provider = MockProvider(script=default_demo_script())
runtime.loop.provider = runtime.provider
runtime.loop.sink = lambda e: events.append(e)
runtime.loop.approver = None  # auto-approve path for this step

replies = list(runtime.loop.handle("How does attention work in transformers?"))
check(len(replies) == 1, f"loop yielded one reply (got {len(replies)})")
reply = replies[0]
check("dot-product" in reply.lower() or "attention" in reply.lower(),
      f"reply mentions attention/dot-product: {reply[:80]!r}")
types_seen = {e.type for e in events}
check("turn_start" in types_seen, "turn_start event emitted")
check("tool_call" in types_seen, "tool_call event emitted")
check("tool_result" in types_seen, "tool_result event emitted")
check("turn_end" in types_seen, "turn_end event emitted")
print(f"        events: {sorted(types_seen)}")

# --- 5. approval gating on a KB write --------------------------------
print("\n[5] Approval gate: write proposed but blocked until approved")
from skynet.core.approval import ApprovalDecision, ApprovalRequest
from skynet.providers.mock_provider import MockProvider
from skynet.runtime import build_runtime

# Fresh runtime + a script that tries to write a KB note, then answers.
script = [
    Frame(calls=[(
        "kb_write_note",
        {
            "note_id": "test-attention-note",
            "title": "Attention is dot-product",
            "body": "Self-attention computes scaled dot products between queries and keys.",
            "tags": ["attention", "test"],
            "sources": ["knowledge_source/attention-is-all-you-need.md#3.2.1"],
            "confidence": "high",
        },
        "distilling what I learned about attention",
    )]),
    Frame(text="Done — I saved a note about attention."),
]
runtime2 = build_runtime(cfg, "default", ingest=False)
runtime2.provider = MockProvider(script=script)
runtime2.loop.provider = runtime2.provider

# Track approval calls.
decisions: list[ApprovalDecision] = []


class TrackingApprover:
    def __init__(self, decision: ApprovalDecision):
        self.decision = decision

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        decisions.append(self.decision)
        return self.decision


# First: REJECT the write.
runtime2.loop.approver = TrackingApprover(ApprovalDecision(approved=False, reason="test reject"))
note_path = cfg.notes_dir / "test-attention-note.md"
if note_path.exists():
    note_path.unlink()
list(runtime2.loop.handle("Write a note about attention."))
check(not note_path.exists(),
      "note file NOT created when write is rejected")
# The loop should have fed a rejection tool result back; the assistant's final
# reply is the second frame regardless, but the tool result message reflects it.
tool_msgs = [m for m in runtime2.loop.session.messages if m.role == "tool"]
check(any("REJECT" in (m.content or "") for m in tool_msgs),
      "rejection fed back to model as a synthetic tool result")

# Now: APPROVE the write on a fresh runtime.
runtime3 = build_runtime(cfg, "default", ingest=False)
runtime3.provider = MockProvider(script=list(script))
runtime3.loop.provider = runtime3.provider
runtime3.loop.approver = TrackingApprover(ApprovalDecision(approved=True))
list(runtime3.loop.handle("Write a note about attention."))
check(note_path.exists(), "note file created when write is approved")
if note_path.exists():
    content = note_path.read_text()
    check("attention" in content.lower(), "approved note content landed")
    check("test-attention-note" in content, "note id in frontmatter")

# --- 6. approved write updates the KB retrieval index ----------------
print("\n[6] Approved KB write becomes immediately retrievable")
runtime4 = build_runtime(cfg, "default", ingest=False)
# The note written in step 5 should now appear in a fresh runtime's KB index.
kb_hits = runtime4.retriever.query("attention dot product", k=3, kinds=["kb_note"])
check(
    any("test-attention-note" == h.source_path for h in kb_hits),
    f"written note is retrievable: {[h.source_path for h in kb_hits]}",
)

# --- 7. secret scrubbing ---------------------------------------------
print("\n[7] Secrets are scrubbed before persistence")
from skynet.core.scrub import scrub, contains_secret

LEAK = "key=sk-abc123def456ghi789jkl012mno345 and ZAI 29182e062b5b419f898f05d7955497ef.snkzmqnqu6IttECK"
scrubbed = scrub(LEAK)
check("sk-abc" not in scrubbed, "sk- key redacted")
check("29182e062b5b419f898f05d7955497ef" not in scrubbed, "ZAI key redacted")
check(contains_secret(LEAK), "contains_secret flags the leak")

# Confirm scrubber runs inside memory_write proposal.
runtime5 = build_runtime(cfg, "default", ingest=False)
runtime5.provider = MockProvider(script=[Frame(calls=[(
    "memory_write",
    {"target": "memory", "action": "append", "content": LEAK},
    "saving a secret by accident",
)]), Frame(text="saved")])
runtime5.loop.provider = runtime5.provider  # loop holds its own provider ref
runtime5.loop.approver = TrackingApprover(ApprovalDecision(approved=True))
list(runtime5.loop.handle("save this."))
mem = (cfg.home / "users" / "default" / "MEMORY.md").read_text()
check("29182e06" not in mem, "secret did NOT land in MEMORY.md")
check("[REDACTED]" in mem, "[REDACTED] placeholder present in MEMORY.md")

# --- 8. TUI mounts headless ------------------------------------------
print("\n[8] Textual app mounts headless")
import asyncio

from skynet.tui.app import SkynetApp

runtime_tui = build_runtime(cfg, "default", ingest=False)
app = SkynetApp(runtime_tui)
async def _mount():
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # The chat pane, feed, and input should all exist.
        assert app.query_one("#chat")
        assert app.query_one("#feed")
        assert app.query_one("#prompt")
        # Status bar reflects user/provider.
        status = app.query_one("#status")
        status_text = status.render() if hasattr(status, "render") else str(status)
        assert "default" in str(status_text)
try:
    asyncio.run(_mount())
    check(True, "app mounted, widgets present, status bar rendered")
except Exception as exc:
    check(False, f"app mount failed: {exc!r}")

# --- cleanup ----------------------------------------------------------
for n in ("test-attention-note",):
    p = cfg.notes_dir / f"{n}.md"
    if p.exists():
        p.unlink()
# reset MEMORY.md to its seeded empty state
(cfg.home / "users" / "default" / "MEMORY.md").write_text(
    "# MEMORY — default\n\n<!-- Durable facts Skynet has learned that should survive every session. -->\n"
)

# --- verdict ----------------------------------------------------------
print("\n" + "=" * 60)
if _failures:
    print(f"❌ {len(_failures)} check(s) FAILED:")
    for f in _failures:
        print(f"   - {f}")
    sys.exit(1)
print("✅ All headless verification checks passed.")
