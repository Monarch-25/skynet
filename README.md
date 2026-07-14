# skynet

A self-evolving, chat-first knowledge base for the terminal. Point it at a folder
of source documents, give it a `user_id`, and it assembles a system prompt from
durable identity/memory files, retrieves relevant chunks from both the source
folder and its own self-written internal KB, runs a tool-calling loop, and
updates its memory. Over sessions the internal KB and per-user memory accumulate
— the wiki "evolves."

Built to the spec in [`skynet_plan.md`](./skynet_plan.md). Model calls go through
the [`gateway/`](./gateway) package (Z.AI today, Bedrock later, behind one seam).

## Quick start

```bash
pip install -e .

# Headless verification (offline, no tokens spent):
python verify.py

# Launch the TUI against the real Z.AI glm-5.2 model:
python -m skynet.tui.app --user default
#   or the installed entry point:
skynet --user default

# Launch offline with the deterministic mock provider:
skynet --user default --model mock
```

The Z.AI API key lives in `skynet_home/.env` (`ZAI_API_KEY=...`), loaded
automatically — never committed.

## Layout

```
skynet_home/                  # data root (SKYNET_HOME)
  config.yaml                 # non-secret config: provider, thresholds, paths
  .env                        # secrets (ZAI_API_KEY)
  SOUL.md                     # agent identity — slot #1 of system prompt
  users/<user_id>/
    USER.md                   # who this user is (agent-curated, bounded)
    MEMORY.md                 # durable facts learned for this user (bounded)
    sessions/<id>.jsonl       # raw turn log
  knowledge_source/           # read-only folder you point skynet at
  internal_kb/notes/*.md      # skynet's self-evolving wiki (provenanced)
  logs/activity.jsonl         # structured event log

src/skynet/
  config.py                   # config load + validation
  runtime.py                  # wires provider+retriever+memory+loop
  core/
    types.py                  # Message / ToolCall / ToolResult (provider-agnostic)
    loop.py                   # the turn loop (§7): bounded tool-calling + approval
    prompt_builder.py         # stable system-prompt prefix (cacheable)
    context_compressor.py     # session compaction (§8)
    memory_manager.py         # USER.md / MEMORY.md orchestration
    tokens.py / scrub.py      # token estimate / secret redaction
    events.py / approval.py   # activity events / approval protocol
  providers/
    base.py                   # LLMProvider ABC
    zai_provider.py           # Z.AI via openai SDK (reuses gateway)
    mock_provider.py          # deterministic, for offline dev + tests
  retrieval/
    base.py                   # Retriever Protocol + Chunk
    bm25_retriever.py         # v1 BM25 (separate source + KB indices)
  ingestion/
    scanner.py / parsers.py   # folder walk, hashing, chunking (pdf/txt/md)
  kb/notes.py                 # KB note CRUD, frontmatter, backlinks, provenance
  tools/registry.py           # 11 tools, read/write/control gating
  tui/
    app.py                    # Textual App: chat + feed + KB tree + approval modal
    widgets/                  # chat_pane, activity_feed, kb_tree, approval_modal
gateway/                      # (separate) Z.AI OpenAI-compatible client
verify.py                     # headless verification — runs offline with mock
```

## How a turn works (plan §7)

1. **System prompt** assembled once at session start (SOUL → user/memory → tool
   conventions → KB-write nudge). Stable prefix → provider prefix caching works.
2. Optional **compaction** if context exceeds threshold (heuristic summary + tool-
   result collapsing; LLM-backed summarize is a drop-in).
3. **Bounded tool-calling loop** (max 8 iterations). Read tools execute
   immediately; **write tools pause for approval** — a modal shows the proposed
   diff and the model's `why`, with Approve / Edit / Reject.
4. **Stream** the final reply token-by-token.
5. Append raw turn to the session log.

## The 11 tools (plan §10)

| tool | kind | gated? |
|---|---|---|
| `source_search`, `source_read`, `kb_search`, `kb_view`, `memory_view` | read | no |
| `kb_write_note`, `kb_link`, `memory_write` | **write** | **yes** |
| `compress_context`, `todo` | read* | no |
| `ask_user` | control | pauses for input |

Every tool takes a `why` field → powers the live activity feed.

## Self-evolving KB (plan §5)

When the agent synthesizes something durable, it calls `kb_write_note` with a
`sources:` field (provenance — a source path or `session:<id>#turn-N`). Approved
writes land in `internal_kb/notes/<slug>.md` AND update the BM25 KB index, so the
note is immediately retrievable. Notes carry frontmatter (tags, links,
confidence) and backlinks are queryable.

## Verification

`python verify.py` runs 8 stages offline (forces the mock provider):
config → ingestion → BM25 retrieval → full turn loop → approval gating (reject
**and** approve paths) → KB-index sync → secret scrubbing → TUI headless mount.

```
✅ All headless verification checks passed.
```

## Status against the phased plan

| Milestone | Status |
|---|---|
| M0 skeleton loop | ✅ |
| M1 ingestion + BM25 + source tools | ✅ |
| M2 identity & memory files | ✅ |
| M3 session compaction | ✅ (heuristic; LLM summarize is a drop-in) |
| M4 internal KB | ✅ |
| M5 TUI + approval workflow | ✅ |
| M6 Bedrock provider | stub (interface ready) |
| M7 embeddings | interface ready (`Retriever` protocol) |
| M8 consolidation pass | not started |

### Known limitation
The Z.AI account had no balance at build time (HTTP 429 / error 1113), so live
model calls are blocked on billing. Everything is verified offline via the mock
provider; switch `provider.active: zai` (already set) and add credit at
https://z.ai/manage-apikey/billing to run against glm-5.2.
