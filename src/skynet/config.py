"""Configuration loading and validation.

Loads ``skynet_home/config.yaml`` (non-secret config) and ``skynet_home/.env``
(secrets). Resolves the ``skynet_home`` root via, in order of precedence:

    1. the ``SKYNET_HOME`` environment variable
    2. a ``./skynet_home`` directory next to the working directory
    3. a ``./skynet_home`` directory next to this package on disk

The config is parsed once and frozen into a ``Config`` object that the rest of
the app consumes; it exposes typed accessors rather than raw dict lookups so
typos fail loudly at the call site instead of silently returning ``None``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional
    def load_dotenv(*_a, **_k):  # type: ignore
        return False


def resolve_home() -> Path:
    """Find the skynet_home directory."""
    env = os.environ.get("SKYNET_HOME")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"SKYNET_HOME={env!r} is not a directory")
        return p
    cwd_candidate = Path.cwd() / "skynet_home"
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()
    pkg_candidate = Path(__file__).resolve().parents[3] / "skynet_home"
    if pkg_candidate.is_dir():
        return pkg_candidate.resolve()
    raise FileNotFoundError(
        "Could not find skynet_home. Set SKYNET_HOME or run from the project root."
    )


@dataclass(frozen=True)
class ProviderCfg:
    active: str
    zai: Dict[str, Any] = field(default_factory=dict)
    bedrock: Dict[str, Any] = field(default_factory=dict)
    mock: Dict[str, Any] = field(default_factory=dict)

    @property
    def active_cfg(self) -> Dict[str, Any]:
        return getattr(self, self.active)


@dataclass(frozen=True)
class ContextCfg:
    threshold_percent: float = 0.5
    protect_tail_tokens: int = 8000
    max_summary_tokens: int = 3000
    max_iterations: int = 8


@dataclass(frozen=True)
class MemoryCfg:
    user_md_char_cap: int = 1500
    memory_md_char_cap: int = 2500


@dataclass(frozen=True)
class RetrievalCfg:
    backend: str = "bm25"
    chunk_tokens: int = 650
    chunk_overlap: int = 100
    top_k_default: int = 6


@dataclass(frozen=True)
class Config:
    home: Path
    provider: ProviderCfg
    context: ContextCfg
    memory: MemoryCfg
    retrieval: RetrievalCfg
    paths: Dict[str, str]

    @property
    def knowledge_source_dir(self) -> Path:
        return (self.home / self.paths.get("knowledge_source", "knowledge_source")).resolve()

    @property
    def internal_kb_dir(self) -> Path:
        return (self.home / self.paths.get("internal_kb", "internal_kb")).resolve()

    @property
    def notes_dir(self) -> Path:
        return self.internal_kb_dir / "notes"

    def user_dir(self, user_id: str) -> Path:
        return (self.home / "users" / user_id).resolve()

    def zai_api_key(self) -> str | None:
        return os.environ.get("ZAI_API_KEY") or os.environ.get("ZAI_KEY")


def load_config(home: Path | None = None) -> Config:
    """Load and validate configuration from ``skynet_home``."""
    home = home or resolve_home()
    # Load .env from home so secrets land in os.environ.
    load_dotenv(home / ".env")

    raw_path = home / "config.yaml"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing config: {raw_path}")
    raw: Dict[str, Any] = yaml.safe_load(raw_path.read_text(encoding="utf-8")) or {}

    provider = ProviderCfg(
        active=raw.get("provider", {}).get("active", "zai"),
        zai=raw.get("provider", {}).get("zai", {}),
        bedrock=raw.get("provider", {}).get("bedrock", {}),
        mock=raw.get("provider", {}).get("mock", {}),
    )
    ctx_raw = raw.get("context", {})
    context = ContextCfg(
        threshold_percent=float(ctx_raw.get("threshold_percent", 0.5)),
        protect_tail_tokens=int(ctx_raw.get("protect_tail_tokens", 8000)),
        max_summary_tokens=int(ctx_raw.get("max_summary_tokens", 3000)),
        max_iterations=int(ctx_raw.get("max_iterations", 8)),
    )
    mem_raw = raw.get("memory", {})
    memory = MemoryCfg(
        user_md_char_cap=int(mem_raw.get("user_md_char_cap", 1500)),
        memory_md_char_cap=int(mem_raw.get("memory_md_char_cap", 2500)),
    )
    ret_raw = raw.get("retrieval", {})
    retrieval = RetrievalCfg(
        backend=ret_raw.get("backend", "bm25"),
        chunk_tokens=int(ret_raw.get("chunk_tokens", 650)),
        chunk_overlap=int(ret_raw.get("chunk_overlap", 100)),
        top_k_default=int(ret_raw.get("top_k_default", 6)),
    )
    paths = raw.get("paths", {})

    cfg = Config(
        home=home,
        provider=provider,
        context=context,
        memory=memory,
        retrieval=retrieval,
        paths=paths,
    )

    # Eager validation so a misconfig fails at startup, not mid-turn.
    cfg.knowledge_source_dir.mkdir(parents=True, exist_ok=True)
    cfg.internal_kb_dir.mkdir(parents=True, exist_ok=True)
    cfg.notes_dir.mkdir(parents=True, exist_ok=True)
    return cfg
