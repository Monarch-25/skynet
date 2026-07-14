"""Core orchestration subpackage.

Holds the shared types (``types.py``), plus the turn loop, prompt builder,
context compressor, memory manager, and token estimator. Importing
``skynet.core`` re-exports the shared message/tool types for convenience, but
deliberately *not* the heavy orchestrators — those are imported explicitly to
avoid a circular import at package load (loop.py imports tools, tools import
memory_manager, etc.).
"""

from __future__ import annotations

from skynet.core.types import (  # noqa: F401
    Message,
    ProviderResponse,
    ToolCall,
    ToolKind,
    ToolResult,
)
