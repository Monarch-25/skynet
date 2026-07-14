"""Gateway package.

A thin, modular abstraction over LLM providers (Z.AI today, Bedrock tomorrow).

Public API:
    from gateway import build_llm, chat, GatewayConfig
"""

from gateway.config import GatewayConfig
from gateway.client import chat, chat_stream, build_llm

__all__ = ["GatewayConfig", "chat", "chat_stream", "build_llm"]

__version__ = "0.1.0"
