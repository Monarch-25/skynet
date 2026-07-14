"""High-level client API.

These helpers hide provider construction so application code only ever imports
``gateway.chat`` / ``gateway.chat_stream`` / ``gateway.build_llm``.

A singleton default config is created lazily from the environment, which means
``chat("hi")`` just works once ``ZAI_API_KEY`` is set. Tests and advanced users
can pass an explicit ``GatewayConfig`` or a pre-built ``BaseChatModel``.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from gateway.config import GatewayConfig
from gateway.providers import build_chat_model

# A message can be supplied as a langchain Message object or a plain string.
MessageLike = Union[str, BaseMessage]

# Module-level cache for the lazily-built default model + config.
_default_llm: Optional[BaseChatModel] = None
_default_config: Optional[GatewayConfig] = None


def _coerce_messages(
    messages_or_text: Union[MessageLike, Iterable[MessageLike]],
    system: Optional[str],
) -> List[BaseMessage]:
    """Normalise the accepted inputs into a list of LangChain messages."""
    if isinstance(messages_or_text, (str, BaseMessage)):
        items: List[MessageLike] = [messages_or_text]
    else:
        items = list(messages_or_text)

    result: List[BaseMessage] = []
    if system:
        result.append(SystemMessage(content=system))
    for item in items:
        if isinstance(item, BaseMessage):
            result.append(item)
        elif isinstance(item, str):
            result.append(HumanMessage(content=item))
        else:
            raise TypeError(
                f"Unsupported message type {type(item).__name__}; "
                "expected str or langchain BaseMessage."
            )
    return result


def _get_default_llm() -> BaseChatModel:
    """Build and cache the default model from environment configuration."""
    global _default_llm, _default_config
    if _default_llm is None:
        _default_config = GatewayConfig.from_env()
        _default_llm = build_chat_model(_default_config)
    return _default_llm


def build_llm(config: Optional[GatewayConfig] = None) -> BaseChatModel:
    """Construct a chat model.

    Args:
        config: Explicit configuration. If omitted, a default is built from the
            environment (cached across calls).

    Returns:
        A ready-to-use LangChain ``BaseChatModel``.
    """
    if config is None:
        return _get_default_llm()
    return build_chat_model(config)


def chat(
    messages: Union[MessageLike, Iterable[MessageLike]],
    *,
    system: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    config: Optional[GatewayConfig] = None,
) -> str:
    """Send messages to the model and return the assistant's reply as a string.

    Args:
        messages: One or more messages. Plain strings are treated as user turns.
        system:  Optional system prompt prepended to the conversation.
        llm:     Pre-built chat model. Mutually exclusive with ``config``.
        config:  Config used to build a model if ``llm`` is not given.

    Returns:
        The assistant's textual response.
    """
    model = _resolve_model(llm, config)
    msgs = _coerce_messages(messages, system)
    response: AIMessage = model.invoke(msgs)
    return response.content if isinstance(response.content, str) else str(response.content)


def chat_stream(
    messages: Union[MessageLike, Iterable[MessageLike]],
    *,
    system: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    config: Optional[GatewayConfig] = None,
) -> Iterator[str]:
    """Stream the model's reply token-by-token as a string iterator.

    Yields content deltas as they arrive. Useful for interactive UIs.
    """
    model = _resolve_model(llm, config)
    msgs = _coerce_messages(messages, system)
    for chunk in model.stream(msgs):
        text = getattr(chunk, "content", "")
        if isinstance(text, list):  # structured content blocks
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        if text:
            yield text


def _resolve_model(
    llm: Optional[BaseChatModel], config: Optional[GatewayConfig]
) -> BaseChatModel:
    if llm is not None and config is not None:
        raise ValueError("Pass either `llm` or `config`, not both.")
    if llm is not None:
        return llm
    return build_llm(config)
