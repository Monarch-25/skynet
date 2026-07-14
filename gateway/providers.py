"""Provider factory.

This module is the single seam that knows how to build a concrete LangChain
``BaseChatModel`` for a given provider. Callers depend only on
``BaseChatModel`` -- the LangChain interface -- not on any specific provider
class. When Bedrock lands, add one branch here and nothing else in the codebase
changes.

Design rationale:
    * LangChain's ``ChatOpenAI`` and ``ChatBedrock`` (from ``langchain-aws``)
      both satisfy ``BaseChatModel``, so returning that type keeps the gateway
      provider-agnostic.
    * Z.AI is OpenAI-compatible, so it reuses ``ChatOpenAI`` pointed at the
      Z.AI ``base_url``. This mirrors the official Z.AI LangChain guide.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from gateway.config import PROVIDER_BEDROCK, PROVIDER_ZAI, GatewayConfig


class UnsupportedProviderError(ValueError):
    """Raised when the config names a provider the gateway cannot build."""


def build_chat_model(config: GatewayConfig) -> BaseChatModel:
    """Return a configured ``BaseChatModel`` for the configured provider.

    Args:
        config: Resolved gateway configuration (key must be present).

    Returns:
        A LangChain chat model. Use ``.invoke(messages)`` to call it.
    """
    if config.provider == PROVIDER_ZAI:
        return _build_zai(config)
    if config.provider == PROVIDER_BEDROCK:
        return _build_bedrock(config)
    raise UnsupportedProviderError(
        f"Unknown provider {config.provider!r}. "
        f"Supported providers: {PROVIDER_ZAI!r}, {PROVIDER_BEDROCK!r}."
    )


def _zai_kwargs(config: GatewayConfig) -> Dict[str, Any]:
    """Translate GatewayConfig into ChatOpenAI constructor kwargs for Z.AI."""
    kwargs: Dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    # Z.AI reasoning mode is passed through OpenAI's `extra_body` extension.
    if config.thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return kwargs


def _build_zai(config: GatewayConfig) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**_zai_kwargs(config))


def _build_bedrock(config: GatewayConfig) -> BaseChatModel:
    """Build a Bedrock-backed chat model.

    Implemented lazily so the rest of the gateway works without
    ``langchain-aws`` installed. When this provider is activated, install
    ``langchain-aws`` and ``boto3``, then flesh out the model id / region map.
    """
    try:
        from langchain_aws import ChatBedrock  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'bedrock' provider requires 'langchain-aws' and 'boto3'. "
            "Install them with: pip install langchain-aws boto3"
        ) from exc

    return ChatBedrock(
        # Map z.ai model id -> Bedrock model id as needed here.
        model_id=config.model,
        temperature=config.temperature,
        # Credentials/region come from the standard boto3 chain.
    )
