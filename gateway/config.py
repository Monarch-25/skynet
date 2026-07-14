"""Configuration for the gateway.

Resolution order for the API key (first non-empty wins):
    1. explicit argument to GatewayConfig(api_key=...)
    2. ZAI_API_KEY environment variable
    3. a .env file in the project root (loaded by python-dotenv)

The key is never printed or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    # Optional dependency; the gateway works without it if env vars are set.
    from dotenv import load_dotenv

    # Look for a .env next to the project root (parent of this package).
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:  # pragma: no cover
    pass


# Provider identifiers understood by the factory in `providers.py`.
PROVIDER_ZAI = "zai"
PROVIDER_BEDROCK = "bedrock"


@dataclass(frozen=True)
class GatewayConfig:
    """Immutable gateway configuration.

    Attributes:
        provider:    Which backend to target (``"zai"`` today).
        model:       Model identifier passed to the provider.
        api_key:     Secret credential. Resolved from arg / env / .env.
        base_url:    OpenAI-compatible endpoint for the Z.AI provider.
        temperature: Sampling temperature (Z.AI supports (0, 1]; 0 is invalid).
        max_tokens:  Cap on generated tokens per response. None = provider default.
        timeout:     Per-request timeout in seconds.
        max_retries: Number of retries the underlying client performs.
        thinking:    If True, enable Z.AI reasoning output (reasoning_content).
    """

    provider: str = PROVIDER_ZAI
    model: str = "glm-5.2"
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: str = "https://api.z.ai/api/paas/v4/"
    temperature: float = 0.6
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 3
    thinking: bool = False

    def __post_init__(self) -> None:
        # Resolve the API key from the environment if not supplied directly.
        if not self.api_key:
            object.__setattr__(
                self, "api_key", os.getenv("ZAI_API_KEY") or os.getenv("ZAI_KEY")
            )
        if not self.api_key:
            raise ValueError(
                "No API key found. Set ZAI_API_KEY (env var or .env file) "
                "or pass GatewayConfig(api_key=...)."
            )

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Build a config purely from environment variables / .env file."""
        return cls(
            provider=os.getenv("GATEWAY_PROVIDER", PROVIDER_ZAI),
            model=os.getenv("GATEWAY_MODEL", "glm-5.2"),
            base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
            temperature=float(os.getenv("GATEWAY_TEMPERATURE", "0.6")),
            max_tokens=int(v) if (v := os.getenv("GATEWAY_MAX_TOKENS")) else None,
            timeout=float(os.getenv("GATEWAY_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("GATEWAY_MAX_RETRIES", "3")),
            thinking=os.getenv("GATEWAY_THINKING", "").lower() in ("1", "true", "yes"),
        )
