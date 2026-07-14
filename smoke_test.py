"""End-to-end smoke test for the gateway against the live Z.AI glm-5.2 model.

Run from the project root:

    python smoke_test.py

It verifies:
  1. The default (env-built) model can answer a simple prompt.
  2. Explicit GatewayConfig construction works.
  3. Streaming yields incremental text.
"""

from __future__ import annotations

import sys

from gateway import GatewayConfig, build_llm, chat, chat_stream


def main() -> int:
    print("=" * 60)
    print(" Z.AI gateway smoke test (model: glm-5.2)")
    print("=" * 60)

    # --- 1. Default (env-loaded) model -----------------------------------
    print("\n[1] Default model via gateway.chat() ...")
    reply = chat(
        "Reply with exactly: PONG",
        system="You are a terse echo bot. Output only what is requested.",
    )
    print(f"    reply: {reply!r}")
    assert "pong" in reply.strip().lower(), f"Expected 'PONG' in reply, got: {reply!r}"
    print("    OK")

    # --- 2. Explicit config ----------------------------------------------
    print("\n[2] Explicit GatewayConfig -> build_llm() ...")
    cfg = GatewayConfig(model="glm-5.2", temperature=0.3)
    llm = build_llm(cfg)
    from langchain_core.language_models import BaseChatModel

    assert isinstance(llm, BaseChatModel)
    reply2 = chat("What is 7 + 5? Reply with just the number.", llm=llm)
    print(f"    reply: {reply2!r}")
    assert "12" in reply2, f"Expected '12' in reply, got: {reply2!r}"
    print("    OK")

    # --- 3. Streaming -----------------------------------------------------
    print("\n[3] Streaming via gateway.chat_stream() ...")
    streamed = ""
    for tok in chat_stream("Count from 1 to 5, comma separated."):
        streamed += tok
        print(f"    chunk: {tok!r}")
    print(f"    full : {streamed!r}")
    assert streamed.strip(), "Stream produced no output"
    print("    OK")

    print("\nAll checks passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
