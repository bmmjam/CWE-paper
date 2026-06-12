"""Entry points for `agentclass-pilot` and other CLI scripts."""

from __future__ import annotations

import argparse

from agentclass.config import settings
from agentclass.llm import chat


def pilot() -> None:
    """Smoke-test that .env is wired and OpenRouter responds."""
    parser = argparse.ArgumentParser(description="Pilot smoke-test for AgentCLASS LLM stack.")
    parser.add_argument("--model", default=settings.llm_dev, help="OpenRouter model id")
    parser.add_argument(
        "--prompt",
        default="Reply with the single word OK.",
        help="Prompt to send",
    )
    args = parser.parse_args()

    response = chat(
        messages=[{"role": "user", "content": args.prompt}],
        model=args.model,
        max_tokens=16,
        use_cache=False,
    )
    content = response["choices"][0]["message"]["content"]
    print(f"model={args.model}  reply={content!r}")


if __name__ == "__main__":
    pilot()
