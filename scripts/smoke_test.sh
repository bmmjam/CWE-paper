#!/usr/bin/env bash
# Quick check that .env is loaded and OpenRouter responds.
# Run from repo root:  ./scripts/smoke_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found at repo root. Copy .env.example and fill OPENROUTER_API_KEY." >&2
    exit 1
fi

if grep -q "PASTE_YOUR_OPENROUTER_KEY_HERE" .env; then
    echo "ERROR: .env still has placeholder OPENROUTER_API_KEY. Fill in a real key." >&2
    exit 1
fi

python -m agentclass.cli --prompt "Reply with the single word OK."
