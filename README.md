# When Does Test-Time Search Help LLMs Classify Vulnerabilities?

Compute-matched comparison of test-time strategies (greedy, self-consistency,
best-of-N, beam, MCTS over the CWE taxonomy, MCTS over a reasoning trajectory)
on hierarchical CWE classification of code patches. The study is diagnostic:
the question is *under what conditions* structured search pays for its tokens,
isolating three axes — policy strength, reward fidelity, and search space.

Headline pilot result: searching a *reasoning trajectory* beats searching the
taxonomy by +12 leaf-accuracy points at half the token budget, and the gain
comes from the reasoning scaffold rather than the amount of search. Full
protocol and the locked experiment policy are in [EXPERIMENTS.md](EXPERIMENTS.md);
the paper draft is in `paper/`.

## Layout

```
src/agentclass/     policy step, reward, strategies (greedy..MCTS), metrics
scripts/            data builders, pilot runners, figure generation
data/               pilot splits (jsonl, tracked) + CWE trees with SHA-256
results/            per-cell CSVs and per-example logs behind every number
paper/              LaTeX source; figures built by scripts/make_figures.py
tests/
```

Raw dataset snapshots (BigVul parquet, MITRE CWE XML) are not tracked; the
build scripts download them on first run.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
cp .env.example .env   # set OPENAI_API_KEY and/or OPENROUTER_API_KEY
./scripts/smoke_test.sh
```

Model routing: ids without `/` (e.g. `gpt-4.1-nano`) go to the OpenAI API, ids
with `/` (e.g. `anthropic/claude-haiku-4.5`) to OpenRouter.

## Reproducing the pilot

All LLM calls are cached on disk (`.cache/llm`, keyed by prompt + model + a
per-sample nonce) and logged; re-running a finished cell is free and
deterministic.

```bash
# 1. CWE trees (full MITRE view-1000 + the 7-category coarse tree)
python scripts/build_cwe_tree.py
python scripts/build_coarse_tree.py

# 2. BigVul diff+commit-message splits (dev 50 / pilot 500)
INCLUDE_MSG=1 python scripts/build_bigvul_diff.py

# 3. Pilot grids (examples; see results/*.csv for the exact cells used)
python scripts/run_pilot.py --limit 300 --seeds 5 --model gpt-4.1-nano \
    --only "greedy,self_consistency:8,mcts_tax:16,mcts_reason:8"
python scripts/run_pilot.py --limit 150 --seeds 3 \
    --model anthropic/claude-haiku-4.5 --only "mcts_tax:16,mcts_reason:8"
python scripts/run_pilot.py --limit 300 --seeds 3 --model gpt-4.1-nano \
    --reward external_judge --only "mcts_tax:16,mcts_reason:8"

# 4. Figures and paper
python scripts/make_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Each run appends one row per (strategy, knob, seed) to `results/pilot_*.csv`
and writes per-example predictions (and MCTS reward rollouts) to
`results/pilot_ex/`, which is what the figures and significance tests consume.

## Development

```bash
ruff check src/ scripts/
pytest
mypy src/agentclass
```

## Notes

- `.env` holds API keys and must never be committed (`git check-ignore .env`
  should match). If a key leaks, revoke it immediately.
- Data sources are public: BigVul (via the `bstee615/bigvul` HF mirror) and
  the MITRE CWE corpus. No private code or labels are used.
