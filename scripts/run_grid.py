#!/usr/bin/env python3
"""Budget-grid driver: strategies x policy models on a PrimeVul split.

Writes incremental results/grid_<split>.csv (one row per cell) plus per-cell
JSONL. Cells are ordered strategy-outer so the fast, informative ones (greedy,
then self-consistency) finish first and populate the policy-strength (alpha)
axis before the slow MCTS cells.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_grid.py --data data/primevul_dev.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agentclass.config import settings
from agentclass.metrics import aggregate
from agentclass.strategies import REGISTRY
from agentclass.tree import CweTree

MODELS = [
    "gpt-4.1-nano",                  # OpenAI, weak end of alpha
    "deepseek/deepseek-chat-v3.1",   # OpenRouter, mid
    "anthropic/claude-haiku-4.5",    # OpenRouter, strong end
]

CELLS = [  # (strategy, knobs) — ordered fast->slow; greedy first to get alpha axis early
    ("greedy", {}),
    ("self_consistency", {"n": 4, "temperature": 0.7}),
    ("mcts_tax", {"iterations": 6, "c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
]

CSV_COLS = ["model", "strategy", "knobs", "n", "leaf_acc", "pillar_acc",
            "hier_f1", "mean_tokens", "total_tokens", "elapsed_s"]


def safe(s: str) -> str:
    return s.replace("/", "_").replace(".", "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/primevul_dev.jsonl")
    ap.add_argument("--tree", default="data/cwe_tree_pilot.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tree = CweTree.load(args.tree)
    data = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    if args.limit:
        data = data[: args.limit]

    split = Path(args.data).stem
    csv_path = Path(f"results/grid_{split}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    grid_dir = Path("results/grid"); grid_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, CSV_COLS).writeheader()

    for strategy, knobs in CELLS:
        fn = REGISTRY[strategy]
        for model in MODELS:
            t0 = time.time()
            rows = []
            for ex in data:
                res = fn(tree, ex["func"], model, **knobs)
                pred_path = res.path[1:] if res.path and res.path[0] == "ROOT" else res.path
                rows.append({"id": ex["id"], "gold": ex["cwe"], "gold_path": ex["cwe_path"],
                             "pred": res.leaf, "pred_path": pred_path,
                             "total_tokens": res.total_tokens})
            summ = aggregate(rows)
            row = {
                "model": model, "strategy": strategy, "knobs": json.dumps(knobs),
                "n": summ["n"], "leaf_acc": summ["leaf_acc"], "pillar_acc": summ["pillar_acc"],
                "hier_f1": summ["hier_f1"], "mean_tokens": summ["mean_tokens"],
                "total_tokens": summ["total_tokens"], "elapsed_s": round(time.time() - t0, 1),
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, CSV_COLS).writerow(row)
            with open(grid_dir / f"{strategy}_{safe(model)}_{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[done] {strategy:16} {model:30} "
                  f"leaf={row['leaf_acc']:.3f} pillar={row['pillar_acc']:.3f} "
                  f"tok={row['mean_tokens']:.0f} t={row['elapsed_s']}s", flush=True)

    print(f"grid complete -> {csv_path}")


if __name__ == "__main__":
    main()
