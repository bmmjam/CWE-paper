#!/usr/bin/env python3
"""Compute-matched iteration/budget sweep for MCTS vs self-consistency.

Answers: does MCTS keep improving with more iterations, or plateau? And, at a
matched token budget, does it beat self-consistency? (EXPERIMENTS.md §5.2, RQ1.)

We sweep MCTS iterations and SC sample-count separately, log mean_tokens per
cell, and align the two curves post-hoc by token budget (the honest
compute-matched comparison). Model-outer ordering: the fast model (nano)
finishes its full curve first, so partial overnight progress is still usable.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_sweep.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agentclass.metrics import aggregate
from agentclass.strategies import REGISTRY
from agentclass.tree import CweTree

MODELS = [
    "gpt-4.1-nano",                # OpenAI, fast -> full curve first
    "anthropic/claude-haiku-4.5",  # OpenRouter, stronger but slower
]

# (strategy, knob_name, knob_value, extra_knobs)
CELLS = [
    ("greedy", "-", 0, {}),
    ("self_consistency", "n", 4, {"temperature": 0.7}),
    ("self_consistency", "n", 8, {"temperature": 0.7}),
    ("self_consistency", "n", 16, {"temperature": 0.7}),
    ("mcts_tax", "iterations", 8, {"c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
    ("mcts_tax", "iterations", 16, {"c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
    ("mcts_tax", "iterations", 32, {"c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
]

COLS = ["model", "strategy", "knob", "knob_val", "n_examples", "leaf_acc",
        "pillar_acc", "hier_f1", "mean_tokens", "total_tokens", "elapsed_s"]


def safe(s: str) -> str:
    return s.replace("/", "_").replace(".", "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/primevul_dev.jsonl")
    ap.add_argument("--tree", default="data/cwe_tree_pilot.json")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    tree = CweTree.load(args.tree)
    data = [json.loads(l) for l in Path(args.data).read_text().splitlines()][: args.limit]
    split = Path(args.data).stem

    csv_path = Path(f"results/sweep_{split}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    grid_dir = Path("results/sweep"); grid_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, COLS).writeheader()

    for model in MODELS:
        for strategy, kname, kval, extra in CELLS:
            knobs = dict(extra)
            if kname == "n":
                knobs["n"] = kval
            elif kname == "iterations":
                knobs["iterations"] = kval
            fn = REGISTRY[strategy]
            t0 = time.time()
            rows = []
            for ex in data:
                res = fn(tree, ex["func"], model, **knobs)
                pp = res.path[1:] if res.path and res.path[0] == "ROOT" else res.path
                rows.append({"id": ex["id"], "gold": ex["cwe"],
                             "gold_path": tree.path_to_root(ex["cwe"]),
                             "pred": res.leaf, "pred_path": pp, "total_tokens": res.total_tokens})
            s = aggregate(rows)
            row = {"model": model, "strategy": strategy, "knob": kname, "knob_val": kval,
                   "n_examples": s["n"], "leaf_acc": s["leaf_acc"], "pillar_acc": s["pillar_acc"],
                   "hier_f1": s["hier_f1"], "mean_tokens": s["mean_tokens"],
                   "total_tokens": s["total_tokens"], "elapsed_s": round(time.time() - t0, 1)}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, COLS).writerow(row)
            with open(grid_dir / f"{strategy}_{kname}{kval}_{safe(model)}_{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[done] {model:30} {strategy:16} {kname}={kval:<3} "
                  f"leaf={row['leaf_acc']:.3f} pillar={row['pillar_acc']:.3f} "
                  f"tok={row['mean_tokens']:.0f} t={row['elapsed_s']}s", flush=True)

    print(f"sweep complete -> {csv_path}")


if __name__ == "__main__":
    main()
