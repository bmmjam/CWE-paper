#!/usr/bin/env python3
"""Pilot-scale, multi-seed, PARALLEL run for the cost-lever result (nano).

Confirms the headline at submission grade: on a cheap local-class model, does
test-time search beat greedy / self-consistency at MATCHED token budget? Runs
greedy + SC{4,8} + MCTS{8,16} (two matched-budget points) over N examples x 3
seeds, parallelised across examples (the harness is otherwise sequential, which
makes pilot scale infeasible at ~4s/call).

Writes results/pilot_<split>.csv: one row per (strategy, knob, seed) plus the
per-cell mean+std across seeds is computed in analysis.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --limit 300 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentclass.metrics import aggregate
from agentclass.strategies import REGISTRY
from agentclass.tree import CweTree

SEEDS = [13, 42, 1337, 2025, 31415]
CELLS = [  # (strategy, knob_name, knob_val, extra)
    ("greedy", "-", 0, {}),
    ("self_consistency", "n", 4, {"temperature": 0.7}),
    ("self_consistency", "n", 8, {"temperature": 0.7}),
    ("best_of_n", "n", 8, {"temperature": 0.7, "reward_source": "self_eval"}),
    ("beam", "width", 4, {}),
    ("mcts_tax", "iterations", 8, {"c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
    ("mcts_tax", "iterations", 16, {"c_exp": 1.0, "reward_source": "self_eval", "rollout_temp": 0.7}),
    ("mcts_reason", "iterations", 2, {"c_exp": 1.0, "reward_source": "self_eval",
                                      "rollout_temp": 0.7, "expand_k": 3}),
    ("mcts_reason", "iterations", 4, {"c_exp": 1.0, "reward_source": "self_eval",
                                      "rollout_temp": 0.7, "expand_k": 3}),
    ("mcts_reason", "iterations", 6, {"c_exp": 1.0, "reward_source": "self_eval",
                                      "rollout_temp": 0.7, "expand_k": 3}),
    ("mcts_reason", "iterations", 8, {"c_exp": 1.0, "reward_source": "self_eval",
                                      "rollout_temp": 0.7, "expand_k": 3}),
]
COLS = ["model", "strategy", "knob", "knob_val", "seed", "n", "leaf_acc",
        "pillar_acc", "hier_f1", "mean_tokens"]


def run_cell(tree, data, fn, knobs, workers, model) -> tuple[dict, int, list]:
    def one(ex):
        try:
            res = fn(tree, ex["func"], model, **knobs)
        except Exception as e:  # noqa: BLE001 - one bad example must not kill the cell
            print(f"  ! example {ex.get('id','?')} failed: {type(e).__name__}", flush=True)
            return None
        pp = res.path[1:] if res.path and res.path[0] == "ROOT" else res.path
        row = {"id": ex["id"], "gold": ex["cwe"], "gold_path": tree.path_to_root(ex["cwe"]),
               "pred": res.leaf, "pred_path": pp, "total_tokens": res.total_tokens}
        if "rollouts" in res.meta:  # (leaf, reward) pairs for reward-AUROC analysis
            row["rollouts"] = res.meta["rollouts"]
        return row
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, data))
    ok = [r for r in rows if r is not None]
    return aggregate(ok), len(rows) - len(ok), ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigvul_diffmsg_pilot.jsonl")
    ap.add_argument("--tree", default="data/cwe_tree_coarse.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds for stochastic cells")
    ap.add_argument("--reward", default=None, choices=["self_eval", "external_judge"],
                    help="override reward_source for mcts/bon cells (RQ3)")
    ap.add_argument("--only", default=None,
                    help="comma list of 'strategy' or 'strategy:knobval' to keep")
    ap.add_argument("--tag", default="",
                    help="extra tag for output filenames (e.g. 'fulltree')")
    args = ap.parse_args()

    model = args.model
    seeds_all = SEEDS[: args.seeds]
    cells = CELLS
    if args.only:
        want = set(args.only.split(","))
        cells = [c for c in CELLS if c[0] in want or f"{c[0]}:{c[2]}" in want]
    tree = CweTree.load(args.tree)
    data = [json.loads(l) for l in Path(args.data).read_text().splitlines()][: args.limit]
    split = Path(args.data).stem
    msafe = model.replace("/", "_").replace(".", "-")
    tag = f"_{args.reward}" if args.reward else ""
    if args.tag:
        tag += f"_{args.tag}"
    if args.only:  # subset runs get their own file, never clobber the full run
        tag += "_only-" + args.only.replace(",", "-").replace(":", "")
    csv_path = Path(f"results/pilot_{split}_{msafe}{tag}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, COLS).writeheader()

    agg: dict = defaultdict(list)  # (strategy,knob_val) -> [leaf per seed]
    for strategy, kname, kval, extra in cells:
        fn = REGISTRY[strategy]
        seeds = [seeds_all[0]] if strategy in ("greedy", "beam") else seeds_all  # deterministic
        for seed in seeds:
            knobs = dict(extra, seed=seed)
            if args.reward and "reward_source" in knobs:
                knobs["reward_source"] = args.reward
            if kname == "n":
                knobs["n"] = kval
            elif kname == "iterations":
                knobs["iterations"] = kval
            elif kname == "width":
                knobs["width"] = kval
            t0 = time.time()
            s, failed, exrows = run_cell(tree, data, fn, knobs, args.workers, model)
            exdir = Path("results/pilot_ex"); exdir.mkdir(parents=True, exist_ok=True)
            exf = exdir / f"{split}_{msafe}{tag}_{strategy}_k{kval}_s{seed}.jsonl"
            with open(exf, "w", encoding="utf-8") as ef:
                for r in exrows:
                    ef.write(json.dumps(r, ensure_ascii=False) + "\n")
            row = {"model": model, "strategy": strategy, "knob": kname, "knob_val": kval,
                   "seed": seed, "n": s["n"], "leaf_acc": s["leaf_acc"],
                   "pillar_acc": s["pillar_acc"], "hier_f1": s["hier_f1"],
                   "mean_tokens": s["mean_tokens"]}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, COLS).writerow(row)
            agg[(strategy, kval)].append((s["leaf_acc"], s["pillar_acc"]))
            print(f"[done] {strategy:16} {kname}={kval:<3} seed={seed} "
                  f"leaf={s['leaf_acc']:.3f} pillar={s['pillar_acc']:.3f} "
                  f"tok={s['mean_tokens']:.0f} n={s['n']} failed={failed} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\n=== per-cell mean across seeds (leaf / pillar) ===")
    for (strat, kv), vals in agg.items():
        leaf = [v[0] for v in vals]; pil = [v[1] for v in vals]
        sd = statistics.pstdev(leaf) if len(leaf) > 1 else 0.0
        print(f"  {strat:16} k={kv:<3}  leaf={statistics.mean(leaf):.3f}±{sd:.3f}  "
              f"pillar={statistics.mean(pil):.3f}  (seeds={len(vals)})")
    print(f"\ncsv -> {csv_path}")


if __name__ == "__main__":
    main()
