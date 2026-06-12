"""Run a test-time-search strategy over a PrimeVul split and report metrics.

Logs every example (prediction + tokens) to results/<run>.jsonl for audit, and
prints the aggregate leaf/pillar/hier-F1 + per-example token budget.

Examples:
  .venv/bin/python -m agentclass.run --data data/primevul_dev.jsonl \
      --strategy greedy --model gpt-4.1-nano --limit 2
  .venv/bin/python -m agentclass.run --data data/primevul_dev.jsonl \
      --strategy mcts_tax --model gpt-4.1-nano --iterations 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentclass.config import settings
from agentclass.metrics import aggregate
from agentclass.strategies import REGISTRY
from agentclass.tree import CweTree


def load_jsonl(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tree", default=None, help="tree json (default: settings.cwe_tree_path)")
    ap.add_argument("--strategy", required=True, choices=list(REGISTRY))
    ap.add_argument("--model", default=settings.llm_cheap)
    ap.add_argument("--limit", type=int, default=None)
    # search knobs (only the relevant one is used per strategy)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--c-exp", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rollout-temp", type=float, default=0.7)
    ap.add_argument("--reward-source", default="self_eval", choices=["self_eval", "external_judge"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tree = CweTree.load(args.tree or settings.cwe_tree_path)
    data = load_jsonl(args.data, args.limit)
    fn = REGISTRY[args.strategy]
    knobs = dict(n=args.n, width=args.width, iterations=args.iterations,
                 c_exp=args.c_exp, temperature=args.temperature,
                 rollout_temp=args.rollout_temp, reward_source=args.reward_source)

    provider = settings.provider_for(args.model)
    print(f"strategy={args.strategy} model={args.model} provider={provider} "
          f"n_examples={len(data)} knobs={ {k:v for k,v in knobs.items()} }")

    rows, t0 = [], time.time()
    for i, ex in enumerate(data):
        res = fn(tree, ex["func"], args.model, **knobs)
        pred_path = res.path[1:] if res.path and res.path[0] == "ROOT" else res.path
        rows.append({
            "id": ex["id"],
            "gold": ex["cwe"],
            "gold_path": tree.path_to_root(ex["cwe"]),  # derive from active tree
            "pred": res.leaf,
            "pred_path": pred_path,
            "total_tokens": res.total_tokens,
            "n_policy_calls": res.n_policy_calls,
            "n_reward_calls": res.n_reward_calls,
            "meta": res.meta,
        })
        ok = "✓" if res.leaf == ex["cwe"] else "·"
        print(f"  [{i+1}/{len(data)}] {ok} gold={ex['cwe']:10} pred={res.leaf:10} "
              f"tok={res.total_tokens:6} calls(p/r)={res.n_policy_calls}/{res.n_reward_calls}")

    summary = aggregate(rows)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out or f"results/{args.strategy}_{Path(args.data).stem}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
