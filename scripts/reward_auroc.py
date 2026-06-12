#!/usr/bin/env python3
"""Reward-calibration analysis: AUROC of the rollout reward as a predictor of
rollout correctness, self-eval vs external judge (RQ3 diagnostics).

Pools (reward, correct) pairs over all logged MCTS rollouts. AUROC is computed
by the rank statistic (Mann-Whitney U with tie correction).

Run:  .venv/bin/python scripts/reward_auroc.py
"""

from __future__ import annotations

import json
from pathlib import Path

RES = Path("results/pilot_ex")
TAGS = {
    "self_eval": "bigvul_diffmsg_pilot_gpt-4-1-nano_only-mcts_tax16-mcts_reason8",
    "judge": "bigvul_diffmsg_pilot_gpt-4-1-nano_external_judge_only-mcts_tax16-mcts_reason8",
}
CELLS = [("mcts_tax", "k16"), ("mcts_reason", "k8")]


def auroc(pairs: list[tuple[float, bool]]) -> float:
    """Rank-based AUROC with midranks for ties."""
    pairs = sorted(pairs, key=lambda x: x[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    pos = [r for r, (_, c) in zip(ranks, pairs) if c]
    npos, nneg = len(pos), n - len(pos)
    if not npos or not nneg:
        return float("nan")
    u = sum(pos) - npos * (npos + 1) / 2
    return u / (npos * nneg)


def main() -> None:
    for source, tag in TAGS.items():
        for strat, k in CELLS:
            pairs_leaf: list[tuple[float, bool]] = []
            files = sorted(RES.glob(f"{tag}_{strat}_{k}_s*.jsonl"))
            for f in files:
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    if "rollouts" not in r:
                        continue
                    gold = r["gold"]
                    for cwe, rew in r["rollouts"]:
                        pairs_leaf.append((float(rew), cwe == gold))
            if not pairs_leaf:
                print(f"{source:10} {strat:12} no rollouts found"); continue
            frac = sum(c for _, c in pairs_leaf) / len(pairs_leaf)
            print(f"{source:10} {strat:12} rollouts={len(pairs_leaf):6d} "
                  f"P(correct)={frac:.3f} AUROC(leaf)={auroc(pairs_leaf):.3f}")


if __name__ == "__main__":
    main()
