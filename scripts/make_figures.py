#!/usr/bin/env python3
"""Build paper figures from pilot results.

  fig:budget  paper/figures/budget_curve.pdf   Acc vs token budget, nano, 5 seeds
  fig:cascade paper/figures/cascade.pdf        first-error level breakdown

Run:  .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path("results")
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)

CSV_MAIN = RES / ("pilot_bigvul_diffmsg_pilot_gpt-4-1-nano_only-greedy-"
                  "self_consistency8-mcts_tax16-mcts_reason8.csv")
CSV_CURVE = RES / ("pilot_bigvul_diffmsg_pilot_gpt-4-1-nano_only-mcts_reason2-"
                   "mcts_reason4-mcts_reason6-mcts_tax8-self_consistency4.csv")
CSV_BON = RES / "pilot_bigvul_diffmsg_pilot_gpt-4-1-nano_only-best_of_n8-beam4.csv"
CSV_BEAM = RES / "pilot_bigvul_diffmsg_pilot_gpt-4-1-nano_only-beam4.csv"

STYLE = {  # strategy -> (label, color, marker)
    "greedy": ("Greedy", "#555555", "s"),
    "self_consistency": ("Self-consistency", "#1f77b4", "o"),
    "best_of_n": ("Best-of-N", "#ff7f0e", "v"),
    "beam": ("Beam", "#9467bd", "P"),
    "mcts_tax": ("MCTS-Tax", "#d62728", "^"),
    "mcts_reason": ("MCTS-Reason", "#2ca02c", "D"),
}


def budget_curve() -> None:
    frames = [pd.read_csv(p) for p in (CSV_MAIN, CSV_CURVE, CSV_BON, CSV_BEAM)
              if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby(["strategy", "knob_val"])
           .agg(leaf=("leaf_acc", "mean"), leaf_sd=("leaf_acc", "std"),
                cat=("pillar_acc", "mean"), cat_sd=("pillar_acc", "std"),
                tok=("mean_tokens", "mean"), seeds=("seed", "nunique"))
           .reset_index().fillna(0.0))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True)
    for ax, metric, sd, title in [(axes[0], "leaf", "leaf_sd", "Leaf accuracy"),
                                  (axes[1], "cat", "cat_sd", "Category accuracy")]:
        for strat, (label, color, marker) in STYLE.items():
            sub = g[g.strategy == strat].sort_values("tok")
            if sub.empty:
                continue
            ax.errorbar(sub.tok / 1000, sub[metric], yerr=sub[sd], color=color,
                        marker=marker, ms=4, lw=1.4, capsize=2, label=label)
            if strat == "greedy":  # extend greedy as a flat reference line
                ax.axhline(sub[metric].iloc[0], color=color, lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tokens / example (thousands)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "budget_curve.pdf", bbox_inches="tight")
    print("budget points:")
    print(g.to_string(index=False))


def first_error(gold_path: list[str], pred_path: list[str]) -> str:
    if not pred_path or pred_path[0] != gold_path[0]:
        return "category"
    if pred_path[-1] != gold_path[-1]:
        return "leaf"
    return "correct"


def cascade() -> None:
    tag = "bigvul_diffmsg_pilot_gpt-4-1-nano_only-greedy-self_consistency8-mcts_tax16-mcts_reason8"
    cells = [("greedy", "k0"), ("mcts_tax", "k16"), ("mcts_reason", "k8")]
    frac: dict[str, dict[str, float]] = {}
    for strat, k in cells:
        counts: dict[str, float] = defaultdict(float)
        files = sorted((RES / "pilot_ex").glob(f"{tag}_{strat}_{k}_s*.jsonl"))
        assert files, f"no pilot_ex files for {strat}"
        for f in files:
            for line in f.read_text().splitlines():
                r = json.loads(line)
                counts[first_error(r["gold_path"], r["pred_path"])] += 1
        total = sum(counts.values())
        frac[strat] = {kk: v / total for kk, v in counts.items()}
        print(strat, {kk: round(v, 3) for kk, v in frac[strat].items()},
              f"(files={len(files)})")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    names = [STYLE[s][0] for s, _ in cells]
    cat = [frac[s].get("category", 0) for s, _ in cells]
    leaf = [frac[s].get("leaf", 0) for s, _ in cells]
    ok = [frac[s].get("correct", 0) for s, _ in cells]
    ax.bar(names, cat, color="#d62728", label="wrong category (unrecoverable)")
    ax.bar(names, leaf, bottom=cat, color="#ff9896", label="right category, wrong leaf")
    ax.bar(names, ok, bottom=[c + l for c, l in zip(cat, leaf)],
           color="#98df8a", label="correct leaf")
    for i, (c, l) in enumerate(zip(cat, leaf)):
        ax.text(i, c / 2, f"{c:.0%}", ha="center", va="center", fontsize=8)
        ax.text(i, c + l / 2, f"{l:.0%}", ha="center", va="center", fontsize=8)
        ax.text(i, c + l + ok[i] / 2, f"{ok[i]:.0%}", ha="center", va="center", fontsize=8)
    ax.set_ylabel("fraction of examples")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "cascade.pdf", bbox_inches="tight")


if __name__ == "__main__":
    budget_curve()
    cascade()
    print(f"figures -> {FIG}/budget_curve.pdf, {FIG}/cascade.pdf")
