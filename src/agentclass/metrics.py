"""Classification metrics (EXPERIMENTS.md §4.1)."""

from __future__ import annotations


def leaf_correct(pred_path: list[str], gold_path: list[str]) -> bool:
    return bool(pred_path) and bool(gold_path) and pred_path[-1] == gold_path[-1]


def pillar_correct(pred_path: list[str], gold_path: list[str]) -> bool:
    return bool(pred_path) and bool(gold_path) and pred_path[0] == gold_path[0]


def hierarchical_f1(pred_path: list[str], gold_path: list[str]) -> float:
    """Set-based hierarchical F1 over path nodes (Kosmopoulos et al. 2015)."""
    p, g = set(pred_path), set(gold_path)
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def aggregate(rows: list[dict]) -> dict:
    """rows: dicts with pred_path, gold_path, total_tokens. Return summary."""
    n = len(rows) or 1
    leaf = sum(leaf_correct(r["pred_path"], r["gold_path"]) for r in rows) / n
    pillar = sum(pillar_correct(r["pred_path"], r["gold_path"]) for r in rows) / n
    hf1 = sum(hierarchical_f1(r["pred_path"], r["gold_path"]) for r in rows) / n
    toks = [r.get("total_tokens", 0) for r in rows]
    return {
        "n": len(rows),
        "leaf_acc": round(leaf, 4),
        "pillar_acc": round(pillar, 4),
        "hier_f1": round(hf1, 4),
        "mean_tokens": round(sum(toks) / n, 1),
        "total_tokens": sum(toks),
    }
