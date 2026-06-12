"""Test-time search strategies (EXPERIMENTS.md §3.1).

All strategies share the same policy step (``policy.rank_children``) and reward
(``reward.score_path``); they differ only in HOW they search. Each returns a
``PredResult`` carrying the predicted path and the per-example token/call budget,
which is what the compute-matched comparison consumes.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from agentclass.policy import rank_children, reason_commit, reason_propose
from agentclass.reward import score_path
from agentclass.tree import ROOT, CweTree


@dataclass
class PredResult:
    path: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_policy_calls: int = 0
    n_reward_calls: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def leaf(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class _Acc:
    """Accumulates tokens and call counts across the calls a strategy makes."""

    def __init__(self) -> None:
        self.pt = self.ct = self.npc = self.nrc = 0

    def policy(self, toks: tuple[int, int]) -> None:
        self.pt += toks[0]; self.ct += toks[1]; self.npc += 1

    def reward(self, toks: tuple[int, int]) -> None:
        self.pt += toks[0]; self.ct += toks[1]; self.nrc += 1

    def result(self, path: list[str], **meta) -> PredResult:
        return PredResult(path, self.pt, self.ct, self.npc, self.nrc, meta)


# --------------------------------------------------------------------------- #
# shared descent
# --------------------------------------------------------------------------- #
def _descend(tree: CweTree, code: str, model: str, start: list[str],
             acc: _Acc, temperature: float, nonce: str | None = None) -> list[str]:
    """Greedy (argmax) descent from ``start`` to a leaf; diversity via LLM temp.

    ``nonce`` distinguishes independent samples so caching does not collapse
    repeated temperature>0 descents to one identical cached response.
    """
    path = list(start)
    while not tree.is_leaf(path[-1]):
        node = path[-1]
        cands = tree.children(node)
        if not cands:
            break
        scores, _, toks = rank_children(tree, code, cands, model,
                                        temperature=temperature, nonce=nonce)
        if toks != (0, 0):
            acc.policy(toks)
        path.append(max(cands, key=lambda c: scores.get(c, 0.0)))
    return path


# --------------------------------------------------------------------------- #
# M0b greedy
# --------------------------------------------------------------------------- #
def greedy(tree: CweTree, code: str, model: str, **_: object) -> PredResult:
    acc = _Acc()
    path = _descend(tree, code, model, [ROOT], acc, temperature=0.0)
    return acc.result(path, strategy="greedy")


# --------------------------------------------------------------------------- #
# M1 self-consistency
# --------------------------------------------------------------------------- #
def self_consistency(tree: CweTree, code: str, model: str, n: int = 8,
                     temperature: float = 0.7, seed: int = 0, **_: object) -> PredResult:
    acc = _Acc()
    leaves: list[str] = []
    paths: dict[str, list[str]] = {}
    for i in range(n):
        p = _descend(tree, code, model, [ROOT], acc, temperature=temperature,
                     nonce=f"sc:{seed}:{i}")
        leaves.append(p[-1]); paths[p[-1]] = p
    winner = Counter(leaves).most_common(1)[0][0]
    return acc.result(paths[winner], strategy="self_consistency", n=n, votes=dict(Counter(leaves)))


# --------------------------------------------------------------------------- #
# M2 best-of-N
# --------------------------------------------------------------------------- #
def best_of_n(tree: CweTree, code: str, model: str, n: int = 8, temperature: float = 0.7,
              reward_source: str = "self_eval", seed: int = 0, **_: object) -> PredResult:
    acc = _Acc()
    best_path, best_r = None, -1.0
    for i in range(n):
        p = _descend(tree, code, model, [ROOT], acc, temperature=temperature,
                     nonce=f"bon:{seed}:{i}")
        r, rtoks = score_path(tree, code, p, source=reward_source, policy_model=model)
        acc.reward(rtoks)
        if r > best_r:
            best_r, best_path = r, p
    return acc.result(best_path or [ROOT], strategy="best_of_n", n=n, reward=best_r)


# --------------------------------------------------------------------------- #
# M3 beam search over the taxonomy
# --------------------------------------------------------------------------- #
def beam(tree: CweTree, code: str, model: str, width: int = 4, **_: object) -> PredResult:
    acc = _Acc()
    beams: list[tuple[list[str], float]] = [([ROOT], 0.0)]  # (path, cumulative log-score)
    while any(not tree.is_leaf(p[-1]) for p, _ in beams):
        nxt: list[tuple[list[str], float]] = []
        for path, cum in beams:
            if tree.is_leaf(path[-1]):
                nxt.append((path, cum)); continue
            cands = tree.children(path[-1])
            scores, _, toks = rank_children(tree, code, cands, model, temperature=0.0)
            if toks != (0, 0):
                acc.policy(toks)
            for c in cands:
                nxt.append((path + [c], cum + math.log(max(scores.get(c, 1e-3), 1e-3))))
        beams = sorted(nxt, key=lambda x: x[1], reverse=True)[:width]
    best = max(beams, key=lambda x: x[1])[0]
    return acc.result(best, strategy="beam", width=width)


# --------------------------------------------------------------------------- #
# M4 MCTS over the taxonomy
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    N: int = 0
    W: float = 0.0
    P: float = 1.0
    children: list[tuple] | None = None  # expanded child path-tuples, or None


def mcts_tax(tree: CweTree, code: str, model: str, iterations: int = 16, c_exp: float = 1.0,
             reward_source: str = "self_eval", rollout_temp: float = 0.7,
             seed: int = 0, **_: object) -> PredResult:
    acc = _Acc()
    root: tuple = (ROOT,)
    stats: dict[tuple, _Node] = {root: _Node()}
    rollouts: list[tuple[str, float]] = []  # (rolled-out leaf, reward) per iteration

    for it in range(iterations):
        # --- SELECT ---
        path = root
        while stats[path].children:
            parent = stats[path]
            best, best_u = None, -math.inf
            for ch in parent.children:
                nd = stats[ch]
                q = (nd.W / nd.N) if nd.N > 0 else 0.0
                u = q + c_exp * nd.P * math.sqrt(parent.N + 1) / (1 + nd.N)
                if u > best_u:
                    best_u, best = u, ch
            path = best
            if tree.is_leaf(path[-1]):
                break
        # --- EXPAND ---
        node = stats[path]
        if not tree.is_leaf(path[-1]) and node.children is None:
            cands = tree.children(path[-1])
            scores, _, toks = rank_children(tree, code, cands, model, temperature=0.0)
            if toks != (0, 0):
                acc.policy(toks)
            tot = sum(max(scores.get(c, 0.0), 0.0) for c in cands) or 1.0
            node.children = []
            for c in cands:
                cp = path + (c,)
                stats[cp] = _Node(P=max(scores.get(c, 1e-3), 1e-3) / tot)
                node.children.append(cp)
            path = max(node.children, key=lambda cp: stats[cp].P)
        # --- EVALUATE (rollout to leaf, then reward) ---
        leaf_path = _descend(tree, code, model, list(path), acc,
                             temperature=rollout_temp, nonce=f"mcts:{seed}:{it}")
        r, rtoks = score_path(tree, code, leaf_path, source=reward_source, policy_model=model)
        acc.reward(rtoks)
        rollouts.append((leaf_path[-1], r))
        # --- BACKPROP ---
        cur = path
        while True:
            nd = stats[cur]; nd.N += 1; nd.W += r
            if cur == root:
                break
            cur = cur[:-1]

    # --- EXTRACT: descend by visit count, then greedy to a leaf ---
    path = root
    while stats.get(path) and stats[path].children:
        path = max(stats[path].children, key=lambda cp: stats[cp].N)
        if tree.is_leaf(path[-1]):
            break
    final = _descend(tree, code, model, list(path), acc, temperature=0.0)
    return acc.result(final, strategy="mcts_tax", iterations=iterations, c_exp=c_exp,
                      rollouts=rollouts)


# --------------------------------------------------------------------------- #
# stubs (scope: implement after de-risk)
# --------------------------------------------------------------------------- #
def flat(tree: CweTree, code: str, model: str, **_: object) -> PredResult:
    raise NotImplementedError(
        "M0a flat: single call over the full leaf list. TODO: decide leaf-list "
        "rendering (716 leaves) before implementing; not needed for de-risk."
    )


def mcts_reason(tree: CweTree, code: str, model: str, iterations: int = 16, c_exp: float = 1.0,
                reward_source: str = "self_eval", rollout_temp: float = 0.7, expand_k: int = 3,
                max_steps: int = 3, seed: int = 0, **_: object) -> PredResult:
    """M5: MCTS over a reasoning trajectory. Tree nodes are reasoning states (a
    list of generated steps); a terminal node commits to a leaf CWE, which maps to
    a taxonomy path. Contrast with mcts_tax, which searches the taxonomy itself."""
    acc = _Acc()
    leaves = tree.leaves
    # node id -> dict(parent, trace tuple, commit cwe|None, N, W, P, children|None)
    nd: dict[int, dict] = {0: {"parent": None, "trace": (), "commit": None,
                               "N": 0, "W": 0.0, "P": 1.0, "children": None}}
    nxt = 1
    rollouts: list[tuple[str, float]] = []  # (committed leaf, reward) per iteration

    def terminal(i: int) -> bool:
        return nd[i]["commit"] is not None or len(nd[i]["trace"]) >= max_steps

    def commit_of(i: int, temp: float, nonce: str | None) -> str:
        if nd[i]["commit"] is not None:
            return nd[i]["commit"]
        cwe, toks = reason_commit(tree, code, list(nd[i]["trace"]), leaves, model,
                                  temperature=temp, nonce=nonce)
        acc.policy(toks)
        return cwe

    for it in range(iterations):
        # SELECT
        i = 0
        while nd[i]["children"]:
            par = nd[i]
            best, bu = par["children"][0], -math.inf
            for ch in par["children"]:
                c = nd[ch]
                q = (c["W"] / c["N"]) if c["N"] > 0 else 0.0
                u = q + c_exp * c["P"] * math.sqrt(par["N"] + 1) / (1 + c["N"])
                if u > bu:
                    bu, best = u, ch
            i = best
            if terminal(i):
                break
        # EXPAND
        if not terminal(i) and nd[i]["children"] is None:
            moves, toks = reason_propose(tree, code, list(nd[i]["trace"]), leaves, model,
                                         k=expand_k, temperature=0.0)
            acc.policy(toks)
            if not moves:
                moves = [("commit", commit_of(i, 0.0, None))]
            nd[i]["children"] = []
            for kind, val in moves:
                nid, nxt = nxt, nxt + 1
                child = {"parent": i, "trace": nd[i]["trace"], "commit": None,
                         "N": 0, "W": 0.0, "P": 1.0 / len(moves), "children": None}
                if kind == "commit":
                    child["commit"] = val
                else:
                    child["trace"] = nd[i]["trace"] + (val,)
                nd[nid] = child
                nd[i]["children"].append(nid)
            i = nd[i]["children"][0]
        # EVALUATE: roll out to a commit, score it
        cwe = commit_of(i, rollout_temp, f"mr:{seed}:{it}")
        r, rtoks = score_path(tree, code, tree.path_to_root(cwe),
                              source=reward_source, policy_model=model)
        acc.reward(rtoks)
        rollouts.append((cwe, r))
        # BACKPROP
        cur: int | None = i
        while cur is not None:
            nd[cur]["N"] += 1
            nd[cur]["W"] += r
            cur = nd[cur]["parent"]

    # EXTRACT: descend by visits to a terminal, then commit
    i = 0
    while nd[i]["children"]:
        i = max(nd[i]["children"], key=lambda c: nd[c]["N"])
        if terminal(i):
            break
    cwe = commit_of(i, 0.0, None)
    return acc.result(["ROOT"] + tree.path_to_root(cwe), strategy="mcts_reason",
                      iterations=iterations, rollouts=rollouts)


REGISTRY = {
    "greedy": greedy,
    "self_consistency": self_consistency,
    "best_of_n": best_of_n,
    "beam": beam,
    "mcts_tax": mcts_tax,
    "flat": flat,
    "mcts_reason": mcts_reason,
}
