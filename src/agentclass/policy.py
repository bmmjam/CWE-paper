"""Policy layer: one shared hierarchical step used by every search strategy.

``rank_children`` asks the policy LLM to rank the candidate child CWEs for the
current node. Greedy takes the top; beam/MCTS use the full ranking as priors;
self-consistency/best-of-N sample it at temperature > 0. Keeping a single step
shared across strategies is what makes the compute-matched comparison clean
(EXPERIMENTS.md §5.2).

Every call returns its (prompt, completion) token counts so strategies can sum
the per-example budget.
"""

from __future__ import annotations

import json

from agentclass.llm import chat, parse_json, tokens_of
from agentclass.tree import CweTree

Tokens = tuple[int, int]

_SYSTEM = (
    "You are a security expert classifying the vulnerability in a code function "
    "according to the CWE taxonomy. Classify the WEAKNESS the code exhibits, not "
    "the fix. You are given the candidate CWE categories at one level of the "
    "taxonomy; choose among THESE candidates only."
)

_RANK_INSTRUCTION = (
    "Score EVERY candidate listed above by how likely it describes the weakness "
    "in the function (0.0 = no, 1.0 = yes); scores need not sum to 1. You MUST "
    'output an entry for every candidate CWE id. Return JSON: {"scores": '
    '{"CWE-XXX": 0.0, "CWE-YYY": 0.0, ...}, "rationale": "one sentence"}.'
)


def _truncate_code(code: str, limit: int = 5000) -> str:
    return code if len(code) <= limit else code[:limit] + "\n/* …truncated… */"


def rank_children(
    tree: CweTree,
    code: str,
    candidates: list[str],
    model: str,
    prior_reasoning: str = "",
    temperature: float = 0.0,
    nonce: str | None = None,
) -> tuple[dict[str, float], str, Tokens]:
    """Return (scores by cwe, rationale, tokens). Robust to malformed output."""
    if len(candidates) == 1:
        return {candidates[0]: 1.0}, "single candidate", (0, 0)

    ctx = f"Prior reasoning: {prior_reasoning}\n\n" if prior_reasoning else ""
    user = (
        f"{ctx}Function:\n```c\n{_truncate_code(code)}\n```\n\n"
        f"Candidate CWE categories:\n{tree.candidate_block(candidates)}\n\n"
        f"{_RANK_INSTRUCTION}"
    )
    resp = chat(
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        model=model,
        response_format={"type": "json_object"},
        temperature=temperature,
        cache_nonce=nonce,
    )
    toks = tokens_of(resp)
    content = resp["choices"][0]["message"]["content"] or "{}"
    scores: dict[str, float] = {}
    rationale = ""
    obj = parse_json(content)
    try:
        rationale = str(obj.get("rationale", ""))[:300]
        raw = obj.get("scores", {})
        if isinstance(raw, dict):
            for cwe, sc in raw.items():
                cwe = str(cwe).strip()
                if cwe in candidates:
                    scores[cwe] = float(sc)
        # fallback: tolerate the older list-of-{cwe,score} shape
        for item in obj.get("ranked", []) if isinstance(obj.get("ranked"), list) else []:
            cwe = str(item.get("cwe", "")).strip()
            if cwe in candidates:
                scores[cwe] = float(item.get("score", 0.0))
    except (ValueError, TypeError, AttributeError):
        pass
    # fill any missing candidate with a small floor so search never crashes
    floor = 1e-3
    for c in candidates:
        scores.setdefault(c, floor)
    return scores, rationale, toks


# --------------------------------------------------------------------------- #
# Reasoning-trajectory policy (M5): search is over generated reasoning steps,
# not over taxonomy children. Each move is either a short next reasoning step or
# a final commit to one leaf CWE.
# --------------------------------------------------------------------------- #
_REASON_SYS = (
    "You are a security expert reasoning step by step about the weakness in a code "
    "patch, in order to classify it into exactly one CWE. Reason about the removed "
    "(vulnerable) lines, not the fix."
)


def _leaf_block(tree: CweTree, leaves: list[str]) -> str:
    return "\n".join(f"- {c}: {tree.name(c)}" for c in leaves)


def _trace_block(trace: list[str]) -> str:
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(trace)) or "(none yet)"


def reason_propose(tree: CweTree, code: str, trace: list[str], leaves: list[str],
                   model: str, k: int = 3, temperature: float = 0.7,
                   nonce: str | None = None) -> tuple[list[tuple[str, str]], Tokens]:
    """Propose up to k next moves: ('step', text) or ('commit', cwe)."""
    user = (
        f"Code patch:\n```c\n{_truncate_code(code)}\n```\n\n"
        f"Reasoning so far:\n{_trace_block(trace)}\n\n"
        f"Candidate CWEs:\n{_leaf_block(tree, leaves)}\n\n"
        f"Propose {k} DISTINCT next moves toward the answer. Each move is either a "
        f"brief next reasoning step, or a final commit to ONE candidate CWE id. "
        'Return JSON: {"moves": [{"type": "step", "text": "..."} or '
        '{"type": "commit", "cwe": "CWE-XXX"}, ...]}.'
    )
    resp = chat(messages=[{"role": "system", "content": _REASON_SYS},
                          {"role": "user", "content": user}],
                model=model, response_format={"type": "json_object"},
                temperature=temperature, cache_nonce=nonce)
    toks = tokens_of(resp)
    obj = parse_json(resp["choices"][0]["message"]["content"])
    moves: list[tuple[str, str]] = []
    for m in (obj.get("moves") or [])[:k] if isinstance(obj.get("moves"), list) else []:
        if not isinstance(m, dict):
            continue
        if str(m.get("type")) == "commit":
            c = str(m.get("cwe", "")).strip()
            if c in leaves:
                moves.append(("commit", c))
        else:
            t = str(m.get("text", "")).strip()
            if t:
                moves.append(("step", t[:300]))
    return moves, toks


def reason_commit(tree: CweTree, code: str, trace: list[str], leaves: list[str],
                  model: str, temperature: float = 0.0,
                  nonce: str | None = None) -> tuple[str, Tokens]:
    """Force a commit to one leaf CWE given the reasoning so far."""
    user = (
        f"Code patch:\n```c\n{_truncate_code(code)}\n```\n\n"
        f"Reasoning so far:\n{_trace_block(trace)}\n\n"
        f"Candidate CWEs:\n{_leaf_block(tree, leaves)}\n\n"
        'Commit to the single best CWE now. Return JSON {"cwe": "CWE-XXX"}.'
    )
    resp = chat(messages=[{"role": "system", "content": _REASON_SYS},
                          {"role": "user", "content": user}],
                model=model, response_format={"type": "json_object"},
                temperature=temperature, cache_nonce=nonce)
    toks = tokens_of(resp)
    obj = parse_json(resp["choices"][0]["message"]["content"])
    c = str(obj.get("cwe", "")).strip()
    return (c if c in leaves else leaves[0]), toks
