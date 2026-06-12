"""Reward signal for search (EXPERIMENTS.md §3.4, RQ3).

Two zero-shot sources, no trained PRM:
  * self_eval     — the policy model scores its own path.
  * external_judge — a stronger model scores the path.

Both score a (possibly partial) root-to-node path in [0, 1].
"""

from __future__ import annotations

import json

from agentclass.config import settings
from agentclass.llm import chat, parse_json, tokens_of
from agentclass.tree import CweTree

Tokens = tuple[int, int]

_JUDGE_SYSTEM = (
    "You are auditing a proposed CWE classification of a code function. Judge how "
    "well the proposed CWE path describes the weakness in the code. Classify the "
    "weakness, not the fix."
)


def _truncate(code: str, limit: int = 5000) -> str:
    return code if len(code) <= limit else code[:limit] + "\n/* …truncated… */"


def score_path(
    tree: CweTree,
    code: str,
    path: list[str],
    source: str = "self_eval",
    policy_model: str | None = None,
) -> tuple[float, Tokens]:
    """Return (reward in [0,1], tokens) for a path under the chosen reward source."""
    model = (policy_model or settings.llm_cheap) if source == "self_eval" else settings.llm_judge
    rendered = " -> ".join(f"{c} ({tree.name(c)})" for c in path)
    leaf = path[-1]
    is_partial = not tree.is_leaf(leaf)
    user = (
        f"Function:\n```c\n{_truncate(code)}\n```\n\n"
        f"Proposed CWE path: {rendered}\n"
        f"This path is {'PARTIAL (not yet a leaf)' if is_partial else 'COMPLETE (leaf)'}.\n\n"
        f"Description of the final node {leaf}: {tree.description(leaf) or 'n/a'}\n\n"
        "Return JSON {\"score\": 0.0-1.0} where score is the probability this path "
        "correctly classifies the weakness in the function."
    )
    resp = chat(
        messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
        model=model,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    toks = tokens_of(resp)
    try:
        obj = parse_json(resp["choices"][0]["message"]["content"])
        score = float(obj.get("score", 0.0))
    except (ValueError, TypeError, AttributeError):
        score = 0.0
    return max(0.0, min(1.0, score)), toks
