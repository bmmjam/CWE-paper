#!/usr/bin/env python3
"""Build a code-aligned coarse 2-level tree: ROOT -> super-category -> dataset-CWE.

Replaces the abstract MITRE pillars with 7 code-recognizable super-categories to
RAISE level-0 policy accuracy (alpha), so we can probe whether test-time search
helps at the HIGH-alpha end (EXPERIMENTS.md RQ2/H2). CWE leaves keep their real
names/descriptions from cwe_tree.json. Grouping approved by the user.

Run:  .venv/bin/python scripts/build_coarse_tree.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FULL = Path("data/cwe_tree.json")
OUT = Path("data/cwe_tree_coarse.json")

CATS = {
    "CAT-BUFFER": ("Spatial memory / out-of-bounds",
                   "Buffer overflow or out-of-bounds read/write: accessing memory outside the bounds of an allocated buffer.",
                   ["CWE-119", "CWE-120", "CWE-125", "CWE-787"]),
    "CAT-LIFETIME": ("Temporal memory / object lifetime",
                     "Use-after-free, double-free, uninitialized use, or null-pointer dereference: incorrect handling of an object's lifetime.",
                     ["CWE-416", "CWE-415", "CWE-908", "CWE-476"]),
    "CAT-RESOURCE": ("Resource leak / exhaustion",
                     "Failure to release resources, memory leaks, or uncontrolled resource consumption.",
                     ["CWE-401", "CWE-400", "CWE-770"]),
    "CAT-NUMERIC": ("Numeric / calculation error",
                    "Integer overflow/wraparound or incorrect type conversion and calculation.",
                    ["CWE-190", "CWE-681"]),
    "CAT-INJECTION": ("Injection / improper input handling",
                      "Improper neutralization or validation of input leading to injection (command/SQL/XSS) or malformed-input bugs.",
                      ["CWE-20", "CWE-78", "CWE-79", "CWE-89"]),
    "CAT-CONTROL": ("Assertion / control-flow defect",
                    "Reachable assertions, infinite loops, and other control-flow / liveness defects.",
                    ["CWE-617", "CWE-835"]),
    "CAT-ACCESS": ("Access control / info exposure / misc",
                   "Authentication, permissions, information exposure, cryptographic, race-condition, and link-following weaknesses.",
                   ["CWE-276", "CWE-287", "CWE-200", "CWE-203", "CWE-59", "CWE-327", "CWE-362"]),
}


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))["nodes"]
    nodes: dict[str, dict] = {}
    for cat, (name, desc, cwes) in CATS.items():
        nodes[cat] = {"id": cat, "name": name, "abstraction": "Category",
                      "description": desc, "parent": None, "depth": 0,
                      "children": sorted(cwes)}
        for c in cwes:
            if c not in full:
                raise SystemExit(f"{c} not in full tree")
            nodes[c] = {"id": c, "name": full[c]["name"], "abstraction": full[c]["abstraction"],
                        "description": full[c]["description"], "parent": cat, "depth": 1,
                        "children": []}
    tree = {"view": "coarse-code-aligned", "source": "code-aligned regrouping of PrimeVul CWE labels",
            "max_depth": 1, "pillars": sorted(CATS), "nodes": nodes}
    payload = json.dumps(tree, ensure_ascii=False, indent=2, sort_keys=True)
    OUT.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode()).hexdigest()
    Path("data/cwe_tree_coarse.sha256").write_text(f"{sha}  {OUT.name}\n")
    print(f"coarse tree: {len(CATS)} super-categories, {sum(len(v[2]) for v in CATS.values())} CWE leaves")
    for cat, (name, _, cwes) in CATS.items():
        print(f"  {cat:14} {name:36} {len(cwes)} CWEs")
    print(f"sha256: {sha}\nwritten: {OUT}")


if __name__ == "__main__":
    main()
