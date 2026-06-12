#!/usr/bin/env python3
"""Build a closed-set 2-level pilot tree: ROOT -> pillar -> dataset-CWE.

Why 2-level: the full Research tree nests some dataset labels (CWE-400 is an
ancestor of CWE-770; CWE-119 of CWE-125/787), which breaks descend-to-leaf. For
the pilot we attach every dataset CWE directly under its full-tree pillar, giving
a tractable closed-set with a real pillar->leaf cascade. Multi-level depth is
deferred to the granularity ablation (EXPERIMENTS.md §6, A6).

Also rewrites dev/pilot jsonl so cwe_path = [pillar, cwe] under this tree.

Run:  .venv/bin/python scripts/build_pilot_tree.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FULL = Path("data/cwe_tree.json")
OUT_TREE = Path("data/cwe_tree_pilot.json")
SPLITS = [Path("data/primevul_dev.jsonl"), Path("data/primevul_pilot.jsonl")]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    fnodes = full["nodes"]

    def pillar_of(cwe: str) -> str:
        cur = cwe
        while fnodes[cur]["parent"] is not None:
            cur = fnodes[cur]["parent"]
        return cur

    # dataset CWE set across all splits
    targets: set[str] = set()
    for sp in SPLITS:
        for ex in load_jsonl(sp):
            targets.add(ex["cwe"])
    targets = {c for c in targets if c in fnodes}

    # build 2-level tree
    pillars: dict[str, list[str]] = {}
    for c in sorted(targets):
        p = pillar_of(c)
        pillars.setdefault(p, []).append(c)

    nodes: dict[str, dict] = {}
    for p, kids in pillars.items():
        nodes[p] = {
            "id": p, "name": fnodes[p]["name"], "abstraction": "Pillar",
            "description": fnodes[p]["description"], "parent": None,
            "depth": 0, "children": sorted(kids),
        }
        for c in kids:
            nodes[c] = {
                "id": c, "name": fnodes[c]["name"],
                "abstraction": fnodes[c]["abstraction"],
                "description": fnodes[c]["description"], "parent": p,
                "depth": 1, "children": [],
            }
    tree = {
        "view": "pilot-2level", "source": "pruned from cwe_tree.json to PrimeVul labels",
        "max_depth": 1, "pillars": sorted(pillars), "nodes": nodes,
    }
    payload = json.dumps(tree, ensure_ascii=False, indent=2, sort_keys=True)
    OUT_TREE.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode()).hexdigest()
    Path("data/cwe_tree_pilot.sha256").write_text(f"{sha}  {OUT_TREE.name}\n")

    # rewrite splits with 2-level cwe_path
    for sp in SPLITS:
        rows = load_jsonl(sp)
        for ex in rows:
            ex["cwe_path"] = [pillar_of(ex["cwe"]), ex["cwe"]]
            ex["pillar"] = pillar_of(ex["cwe"])
        sp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                      encoding="utf-8")

    print(f"pilot tree: {len(pillars)} pillars, {len(targets)} dataset-CWE leaves")
    for p in sorted(pillars):
        print(f"  {p} {fnodes[p]['name'][:42]:42} -> {len(pillars[p])} CWEs: {sorted(pillars[p])}")
    print(f"sha256: {sha}")
    print(f"rewrote splits: {[s.name for s in SPLITS]}")


if __name__ == "__main__":
    main()
