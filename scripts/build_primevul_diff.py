#!/usr/bin/env python3
"""Build a DIFF-based PrimeVul dataset (option A): classify the localized patch.

PrimeVul paired data gives, per fix, a vulnerable function (is_vulnerable=True,
carries the CWE label) and its benign/fixed counterpart (False), linked by
big_vul_idx. We emit a unified diff (vulnerable -> fixed) so the vulnerable lines
are localized — the setup that let prior work reach ~67% vs ~40% on raw funcs.

The diff text is stored under "func" so the existing harness/prompt work
unchanged. Stratified by CWE; CWE must map into the active tree's leaves.

Run:  .venv/bin/python scripts/build_primevul_diff.py
"""

from __future__ import annotations

import difflib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

PAIRS = {"dev": "data/primevul_valid_paired.parquet",
         "pilot": "data/primevul_test_paired.parquet"}
TREE = json.loads(Path("data/cwe_tree_coarse.json").read_text())
LEAVES = {c for c, n in TREE["nodes"].items() if not n["children"]}
SEED, DEV_N, PILOT_N = 42, 50, 500


def norm(x) -> list[str]:
    items = x if (hasattr(x, "__iter__") and not isinstance(x, str)) else [x]
    out = []
    for c in items:
        c = str(c).strip()
        if c.upper().startswith("CWE-") and c.split("-")[1].isdigit():
            out.append("CWE-" + c.split("-")[1])
    return out


def make_diff(vuln: str, benign: str) -> str:
    d = difflib.unified_diff(vuln.splitlines(), benign.splitlines(),
                             fromfile="vulnerable.c", tofile="fixed.c", lineterm="")
    body = "\n".join(d)
    header = ("// Unified diff (vulnerable -> fixed). Lines marked '-' are the "
              "VULNERABLE version; '+' are the fix. Classify the weakness in the "
              "removed/vulnerable code.\n")
    return header + body


def records(parquet: str) -> list[dict]:
    df = pd.read_parquet(parquet)
    by_pair: dict = defaultdict(dict)
    for _, r in df.iterrows():
        key = r["big_vul_idx"]
        by_pair[key]["vuln" if r["is_vulnerable"] else "benign"] = r
    recs = []
    for key, pair in by_pair.items():
        if "vuln" not in pair or "benign" not in pair:
            continue
        cwes = norm(pair["vuln"]["cwe"])
        if len(cwes) != 1 or cwes[0] not in LEAVES:
            continue
        diff = make_diff(pair["vuln"]["func"], pair["benign"]["func"])
        if diff.count("\n") < 2:  # empty/degenerate diff
            continue
        recs.append({"id": f"{pair['vuln']['project']}:{int(key)}", "func": diff,
                     "cwe": cwes[0], "project": pair["vuln"]["project"],
                     "commit_id": pair["vuln"]["commit_id"], "language": "c_cpp_diff"})
    return recs


def stratified(recs: list[dict], n: int) -> list[dict]:
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in recs:
        by[r["cwe"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, cur, keys = [], {c: 0 for c in by}, list(by)
    while len(out) < n and any(cur[c] < len(by[c]) for c in keys):
        for c in keys:
            if len(out) >= n:
                break
            if cur[c] < len(by[c]):
                out.append(by[c][cur[c]]); cur[c] += 1
    rng.shuffle(out)
    return out


def main() -> None:
    for split, pq in PAIRS.items():
        recs = records(pq)
        n = DEV_N if split == "dev" else PILOT_N
        sample = stratified(recs, n)
        out = Path(f"data/primevul_diff_{split}.jsonl")
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sample) + "\n")
        dist = defaultdict(int)
        for r in sample:
            dist[r["cwe"]] += 1
        print(f"{split}: usable_pairs={len(recs)} -> sampled={len(sample)}, {len(dist)} CWEs")


if __name__ == "__main__":
    main()
