#!/usr/bin/env python3
"""Build a DIFF-based dataset from BigVul (option A, richer than PrimeVul-paired).

BigVul natively has func_before (vulnerable) + func_after (fixed) + CWE ID, so we
emit a unified diff that localizes the vulnerable lines — the setup that lets
models exceed the ~40% ceiling seen on raw functions. The diff is stored under
"func" so the harness/prompt work unchanged.

  dev   (50)  <- bigvul validation
  pilot (500) <- bigvul test

Run:  .venv/bin/python scripts/build_bigvul_diff.py
"""

from __future__ import annotations

import difflib
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Auto-fetched from the public HF mirror if missing (see ensure_parquet).
HF_BASE = "https://huggingface.co/api/datasets/bstee615/bigvul/parquet/default"
SRC = {"dev": "data/bigvul_validation.parquet", "pilot": "data/bigvul_test.parquet"}
HF_SPLIT = {"dev": "validation", "pilot": "test"}


def ensure_parquet(split: str) -> str:
    dest = Path(SRC[split])
    if not dest.exists():
        url = f"{HF_BASE}/{HF_SPLIT[split]}/0.parquet"
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return str(dest)
TREE = json.loads(Path("data/cwe_tree_coarse.json").read_text())
LEAVES = {c for c, n in TREE["nodes"].items() if not n["children"]}
SEED, DEV_N, PILOT_N = 42, 50, 500
INCLUDE_MSG = __import__("os").environ.get("INCLUDE_MSG", "0") == "1"
NAME = "diffmsg" if INCLUDE_MSG else "diff"


def norm_cwe(v) -> str | None:
    s = str(v).strip()
    if s.upper().startswith("CWE-") and s.split("-")[1].isdigit():
        return "CWE-" + s.split("-")[1]
    return None


def make_diff(before, after) -> str:
    d = difflib.unified_diff(str(before).splitlines(), str(after).splitlines(),
                             fromfile="vulnerable.c", tofile="fixed.c", lineterm="")
    header = ("// Unified diff (vulnerable -> fixed). '-' lines are the VULNERABLE "
              "code, '+' are the fix. Classify the weakness in the removed/vulnerable code.\n")
    return header + "\n".join(d)


def records(pq: str) -> list[dict]:
    df = pd.read_parquet(pq)
    df = df[df["vul"] == 1]
    recs = []
    for i, r in df.iterrows():
        cwe = norm_cwe(r["CWE ID"])
        if cwe not in LEAVES:
            continue
        diff = make_diff(r["func_before"], r["func_after"])
        if diff.count("\n") < 2:
            continue
        if INCLUDE_MSG:
            msg = str(r.get("commit_message", "")).strip().replace("\n", " ")[:500]
            diff = f"// Commit message: {msg}\n{diff}"
        recs.append({"id": f'{r["project"]}:{str(r["commit_id"])[:10]}:{i}', "func": diff,
                     "cwe": cwe, "project": r["project"], "commit_id": str(r["commit_id"]),
                     "language": "c_cpp_diff"})
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
    for split in SRC:
        recs = records(ensure_parquet(split))
        sample = stratified(recs, DEV_N if split == "dev" else PILOT_N)
        out = Path(f"data/bigvul_{NAME}_{split}.jsonl")
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sample) + "\n")
        dist = defaultdict(int)
        for r in sample:
            dist[r["cwe"]] += 1
        print(f"{split}: usable={len(recs)} -> sampled={len(sample)}, {len(dist)} CWEs  "
              f"top={sorted(dist.items(), key=lambda x:-x[1])[:6]}")


if __name__ == "__main__":
    main()
