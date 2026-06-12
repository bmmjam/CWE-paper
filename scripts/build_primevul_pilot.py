#!/usr/bin/env python3
"""Build PrimeVul pilot/dev splits for the test-time-search study.

Source: ASSERT-KTH/PrimeVul (public parquet mirror). We keep single-label
vulnerable functions whose CWE maps into data/cwe_tree.json, attach the
root-to-leaf cwe_path, and stratified-sample by CWE.

  dev   (~50)  <- valid split  (prompt iteration; never tune on test)
  pilot (~500) <- test  split  (the $5-10 de-risk + pilot eval pool)

Run:  .venv/bin/python scripts/build_primevul_pilot.py
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/ASSERT-KTH/PrimeVul/resolve/main/data"
FILES = {
    "valid": "valid_unpaired-00000-of-00001.parquet",
    "test": "test_unpaired-00000-of-00001.parquet",
}
TREE_PATH = Path("data/cwe_tree.json")
SEED = 42
DEV_N = 50
PILOT_N = 500


def ensure_parquet(split: str) -> Path:
    dest = Path(f"data/primevul_{split}_unpaired.parquet")
    if not dest.exists():
        url = f"{BASE}/{FILES[split]}"
        print(f"Downloading {split} <- {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_tree() -> dict:
    return json.load(open(TREE_PATH))


def cwe_path(cid: str, nodes: dict) -> list[str]:
    path = []
    while cid:
        path.append(cid)
        cid = nodes[cid]["parent"]
    return list(reversed(path))


def norm_cwes(x) -> list[str]:
    items = x if (hasattr(x, "__iter__") and not isinstance(x, str)) else [x]
    out = []
    for c in items:
        c = str(c).strip()
        if c.upper().startswith("CWE-") and c.split("-")[1].isdigit():
            out.append("CWE-" + c.split("-")[1])
        elif c.isdigit():
            out.append(f"CWE-{c}")
    return out


def to_records(parquet: Path, nodes: dict) -> list[dict]:
    df = pd.read_parquet(parquet)
    df = df[df["is_vulnerable"] == True]  # noqa: E712
    records = []
    for _, r in df.iterrows():
        cwes = norm_cwes(r["cwe"])
        if len(cwes) != 1:
            continue
        cid = cwes[0]
        if cid not in nodes:
            continue
        path = cwe_path(cid, nodes)
        records.append(
            {
                "id": f"{r['project']}:{r['hash']}",
                "func": r["func"],
                "cwe": cid,
                "cwe_path": path,
                "pillar": path[0],
                "project": r["project"],
                "commit_id": r["commit_id"],
                "language": "c_cpp",
            }
        )
    return records


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Sample ~n records, ensuring every CWE present, then fill proportionally."""
    rng = random.Random(seed)
    by_cwe: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cwe[r["cwe"]].append(r)
    for v in by_cwe.values():
        rng.shuffle(v)
    chosen: list[dict] = []
    # round-robin across CWEs so rare classes are represented
    cursors = {c: 0 for c in by_cwe}
    cwes = list(by_cwe)
    while len(chosen) < n and any(cursors[c] < len(by_cwe[c]) for c in cwes):
        for c in cwes:
            if len(chosen) >= n:
                break
            if cursors[c] < len(by_cwe[c]):
                chosen.append(by_cwe[c][cursors[c]])
                cursors[c] += 1
    rng.shuffle(chosen)
    return chosen


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(name: str, records: list[dict]) -> None:
    cnt: dict[str, int] = defaultdict(int)
    pil: dict[str, int] = defaultdict(int)
    for r in records:
        cnt[r["cwe"]] += 1
        pil[r["pillar"]] += 1
    print(f"  {name}: {len(records)} rows, {len(cnt)} CWEs, {len(pil)} pillars")
    print(f"    pillars: {dict(sorted(pil.items(), key=lambda kv: -kv[1]))}")


def main() -> None:
    nodes = load_tree()["nodes"]
    dev_recs = to_records(ensure_parquet("valid"), nodes)
    pilot_recs = to_records(ensure_parquet("test"), nodes)
    print(f"usable labeled: valid={len(dev_recs)}  test={len(pilot_recs)}")

    dev = stratified_sample(dev_recs, DEV_N, SEED)
    pilot = stratified_sample(pilot_recs, PILOT_N, SEED)

    write_jsonl(dev, Path("data/primevul_dev.jsonl"))
    write_jsonl(pilot, Path("data/primevul_pilot.jsonl"))
    print("--- built ---")
    summarize("dev  ", dev)
    summarize("pilot", pilot)


if __name__ == "__main__":
    main()
