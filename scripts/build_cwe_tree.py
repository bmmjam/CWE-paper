#!/usr/bin/env python3
"""Build the CWE classification tree from the MITRE CWE corpus.

Downloads the official CWE XML, extracts the Research-Concepts view (View-1000)
ChildOf hierarchy whose roots are the Pillars, resolves multi-parent nodes to the
parent that yields the smallest depth, keeps Pillar..depth-3, and writes a clean
adjacency JSON plus its SHA-256.

Output schema (data/cwe_tree.json):
    {
      "view": "1000",
      "source": "cwec_latest.xml",
      "max_depth": 3,
      "pillars": ["CWE-664", ...],
      "nodes": {
        "CWE-664": {"id","name","abstraction","description",
                    "parent": null|str, "depth": int, "children": [str,...]},
        ...
      }
    }

Run:  python scripts/build_cwe_tree.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

CWE_XML_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
RESEARCH_VIEW = "1000"
# Full Research-Concepts depth: a hard depth cut drops TIER-1 leaves such as
# CWE-416 (Use-After-Free, depth 4). We keep the complete view-1000 tree; the
# shallow-vs-deep question is handled by the granularity ablation (A6).
MAX_DEPTH = 99
CACHE_ZIP = Path("data/cwec_latest.xml.zip")
OUT_JSON = Path("data/cwe_tree.json")
OUT_SHA = Path("data/cwe_tree.sha256")


def _local(tag: str) -> str:
    """Strip XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def fetch_xml() -> bytes:
    if CACHE_ZIP.exists():
        print(f"Using cached {CACHE_ZIP}")
        blob = CACHE_ZIP.read_bytes()
    else:
        print(f"Downloading {CWE_XML_URL} ...")
        with urllib.request.urlopen(CWE_XML_URL, timeout=60) as resp:
            blob = resp.read()
        CACHE_ZIP.parent.mkdir(parents=True, exist_ok=True)
        CACHE_ZIP.write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xml"))
        print(f"  unzipped {name} ({zf.getinfo(name).file_size} bytes)")
        return zf.read(name)


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def parse_weaknesses(xml: bytes) -> dict[str, dict]:
    """Return {cwe_id: {name, abstraction, description, parents:[view-1000 ChildOf]}}."""
    root = ET.fromstring(xml)
    weaknesses: dict[str, dict] = {}
    for el in root.iter():
        if _local(el.tag) != "Weakness":
            continue
        cid = el.get("ID")
        if not cid:
            continue
        node = {
            "id": f"CWE-{cid}",
            "name": el.get("Name", ""),
            "abstraction": el.get("Abstraction", ""),
            "description": "",
            "parents": [],
        }
        for child in el.iter():
            lt = _local(child.tag)
            if lt == "Description" and not node["description"]:
                node["description"] = clean_text("".join(child.itertext()))
            elif lt == "Related_Weakness":
                if child.get("Nature") == "ChildOf" and child.get("View_ID") == RESEARCH_VIEW:
                    pid = child.get("CWE_ID")
                    if pid:
                        node["parents"].append(f"CWE-{pid}")
        weaknesses[node["id"]] = node
    return weaknesses


def build_tree(weaknesses: dict[str, dict]) -> dict:
    pillars = sorted(
        cid for cid, n in weaknesses.items() if n["abstraction"] == "Pillar"
    )
    # BFS from pillars; first time we reach a node = its minimal depth, and the
    # discovering parent becomes the primary parent.
    depth: dict[str, int] = {p: 0 for p in pillars}
    parent: dict[str, str | None] = {p: None for p in pillars}

    # children map restricted to view-1000 ChildOf edges
    children_of: dict[str, list[str]] = {cid: [] for cid in weaknesses}
    for cid, n in weaknesses.items():
        for p in n["parents"]:
            if p in children_of:
                children_of[p].append(cid)

    q = deque(pillars)
    while q:
        cur = q.popleft()
        d = depth[cur]
        if d >= MAX_DEPTH:
            continue
        for ch in children_of.get(cur, []):
            if ch not in depth:  # first (=shortest) discovery wins
                depth[ch] = d + 1
                parent[ch] = cur
                q.append(ch)

    kept = {cid for cid, d in depth.items() if d <= MAX_DEPTH}
    nodes: dict[str, dict] = {}
    for cid in kept:
        w = weaknesses[cid]
        nodes[cid] = {
            "id": cid,
            "name": w["name"],
            "abstraction": w["abstraction"],
            "description": w["description"][:600],
            "parent": parent[cid],
            "depth": depth[cid],
            "children": [],
        }
    # fill children only with kept nodes whose primary parent is this node
    for cid, node in nodes.items():
        par = node["parent"]
        if par in nodes:
            nodes[par]["children"].append(cid)
    for node in nodes.values():
        node["children"].sort()

    return {
        "view": RESEARCH_VIEW,
        "source": "cwec_latest.xml (MITRE Research Concepts view)",
        "max_depth": MAX_DEPTH,
        "pillars": pillars,
        "nodes": nodes,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    xml = fetch_xml()
    weaknesses = parse_weaknesses(xml)
    print(f"Parsed {len(weaknesses)} weaknesses from XML.")
    tree = build_tree(weaknesses)

    payload = json.dumps(tree, ensure_ascii=False, indent=2, sort_keys=True)
    OUT_JSON.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    OUT_SHA.write_text(f"{sha}  {OUT_JSON.name}\n", encoding="utf-8")

    nodes = tree["nodes"]
    leaves = [c for c, n in nodes.items() if not n["children"]]
    by_depth: dict[int, int] = {}
    for n in nodes.values():
        by_depth[n["depth"]] = by_depth.get(n["depth"], 0) + 1
    print("--- CWE tree built ---")
    print(f"pillars     : {len(tree['pillars'])}")
    print(f"total nodes : {len(nodes)}")
    print(f"leaves      : {len(leaves)}")
    print(f"max depth   : {max(n['depth'] for n in nodes.values())}")
    print(f"by depth    : {dict(sorted(by_depth.items()))}")
    print(f"sha256      : {sha}")
    print(f"written     : {OUT_JSON}  +  {OUT_SHA}")


if __name__ == "__main__":
    main()
