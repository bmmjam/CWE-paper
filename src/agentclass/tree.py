"""CWE taxonomy tree loaded from data/cwe_tree.json.

The MITRE Research view is a *forest* of 10 pillars, so we add a synthetic
``ROOT`` whose children are the pillars. Hierarchical strategies descend
ROOT -> pillar -> ... -> leaf.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

ROOT = "ROOT"


class CweTree:
    def __init__(self, data: dict):
        self.view = data.get("view", "1000")
        self._nodes: dict[str, dict] = data["nodes"]
        self.pillars: list[str] = list(data["pillars"])
        # synthetic root
        self._children: dict[str, list[str]] = {ROOT: list(self.pillars)}
        self._parent: dict[str, str | None] = {ROOT: None}
        for cid, n in self._nodes.items():
            self._children[cid] = list(n["children"])
            self._parent[cid] = n["parent"] if n["parent"] is not None else ROOT

    @classmethod
    def load(cls, path: str | Path) -> "CweTree":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- structure ---
    def children(self, node: str) -> list[str]:
        return self._children.get(node, [])

    def is_leaf(self, node: str) -> bool:
        return node != ROOT and not self._children.get(node)

    def parent(self, node: str) -> str | None:
        return self._parent.get(node)

    def path_to_root(self, node: str) -> list[str]:
        """Root-to-node path WITHOUT the synthetic ROOT (i.e. pillar..node)."""
        path: list[str] = []
        cur: str | None = node
        while cur is not None and cur != ROOT:
            path.append(cur)
            cur = self._parent.get(cur)
        return list(reversed(path))

    @cached_property
    def leaves(self) -> list[str]:
        return [c for c in self._nodes if not self._children.get(c)]

    # --- presentation for prompts ---
    def name(self, node: str) -> str:
        return self._nodes.get(node, {}).get("name", node)

    def description(self, node: str) -> str:
        return self._nodes.get(node, {}).get("description", "")

    def candidate_block(self, nodes: list[str], with_desc: bool = True) -> str:
        """Render a contrastive candidate list for a policy prompt."""
        lines = []
        for c in nodes:
            line = f"- {c}: {self.name(c)}"
            if with_desc and self.description(c):
                line += f" — {self.description(c)[:240]}"
            lines.append(line)
        return "\n".join(lines)
