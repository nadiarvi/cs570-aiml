"""
Rico JSON -> flattened hierarchy records.
Node traversal uses DFS pre-order, giving stable node_id assignments.
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field


@dataclass
class FlattenedHierarchy:
    nodes: list[dict]
    containment_edges: list[tuple[int, int]]  # directed (parent_idx, child_idx)
    sibling_edges: list[tuple[int, int]]       # unordered (child_i, child_j) pairs
    parent_index: list[int | None]
    ancestor_indices: list[list[int]]


def load_hierarchy(json_path: str) -> dict:
    """Load raw Rico JSON and return the root node dict."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Rico JSONs are typically {"activity": {"root": {...}}} or directly a node dict
    if "activity" in data and "root" in data["activity"]:
        return data["activity"]["root"]
    if "root" in data:
        return data["root"]
    return data


def flatten_hierarchy(root: dict) -> FlattenedHierarchy:
    """
    Flatten a Rico hierarchy into stable node records and graph structure.
    Uses DFS pre-order traversal for deterministic node_id assignment.

    Returns:
      - nodes: node attribute dicts, one per UI element (with node_id, depth, child_count, sibling_count)
      - containment_edges: directed (parent_idx, child_idx) pairs
      - sibling_edges: unordered (child_i_idx, child_j_idx) one pair per sibling pair
      - parent_index: parent index for each node, None for root
      - ancestor_indices: root-to-parent index chain for each node
    """
    nodes: list[dict] = []
    containment_edges: list[tuple[int, int]] = []
    sibling_edges: list[tuple[int, int]] = []
    parent_index: list[int | None] = []
    ancestor_indices: list[list[int]] = []

    # DFS pre-order: (node_dict, parent_idx, ancestor_chain, depth)
    stack = deque()
    stack.append((root, None, [], 0))

    while stack:
        node, par_idx, ancestors, depth = stack.pop()

        node_idx = len(nodes)
        children = node.get("children") or []
        sibling_count = len(children) - 1  # siblings of this node = parent's children - 1

        node_record = {k: v for k, v in node.items() if k != "children"}
        node_record["node_id"] = node_idx
        node_record["depth"] = depth
        node_record["child_count"] = len(children)
        # sibling_count is filled in when we process within parent context
        node_record["sibling_count"] = 0  # will patch below for non-root

        nodes.append(node_record)
        parent_index.append(par_idx)
        ancestor_indices.append(list(ancestors))

        if par_idx is not None:
            containment_edges.append((par_idx, node_idx))

        # Push children in reverse order so leftmost child is processed first
        child_start_idx = node_idx + 1  # indices assigned in DFS pre-order
        # We need to know child indices to build sibling edges, but they aren't
        # assigned yet. Collect them in a second pass using a BFS-like approach.
        # Instead, build sibling edges after assigning all child node_ids.
        # We use a deferred list: store (parent_idx, [children_in_order]).
        new_ancestors = ancestors + [node_idx]
        # push children in reverse so stack pops left-to-right
        for child in reversed(children):
            stack.append((child, node_idx, new_ancestors, depth + 1))

    # Second pass: patch sibling_count and build sibling edges.
    # Group children by parent.
    from collections import defaultdict
    parent_to_children: dict[int, list[int]] = defaultdict(list)
    for idx, par in enumerate(parent_index):
        if par is not None:
            parent_to_children[par].append(idx)

    # DFS pre-order guarantees children of a parent are contiguous only if the
    # tree has no crossings, which Rico trees don't. However, the order in
    # parent_to_children will be DFS pre-order among siblings, which is the
    # traversal order — good enough for stable sibling assignment.
    for par, children_idxs in parent_to_children.items():
        n = len(children_idxs)
        for c in children_idxs:
            nodes[c]["sibling_count"] = n - 1
        for i in range(n):
            for j in range(i + 1, n):
                sibling_edges.append((children_idxs[i], children_idxs[j]))

    return FlattenedHierarchy(
        nodes=nodes,
        containment_edges=containment_edges,
        sibling_edges=sibling_edges,
        parent_index=parent_index,
        ancestor_indices=ancestor_indices,
    )


def get_app_id(json_path: str) -> str:
    """Extract the app/package ID from a Rico JSON path.

    Expects paths like: data/raw/<app_id>/<screen_id>.json
    Falls back to the parent directory name.
    """
    return os.path.basename(os.path.dirname(json_path))
