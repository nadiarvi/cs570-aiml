import json
from pathlib import Path
from collections import deque


def load_hierarchy(json_path: str) -> dict:
    """Load raw Rico JSON. Return root node dict."""
    with open(json_path) as f:
        data = json.load(f)
    root = data
    if "activity" in root:
        root = root["activity"]
    if "root" in root:
        root = root["root"]
    return root


def flatten_hierarchy(root: dict) -> tuple:
    """
    BFS traversal. Returns (nodes, containment_edges, sibling_edges).
    Per-node: node["depth"], node["sibling_count"], node["child_count"], node["ancestors"]
    Containment edges: (parent_idx, child_idx)
    Sibling edges: all C(k,2) undirected pairs for each parent with k children.
    """
    nodes = []
    containment_edges = []
    sibling_edges = []

    # BFS queue: (node_data, parent_idx, depth, ancestors_list, num_siblings)
    queue = deque()
    queue.append((root, -1, 0, [], 0))

    while queue:
        node_data, parent_idx, depth, ancestors, num_siblings = queue.popleft()
        idx = len(nodes)

        children = node_data.get("children") or []

        node = {k: v for k, v in node_data.items() if k != "children"}
        node["depth"] = depth
        node["child_count"] = len(children)
        node["sibling_count"] = num_siblings
        node["ancestors"] = ancestors

        if not node.get("text"):
            node["text"] = ""
        bounds = node.get("bounds")
        if not bounds or len(bounds) < 4:
            node["bounds"] = [0, 0, 0, 0]
        if not node.get("resource-id"):
            node["resource-id"] = ""
        if not node.get("content-desc"):
            node["content-desc"] = ""
        if not node.get("class"):
            node["class"] = "Unknown"

        nodes.append(node)

        if parent_idx >= 0:
            containment_edges.append((parent_idx, idx))

        # Pre-compute future indices for children using BFS invariant:
        # first child index = len(nodes) + len(queue)
        first_child_idx = len(nodes) + len(queue)
        child_indices = list(range(first_child_idx, first_child_idx + len(children)))

        # Sibling edges: all C(k,2) undirected pairs
        for i in range(len(child_indices)):
            for j in range(i + 1, len(child_indices)):
                sibling_edges.append((child_indices[i], child_indices[j]))
                sibling_edges.append((child_indices[j], child_indices[i]))

        ancestors_next = ancestors + [node]
        child_num_siblings = len(children) - 1
        for child in children:
            queue.append((child, idx, depth + 1, ancestors_next, child_num_siblings))

    return nodes, containment_edges, sibling_edges


def get_app_id(json_path: str) -> str:
    """Extract app package name — used for app-level splits.
    Tries path first, then reads the package field from inside the JSON.
    """
    path = Path(json_path)
    parent = path.parent.name
    if parent not in ("raw", ".", "", "unknown_app"):
        return parent

    # Read package name from inside the JSON (Rico always stores it there)
    try:
        with open(json_path) as f:
            data = json.load(f)
        pkg = (
            data.get("package")
            or data.get("app_package_name")
            or data.get("packageName")
            or (data.get("activity") or {}).get("package")
        )
        if pkg:
            return str(pkg)
    except Exception:
        pass

    # Last resort: use the filename stem
    return path.stem
