"""
Assemble per-node features, labels, and edge tensors into a graph dict.
Edge conventions:
  - containment_edges from rico_loader are directed (parent -> child)
  - build_graph makes them bidirectional in edge_index
  - sibling_edges from rico_loader are unordered pairs
  - build_graph makes them bidirectional in edge_index
  - Self-loops are NOT added here; GCN/GAT layers add them internally.
"""

import torch

from src.data.rico_loader import FlattenedHierarchy


def build_graph(
    flattened: FlattenedHierarchy,
    features: torch.Tensor,
    labels: list[int],
    include_sibling_edges: bool = True,
    feature_slices: dict | None = None,
    screen_id: str = "",
    app_id: str = "",
    label_mode: str = "",
    source_json_path: str = "",
) -> dict:
    """
    Build a graph dict from flattened hierarchy, features, and heuristic labels.

    Returns a dict with:
      x, y, edge_index, containment_edge_index, sibling_edge_index,
      node_ids, parent_index, ancestor_indices, feature_slices, num_nodes,
      screen_id, app_id, label_mode, source_json_path.
    """
    N = len(flattened.nodes)
    assert features.shape[0] == N
    assert len(labels) == N

    # Containment edges: bidirectional
    cont_src, cont_dst = [], []
    for p, c in flattened.containment_edges:
        cont_src.extend([p, c])
        cont_dst.extend([c, p])

    if cont_src:
        containment_edge_index = torch.tensor(
            [cont_src, cont_dst], dtype=torch.long
        )
    else:
        containment_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Sibling edges: bidirectional
    sib_src, sib_dst = [], []
    for a, b in flattened.sibling_edges:
        sib_src.extend([a, b])
        sib_dst.extend([b, a])

    if sib_src:
        sibling_edge_index = torch.tensor(
            [sib_src, sib_dst], dtype=torch.long
        )
    else:
        sibling_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Combined edge_index
    parts = [containment_edge_index]
    if include_sibling_edges and sib_src:
        parts.append(sibling_edge_index)
    if any(p.shape[1] > 0 for p in parts):
        valid = [p for p in parts if p.shape[1] > 0]
        edge_index = torch.cat(valid, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor(labels, dtype=torch.long)
    node_ids = [str(n["node_id"]) for n in flattened.nodes]

    return {
        "x": features,
        "y": y,
        "edge_index": edge_index,
        "containment_edge_index": containment_edge_index,
        "sibling_edge_index": sibling_edge_index,
        "node_ids": node_ids,
        "parent_index": flattened.parent_index,
        "ancestor_indices": flattened.ancestor_indices,
        "feature_slices": feature_slices or {},
        "num_nodes": N,
        "screen_id": screen_id,
        "app_id": app_id,
        "label_mode": label_mode,
        "source_json_path": source_json_path,
    }
