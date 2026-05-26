"""
Lazy graph dataset and batching utilities.
Graphs are stored as individual .pt files and loaded on demand.
"""

import torch
from torch.utils.data import Dataset


class RicoGraphDataset(Dataset):
    def __init__(
        self,
        graph_paths: list[str],
        include_sibling_edges: bool = True,
        feature_groups: list[str] | None = None,
    ):
        self.graph_paths = graph_paths
        self.include_sibling_edges = include_sibling_edges
        self.feature_groups = feature_groups

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> dict:
        graph = torch.load(self.graph_paths[idx], weights_only=False)
        graph = _apply_options(graph, self.include_sibling_edges, self.feature_groups)
        return graph


def _apply_options(
    graph: dict,
    include_sibling_edges: bool,
    feature_groups: list[str] | None,
) -> dict:
    """Re-slice features and rebuild edge_index from stored sub-indices."""
    result = {k: v for k, v in graph.items()}

    # Feature group ablation using saved feature_slices
    if feature_groups is not None:
        slices = graph.get("feature_slices", {})
        x = graph["x"]
        cols = []
        for g in feature_groups:
            if g in slices:
                start, end = slices[g]
                cols.append(x[:, start:end])
        if cols:
            result["x"] = torch.cat(cols, dim=1)
        else:
            result["x"] = torch.zeros(x.shape[0], 0)

    # Rebuild combined edge_index
    cont = graph.get("containment_edge_index", torch.zeros((2, 0), dtype=torch.long))
    sib = graph.get("sibling_edge_index", torch.zeros((2, 0), dtype=torch.long))

    if include_sibling_edges and sib.shape[1] > 0:
        parts = [cont, sib] if cont.shape[1] > 0 else [sib]
        result["edge_index"] = torch.cat(parts, dim=1) if cont.shape[1] > 0 else sib
    else:
        result["edge_index"] = cont

    return result


def collate_graphs(batch: list[dict]) -> dict:
    """
    Concatenate a list of graph dicts into one disconnected super-graph.
    Edge indices are offset by cumulative node counts.
    """
    xs, ys, edge_indices, batch_indices = [], [], [], []
    graph_ids, node_ids_all = [], []
    screen_ids, app_ids = [], []

    node_offset = 0
    for graph_idx, graph in enumerate(batch):
        N = graph["num_nodes"]
        xs.append(graph["x"])
        ys.append(graph["y"])

        ei = graph["edge_index"]
        if ei.shape[1] > 0:
            edge_indices.append(ei + node_offset)

        batch_indices.append(torch.full((N,), graph_idx, dtype=torch.long))
        graph_ids.append(graph.get("screen_id", str(graph_idx)))
        node_ids_all.extend(graph.get("node_ids", [str(i) for i in range(N)]))
        screen_ids.append(graph.get("screen_id", ""))
        app_ids.append(graph.get("app_id", ""))

        node_offset += N

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    batch_index = torch.cat(batch_indices, dim=0)

    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "batch_index": batch_index,
        "graph_ids": graph_ids,
        "node_ids": node_ids_all,
        "screen_ids": screen_ids,
        "app_ids": app_ids,
        "num_nodes": int(x.shape[0]),
    }
