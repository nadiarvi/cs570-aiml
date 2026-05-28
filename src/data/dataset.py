import torch
from torch.utils.data import Dataset

from src.data.features import FEATURE_DIMS, TOTAL_DIM


class RicoGraphDataset(Dataset):
    def __init__(
        self,
        graph_paths,
        include_sibling_edges=True,
        feature_groups=None,
        label_source="heuristic",
    ):
        self.graph_paths = [str(p) for p in graph_paths]
        self.include_sibling_edges = include_sibling_edges
        self.label_source = label_source

        # Normalise feature_groups
        if feature_groups in (None, "all"):
            self.feature_groups = None
        elif feature_groups == "no_text":
            self.feature_groups = ["visual", "structural", "type"]
        else:
            self.feature_groups = list(feature_groups)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        graph = torch.load(self.graph_paths[idx], weights_only=False)
        graph["y"] = graph[f"y_{self.label_source}"]

        # Feature slicing for ablation
        if self.feature_groups is not None:
            slices = [
                graph["x"][:, FEATURE_DIMS[g][0]:FEATURE_DIMS[g][1]]
                for g in self.feature_groups
            ]
            graph["x"] = torch.cat(slices, dim=1)

        # Edge selection
        if not self.include_sibling_edges:
            graph["edge_index"] = graph["containment_edge_index"]

        return graph


def collate_graphs(batch) -> dict:
    """Merge N graphs into one disconnected super-graph, offsetting edge indices."""
    xs = [g["x"] for g in batch]
    ys = [g["y"] for g in batch]
    edge_indices = [g["edge_index"] for g in batch]
    num_nodes_list = [g["num_nodes"] for g in batch]

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    # Cumulative offsets: [0, N0, N0+N1, ...]
    offsets = [0]
    for n in num_nodes_list[:-1]:
        offsets.append(offsets[-1] + n)

    ei_list = []
    for ei, off in zip(edge_indices, offsets):
        if ei.shape[1] > 0:
            ei_list.append(ei + off)
    if ei_list:
        edge_index = torch.cat(ei_list, dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    batch_mask = torch.cat([
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(num_nodes_list)
    ])

    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "batch_mask": batch_mask,
        "num_nodes": sum(num_nodes_list),
    }
