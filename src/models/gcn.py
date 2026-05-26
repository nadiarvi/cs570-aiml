"""
Mean-aggregation GCN implemented from scratch in PyTorch (no PyG).

Aggregation order (per plan spec):
  1. Add self-loops to edge_index.
  2. Treat edge_index[0] as source, edge_index[1] as destination.
  3. Mean-aggregate source features into each destination: agg shape = [N, in_dim].
  4. Apply linear: out = agg @ W.T + b.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        N = x.shape[0]

        # 1. Add self-loops
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).expand(2, -1)
        ei = torch.cat([edge_index, self_loops], dim=1)  # [2, E + N]

        src, dst = ei[0], ei[1]  # source and destination

        # 2. Mean-aggregate: agg[dst] += x[src]; then divide by degree of dst
        # Accumulate in [N, in_dim] before applying linear
        agg = torch.zeros(N, x.shape[1], dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, x[src])

        # Degree of each destination node (including self-loop)
        degree = torch.zeros(N, dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype, device=x.device))
        degree = degree.clamp(min=1.0).unsqueeze(1)  # [N, 1]

        agg = agg / degree  # [N, in_dim]

        # 3. Linear transform
        return self.linear(agg)  # [N, out_dim]


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.layers.append(GCNLayer(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """Return [N, num_classes]."""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
