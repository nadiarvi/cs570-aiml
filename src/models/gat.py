"""
Graph Attention Network implemented from scratch in PyTorch (no PyG).

Per-layer behavior:
  1. Add self-loops.
  2. Project node features to multi-head key/value space.
  3. Compute attention scores using a shared attention vector, with destination-wise softmax.
  4. Aggregate weighted neighbor features; concatenate heads (or average for last layer).
  5. Final layer produces raw logits, no activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
    ):
        """
        concat=True  → output dim is out_dim * num_heads (intermediate layers).
        concat=False → output dim is out_dim (average heads; use for final layer).
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        # Attention vector: [2 * out_dim] per head, concatenated across heads
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        N = x.shape[0]
        device = x.device

        # 1. Add self-loops
        self_loops = torch.arange(N, device=device).unsqueeze(0).expand(2, -1)
        ei = torch.cat([edge_index, self_loops], dim=1)  # [2, E+N]
        src, dst = ei[0], ei[1]

        # 2. Project: [N, num_heads, out_dim]
        h = self.W(x).view(N, self.num_heads, self.out_dim)

        # 3. Attention scores
        # For each edge (src -> dst): concat [h_src, h_dst] per head, dot with a
        h_src = h[src]   # [E+N, num_heads, out_dim]
        h_dst = h[dst]   # [E+N, num_heads, out_dim]
        alpha = torch.cat([h_src, h_dst], dim=-1)  # [E+N, num_heads, 2*out_dim]
        # dot with attention vector a: [num_heads, 2*out_dim]
        alpha = (alpha * self.a.unsqueeze(0)).sum(dim=-1)  # [E+N, num_heads]
        alpha = self.leaky_relu(alpha)

        # 4. Destination-wise softmax (numerically stable, per head)
        # Scatter-max for stability
        num_edges = ei.shape[1]
        alpha_max = torch.full((N, self.num_heads), float("-inf"), device=device)
        alpha_max.scatter_reduce_(
            0,
            dst.unsqueeze(1).expand(-1, self.num_heads),
            alpha,
            reduce="amax",
            include_self=True,
        )
        alpha_shifted = alpha - alpha_max[dst]  # [E+N, num_heads]
        alpha_exp = torch.exp(alpha_shifted)

        # Apply edge dropout during training
        if self.training and self.dropout > 0:
            alpha_exp = F.dropout(alpha_exp, p=self.dropout)

        alpha_sum = torch.zeros(N, self.num_heads, device=device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-16)  # [E+N, num_heads]

        # 5. Weighted sum of neighbor features
        out = torch.zeros(N, self.num_heads, self.out_dim, device=device, dtype=x.dtype)
        weighted = h_src * alpha_norm.unsqueeze(-1)  # [E+N, num_heads, out_dim]
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.out_dim),
            weighted,
        )

        if self.concat:
            return out.view(N, self.num_heads * self.out_dim)
        else:
            return out.mean(dim=1)  # [N, out_dim]


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            is_last = i == num_layers - 1
            if i == 0:
                in_d = in_dim
            else:
                in_d = hidden_dim * num_heads  # concat heads from previous layer

            if is_last:
                out_d = num_classes
                layer = GATLayer(in_d, out_d, num_heads=num_heads, dropout=dropout, concat=False)
            else:
                out_d = hidden_dim
                layer = GATLayer(in_d, out_d, num_heads=num_heads, dropout=dropout, concat=True)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """Return [N, num_classes]."""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # raw logits, no activation on final layer
