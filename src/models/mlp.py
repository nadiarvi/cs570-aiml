"""
Per-node MLP baseline — ignores graph structure entirely.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 1
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Ignores edge_index and returns [N, num_classes]."""
        return self.net(x)
