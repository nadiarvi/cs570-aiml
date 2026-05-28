import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        N = x.size(0)
        in_dim = x.size(1)

        # Add self-loops
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).expand(2, -1)
        ei = torch.cat([edge_index, self_loops], dim=1)

        src, dst = ei

        # Mean aggregation via scatter-add + degree normalization
        agg = torch.zeros(N, in_dim, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, in_dim), x[src])
        deg = torch.bincount(dst, minlength=N).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)

        return agg @ self.weight.T + self.bias


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
