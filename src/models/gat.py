import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.dropout = dropout

        # Per-head: W [in_dim, head_dim], a [2 * head_dim]
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, self.head_dim))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.view(self.num_heads * self.W.shape[1], self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0)).squeeze_(0)

    def forward(self, x, edge_index):
        N = x.size(0)
        if edge_index.shape[1] == 0:
            return torch.zeros(N, self.out_dim, device=x.device)

        src, dst = edge_index  # [E]

        head_outputs = []
        for k in range(self.num_heads):
            # Linear projection: [N, in_dim] × [in_dim, head_dim] → [N, head_dim]
            h = x @ self.W[k]

            # Attention scores: e_ij = LeakyReLU(a_k^T [h_i || h_j])
            h_cat = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2*head_dim]
            e = self.leaky_relu(h_cat @ self.a[k])       # [E]

            # Numerically-stable scatter softmax
            max_e = torch.zeros(N, device=x.device)
            max_e.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
            e_stable = e - max_e[dst]
            alpha_num = torch.exp(e_stable)              # [E]
            alpha_sum = torch.zeros(N, device=x.device)
            alpha_sum.scatter_add_(0, dst, alpha_num)
            alpha = alpha_num / (alpha_sum[dst] + 1e-9)  # [E]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Weighted aggregation: for each dst, sum alpha * h_src
            agg = torch.zeros(N, self.head_dim, device=x.device)
            agg.scatter_add_(
                0,
                dst.unsqueeze(1).expand(-1, self.head_dim),
                alpha.unsqueeze(1) * h[src],
            )
            head_outputs.append(agg)

        return torch.cat(head_outputs, dim=-1)  # [N, out_dim]


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes=3,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            out = dims[i + 1]
            heads = num_heads if i < len(dims) - 2 else 1
            # Ensure out_dim is divisible by heads; for last layer use 1 head
            self.layers.append(GATLayer(dims[i], out, num_heads=heads, dropout=dropout))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
