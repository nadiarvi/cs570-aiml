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

        self.W = nn.Parameter(torch.empty(num_heads, in_dim, self.head_dim))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data.view(-1, self.head_dim))
        nn.init.xavier_uniform_(self.a.data)  # already 2D: [num_heads, 2*head_dim]

    def forward(self, x, edge_index):
        N = x.size(0)

        # Self-loops: every node attends to itself; also prevents empty edge_index
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).expand(2, -1)
        ei = torch.cat([edge_index, self_loops], dim=1)
        src, dst = ei

        head_outputs = []
        for k in range(self.num_heads):
            h = x @ self.W[k]  # [N, head_dim]

            h_cat = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2*head_dim]
            e = self.leaky_relu(h_cat @ self.a[k])       # [E]

            # Max per dst for numerical stability — no grad needed here
            with torch.no_grad():
                e_max = torch.zeros(N, device=x.device)
                e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
            e_stable = e - e_max[dst]  # grad flows through e

            exp_e = torch.exp(e_stable)  # [E]

            # Out-of-place scatter_add preserves grad_fn through exp_e
            exp_sum = torch.zeros(N, device=x.device).scatter_add(0, dst, exp_e)
            alpha = exp_e / (exp_sum[dst] + 1e-9)  # [E]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Out-of-place aggregation preserves grad_fn through weighted
            weighted = alpha.unsqueeze(1) * h[src]  # [E, head_dim]
            agg = torch.zeros(N, self.head_dim, device=x.device).scatter_add(
                0, dst.unsqueeze(1).expand(-1, self.head_dim), weighted
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
            heads = num_heads if i < len(dims) - 2 else 1
            self.layers.append(GATLayer(dims[i], dims[i + 1], num_heads=heads, dropout=dropout))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
