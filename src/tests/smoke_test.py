"""
Smoke test: verifies MLP, GCN, GAT forward pass + backward pass on dummy tensors.
Run from project root: python src/tests/smoke_test.py
Expected output: MLP ✓   GCN ✓   GAT ✓   (< 5 seconds on CPU)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from src.models.gat import GAT
from src.models.gcn import GCN
from src.models.mlp import MLP

N = 50
x = torch.randn(N, 420)
edge_index = torch.randint(0, N, (2, 100)).long()
y = torch.randint(0, 3, (N,)).long()
criterion = nn.CrossEntropyLoss()

for Cls, kw in [
    (MLP, {}),
    (GCN, {}),
    (GAT, {"num_heads": 4}),
]:
    model = Cls(in_dim=420, hidden_dim=256, num_classes=3, **kw)
    logits = model(x, edge_index)
    assert logits.shape == (N, 3), f"{Cls.__name__}: expected ({N}, 3), got {logits.shape}"
    loss = criterion(logits, y)
    loss.backward()
    print(f"{Cls.__name__} ✓")

print("All smoke tests passed.")
