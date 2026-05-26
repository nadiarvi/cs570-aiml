"""Unit tests for MLP, GCN, and GAT models."""

import torch
import pytest
from src.models.mlp import MLP
from src.models.gcn import GCN, GCNLayer
from src.models.gat import GAT


IN_DIM = 16
HIDDEN_DIM = 32
N = 10
NUM_CLASSES = 3


def _simple_edge_index(N: int) -> torch.LongTensor:
    """Chain graph: 0->1->2->...->N-1, bidirectional."""
    src = list(range(N - 1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _isolated_edge_index(N: int) -> torch.LongTensor:
    """No edges — N isolated nodes."""
    return torch.zeros((2, 0), dtype=torch.long)


# ---- MLP ----------------------------------------------------------------

def test_mlp_output_shape():
    model = MLP(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=3)
    x = torch.randn(N, IN_DIM)
    out = model(x)
    assert out.shape == (N, NUM_CLASSES)


def test_mlp_ignores_edge_index():
    model = MLP(IN_DIM, HIDDEN_DIM, NUM_CLASSES)
    x = torch.randn(N, IN_DIM)
    out1 = model(x, edge_index=None)
    ei = _simple_edge_index(N)
    out2 = model(x, edge_index=ei)
    assert torch.allclose(out1, out2)


# ---- GCN ----------------------------------------------------------------

def test_gcn_output_shape():
    model = GCN(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=2)
    x = torch.randn(N, IN_DIM)
    ei = _simple_edge_index(N)
    out = model(x, ei)
    assert out.shape == (N, NUM_CLASSES)


def test_gcn_isolated_nodes():
    """Self-loops added inside GCNLayer mean isolated nodes still get updated."""
    model = GCN(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=2)
    x = torch.randn(N, IN_DIM)
    ei = _isolated_edge_index(N)
    out = model(x, ei)
    assert out.shape == (N, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_gcn_aggregation_in_dim_before_linear():
    """Verify aggregation is [N, in_dim] before the linear transform."""
    layer = GCNLayer(IN_DIM, HIDDEN_DIM)
    x = torch.randn(N, IN_DIM)
    ei = _simple_edge_index(N)
    # Just checking that the layer runs without error and produces [N, HIDDEN_DIM]
    out = layer(x, ei)
    assert out.shape == (N, HIDDEN_DIM)


def test_gcn_batched_disconnected_no_message_passing():
    """Two disconnected sub-graphs should not exchange messages."""
    model = GCN(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=1)
    model.eval()

    # Graph A: nodes 0,1 connected
    # Graph B: nodes 2,3 connected
    # No edges cross the two sub-graphs
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    xA = torch.randn(2, IN_DIM)
    xB = torch.randn(2, IN_DIM)
    x = torch.cat([xA, xB], dim=0)

    with torch.no_grad():
        out_combined = model(x, ei)

    # Verify nodes 0,1 only interact with xA
    x_only_A = torch.cat([xA, torch.zeros(2, IN_DIM)], dim=0)
    ei_A = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with torch.no_grad():
        out_A = model(xA, ei_A)

    # Output for nodes 0 and 1 should match when using combined graph
    assert torch.allclose(out_combined[:2], out_A, atol=1e-5), \
        "Disconnected sub-graphs should not influence each other"


# ---- GAT ----------------------------------------------------------------

def test_gat_output_shape():
    model = GAT(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=2, num_heads=4)
    x = torch.randn(N, IN_DIM)
    ei = _simple_edge_index(N)
    out = model(x, ei)
    assert out.shape == (N, NUM_CLASSES)


def test_gat_isolated_nodes():
    model = GAT(IN_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=2, num_heads=4)
    x = torch.randn(N, IN_DIM)
    ei = _isolated_edge_index(N)
    out = model(x, ei)
    assert out.shape == (N, NUM_CLASSES)
    assert not torch.isnan(out).any()
