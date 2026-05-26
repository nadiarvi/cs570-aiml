"""Unit tests for graph_builder.py"""

import torch
import pytest
from src.data.rico_loader import flatten_hierarchy
from src.data.graph_builder import build_graph


def _flat_and_features(tree):
    from src.data.features import extract_features
    flat = flatten_hierarchy(tree)
    N = len(flat.nodes)
    # Use only structural features (no sentence-transformers) for fast testing
    features, slices = extract_features(flat, feature_groups=["structural"])
    labels = [0] * N
    return flat, features, labels, slices


def _simple_tree():
    return {
        "class": "android.widget.FrameLayout",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {
                "class": "android.widget.TextView",
                "bounds": [0, 0, 540, 100],
                "text": "hi",
                "children": [],
            },
            {
                "class": "android.widget.TextView",
                "bounds": [540, 0, 1080, 100],
                "text": "there",
                "children": [],
            },
        ],
    }


def test_containment_edges_bidirectional():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g = build_graph(flat, features, labels, include_sibling_edges=False, feature_slices=slices)
    ei = g["edge_index"]
    assert ei.dtype == torch.long
    # Each directed containment pair should appear in both directions
    edges_set = set(zip(ei[0].tolist(), ei[1].tolist()))
    for p, c in flat.containment_edges:
        assert (p, c) in edges_set
        assert (c, p) in edges_set


def test_sibling_edges_bidirectional():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g = build_graph(flat, features, labels, include_sibling_edges=True, feature_slices=slices)
    ei = g["edge_index"]
    edges_set = set(zip(ei[0].tolist(), ei[1].tolist()))
    for a, b in flat.sibling_edges:
        assert (a, b) in edges_set
        assert (b, a) in edges_set


def test_include_sibling_edges_false_removes_sibling_from_combined():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g_with = build_graph(flat, features, labels, include_sibling_edges=True, feature_slices=slices)
    g_without = build_graph(flat, features, labels, include_sibling_edges=False, feature_slices=slices)
    # The sibling_edge_index should still be stored
    assert g_without["sibling_edge_index"].shape[1] > 0
    # But combined edge_index should differ
    assert g_without["edge_index"].shape[1] < g_with["edge_index"].shape[1]


def test_no_self_loops():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g = build_graph(flat, features, labels, include_sibling_edges=True, feature_slices=slices)
    ei = g["edge_index"]
    src, dst = ei[0], ei[1]
    assert (src == dst).sum().item() == 0, "build_graph should not add self-loops"


def test_edge_index_dtype():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g = build_graph(flat, features, labels, feature_slices=slices)
    assert g["edge_index"].dtype == torch.long
    assert g["containment_edge_index"].dtype == torch.long
    assert g["sibling_edge_index"].dtype == torch.long


def test_num_nodes():
    flat, features, labels, slices = _flat_and_features(_simple_tree())
    g = build_graph(flat, features, labels, feature_slices=slices)
    assert g["num_nodes"] == 3
