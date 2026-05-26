"""Unit tests for rico_loader.py"""

import pytest
from src.data.rico_loader import flatten_hierarchy, FlattenedHierarchy


def _make_tree():
    """
    Build a small synthetic Rico-style hierarchy:
        root (0)
         ├── child_A (1)
         │    ├── grandchild_AA (2)
         │    └── grandchild_AB (3)
         └── child_B (4)
    """
    return {
        "class": "android.widget.FrameLayout",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {
                "class": "android.widget.LinearLayout",
                "bounds": [0, 0, 540, 1920],
                "children": [
                    {
                        "class": "android.widget.TextView",
                        "bounds": [0, 0, 270, 100],
                        "text": "Hello",
                        "children": [],
                    },
                    {
                        "class": "android.widget.TextView",
                        "bounds": [270, 0, 540, 100],
                        "text": "World",
                        "children": [],
                    },
                ],
            },
            {
                "class": "android.widget.ImageView",
                "bounds": [540, 0, 1080, 1920],
                "children": [],
            },
        ],
    }


def test_node_count():
    tree = _make_tree()
    flat = flatten_hierarchy(tree)
    assert len(flat.nodes) == 5


def test_stable_node_id():
    tree = _make_tree()
    flat = flatten_hierarchy(tree)
    for i, node in enumerate(flat.nodes):
        assert node["node_id"] == i


def test_parent_index_root():
    flat = flatten_hierarchy(_make_tree())
    assert flat.parent_index[0] is None


def test_parent_index_children():
    flat = flatten_hierarchy(_make_tree())
    # child_A's parent is root (0)
    assert flat.parent_index[1] == 0
    # child_B's parent is root (0)
    assert flat.parent_index[4] == 0


def test_ancestor_indices_root():
    flat = flatten_hierarchy(_make_tree())
    assert flat.ancestor_indices[0] == []


def test_ancestor_indices_grandchild():
    flat = flatten_hierarchy(_make_tree())
    # grandchild_AA (idx 2) should have ancestors [root=0, child_A=1]
    assert flat.ancestor_indices[2] == [0, 1]


def test_depth():
    flat = flatten_hierarchy(_make_tree())
    assert flat.nodes[0]["depth"] == 0  # root
    assert flat.nodes[1]["depth"] == 1  # child_A
    assert flat.nodes[2]["depth"] == 2  # grandchild_AA


def test_child_count():
    flat = flatten_hierarchy(_make_tree())
    assert flat.nodes[0]["child_count"] == 2   # root has 2 children
    assert flat.nodes[1]["child_count"] == 2   # child_A has 2 children
    assert flat.nodes[2]["child_count"] == 0   # leaf


def test_sibling_count():
    flat = flatten_hierarchy(_make_tree())
    # child_A and child_B are siblings; each has sibling_count=1
    assert flat.nodes[1]["sibling_count"] == 1
    assert flat.nodes[4]["sibling_count"] == 1
    # grandchildren of child_A are siblings of each other
    assert flat.nodes[2]["sibling_count"] == 1
    assert flat.nodes[3]["sibling_count"] == 1
    # root has no siblings
    assert flat.nodes[0]["sibling_count"] == 0


def test_containment_edges_are_parent_child():
    flat = flatten_hierarchy(_make_tree())
    parent_idxs = {e[0] for e in flat.containment_edges}
    child_idxs = {e[1] for e in flat.containment_edges}
    # All containment edges should have valid parent/child node indices
    all_idxs = set(range(len(flat.nodes)))
    assert parent_idxs.issubset(all_idxs)
    assert child_idxs.issubset(all_idxs)
    # Root should never be a child
    assert 0 not in child_idxs


def test_sibling_edges_are_unordered():
    flat = flatten_hierarchy(_make_tree())
    # Unordered: (i, j) appears only once, not both (i,j) and (j,i)
    seen = set()
    for a, b in flat.sibling_edges:
        key = (min(a, b), max(a, b))
        assert key not in seen, f"Duplicate unordered sibling pair: {key}"
        seen.add(key)


def test_single_node():
    tree = {"class": "android.widget.TextView", "bounds": [0, 0, 100, 100], "text": "hi"}
    flat = flatten_hierarchy(tree)
    assert len(flat.nodes) == 1
    assert flat.parent_index[0] is None
    assert flat.ancestor_indices[0] == []
    assert flat.containment_edges == []
    assert flat.sibling_edges == []
