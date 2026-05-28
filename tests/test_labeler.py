"""Unit tests for labeler.py — verifies contextual vs local_only mode behavior."""

import pytest
from src.data.labeler import assign_label, label_graph
from src.data.rico_loader import flatten_hierarchy


# ---- assign_label tests ------------------------------------------------------

def _node(**kwargs):
    defaults = {"class": "android.widget.TextView", "child_count": 0, "depth": 3}
    defaults.update(kwargs)
    return defaults


def test_canonical_resource_id_local():
    node = _node(**{"resource-id": "com.example:id/price_label", "text": "$9.99"})
    assert assign_label(node, [], "local_only") == 0


def test_canonical_resource_id_contextual():
    node = _node(**{"resource-id": "com.example:id/price_label", "text": "$9.99"})
    assert assign_label(node, [], "contextual") == 0


def test_canonical_email_local():
    node = _node(text="user@example.com")
    assert assign_label(node, [], "local_only") == 0


def test_ancestor_only_canonical_rule_fires_in_contextual():
    """Currency text node with payment ancestor → canonical in contextual mode."""
    node = _node(text="$24.99")
    ancestor = _node(**{"resource-id": "com.example:id/checkout_total"})
    result = assign_label(node, [ancestor], "contextual")
    assert result == 0, f"Expected canonical (0), got {result}"


def test_ancestor_only_canonical_rule_does_not_fire_in_local_only():
    """Same setup but local_only — ancestor rules must not fire."""
    node = _node(text="$24.99")
    ancestor = _node(**{"resource-id": "com.example:id/checkout_total"})
    result = assign_label(node, [ancestor], "local_only")
    # Without the resource-id on the node itself, should NOT be canonical
    assert result != 0, f"Expected non-canonical in local_only mode, got {result}"


def test_translatable_header_resource_id():
    node = _node(**{"resource-id": "com.example:id/section_title", "text": "Settings"})
    result = assign_label(node, [], "local_only")
    assert result == 1


def test_translatable_textview_leaf():
    node = _node(**{"class": "android.widget.TextView", "text": "Click here", "depth": 4, "child_count": 0})
    result = assign_label(node, [], "local_only")
    assert result == 1


def test_open_image():
    node = _node(**{"class": "android.widget.ImageView", "child_count": 0})
    result = assign_label(node, [], "local_only")
    assert result == 2


def test_excluded_no_text_leaf():
    node = _node(**{"class": "android.widget.View", "child_count": 0, "text": "", "content-desc": ""})
    result = assign_label(node, [], "local_only")
    assert result is None


def test_missing_fields_no_crash():
    node = {"node_id": 0, "depth": 2, "child_count": 0}
    result = assign_label(node, [], "local_only")
    # Should not raise; may return None or a valid label
    assert result in (None, 0, 1, 2)


def test_list_valued_text_fields_no_crash():
    node = _node(
        **{
            "text": ["Click", "here"],
            "content-desc": ["account", None, "details"],
            "resource-id": ["com.example:id", "profile_button"],
        }
    )
    assert assign_label(node, [], "local_only") == 0


# ---- label_graph tests -------------------------------------------------------

def test_label_graph_returns_minus_one_for_excluded():
    tree = {
        "class": "android.widget.FrameLayout",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {"class": "android.widget.View", "bounds": [0, 0, 100, 100], "children": []},
        ],
    }
    flat = flatten_hierarchy(tree)
    labels = label_graph(flat, "local_only")
    assert len(labels) == 2
    assert all(l in (-1, 0, 1, 2) for l in labels)


def test_label_graph_contextual_vs_local_only_differ():
    """
    Build a tree where a leaf has text '$9.99' and the parent has checkout resource-id.
    contextual should label it canonical; local_only should not (no canonical resource-id on leaf).
    """
    tree = {
        "class": "android.widget.LinearLayout",
        "bounds": [0, 0, 1080, 1920],
        "resource-id": "com.ex:id/checkout_section",
        "children": [
            {
                "class": "android.widget.TextView",
                "bounds": [0, 0, 200, 50],
                "text": "$9.99",
                "children": [],
            }
        ],
    }
    flat = flatten_hierarchy(tree)
    labels_ctx = label_graph(flat, "contextual")
    labels_local = label_graph(flat, "local_only")
    leaf_idx = 1  # child in DFS pre-order
    assert labels_ctx[leaf_idx] == 0, "contextual should label price-in-checkout as canonical"
    assert labels_local[leaf_idx] != 0, "local_only must not use ancestor context"
