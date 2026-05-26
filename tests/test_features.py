"""Unit tests for features.py — skips text embedding tests to avoid downloading models."""

import torch
import pytest
from src.data.rico_loader import flatten_hierarchy
from src.data.features import (
    extract_features,
    VISUAL_DIM, STRUCTURAL_DIM, TYPE_DIM, TEXT_DIM,
    ALL_GROUPS,
)


def _simple_tree():
    return {
        "class": "android.widget.FrameLayout",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {
                "class": "android.widget.TextView",
                "bounds": [0, 0, 540, 100],
                "text": "hello",
                "children": [],
            },
        ],
    }


def test_feature_dim_structural_only():
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["structural"])
    assert features.shape == (2, STRUCTURAL_DIM)
    assert "structural" in slices
    assert slices["structural"] == (0, STRUCTURAL_DIM)


def test_feature_dim_visual_structural():
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["visual", "structural"])
    expected_dim = VISUAL_DIM + STRUCTURAL_DIM
    assert features.shape == (2, expected_dim)
    assert slices["visual"] == (0, VISUAL_DIM)
    assert slices["structural"] == (VISUAL_DIM, VISUAL_DIM + STRUCTURAL_DIM)


def test_feature_dim_visual_structural_type():
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["visual", "structural", "type"])
    expected_dim = VISUAL_DIM + STRUCTURAL_DIM + TYPE_DIM
    assert features.shape == (2, expected_dim)


def test_feature_slices_correct_offsets():
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["visual", "structural", "type"])
    # Slices should be contiguous starting from 0
    start = 0
    for g in ["visual", "structural", "type"]:
        assert slices[g][0] == start
        start = slices[g][1]


def test_empty_text_gives_zero_structural():
    """Nodes with no text should still produce valid structural features."""
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["structural"])
    assert not torch.isnan(features).any()


def test_bounds_normalize_by_screen_size():
    """Visual feature values should be in [0, 1] for valid bounds."""
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["visual"])
    assert features.shape[1] == VISUAL_DIM
    # x, y, w, h components should be in [0, 1]
    assert (features[:, :6] >= 0).all()
    assert (features[:, :6] <= 1.01).all()  # tiny float tolerance


def test_feature_dtype():
    flat = flatten_hierarchy(_simple_tree())
    features, slices = extract_features(flat, feature_groups=["visual", "structural"])
    assert features.dtype == torch.float32
