"""Unit tests for gold.py"""

import io
import os
import tempfile

import pandas as pd
import pytest

from src.data.gold import (
    load_annotations,
    compute_agreement,
    write_disagreement_report,
    load_gold_test_labels,
    attach_gold_labels,
)


def _make_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return its path."""
    df = pd.DataFrame(rows)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


def test_load_annotations_valid():
    path = _make_csv([
        {"screen_id": "s1", "node_id": "0", "label": "canonical", "annotator_id": "A"},
        {"screen_id": "s1", "node_id": "1", "label": "translatable", "annotator_id": "A"},
    ])
    df = load_annotations(path)
    assert "label_int" in df.columns
    assert list(df["label_int"]) == [0, 1]
    os.unlink(path)


def test_load_annotations_missing_field():
    path = _make_csv([
        {"screen_id": "s1", "node_id": "0", "label": "canonical"},  # missing annotator_id
    ])
    with pytest.raises(ValueError, match="missing required fields"):
        load_annotations(path)
    os.unlink(path)


def test_compute_agreement_kappa():
    rows = [
        {"screen_id": "s1", "node_id": "0", "label": "canonical",    "annotator_id": "A"},
        {"screen_id": "s1", "node_id": "0", "label": "canonical",    "annotator_id": "B"},
        {"screen_id": "s1", "node_id": "1", "label": "translatable", "annotator_id": "A"},
        {"screen_id": "s1", "node_id": "1", "label": "open",         "annotator_id": "B"},
    ]
    path = _make_csv(rows)
    df = load_annotations(path)
    result = compute_agreement(df)
    assert "kappa" in result
    assert -1.0 <= result["kappa"] <= 1.0
    os.unlink(path)


def test_write_disagreement_report():
    rows = [
        {"screen_id": "s1", "node_id": "0", "label": "canonical",    "annotator_id": "A"},
        {"screen_id": "s1", "node_id": "0", "label": "translatable", "annotator_id": "B"},
        {"screen_id": "s1", "node_id": "1", "label": "open",         "annotator_id": "A"},
        {"screen_id": "s1", "node_id": "1", "label": "open",         "annotator_id": "B"},  # agree
    ]
    path = _make_csv(rows)
    df = load_annotations(path)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out_f:
        out_path = out_f.name

    write_disagreement_report(df, out_path)
    report = pd.read_csv(out_path)
    # Only node_id=0 disagrees
    assert len(report) == 1
    os.unlink(path)
    os.unlink(out_path)


def test_load_gold_test_labels_no_duplicates():
    rows = [
        {"screen_id": "s1", "node_id": "0", "label": "canonical",    "annotator_id": "resolved"},
        {"screen_id": "s1", "node_id": "1", "label": "translatable", "annotator_id": "resolved"},
    ]
    path = _make_csv(rows)
    df = load_gold_test_labels(path)
    assert len(df) == 2
    os.unlink(path)


def test_load_gold_test_labels_raises_on_duplicates():
    rows = [
        {"screen_id": "s1", "node_id": "0", "label": "canonical",    "annotator_id": "resolved"},
        {"screen_id": "s1", "node_id": "0", "label": "translatable", "annotator_id": "resolved"},
    ]
    path = _make_csv(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_gold_test_labels(path)
    os.unlink(path)


def test_attach_gold_labels_aligns_by_screen_node():
    import torch

    gold_rows = [
        {"screen_id": "s1", "node_id": "1", "label": "translatable", "annotator_id": "r"},
    ]
    path = _make_csv(gold_rows)
    gold_df = load_gold_test_labels(path)

    graph = {
        "screen_id": "s1",
        "node_ids": ["0", "1", "2"],
        "num_nodes": 3,
        "x": torch.zeros(3, 4),
        "y": torch.zeros(3, dtype=torch.long),
    }

    result = attach_gold_labels(graph, gold_df)
    y_gold = result["y_gold"]
    assert y_gold[0].item() == -1   # not in gold
    assert y_gold[1].item() == 1    # translatable
    assert y_gold[2].item() == -1   # not in gold
    os.unlink(path)
