"""
Gold annotation validation, inter-annotator agreement, and merge utilities.
"""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABEL_MAP = {"canonical": 0, "translatable": 1, "open": 2, 0: 0, 1: 1, 2: 2}
REQUIRED_FIELDS = {"screen_id", "node_id", "label", "annotator_id"}


def _normalize_label(val) -> int:
    """Convert string or int label to integer 0/1/2."""
    if isinstance(val, str):
        val = val.strip().lower()
    if val in LABEL_MAP:
        return LABEL_MAP[val]
    raise ValueError(f"Unknown label value: {val!r}")


def load_annotations(path: str) -> pd.DataFrame:
    """
    Load and validate raw annotation CSV or JSONL.
    Raises ValueError if required fields are missing.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif ext in (".jsonl", ".json"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported annotation file format: {ext}")

    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        raise ValueError(f"Annotation file missing required fields: {missing}")

    df["label_int"] = df["label"].apply(_normalize_label)
    df["screen_id"] = df["screen_id"].astype(str)
    df["node_id"] = df["node_id"].astype(str)
    return df


def compute_agreement(overlap_annotations: pd.DataFrame) -> dict:
    """
    Compute Cohen's kappa and per-class agreement on the overlap subset.
    Expects at least two distinct annotators for overlapping (screen_id, node_id) pairs.

    Returns dict with kappa, per_class_agreement, num_pairs, annotators.
    """
    annotators = overlap_annotations["annotator_id"].unique().tolist()
    if len(annotators) < 2:
        raise ValueError("Need at least 2 annotators to compute agreement.")

    # Pivot: one row per (screen_id, node_id), one column per annotator
    pivot = overlap_annotations.pivot_table(
        index=["screen_id", "node_id"],
        columns="annotator_id",
        values="label_int",
        aggfunc="first",
    )
    # Keep only rows annotated by both of the first two annotators
    a1, a2 = annotators[0], annotators[1]
    valid = pivot[[a1, a2]].dropna()
    if len(valid) == 0:
        raise ValueError("No overlapping (screen_id, node_id) pairs found between annotators.")

    labels_a = valid[a1].astype(int).tolist()
    labels_b = valid[a2].astype(int).tolist()

    kappa = cohen_kappa_score(labels_a, labels_b)

    # Per-class agreement
    labels_a_arr = np.array(labels_a)
    labels_b_arr = np.array(labels_b)
    per_class = {}
    for cls, name in enumerate(["canonical", "translatable", "open"]):
        mask = (labels_a_arr == cls) | (labels_b_arr == cls)
        if mask.sum() == 0:
            per_class[name] = None
        else:
            agree = (labels_a_arr[mask] == labels_b_arr[mask]).mean()
            per_class[name] = float(agree)

    return {
        "kappa": float(kappa),
        "per_class_agreement": per_class,
        "num_pairs": len(valid),
        "annotators": [a1, a2],
    }


def write_disagreement_report(overlap_annotations: pd.DataFrame, out_path: str) -> None:
    """Write rows where annotators disagree on the same (screen_id, node_id)."""
    annotators = overlap_annotations["annotator_id"].unique().tolist()
    if len(annotators) < 2:
        return

    pivot = overlap_annotations.pivot_table(
        index=["screen_id", "node_id"],
        columns="annotator_id",
        values="label_int",
        aggfunc="first",
    )
    a1, a2 = annotators[0], annotators[1]
    valid = pivot[[a1, a2]].dropna()
    disagreements = valid[valid[a1] != valid[a2]].reset_index()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    disagreements.to_csv(out_path, index=False)


def load_gold_test_labels(path: str) -> pd.DataFrame:
    """
    Load final resolved labels. Must contain exactly one label per (screen_id, node_id).
    Raises ValueError on duplicates or missing fields.
    """
    df = load_annotations(path)
    dups = df.duplicated(subset=["screen_id", "node_id"])
    if dups.any():
        raise ValueError(f"Duplicate (screen_id, node_id) pairs in gold labels: {dups.sum()} rows.")
    return df


def attach_gold_labels(graph: dict, gold_labels: pd.DataFrame) -> dict:
    """
    Return a copy of graph with y_gold populated for matching nodes and -1 otherwise.
    Matches on (screen_id, node_id).
    """
    import torch

    screen_id = str(graph.get("screen_id", ""))
    node_ids = [str(nid) for nid in graph.get("node_ids", [])]
    N = graph["num_nodes"]

    subset = gold_labels[gold_labels["screen_id"] == screen_id]
    gold_map = dict(zip(subset["node_id"].astype(str), subset["label_int"]))

    y_gold = torch.full((N,), -1, dtype=torch.long)
    for i, nid in enumerate(node_ids):
        if nid in gold_map:
            y_gold[i] = gold_map[nid]

    result = {k: v for k, v in graph.items()}
    result["y_gold"] = y_gold
    return result
