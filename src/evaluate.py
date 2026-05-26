"""
Gold evaluation, metrics, bootstrap comparison, and qualitative output.
Gold labels are only used here, never during training.
"""

import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["canonical", "translatable", "open"]


def evaluate(model, loader, device) -> dict:
    """
    Run model over loader and compute gold-label metrics.
    Filters out nodes with label -1.

    Returns dict with macro_f1, accuracy, per-class metrics, confusion_matrix,
    predictions, labels, screen_ids, app_ids, node_ids.
    """
    model.eval()
    all_preds, all_labels = [], []
    all_screen_ids, all_app_ids, all_node_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            ei = batch["edge_index"].to(device)

            # Use y_gold if present, else fall back to y
            y = batch.get("y_gold", batch["y"])
            if isinstance(y, torch.Tensor):
                y = y
            else:
                y = torch.tensor(y, dtype=torch.long)
            y = y.to(device)

            logits = model(x, ei)
            preds = logits.argmax(dim=-1)

            mask = y != -1
            preds_masked = preds[mask].cpu().tolist()
            labels_masked = y[mask].cpu().tolist()

            all_preds.extend(preds_masked)
            all_labels.extend(labels_masked)

            # Expand per-node metadata
            node_ids = batch.get("node_ids", [])
            screen_ids_per_graph = batch.get("screen_ids", [])
            app_ids_per_graph = batch.get("app_ids", [])
            batch_index = batch.get("batch_index")

            if batch_index is not None:
                mask_cpu = mask.cpu()
                node_ids_arr = np.array(node_ids)
                batch_arr = batch_index.cpu().numpy()

                for i, (m, nid, bidx) in enumerate(zip(mask_cpu, node_ids_arr, batch_arr)):
                    if m:
                        all_node_ids.append(str(nid))
                        all_screen_ids.append(screen_ids_per_graph[bidx] if bidx < len(screen_ids_per_graph) else "")
                        all_app_ids.append(app_ids_per_graph[bidx] if bidx < len(app_ids_per_graph) else "")

    if not all_labels:
        return {"macro_f1": 0.0, "accuracy": 0.0, "num_nodes": 0}

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    acc = float(accuracy_score(all_labels, all_preds))

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], average=None, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    screen_set = set(all_screen_ids)
    app_set = set(all_app_ids)

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "screen_ids": all_screen_ids,
        "app_ids": all_app_ids,
        "node_ids": all_node_ids,
        "num_nodes": len(all_labels),
        "num_screens": len(screen_set),
        "num_apps": len(app_set),
    }


def bootstrap_compare(
    predictions_a: list[int],
    predictions_b: list[int],
    labels: list[int],
    group_ids: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test comparing two models on gold labels.
    Resamples by group_ids (screen_id or app_id).

    Returns metric_diff, ci_lower, ci_upper, p_value (tail probability).
    """
    rng = np.random.default_rng(seed)

    preds_a = np.array(predictions_a)
    preds_b = np.array(predictions_b)
    lbls = np.array(labels)
    groups = np.array(group_ids)

    unique_groups = np.unique(groups)

    def _macro_f1(p, l):
        return float(f1_score(l, p, average="macro", zero_division=0))

    observed_a = _macro_f1(preds_a, lbls)
    observed_b = _macro_f1(preds_b, lbls)
    observed_diff = observed_a - observed_b

    boot_diffs = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, sampled)
        if mask.sum() == 0:
            continue
        f1_a = _macro_f1(preds_a[mask], lbls[mask])
        f1_b = _macro_f1(preds_b[mask], lbls[mask])
        boot_diffs.append(f1_a - f1_b)

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    # p-value: fraction of bootstrap diffs on the wrong side of 0
    p_value = float(np.mean(boot_diffs <= 0)) if observed_diff > 0 else float(np.mean(boot_diffs >= 0))

    return {
        "macro_f1_a": observed_a,
        "macro_f1_b": observed_b,
        "metric_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "n_groups": len(unique_groups),
    }


def save_qualitative_examples(
    eval_results_gcn: dict,
    eval_results_mlp: dict,
    out_path: str,
    max_examples: int = 50,
) -> None:
    """
    Save examples where GCN and MLP disagree, prioritizing gold-label mismatches.
    """
    preds_gcn = np.array(eval_results_gcn["predictions"])
    preds_mlp = np.array(eval_results_mlp["predictions"])
    labels = np.array(eval_results_gcn["labels"])
    node_ids = eval_results_gcn.get("node_ids", [])
    screen_ids = eval_results_gcn.get("screen_ids", [])

    disagree_mask = preds_gcn != preds_mlp
    indices = np.where(disagree_mask)[0]

    examples = []
    for i in indices[:max_examples]:
        examples.append({
            "node_id": node_ids[i] if i < len(node_ids) else "",
            "screen_id": screen_ids[i] if i < len(screen_ids) else "",
            "gold_label": CLASS_NAMES[labels[i]],
            "gcn_pred": CLASS_NAMES[preds_gcn[i]],
            "mlp_pred": CLASS_NAMES[preds_mlp[i]],
            "gcn_correct": bool(preds_gcn[i] == labels[i]),
            "mlp_correct": bool(preds_mlp[i] == labels[i]),
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info("Saved %d qualitative examples to %s", len(examples), out_path)
