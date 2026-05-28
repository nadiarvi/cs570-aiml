"""
Fixed ablation experiment matrix.
Trains each config and evaluates on gold labels.
Writes results/ablation_results.csv.
"""

import csv
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RicoGraphDataset, collate_graphs
from src.data.gold import load_gold_test_labels, attach_gold_labels
from src.evaluate import evaluate
from src.models.mlp import MLP
from src.models.gcn import GCN
from src.models.gat import GAT
from src.train import train, _build_model, collect_graph_paths

logger = logging.getLogger(__name__)

ABLATION_CONFIGS = [
    {"name": "mlp_all_contextual",        "model": "mlp", "label_mode": "contextual",  "edges": "all",         "features": "all",     "layers": 3},
    {"name": "gcn_2l_all_contextual",     "model": "gcn", "label_mode": "contextual",  "edges": "all",         "features": "all",     "layers": 2},
    {"name": "gat_2l_all_contextual",     "model": "gat", "label_mode": "contextual",  "edges": "all",         "features": "all",     "layers": 2},
    {"name": "gcn_2l_contain_contextual", "model": "gcn", "label_mode": "contextual",  "edges": "containment", "features": "all",     "layers": 2},
    {"name": "gcn_2l_notext_contextual",  "model": "gcn", "label_mode": "contextual",  "edges": "all",         "features": "no_text", "layers": 2},
    {"name": "gcn_1l_all_contextual",     "model": "gcn", "label_mode": "contextual",  "edges": "all",         "features": "all",     "layers": 1},
    {"name": "gcn_3l_all_contextual",     "model": "gcn", "label_mode": "contextual",  "edges": "all",         "features": "all",     "layers": 3},
    {"name": "mlp_all_local_only",        "model": "mlp", "label_mode": "local_only",  "edges": "all",         "features": "all",     "layers": 3},
    {"name": "gcn_2l_all_local_only",     "model": "gcn", "label_mode": "local_only",  "edges": "all",         "features": "all",     "layers": 2},
]

CSV_COLUMNS = [
    "name", "model", "label_mode", "edges", "features", "layers",
    "heuristic_val_macro_f1",
    "gold_macro_f1", "gold_accuracy",
    "gold_precision_canonical", "gold_recall_canonical", "gold_f1_canonical",
    "gold_precision_translatable", "gold_recall_translatable", "gold_f1_translatable",
    "gold_precision_open", "gold_recall_open", "gold_f1_open",
    "gold_num_nodes", "gold_num_screens", "gold_num_apps",
    "checkpoint_path",
]

_FEATURE_GROUP_MAP = {
    "all": ["visual", "structural", "type", "text"],
    "no_text": ["visual", "structural", "type"],
}


def _config_for_ablation(abl: dict, base_config: dict) -> dict:
    cfg = {**base_config}
    cfg["name"] = abl["name"]
    cfg["model_type"] = abl["model"]
    cfg["label_mode"] = abl["label_mode"]
    cfg["num_layers"] = abl["layers"]
    cfg["include_sibling_edges"] = abl["edges"] != "containment"
    cfg["feature_groups"] = _FEATURE_GROUP_MAP.get(abl["features"], None)
    cfg["save_dir"] = os.path.join(base_config.get("save_root", "results/checkpoints"), abl["name"])
    return cfg


def _load_gold_graphs(
    processed_dir: str,
    gold_df,
    label_mode: str,
    gold_app_ids: set[str],
    include_sibling_edges: bool,
    feature_groups: list[str] | None,
    device: torch.device,
) -> DataLoader | None:
    """Build a DataLoader over gold-app graphs with gold labels attached."""
    pt_paths = []
    for app_id in gold_app_ids:
        pattern = os.path.join(processed_dir, "**", label_mode, app_id, "*.pt")
        pt_paths.extend(collect_graph_paths(pattern))
        # Also check train/val dirs for gold apps that were processed
        for partition in ("train", "val", "other"):
            p = os.path.join(processed_dir, partition, label_mode, app_id, "*.pt")
            pt_paths.extend(collect_graph_paths(p))

    pt_paths = sorted(set(pt_paths))
    if not pt_paths:
        logger.warning("No gold graphs found for evaluation.")
        return None

    # Attach gold labels to each graph
    gold_graphs = []
    for path in pt_paths:
        g = torch.load(path, weights_only=False)
        g = attach_gold_labels(g, gold_df)
        gold_graphs.append(g)

    # Wrap in a simple list-dataset
    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, graphs, include_sibling_edges, feature_groups):
            self.graphs = graphs
            self.include_sibling_edges = include_sibling_edges
            self.feature_groups = feature_groups

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            from src.data.dataset import _apply_options
            g = self.graphs[idx]
            g = _apply_options(g, self.include_sibling_edges, self.feature_groups)
            # Make y the gold labels for evaluation
            if "y_gold" in g:
                g["y"] = g["y_gold"]
            return g

    dataset = _ListDataset(gold_graphs, include_sibling_edges, feature_groups)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_graphs)
    return loader


def run_ablation(base_config: dict, out_csv: str = "results/ablation_results.csv") -> None:
    """
    Train and evaluate each ablation config.
    Appends results to out_csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    gold_labels_path = base_config.get("gold_labels_path", "data/gold/gold_test_labels.csv")
    processed_dir = base_config["processed_dir"]
    device = torch.device(base_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    gold_df = load_gold_test_labels(gold_labels_path)
    gold_app_ids = set(
        gold_df["screen_id"].apply(lambda s: s.split("_")[0])
        if "app_id" not in gold_df.columns
        else gold_df["app_id"]
    )

    rows = []
    for abl in ABLATION_CONFIGS:
        cfg = _config_for_ablation(abl, base_config)
        logger.info("Starting ablation: %s", cfg["name"])

        try:
            metadata = train(cfg)
            val_f1 = metadata["best_val_macro_f1"]
            checkpoint_path = metadata["checkpoint_path"]
            in_dim = metadata["in_dim"]
        except Exception as e:
            logger.error("Training failed for %s: %s", cfg["name"], e)
            continue

        # Load best model
        model = _build_model(cfg, in_dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Evaluate on gold
        include_sibling_edges = cfg.get("include_sibling_edges", True)
        feature_groups = cfg.get("feature_groups", None)

        gold_loader = _load_gold_graphs(
            processed_dir, gold_df, cfg["label_mode"], gold_app_ids,
            include_sibling_edges, feature_groups, device,
        )

        if gold_loader is None:
            logger.warning("Skipping gold evaluation for %s (no gold graphs found)", cfg["name"])
            gold_metrics = {}
        else:
            gold_metrics = evaluate(model, gold_loader, device)

        pc = gold_metrics.get("per_class", {})
        row = {
            "name": abl["name"],
            "model": abl["model"],
            "label_mode": abl["label_mode"],
            "edges": abl["edges"],
            "features": abl["features"],
            "layers": abl["layers"],
            "heuristic_val_macro_f1": val_f1,
            "gold_macro_f1": gold_metrics.get("macro_f1", ""),
            "gold_accuracy": gold_metrics.get("accuracy", ""),
            "gold_precision_canonical": pc.get("canonical", {}).get("precision", ""),
            "gold_recall_canonical": pc.get("canonical", {}).get("recall", ""),
            "gold_f1_canonical": pc.get("canonical", {}).get("f1", ""),
            "gold_precision_translatable": pc.get("translatable", {}).get("precision", ""),
            "gold_recall_translatable": pc.get("translatable", {}).get("recall", ""),
            "gold_f1_translatable": pc.get("translatable", {}).get("f1", ""),
            "gold_precision_open": pc.get("open", {}).get("precision", ""),
            "gold_recall_open": pc.get("open", {}).get("recall", ""),
            "gold_f1_open": pc.get("open", {}).get("f1", ""),
            "gold_num_nodes": gold_metrics.get("num_nodes", ""),
            "gold_num_screens": gold_metrics.get("num_screens", ""),
            "gold_num_apps": gold_metrics.get("num_apps", ""),
            "checkpoint_path": checkpoint_path,
        }
        rows.append(row)
        logger.info(
            "Ablation %s done: val_f1=%.4f  gold_f1=%s",
            abl["name"], val_f1, row["gold_macro_f1"],
        )

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Ablation results written to %s", out_csv)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config JSON path")
    parser.add_argument("--out_csv", default="results/ablation_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config) as f:
        base_config = json.load(f)
    run_ablation(base_config, out_csv=args.out_csv)
