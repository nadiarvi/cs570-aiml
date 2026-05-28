"""
Run all ablation experiments and collect results into a single CSV.

Usage:
  python src/ablation.py --output results/ablation_results.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import run_training

ABLATION_CONFIGS = [
    # Track A — heuristic labels
    {"name": "mlp_all",        "model": "mlp", "edges": "all",         "features": "all",     "layers": 3, "label_source": "heuristic"},
    {"name": "gcn_2l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gat_2l_all",     "model": "gat", "edges": "all",         "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_2l_contain", "model": "gcn", "edges": "containment", "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_2l_notext",  "model": "gcn", "edges": "all",         "features": "no_text", "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_1l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 1, "label_source": "heuristic"},
    {"name": "gcn_3l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 3, "label_source": "heuristic"},
    # Track B — LLM labels (run after add_llm_labels.sh completes)
    {"name": "mlp_all_llm",    "model": "mlp", "edges": "all",         "features": "all",     "layers": 3, "label_source": "llm"},
    {"name": "gcn_2l_all_llm", "model": "gcn", "edges": "all",         "features": "all",     "layers": 2, "label_source": "llm"},
    {"name": "gat_2l_all_llm", "model": "gat", "edges": "all",         "features": "all",     "layers": 2, "label_source": "llm"},
]

CSV_FIELDS = [
    "name", "model", "edges", "features", "layers", "label_source",
    "macro_f1", "f1_canonical", "f1_translatable", "f1_open",
    "p_canonical", "r_canonical", "p_translatable", "r_translatable",
    "p_open", "r_open", "per_node_accuracy",
]


def ablation_to_train_config(ab: dict) -> dict:
    feature_groups = None if ab["features"] == "all" else ab["features"]
    return {
        "model_type": ab["model"],
        "hidden_dim": 256,
        "num_layers": ab["layers"],
        "num_heads": 4,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 15,
        "batch_size": 32,
        "include_sibling_edges": ab["edges"] == "all",
        "feature_groups": feature_groups,
        "label_source": ab["label_source"],
        "device": "cuda",
        "save_dir": f"results/checkpoints/{ab['name']}",
    }


def metrics_to_row(name: str, ab: dict, metrics: dict) -> dict:
    pf1 = metrics.get("per_class_f1", {})
    pp  = metrics.get("per_class_precision", {})
    pr  = metrics.get("per_class_recall", {})
    return {
        "name": name,
        "model": ab["model"],
        "edges": ab["edges"],
        "features": ab["features"],
        "layers": ab["layers"],
        "label_source": ab["label_source"],
        "macro_f1": round(metrics.get("macro_f1", 0), 4),
        "f1_canonical":    round(pf1.get(0, 0), 4),
        "f1_translatable": round(pf1.get(1, 0), 4),
        "f1_open":         round(pf1.get(2, 0), 4),
        "p_canonical":     round(pp.get(0, 0), 4),
        "r_canonical":     round(pr.get(0, 0), 4),
        "p_translatable":  round(pp.get(1, 0), 4),
        "r_translatable":  round(pr.get(1, 0), 4),
        "p_open":          round(pp.get(2, 0), 4),
        "r_open":          round(pr.get(2, 0), 4),
        "per_node_accuracy": round(metrics.get("accuracy", 0), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ablation_results.csv")
    parser.add_argument(
        "--names", nargs="*", default=None,
        help="Run only these ablation names (default: all)"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs_to_run = ABLATION_CONFIGS
    if args.names:
        configs_to_run = [c for c in ABLATION_CONFIGS if c["name"] in args.names]

    rows = []
    for ab in configs_to_run:
        name = ab["name"]
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")
        train_cfg = ablation_to_train_config(ab)
        try:
            metrics = run_training(train_cfg, run_name=name)
            row = metrics_to_row(name, ab, metrics)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            row = {k: "" for k in CSV_FIELDS}
            row["name"] = name
            row["model"] = ab["model"]
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
