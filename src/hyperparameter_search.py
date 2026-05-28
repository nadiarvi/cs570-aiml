"""Small deterministic hyperparameter sweep for MLP and GCN models."""

import argparse
import csv
import json
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

FEATURES_ALL = ["visual", "structural", "type", "text"]

CSV_COLUMNS = [
    "name",
    "model_type",
    "num_layers",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "include_sibling_edges",
    "feature_groups",
    "best_val_macro_f1",
    "best_epoch",
    "total_epochs",
    "checkpoint_path",
    "config_path",
]


def _base_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("label_mode", "contextual")
    cfg.setdefault("feature_groups", FEATURES_ALL)
    cfg.setdefault("include_sibling_edges", True)
    cfg.setdefault("model_type", "gcn")
    cfg.setdefault("name", "hparam")
    return cfg


def _quick_specs() -> list[dict]:
    """Curated first-pass sweep that stays reasonably small."""
    specs = []

    for hidden_dim in (128, 256):
        for dropout in (0.1, 0.3, 0.5):
            specs.append({
                "model_type": "mlp",
                "num_layers": 3,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "include_sibling_edges": True,
                "feature_groups": FEATURES_ALL,
            })

    specs.extend([
        {
            "model_type": "gcn",
            "num_layers": layers,
            "hidden_dim": 256,
            "dropout": 0.3,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "include_sibling_edges": True,
            "feature_groups": FEATURES_ALL,
        }
        for layers in (1, 2, 3)
    ])

    specs.extend([
        {
            "model_type": "gcn",
            "num_layers": 2,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": 0.0001,
            "include_sibling_edges": include_sibling_edges,
            "feature_groups": FEATURES_ALL,
        }
        for hidden_dim, dropout, lr, include_sibling_edges in (
            (128, 0.3, 0.001, True),
            (256, 0.1, 0.001, True),
            (256, 0.5, 0.001, True),
            (256, 0.3, 0.0005, True),
            (256, 0.3, 0.001, False),
            (128, 0.1, 0.0005, False),
        )
    ])

    return specs


def _full_specs() -> list[dict]:
    specs = []

    for model_type in ("mlp", "gcn"):
        for num_layers in (1, 2, 3):
            for hidden_dim in (64, 128, 256):
                for dropout in (0.1, 0.3, 0.5):
                    for lr in (0.001, 0.0005):
                        if model_type == "mlp":
                            edge_options = (True,)
                        else:
                            edge_options = (True, False)
                        for include_sibling_edges in edge_options:
                            specs.append({
                                "model_type": model_type,
                                "num_layers": num_layers,
                                "hidden_dim": hidden_dim,
                                "dropout": dropout,
                                "lr": lr,
                                "weight_decay": 0.0001,
                                "include_sibling_edges": include_sibling_edges,
                                "feature_groups": FEATURES_ALL,
                            })

    return specs


def _name_for_spec(spec: dict) -> str:
    edge_name = "all_edges" if spec["include_sibling_edges"] else "containment"
    lr_name = str(spec["lr"]).replace(".", "p")
    wd_name = str(spec["weight_decay"]).replace(".", "p")
    dropout_name = str(spec["dropout"]).replace(".", "p")
    return (
        f"hparam_{spec['model_type']}_"
        f"l{spec['num_layers']}_h{spec['hidden_dim']}_"
        f"d{dropout_name}_lr{lr_name}_wd{wd_name}_{edge_name}"
    )


def _write_config(path: str, cfg: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _metadata_row(cfg: dict, metadata: dict, config_path: str) -> dict:
    return {
        "name": cfg["name"],
        "model_type": cfg["model_type"],
        "num_layers": cfg["num_layers"],
        "hidden_dim": cfg["hidden_dim"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "include_sibling_edges": cfg["include_sibling_edges"],
        "feature_groups": "+".join(cfg.get("feature_groups") or []),
        "best_val_macro_f1": metadata["best_val_macro_f1"],
        "best_epoch": metadata["best_epoch"],
        "total_epochs": metadata["total_epochs"],
        "checkpoint_path": metadata["checkpoint_path"],
        "config_path": config_path,
    }


def _write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_best_configs(rows: list[dict], config_dir: str) -> None:
    for model_type in ("mlp", "gcn"):
        model_rows = [r for r in rows if r["model_type"] == model_type]
        if not model_rows:
            continue
        best = max(model_rows, key=lambda r: float(r["best_val_macro_f1"]))
        best_config_path = os.path.join(config_dir, f"best_{model_type}.json")
        with open(best["config_path"], "r") as f:
            cfg = json.load(f)
        cfg["name"] = f"best_{model_type}_contextual"
        cfg["save_dir"] = os.path.join("results", "checkpoints", cfg["name"])
        _write_config(best_config_path, cfg)
        logger.info(
            "Best %s: val_macro_f1=%.4f config=%s",
            model_type,
            float(best["best_val_macro_f1"]),
            best_config_path,
        )


def run_search(
    base_config_path: str,
    out_csv: str,
    config_dir: str,
    save_root: str,
    search_space: str,
    max_runs: int | None,
    resume: bool,
    skip_completed: bool,
) -> None:
    from src.train import train

    base = _base_config(base_config_path)
    specs = _quick_specs() if search_space == "quick" else _full_specs()
    if max_runs is not None:
        specs = specs[:max_runs]

    rows = []
    for run_idx, spec in enumerate(specs, start=1):
        cfg = deepcopy(base)
        cfg.update(spec)
        cfg["name"] = _name_for_spec(spec)
        cfg["save_dir"] = os.path.join(save_root, cfg["name"])
        cfg["resume"] = resume

        config_path = os.path.join(config_dir, f"{cfg['name']}.json")
        metadata_path = os.path.join(cfg["save_dir"], "run_metadata.json")
        _write_config(config_path, cfg)

        logger.info("Starting run %d/%d: %s", run_idx, len(specs), cfg["name"])
        if skip_completed and os.path.isfile(metadata_path):
            logger.info("Using existing metadata for completed run: %s", metadata_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = train(cfg)

        rows.append(_metadata_row(cfg, metadata, config_path))
        _write_rows(out_csv, rows)

    _write_best_configs(rows, config_dir)
    logger.info("Hyperparameter search results written to %s", out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP and GCN hyperparameter search.")
    parser.add_argument("--base_config", default="experiments/configs/ablation_base.json")
    parser.add_argument("--out_csv", default="results/hparam_search_results.csv")
    parser.add_argument("--config_dir", default="experiments/generated_configs/hparam")
    parser.add_argument("--save_root", default="results/checkpoints/hparam")
    parser.add_argument("--search_space", choices=["quick", "full"], default="quick")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_search(
        base_config_path=args.base_config,
        out_csv=args.out_csv,
        config_dir=args.config_dir,
        save_root=args.save_root,
        search_space=args.search_space,
        max_runs=args.max_runs,
        resume=args.resume,
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
