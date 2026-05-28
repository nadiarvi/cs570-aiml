"""
Train MLP / GCN / GAT from a JSON config file.

Usage:
  python src/train.py --config experiments/configs/gcn_baseline.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import RicoGraphDataset, collate_graphs
from src.data.features import get_feature_dim
from src.data.splits import load_splits
from src.evaluate import evaluate, evaluate_app_holdout, plot_confusion_matrix
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.mlp import MLP


def build_model(config, in_dim):
    mtype = config["model_type"]
    kw = dict(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=3,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    if mtype == "mlp":
        return MLP(**kw)
    elif mtype == "gcn":
        return GCN(**kw)
    elif mtype == "gat":
        kw["num_heads"] = config.get("num_heads", 4)
        return GAT(**kw)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")


def run_training(config: dict, run_name: str = None) -> dict:
    """
    Full training + test evaluation. Returns test metrics dict.
    Config keys: model_type, hidden_dim, num_layers, num_heads, dropout,
                 lr, weight_decay, epochs, patience, batch_size,
                 include_sibling_edges, feature_groups, label_source,
                 device, save_dir.
    """
    device = torch.device(config.get("device", "cpu"))
    save_dir = Path(config.get("save_dir", "results/checkpoints/run"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = save_dir.name

    # Feature groups
    fg = config.get("feature_groups")
    if fg == "no_text":
        fg = ["visual", "structural", "type"]
    in_dim = get_feature_dim(fg)

    label_source = config.get("label_source", "heuristic")
    include_sibling = config.get("include_sibling_edges", True)

    # Load splits
    train_paths, val_paths, test_paths = load_splits()

    train_ds = RicoGraphDataset(train_paths, include_sibling, fg, label_source)
    val_ds   = RicoGraphDataset(val_paths,   include_sibling, fg, label_source)
    test_ds  = RicoGraphDataset(test_paths,  include_sibling, fg, label_source)

    bs = config.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_graphs)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate_graphs)

    # Class weights from training labels
    print("Computing class weights...")
    all_labels = []
    for path in train_ds.graph_paths:
        g = torch.load(path, weights_only=False)
        y = g[f"y_{label_source}"]
        all_labels.append(y)
    all_labels = torch.cat(all_labels)
    all_labels = all_labels[all_labels != -1]
    counts = torch.bincount(all_labels, minlength=3).float()
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)

    model = build_model(config, in_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    epochs = config.get("epochs", 100)
    patience = config.get("patience", 15)
    best_val_f1 = -1.0
    patience_counter = 0
    best_ckpt = save_dir / "best_model.pt"

    log_path = Path("results/logs") / f"{run_name}_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []

    epoch_bar = tqdm(range(epochs), desc=f"[{run_name}] Epochs")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False)

        for batch in batch_bar:
            x  = batch["x"].to(device)
            ei = batch["edge_index"].to(device)
            y  = batch["y"].to(device)

            logits = model(x, ei)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)
        epoch_bar.set_postfix(val_f1=f"{val_f1:.4f}")

        log_rows.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_macro_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch + 1,
                 "val_f1": val_f1, "config": config},
                best_ckpt,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Write log CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Load best model and evaluate on test
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_app_holdout(model, test_loader, device)

    print(f"\n[{run_name}] Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Confusion matrix figure
    cm = test_metrics.get("confusion_matrix")
    if cm is not None:
        plot_confusion_matrix(cm, run_name, f"results/figures/{run_name}_confusion.png")

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_name = Path(args.config).stem
    run_training(config, run_name)


if __name__ == "__main__":
    main()
