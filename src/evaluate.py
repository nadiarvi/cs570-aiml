import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate(model, loader, device) -> dict:
    """
    Returns macro_f1, per_class_f1, per_class_precision, per_class_recall,
    accuracy, confusion_matrix — all computed on nodes where y != -1.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            ei = batch["edge_index"].to(device)
            y = batch["y"]

            logits = model(x, ei)
            preds = logits.argmax(dim=-1).cpu()

            mask = y != -1
            all_preds.append(preds[mask])
            all_labels.append(y[mask])

    if not all_preds:
        return {"macro_f1": 0.0}

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    macro_f1 = f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0)
    per_f1 = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {i: float(per_f1[i]) for i in range(3)},
        "per_class_precision": {i: float(prec[i]) for i in range(3)},
        "per_class_recall": {i: float(rec[i]) for i in range(3)},
        "accuracy": float(acc),
        "confusion_matrix": cm,
    }


def evaluate_app_holdout(model, test_loader, device) -> dict:
    """Evaluate on app-level test holdout. This is the ONLY number reported in the paper."""
    return evaluate(model, test_loader, device)


def plot_confusion_matrix(cm, run_name, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    labels = ["canonical", "translatable", "open"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(log_csv, run_name, save_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df["epoch"], df["train_loss"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{run_name} — Train Loss")

    ax2.plot(df["epoch"], df["val_macro_f1"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Macro-F1")
    ax2.set_title(f"{run_name} — Val Macro-F1")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_macro_f1_bar(results_csv, save_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    main_runs = df[df["name"].isin(["mlp_all", "gcn_2l_all", "gat_2l_all"])]

    plt.figure(figsize=(7, 5))
    plt.bar(main_runs["name"], main_runs["macro_f1"])
    plt.xlabel("Model")
    plt.ylabel("Macro-F1")
    plt.title("Model Comparison — Test Macro-F1 (app-level holdout)")
    plt.ylim(0, 1)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ablation_bar(results_csv, save_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    plt.figure(figsize=(14, 5))
    bars = plt.bar(df["name"], df["macro_f1"])
    plt.xlabel("Config")
    plt.ylabel("Macro-F1")
    plt.title("Ablation Study — Test Macro-F1")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results/ablation_results.csv")
    parser.add_argument("--logs_dir", default="results/logs")
    parser.add_argument("--figures_dir", default="results/figures")
    args = parser.parse_args()

    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    results_csv = args.results_csv
    if Path(results_csv).exists():
        plot_macro_f1_bar(results_csv, figs / "model_comparison.png")
        plot_ablation_bar(results_csv, figs / "ablation_comparison.png")
        print("Model comparison and ablation bar charts saved.")

    logs_dir = Path(args.logs_dir)
    for log_csv in logs_dir.glob("*_log.csv"):
        run_name = log_csv.stem.replace("_log", "")
        plot_training_curves(str(log_csv), run_name, figs / f"{run_name}_curves.png")
        print(f"Training curves saved for {run_name}.")


if __name__ == "__main__":
    main()
