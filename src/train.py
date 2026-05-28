"""
Deterministic training loop.
Trains on heuristic train split, early-stops on heuristic validation split.
Gold labels are never touched during training.
"""

import json
import logging
import os
import random
import subprocess
import time
import glob

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.data.dataset import RicoGraphDataset, collate_graphs
from src.models.mlp import MLP
from src.models.gcn import GCN
from src.models.gat import GAT

logger = logging.getLogger(__name__)


def collect_graph_paths(pattern: str) -> list[str]:
    """Return sorted regular graph files matching a glob pattern."""
    paths = sorted(glob.glob(pattern, recursive=True))
    skipped_dirs = [p for p in paths if os.path.isdir(p)]
    if skipped_dirs:
        logger.warning(
            "Ignoring %d directory path(s) matched by graph glob, e.g. %s",
            len(skipped_dirs),
            skipped_dirs[0],
        )
    return [p for p in paths if os.path.isfile(p)]


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_model(config: dict, in_dim: int) -> nn.Module:
    model_type = config["model_type"]
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.3)
    num_classes = config.get("num_classes", 3)

    if model_type == "mlp":
        return MLP(in_dim, hidden_dim, num_classes, num_layers, dropout)
    elif model_type == "gcn":
        return GCN(in_dim, hidden_dim, num_classes, num_layers, dropout)
    elif model_type == "gat":
        return GAT(in_dim, hidden_dim, num_classes, num_layers,
                   num_heads=config.get("num_heads", 4), dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _make_summary_writer(config: dict, save_dir: str, purge_step: int | None = None):
    if not config.get("enable_tensorboard", True):
        return None, None
    log_dir = config.get("tensorboard_log_dir", os.path.join(save_dir, "tensorboard"))
    if SummaryWriter is None:
        logger.warning(
            "TensorBoard logging disabled because the tensorboard package is not installed."
        )
        return None, log_dir
    return SummaryWriter(log_dir=log_dir, purge_step=purge_step), log_dir


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_training_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_f1: float,
    best_epoch: int,
    patience_counter: int,
    train_loss_history: list[float],
    val_f1_history: list[float],
    config: dict,
    in_dim: int,
    checkpoint_path: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "train_loss_history": train_loss_history,
            "val_f1_history": val_f1_history,
            "config": config,
            "in_dim": in_dim,
            "checkpoint_path": checkpoint_path,
            "rng_state": _rng_state(),
        },
        path,
    )


def _load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def _compute_class_weights(
    dataset: RicoGraphDataset,
    num_classes: int = 3,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute inverse-frequency weights from training labels only."""
    counts = torch.zeros(num_classes)
    iterator = tqdm(
        dataset,
        desc="Computing class weights",
        unit="graph",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for graph in iterator:
        y = graph["y"]
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    total = counts.sum().clamp(min=1)
    weights = total / (counts.clamp(min=1) * num_classes)
    return weights


def _eval_split(
    model,
    loader,
    device,
    num_classes: int = 3,
    show_progress: bool = False,
) -> tuple[float, float]:
    """Return (macro_f1, accuracy) on a validation split."""
    model.eval()
    all_preds, all_labels = [], []
    iterator = tqdm(
        loader,
        desc="Validation",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for batch in iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            ei = batch["edge_index"].to(device)
            logits = model(x, ei)
            preds = logits.argmax(dim=-1)
            mask = y != -1
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(y[mask].cpu().tolist())

    if not all_labels:
        return 0.0, 0.0

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return float(macro_f1), float(acc)


def train(config: dict) -> dict:
    """
    Train on heuristic train split, early-stop on heuristic validation Macro-F1.
    Returns run metadata dict.
    """
    _seed_everything(config.get("seed", 42))

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Training %s on %s", config.get("name", "run"), device)

    label_mode = config.get("label_mode", "contextual")
    processed_dir = config["processed_dir"]
    include_sibling_edges = config.get("include_sibling_edges", True)
    feature_groups = config.get("feature_groups", None)

    train_pt_paths = collect_graph_paths(
        os.path.join(processed_dir, "train", label_mode, "**", "*.pt")
    )
    val_pt_paths = collect_graph_paths(
        os.path.join(processed_dir, "val", label_mode, "**", "*.pt")
    )

    if not train_pt_paths:
        raise FileNotFoundError(
            f"No training graphs found in {processed_dir}/train/{label_mode}/**/*.pt"
        )

    train_dataset = RicoGraphDataset(train_pt_paths, include_sibling_edges, feature_groups)
    val_dataset = RicoGraphDataset(val_pt_paths, include_sibling_edges, feature_groups)

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_graphs, num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_graphs, num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    # Determine feature dimension from first graph
    sample = train_dataset[0]
    in_dim = sample["x"].shape[1]

    model = _build_model(config, in_dim).to(device)

    # Class weights from training labels
    class_weights = _compute_class_weights(train_dataset, show_progress=True)
    class_weights = class_weights.to(device)
    logger.info("Class weights: %s", class_weights.cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
    )

    epochs = config.get("epochs", 100)
    patience = config.get("patience", 15)
    save_dir = config.get("save_dir", "results/checkpoints/run")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    latest_checkpoint_path = config.get(
        "resume_checkpoint_path",
        os.path.join(save_dir, "latest_checkpoint.pt"),
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_loss_history, val_f1_history = [], []
    start_epoch = 1
    resumed_from_checkpoint = False

    if config.get("resume", False):
        if os.path.isfile(latest_checkpoint_path):
            checkpoint = _load_training_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_f1 = float(checkpoint.get("best_val_f1", best_val_f1))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            patience_counter = int(checkpoint.get("patience_counter", patience_counter))
            train_loss_history = list(checkpoint.get("train_loss_history", []))
            val_f1_history = list(checkpoint.get("val_f1_history", []))
            checkpoint_path = checkpoint.get("checkpoint_path", checkpoint_path)
            resumed_from_checkpoint = True
            logger.info(
                "Resumed from %s at epoch %d; next epoch is %d",
                latest_checkpoint_path,
                checkpoint["epoch"],
                start_epoch,
            )
        else:
            logger.warning(
                "Resume requested, but no checkpoint found at %s. Starting from scratch.",
                latest_checkpoint_path,
            )

    writer, tensorboard_log_dir = _make_summary_writer(
        config,
        save_dir,
        purge_step=start_epoch if resumed_from_checkpoint else None,
    )
    if tensorboard_log_dir:
        logger.info("TensorBoard logs: %s", tensorboard_log_dir)

    start_time = time.time()
    last_epoch = start_epoch - 1

    epoch_iterator = tqdm(
        range(start_epoch, epochs + 1),
        desc=f"Training {config.get('name', 'run')}",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_iterator:
        last_epoch = epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in train_iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            ei = batch["edge_index"].to(device)

            optimizer.zero_grad()
            logits = model(x, ei)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            train_iterator.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        val_f1, val_acc = _eval_split(model, val_loader, device, show_progress=True)

        should_stop = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                should_stop = True

        train_loss_history.append(avg_loss)
        val_f1_history.append(val_f1)

        if writer is not None:
            writer.add_scalar("loss/train", avg_loss, epoch)
            writer.add_scalar("f1/val_macro", val_f1, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("early_stopping/patience_counter", patience_counter, epoch)

        _save_training_checkpoint(
            latest_checkpoint_path,
            epoch,
            model,
            optimizer,
            best_val_f1,
            best_epoch,
            patience_counter,
            train_loss_history,
            val_f1_history,
            config,
            in_dim,
            checkpoint_path,
        )

        logger.info(
            "Epoch %d/%d  loss=%.4f  val_macro_f1=%.4f  val_acc=%.4f",
            epoch, epochs, avg_loss, val_f1, val_acc,
        )
        epoch_iterator.set_postfix(
            loss=f"{avg_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_f1:.4f}",
        )

        if should_stop:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if start_epoch > epochs:
        logger.info(
            "Checkpoint is already at epoch %d, which is >= configured epochs=%d.",
            start_epoch - 1,
            epochs,
        )

    elapsed = time.time() - start_time
    if writer is not None:
        writer.close()

    metadata = {
        "config": config,
        "label_mode": label_mode,
        "in_dim": in_dim,
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "total_epochs": last_epoch,
        "class_weights": class_weights.cpu().tolist(),
        "checkpoint_path": checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "train_loss_history": train_loss_history,
        "val_f1_history": val_f1_history,
        "train_screens": len(train_pt_paths),
        "val_screens": len(val_pt_paths),
        "git_hash": _get_git_hash(),
        "elapsed_seconds": elapsed,
        "tensorboard_log_dir": tensorboard_log_dir,
    }

    with open(os.path.join(save_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Training complete. Best val Macro-F1=%.4f at epoch %d. Checkpoint: %s",
        best_val_f1, best_epoch, checkpoint_path,
    )
    return metadata


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest full training checkpoint for this run.",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        default=None,
        help="Optional path to a full training checkpoint to resume from.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.resume:
        cfg["resume"] = True
    if args.resume_checkpoint_path:
        cfg["resume"] = True
        cfg["resume_checkpoint_path"] = args.resume_checkpoint_path
    train(cfg)
