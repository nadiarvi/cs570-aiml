"""
App-level split generation with gold-app exclusion.
Splits are by app, never by screen, to prevent leakage from repeated app layouts.
"""

import json
import os
import random
from collections import defaultdict

from src.data.rico_loader import get_app_id


def make_splits(
    all_json_paths: list[str],
    gold_app_ids: set[str],
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    seed: int = 42,
    out_path: str | None = None,
) -> dict:
    """
    Group by app_id, exclude gold apps, split remaining into heuristic train/val.

    Returns dict with:
      train_paths, val_paths, train_app_ids, val_app_ids, gold_app_ids,
      counts (apps, screens, labeled_nodes placeholder).
    Saves metadata to out_path if provided.
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"

    # Group paths by app
    app_to_paths: dict[str, list[str]] = defaultdict(list)
    for p in all_json_paths:
        app_id = get_app_id(p)
        app_to_paths[app_id].append(p)

    # Exclude gold apps
    non_gold_apps = sorted([a for a in app_to_paths if a not in gold_app_ids])

    rng = random.Random(seed)
    rng.shuffle(non_gold_apps)

    n_train = max(1, round(len(non_gold_apps) * train_ratio))
    train_apps = non_gold_apps[:n_train]
    val_apps = non_gold_apps[n_train:]

    train_paths = [p for app in train_apps for p in app_to_paths[app]]
    val_paths = [p for app in val_apps for p in app_to_paths[app]]

    result = {
        "seed": seed,
        "train_paths": sorted(train_paths),
        "val_paths": sorted(val_paths),
        "train_app_ids": sorted(train_apps),
        "val_app_ids": sorted(val_apps),
        "gold_app_ids": sorted(gold_app_ids),
        "counts": {
            "total_apps": len(app_to_paths),
            "gold_apps": len(gold_app_ids),
            "train_apps": len(train_apps),
            "val_apps": len(val_apps),
            "train_screens": len(train_paths),
            "val_screens": len(val_paths),
        },
    }

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def load_splits(split_path: str) -> dict:
    """Load a previously saved split JSON."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)
