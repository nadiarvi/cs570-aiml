import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.rico_loader import get_app_id


def make_splits(
    all_json_paths,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    CRITICAL: split at APP level, not screen level.
    Groups by get_app_id() → shuffles app IDs → assigns whole apps → flattens.
    Returns (train_paths, val_paths, test_paths) as lists of str.
    """
    app_to_paths = defaultdict(list)
    for path in all_json_paths:
        app_id = get_app_id(str(path))
        app_to_paths[app_id].append(str(path))

    rng = random.Random(seed)
    app_ids = list(app_to_paths.keys())
    rng.shuffle(app_ids)

    n = len(app_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_apps = app_ids[:n_train]
    val_apps = app_ids[n_train:n_train + n_val]
    test_apps = app_ids[n_train + n_val:]

    train_paths = [p for app in train_apps for p in app_to_paths[app]]
    val_paths = [p for app in val_apps for p in app_to_paths[app]]
    test_paths = [p for app in test_apps for p in app_to_paths[app]]

    return train_paths, val_paths, test_paths


def save_splits(train_paths, val_paths, test_paths, out_path="data/splits/split_seed42.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"train": train_paths, "val": val_paths, "test": test_paths}, f)


def load_splits(split_path="data/splits/split_seed42.json"):
    with open(split_path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]
