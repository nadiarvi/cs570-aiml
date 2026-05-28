"""
Two-pass preprocessing script.

Pass 1 — heuristic labels (fast):
  python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 4

Pass 2 — add LLM labels to existing .pt files:
  python src/data/preprocess.py --processed_dir data/processed --add_llm --max_screens 5000
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.features import (
    extract_features,
    load_embed_cache,
    save_embed_cache,
    _embed_cache_dirty,
)
from src.data.graph_builder import build_graph
from src.data.labeler import assign_label
from src.data.labeler_llm import label_nodes_llm, load_cache, save_cache
from src.data.rico_loader import flatten_hierarchy, load_hierarchy
from src.data.splits import load_splits, make_splits, save_splits


def _json_to_pt(json_path: str, out_dir: Path) -> Path:
    rel = Path(json_path).stem
    parent_name = Path(json_path).parent.name
    if parent_name not in ("raw", ".", ""):
        return out_dir / parent_name / (rel + ".pt")
    return out_dir / (rel + ".pt")


def process_one(json_path: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    try:
        nodes, c_edges, s_edges = flatten_hierarchy(load_hierarchy(json_path))
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return

    try:
        features = extract_features(nodes)
    except Exception as e:
        print(f"[WARN] Feature extraction failed for {json_path}: {e}")
        return

    labels_h = [assign_label(n, n["ancestors"]) for n in nodes]
    graph = build_graph(nodes, c_edges, s_edges, features, labels_h)
    torch.save(graph, out_path)


def pass1(args):
    rico_dir = Path(args.rico_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_json_paths = sorted(rico_dir.rglob("*.json"))
    if args.max_screens:
        all_json_paths = all_json_paths[: args.max_screens]

    if not all_json_paths:
        print(f"No JSON files found in {rico_dir}")
        return

    # App-level splits
    split_path = "data/splits/split_seed42.json"
    if not Path(split_path).exists():
        print("Creating app-level splits...")
        train_json, val_json, test_json = make_splits(all_json_paths)
        # Convert JSON paths → .pt paths for the split file
        train_pt = [str(_json_to_pt(p, out_dir)) for p in train_json]
        val_pt = [str(_json_to_pt(p, out_dir)) for p in val_json]
        test_pt = [str(_json_to_pt(p, out_dir)) for p in test_json]
        save_splits(train_pt, val_pt, test_pt, split_path)
        print(f"Splits saved: {len(train_pt)} train / {len(val_pt)} val / {len(test_pt)} test")
    else:
        print(f"Split file already exists: {split_path}")

    # Load embedding cache once (shared across all processed files)
    load_embed_cache()

    for json_path in tqdm(all_json_paths, desc="Preprocessing"):
        out_path = _json_to_pt(str(json_path), out_dir)
        process_one(str(json_path), out_path)

    import src.data.features as _fm
    if _fm._embed_cache_dirty:
        save_embed_cache()
        print("Embedding cache saved.")


def pass2(args):
    processed_dir = Path(args.processed_dir)
    all_pt_paths = sorted(processed_dir.rglob("*.pt"))
    if args.max_screens:
        all_pt_paths = all_pt_paths[: args.max_screens]

    if not all_pt_paths:
        print(f"No .pt files found in {processed_dir}")
        return

    cache = load_cache()

    for pt_path in tqdm(all_pt_paths, desc="Adding LLM labels"):
        graph = torch.load(pt_path, weights_only=False)

        # Skip if already labeled
        if "y_llm" in graph and (graph["y_llm"] != -1).any():
            continue

        labels_l = label_nodes_llm(graph["nodes"], cache)
        graph["y_llm"] = torch.tensor(
            [-1 if lbl is None else lbl for lbl in labels_l], dtype=torch.long
        )
        torch.save(graph, pt_path)

    print(f"LLM labeling complete. Cache at {args.cache_path if hasattr(args, 'cache_path') else 'data/llm_label_cache.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rico_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_screens", type=int, default=None)
    parser.add_argument("--add_llm", action="store_true", help="Run Pass 2: add LLM labels")
    args = parser.parse_args()

    if args.add_llm:
        pass2(args)
    else:
        pass1(args)


if __name__ == "__main__":
    main()
