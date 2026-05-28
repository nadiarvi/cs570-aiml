"""
End-to-end preprocessing entry point.
Processes Rico JSON files into graph .pt files.

Usage:
  python src/data/preprocess.py \
    --rico_dir data/raw \
    --out_dir data/processed \
    --split_path data/splits/split_seed42.json \
    --label_mode contextual \
    --workers 8
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import traceback

import torch
from tqdm import tqdm

from src.data.rico_loader import load_hierarchy, flatten_hierarchy, get_app_id
from src.data.features import (
    extract_features,
    precompute_text_embedding_cache,
    release_sentence_model,
    _build_text_string,
)
from src.data.labeler import label_graph
from src.data.graph_builder import build_graph
from src.data.splits import make_splits, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _partition_for_app(app_id: str, split: dict) -> str:
    train_app_ids = set(split.get("train_app_ids", []))
    val_app_ids = set(split.get("val_app_ids", []))

    if app_id in train_app_ids:
        return "train"
    if app_id in val_app_ids:
        return "val"
    return "other"


def _output_path(
    json_path: str,
    out_dir: str,
    label_mode: str,
    split: dict,
) -> str:
    screen_id = os.path.splitext(os.path.basename(json_path))[0]
    app_id = get_app_id(json_path)
    partition = _partition_for_app(app_id, split)
    return os.path.join(out_dir, partition, label_mode, app_id, f"{screen_id}.pt")


def _collect_texts_for_precompute(json_paths: list[str]) -> list[str]:
    texts: list[str] = []
    for json_path in tqdm(
        json_paths,
        total=len(json_paths),
        desc="Collecting text for embeddings",
        unit="screen",
    ):
        try:
            root = load_hierarchy(json_path)
            flattened = flatten_hierarchy(root)
            texts.extend(_build_text_string(node) for node in flattened.nodes)
        except Exception:
            logger.warning("Failed to collect text from %s:\n%s", json_path, traceback.format_exc())
    return texts


def _process_one(args: tuple) -> str | None:
    """Process a single Rico JSON file. Returns output path or None on failure."""
    (
        json_path,
        out_dir,
        label_mode,
        embedding_cache_path,
        split,
    ) = args

    try:
        screen_id = os.path.splitext(os.path.basename(json_path))[0]
        app_id = get_app_id(json_path)
        out_path = _output_path(json_path, out_dir, label_mode, split)
        if os.path.isfile(out_path):
            return out_path
        if os.path.isdir(out_path):
            logger.warning("Expected graph output path is a directory, skipping: %s", out_path)
            return None

        root = load_hierarchy(json_path)
        flattened = flatten_hierarchy(root)

        if len(flattened.nodes) == 0:
            return None

        features, feature_slices = extract_features(
            flattened,
            embedding_cache_path=embedding_cache_path,
        )
        labels = label_graph(flattened, mode=label_mode)

        graph = build_graph(
            flattened=flattened,
            features=features,
            labels=labels,
            feature_slices=feature_slices,
            screen_id=screen_id,
            app_id=app_id,
            label_mode=label_mode,
            source_json_path=json_path,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(graph, out_path)
        return out_path

    except Exception:
        logger.warning("Failed to process %s:\n%s", json_path, traceback.format_exc())
        return None


def preprocess(
    rico_dir: str,
    out_dir: str,
    split_path: str,
    label_mode: str = "contextual",
    workers: int = 4,
    embedding_cache_path: str | None = None,
    precompute_embeddings: bool = True,
    embedding_batch_size: int = 256,
    gold_app_ids: set[str] | None = None,
    max_screens: int | None = None,
) -> dict:
    """
    Preprocess Rico JSON files into graph .pt files.
    Returns paths grouped by partition.
    """
    # Collect JSON paths
    json_paths = sorted(glob.glob(os.path.join(rico_dir, "**", "*.json"), recursive=True))
    if max_screens:
        json_paths = json_paths[:max_screens]

    logger.info("Found %d JSON files in %s", len(json_paths), rico_dir)

    # Load or create splits
    if os.path.exists(split_path):
        split = load_splits(split_path)
        logger.info("Loaded existing split from %s", split_path)
    else:
        gold_app_ids = gold_app_ids or set()
        split = make_splits(
            all_json_paths=json_paths,
            gold_app_ids=gold_app_ids,
            out_path=split_path,
        )
        logger.info(
            "Created new split: %d train apps, %d val apps",
            len(split["train_app_ids"]),
            len(split["val_app_ids"]),
        )

    # Build task args
    tasks = [
        (p, out_dir, label_mode, embedding_cache_path, split)
        for p in json_paths
    ]

    pending_json_paths = [
        p for p in json_paths
        if not os.path.isfile(_output_path(p, out_dir, label_mode, split))
    ]

    if embedding_cache_path and precompute_embeddings and pending_json_paths:
        texts = _collect_texts_for_precompute(pending_json_paths)
        stats = precompute_text_embedding_cache(
            texts,
            embedding_cache_path=embedding_cache_path,
            batch_size=embedding_batch_size,
        )
        logger.info(
            "Text embedding precompute complete: %d unique, %d newly encoded, %d cached",
            stats["unique_texts"],
            stats["missing_texts"],
            stats["cache_size"],
        )
        release_sentence_model()

    if workers > 1:
        with mp.Pool(workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_one, tasks),
                    total=len(tasks),
                    desc="Preprocessing screens",
                    unit="screen",
                )
            )
    else:
        results = [
            _process_one(t)
            for t in tqdm(
                tasks,
                total=len(tasks),
                desc="Preprocessing screens",
                unit="screen",
            )
        ]

    successes = [r for r in results if r is not None]
    failures = len(results) - len(successes)
    logger.info("Processed %d screens, %d failures", len(successes), failures)

    # Categorize outputs
    train_pt_paths = [p for p in successes if os.path.isfile(p) and "/train/" in p]
    val_pt_paths = [p for p in successes if os.path.isfile(p) and "/val/" in p]

    return {
        "train_paths": sorted(train_pt_paths),
        "val_paths": sorted(val_pt_paths),
        "total_processed": len(successes),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Rico UI hierarchy JSONs.")
    parser.add_argument("--rico_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--label_mode", default="contextual", choices=["contextual", "local_only"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embedding_cache_path", default=None)
    parser.add_argument(
        "--skip_embedding_precompute",
        action="store_true",
        help="Disable bulk text embedding precompute before graph processing.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--max_screens", type=int, default=None)
    args = parser.parse_args()

    preprocess(
        rico_dir=args.rico_dir,
        out_dir=args.out_dir,
        split_path=args.split_path,
        label_mode=args.label_mode,
        workers=args.workers,
        embedding_cache_path=args.embedding_cache_path,
        precompute_embeddings=not args.skip_embedding_precompute,
        embedding_batch_size=args.embedding_batch_size,
        max_screens=args.max_screens,
    )


if __name__ == "__main__":
    main()
