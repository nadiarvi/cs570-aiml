"""
Feature extraction for Rico UI hierarchy nodes.
Feature groups: visual (12), structural (4), type (20), text (384) = 420 total.
"""

import os
import hashlib
import json
import logging

import numpy as np
import torch

from src.data.rico_loader import FlattenedHierarchy

logger = logging.getLogger(__name__)

# Fixed widget class vocabulary (20 classes)
WIDGET_CLASSES = [
    "android.widget.TextView",
    "android.widget.ImageView",
    "android.widget.Button",
    "android.widget.EditText",
    "android.widget.LinearLayout",
    "android.widget.RelativeLayout",
    "android.widget.FrameLayout",
    "android.widget.ScrollView",
    "android.widget.ListView",
    "android.widget.RecyclerView",
    "android.widget.CheckBox",
    "android.widget.RadioButton",
    "android.widget.Switch",
    "android.widget.ImageButton",
    "android.view.View",
    "android.widget.ProgressBar",
    "android.widget.SeekBar",
    "android.widget.Spinner",
    "android.widget.WebView",
    "__other__",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(WIDGET_CLASSES)}
TYPE_DIM = len(WIDGET_CLASSES)  # 20
VISUAL_DIM = 12
STRUCTURAL_DIM = 4
TEXT_DIM = 384  # MiniLM-L6-v2

DEFAULT_SCREEN_W = 1440
DEFAULT_SCREEN_H = 2560

ALL_GROUPS = ["visual", "structural", "type", "text"]


def _parse_bounds(node: dict) -> tuple[float, float, float, float] | None:
    """Return (x1, y1, x2, y2) from node bounds, or None if missing/invalid."""
    bounds = node.get("bounds")
    if not bounds or len(bounds) < 4:
        return None
    x1, y1, x2, y2 = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    return x1, y1, x2, y2


def _visual_features(
    node: dict,
    parent_node: dict | None,
    screen_w: int,
    screen_h: int,
) -> np.ndarray:
    """
    12-dim visual feature vector:
      [x1, y1, x2, y2, w, h, area, aspect_ratio, cx, cy, rel_x, rel_y]
    All normalized by screen dimensions. Parent-relative features are zeros if
    parent bounds are missing.
    """
    feats = np.zeros(VISUAL_DIM, dtype=np.float32)
    b = _parse_bounds(node)
    if b is None:
        return feats

    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / (h + 1e-6)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sw, sh = float(screen_w), float(screen_h)
    feats[0] = x1 / sw
    feats[1] = y1 / sh
    feats[2] = x2 / sw
    feats[3] = y2 / sh
    feats[4] = w / sw
    feats[5] = h / sh
    feats[6] = area / (sw * sh)
    feats[7] = min(aspect, 10.0) / 10.0  # clamp extreme ratios
    feats[8] = cx / sw
    feats[9] = cy / sh

    if parent_node is not None:
        pb = _parse_bounds(parent_node)
        if pb is not None:
            px1, py1, px2, py2 = pb
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            feats[10] = (cx - px1) / pw
            feats[11] = (cy - py1) / ph

    return feats


def _structural_features(node: dict) -> np.ndarray:
    """4-dim: [depth, sibling_count, child_count, is_leaf]."""
    feats = np.zeros(STRUCTURAL_DIM, dtype=np.float32)
    feats[0] = min(node.get("depth", 0), 20) / 20.0
    feats[1] = min(node.get("sibling_count", 0), 20) / 20.0
    feats[2] = min(node.get("child_count", 0), 20) / 20.0
    feats[3] = 1.0 if node.get("child_count", 0) == 0 else 0.0
    return feats


def _type_features(node: dict) -> np.ndarray:
    """20-dim one-hot over widget class vocabulary."""
    feats = np.zeros(TYPE_DIM, dtype=np.float32)
    cls = node.get("class", "")
    idx = CLASS_TO_IDX.get(cls, CLASS_TO_IDX["__other__"])
    feats[idx] = 1.0
    return feats


def _build_text_string(node: dict) -> str:
    """Concatenate non-empty text and content-desc."""
    parts = []
    for key in ("text", "content-desc"):
        raw_val = node.get(key, "")
        if isinstance(raw_val, list):
            val = " ".join(str(v).strip() for v in raw_val if v is not None)
        elif raw_val is None:
            val = ""
        else:
            val = str(raw_val)
        val = val.strip()
        if val:
            parts.append(val)
    return " ".join(parts)


def _text_cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_embedding_cache(cache_path: str) -> dict:
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid embedding cache JSON: %s", cache_path)
    return {}


def _save_embedding_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp_path, cache_path)


_SENTENCE_MODEL = None


def _sentence_model_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        device = _sentence_model_device()
        logger.info("Loading SentenceTransformer on %s", device)
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _SENTENCE_MODEL


def _text_features_batch(
    texts: list[str],
    cache: dict,
) -> tuple[np.ndarray, dict]:
    """
    Embed a list of text strings. Returns [N, 384] float32 array and updated cache.
    Empty strings produce zero vectors.
    """
    embeddings = np.zeros((len(texts), TEXT_DIM), dtype=np.float32)
    to_encode: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        if not text:
            continue
        key = _text_cache_key(text)
        if key in cache:
            embeddings[i] = np.array(cache[key], dtype=np.float32)
        else:
            to_encode.append((i, text))

    if to_encode:
        model = _get_sentence_model()
        idxs, raw_texts = zip(*to_encode)
        vecs = model.encode(list(raw_texts), batch_size=64, show_progress_bar=False)
        for i, vec in zip(idxs, vecs):
            text = texts[i]
            key = _text_cache_key(text)
            cache[key] = vec.tolist()
            embeddings[i] = vec.astype(np.float32)

    return embeddings, cache


def extract_features(
    flattened: FlattenedHierarchy,
    feature_groups: list[str] | None = None,
    screen_width: int | None = None,
    screen_height: int | None = None,
    embedding_cache_path: str | None = None,
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """
    Extract per-node features for a flattened Rico hierarchy.

    Returns:
      features: [num_nodes, dim] float32 tensor
      feature_slices: {"visual": (0, 12), "structural": (12, 16), "type": (16, 36), "text": (36, 420)}
    """
    if feature_groups is None:
        feature_groups = ALL_GROUPS

    nodes = flattened.nodes
    parent_index = flattened.parent_index
    N = len(nodes)

    # Determine screen dimensions from root node bounds if not provided
    if screen_width is None or screen_height is None:
        root_bounds = _parse_bounds(nodes[0]) if nodes else None
        if root_bounds is not None:
            screen_width = screen_width or int(root_bounds[2])
            screen_height = screen_height or int(root_bounds[3])
    screen_width = screen_width or DEFAULT_SCREEN_W
    screen_height = screen_height or DEFAULT_SCREEN_H

    parts: list[np.ndarray] = []
    feature_slices: dict[str, tuple[int, int]] = {}
    offset = 0

    if "visual" in feature_groups:
        vis = np.zeros((N, VISUAL_DIM), dtype=np.float32)
        for i, node in enumerate(nodes):
            par_idx = parent_index[i]
            parent_node = nodes[par_idx] if par_idx is not None else None
            vis[i] = _visual_features(node, parent_node, screen_width, screen_height)
        parts.append(vis)
        feature_slices["visual"] = (offset, offset + VISUAL_DIM)
        offset += VISUAL_DIM

    if "structural" in feature_groups:
        struct = np.zeros((N, STRUCTURAL_DIM), dtype=np.float32)
        for i, node in enumerate(nodes):
            struct[i] = _structural_features(node)
        parts.append(struct)
        feature_slices["structural"] = (offset, offset + STRUCTURAL_DIM)
        offset += STRUCTURAL_DIM

    if "type" in feature_groups:
        typ = np.zeros((N, TYPE_DIM), dtype=np.float32)
        for i, node in enumerate(nodes):
            typ[i] = _type_features(node)
        parts.append(typ)
        feature_slices["type"] = (offset, offset + TYPE_DIM)
        offset += TYPE_DIM

    if "text" in feature_groups:
        cache = _load_embedding_cache(embedding_cache_path) if embedding_cache_path else {}
        texts = [_build_text_string(node) for node in nodes]
        text_embs, cache = _text_features_batch(texts, cache)
        if embedding_cache_path:
            _save_embedding_cache(cache, embedding_cache_path)
        parts.append(text_embs)
        feature_slices["text"] = (offset, offset + TEXT_DIM)
        offset += TEXT_DIM

    if not parts:
        features = torch.zeros(N, 0, dtype=torch.float32)
    else:
        features = torch.from_numpy(np.concatenate(parts, axis=1))

    return features, feature_slices
