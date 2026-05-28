import math
import pickle
from pathlib import Path

import torch

EMBED_CACHE_PATH = Path("data/embed_cache.pkl")

WIDGET_CLASSES = [
    "TextView", "ImageView", "Button", "EditText", "LinearLayout",
    "RelativeLayout", "FrameLayout", "ScrollView", "ListView", "RecyclerView",
    "ViewPager", "CheckBox", "RadioButton", "Switch", "ImageButton",
    "WebView", "ProgressBar", "SeekBar", "Other", "Unknown",
]
_CLASS_TO_IDX = {c.lower(): i for i, c in enumerate(WIDGET_CLASSES)}

FEATURE_DIMS = {
    "visual":     (0, 12),
    "structural": (12, 16),
    "type":       (16, 36),
    "text":       (36, 420),
}
TOTAL_DIM = 420

_embed_model = None
_embed_cache: dict = {}
_embed_cache_dirty = False


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def load_embed_cache() -> dict:
    global _embed_cache
    if EMBED_CACHE_PATH.exists():
        with open(EMBED_CACHE_PATH, "rb") as f:
            _embed_cache = pickle.load(f)
    return _embed_cache


def save_embed_cache():
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_CACHE_PATH, "wb") as f:
        pickle.dump(_embed_cache, f)


def _get_embedding(text: str) -> list:
    global _embed_cache, _embed_cache_dirty
    if text in _embed_cache:
        return _embed_cache[text]
    model = _get_embed_model()
    emb = model.encode(text, normalize_embeddings=False).tolist()
    _embed_cache[text] = emb
    _embed_cache_dirty = True
    return emb


def _visual_features(node: dict, parent: dict | None) -> list:
    bounds = node.get("bounds") or [0, 0, 0, 0]
    x1, y1, x2, y2 = bounds
    W, H = 1080.0, 1920.0

    x_norm = x1 / W
    y_norm = y1 / H
    w_norm = (x2 - x1) / W
    h_norm = (y2 - y1) / H
    area_norm = w_norm * h_norm
    aspect_ratio = w_norm / (h_norm + 1e-6)
    cx_norm = ((x1 + x2) / 2) / W
    cy_norm = ((y1 + y2) / 2) / H

    if parent is not None:
        pb = parent.get("bounds") or [0, 0, 0, 0]
        px1, py1, px2, py2 = pb
        dx1 = (x1 - px1) / W
        dy1 = (y1 - py1) / H
        dx2 = (x2 - px1) / W
        dy2 = (y2 - py1) / H
    else:
        dx1, dy1, dx2, dy2 = 0.0, 0.0, 0.0, 0.0

    return [x_norm, y_norm, w_norm, h_norm, area_norm, aspect_ratio,
            cx_norm, cy_norm, dx1, dy1, dx2, dy2]


def _structural_features(node: dict, max_depth: int) -> list:
    depth = node.get("depth", 0)
    sibling_count = node.get("sibling_count", 0)
    child_count = node.get("child_count", 0)
    is_leaf = 1.0 if child_count == 0 else 0.0
    return [
        depth / max(max_depth, 1),
        math.log(1 + sibling_count),
        math.log(1 + child_count),
        is_leaf,
    ]


def _type_onehot(node: dict) -> list:
    cls_full = node.get("class") or "Unknown"
    cls_short = cls_full.split(".")[-1].lower()
    idx = _CLASS_TO_IDX.get(cls_short)
    if idx is None:
        # Try prefix match
        for name, i in _CLASS_TO_IDX.items():
            if name in cls_short or cls_short in name:
                idx = i
                break
        if idx is None:
            idx = _CLASS_TO_IDX.get("other", len(WIDGET_CLASSES) - 2)
    vec = [0.0] * len(WIDGET_CLASSES)
    vec[idx] = 1.0
    return vec


def _text_embedding(node: dict) -> list:
    text = (node.get("text") or "").strip()
    desc = (node.get("content-desc") or "").strip()
    combined = " ".join(filter(None, [text, desc]))
    if not combined:
        return [0.0] * 384
    return _get_embedding(combined)


def get_feature_dim(feature_groups) -> int:
    if feature_groups is None:
        return TOTAL_DIM
    if feature_groups == "all":
        return TOTAL_DIM
    if feature_groups == "no_text":
        feature_groups = ["visual", "structural", "type"]
    total = 0
    for g in feature_groups:
        s, e = FEATURE_DIMS[g]
        total += e - s
    return total


def extract_features(nodes: list, feature_groups=None) -> torch.FloatTensor:
    """
    feature_groups: None/"all" = all groups, "no_text" or list subset.
    Returns FloatTensor [N, d].
    """
    if feature_groups == "all":
        feature_groups = None
    if feature_groups == "no_text":
        feature_groups = ["visual", "structural", "type"]

    use_all = feature_groups is None
    use_visual = use_all or "visual" in feature_groups
    use_structural = use_all or "structural" in feature_groups
    use_type = use_all or "type" in feature_groups
    use_text = use_all or "text" in feature_groups

    max_depth = max((n.get("depth", 0) for n in nodes), default=0)
    rows = []

    for node in nodes:
        ancestors = node.get("ancestors") or []
        parent = ancestors[-1] if ancestors else None
        parts = []
        if use_visual:
            parts.extend(_visual_features(node, parent))
        if use_structural:
            parts.extend(_structural_features(node, max_depth))
        if use_type:
            parts.extend(_type_onehot(node))
        if use_text:
            parts.extend(_text_embedding(node))
        rows.append(parts)

    return torch.tensor(rows, dtype=torch.float)
