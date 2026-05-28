"""
Heuristic labeling rules for Rico UI nodes.
Labels: 0=canonical, 1=translatable, 2=open, -1=excluded/unlabelable.

Two modes:
  contextual  - may use ancestor context (classes, resource-ids, content-desc)
  local_only  - uses only node-local fields; ancestor-only rules must not fire
"""

import logging
import re
from typing import Literal

from src.data.rico_loader import FlattenedHierarchy

logger = logging.getLogger(__name__)

# ---- Pattern constants -------------------------------------------------------

_CANONICAL_RESOURCE_PATTERNS = re.compile(
    r"(price|amount|total|account|uid|balance|transaction|payment|checkout|order"
    r"|billing|card|cvv|ssn|passport|license|tin|ein|routing|iban|swift|bic"
    r"|email|username|password|phone|mobile|captcha|recaptcha)",
    re.IGNORECASE,
)

_CANONICAL_CONTEXT_PATTERNS = re.compile(
    r"(checkout|cart|payment|order|price|billing|transaction|account|balance"
    r"|credential|auth|login|signin|wallet|bank|financial)",
    re.IGNORECASE,
)

_CANONICAL_CONTENT_DESC_PATTERNS = re.compile(
    r"(account|profile|order|transaction|payment|balance|credential|verification)",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

_TRANSLATABLE_RESOURCE_PATTERNS = re.compile(
    r"(header|title|section|label|heading|subtitle|caption|description)",
    re.IGNORECASE,
)

_OPEN_RESOURCE_PATTERNS = re.compile(
    r"(banner|promo|ad|recommendation|hero|carousel|featured|offer|deal|sale"
    r"|advertisement|sponsored)",
    re.IGNORECASE,
)

_NAV_CONTEXT_PATTERNS = re.compile(
    r"(navigation|nav|menu|tab|toolbar|drawer|bottombar|actionbar)",
    re.IGNORECASE,
)

_IMAGE_CLASSES = {
    "android.widget.ImageView",
    "android.widget.ImageButton",
}

_TEXT_CLASSES = {
    "android.widget.TextView",
    "android.widget.Button",
}


# ---- Helper functions --------------------------------------------------------

def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if v is not None).strip()
    return str(value).strip()


def _get_text(node: dict) -> str:
    return _as_text(node.get("text"))


def _get_content_desc(node: dict) -> str:
    return _as_text(node.get("content-desc"))


def _get_resource_id(node: dict) -> str:
    return _as_text(node.get("resource-id"))


def _get_class(node: dict) -> str:
    return _as_text(node.get("class"))


def _is_leaf(node: dict) -> bool:
    return node.get("child_count", 0) == 0


def _has_text_content(node: dict) -> bool:
    return bool(_get_text(node) or _get_content_desc(node))


# ---- Core labeling function --------------------------------------------------

def assign_label(
    node: dict,
    ancestors: list[dict],
    mode: Literal["contextual", "local_only"],
) -> int | None:
    """
    Return 0=canonical, 1=translatable, 2=open, or None for excluded nodes.
    Rules are evaluated in order; first match wins.
    """
    text = _get_text(node)
    content_desc = _get_content_desc(node)
    resource_id = _get_resource_id(node)
    cls = _get_class(node)

    # --- CANONICAL rules ---

    # 1. CAPTCHA
    if "captcha" in resource_id.lower() or "recaptcha" in resource_id.lower():
        return 0
    if "captcha" in content_desc.lower():
        return 0

    # 2. Canonical resource-id patterns (local)
    if resource_id and _CANONICAL_RESOURCE_PATTERNS.search(resource_id):
        return 0

    # 3. Email-like text (local)
    if text and _EMAIL_PATTERN.search(text):
        return 0

    # 4. Canonical content-desc patterns (local)
    if content_desc and _CANONICAL_CONTENT_DESC_PATTERNS.search(content_desc):
        return 0

    # 5. Currency/price text in canonical context (contextual only: ancestor check)
    if mode == "contextual" and text:
        # Check if any ancestor has a canonical context resource-id or class
        for anc in ancestors:
            anc_rid = _get_resource_id(anc)
            anc_cdesc = _get_content_desc(anc)
            if anc_rid and _CANONICAL_CONTEXT_PATTERNS.search(anc_rid):
                return 0
            if anc_cdesc and _CANONICAL_CONTEXT_PATTERNS.search(anc_cdesc):
                return 0
            anc_cls = _get_class(anc)
            if _CANONICAL_CONTEXT_PATTERNS.search(anc_cls):
                return 0

    # --- TRANSLATABLE rules ---

    # 6. Translatable resource-id (local)
    if resource_id and _TRANSLATABLE_RESOURCE_PATTERNS.search(resource_id):
        if text or content_desc:
            return 1

    # 7. Buttons/labels in nav/menu/tab context (contextual only)
    if mode == "contextual" and cls in _TEXT_CLASSES and _is_leaf(node):
        for anc in ancestors:
            anc_rid = _get_resource_id(anc)
            anc_cdesc = _get_content_desc(anc)
            if (anc_rid and _NAV_CONTEXT_PATTERNS.search(anc_rid)) or \
               (anc_cdesc and _NAV_CONTEXT_PATTERNS.search(anc_cdesc)):
                return 1

    # 8. TextView leaf nodes at moderate depth (local)
    if cls == "android.widget.TextView" and _is_leaf(node) and text:
        depth = node.get("depth", 0)
        if depth >= 2:
            return 1

    # 9. Button with text (local)
    if cls == "android.widget.Button" and text:
        return 1

    # --- OPEN rules ---

    # 10. Open resource-id patterns (banner, promo, ad, etc.)
    if resource_id and _OPEN_RESOURCE_PATTERNS.search(resource_id):
        return 2

    # 11. Non-critical images (ImageView not already canonical/translatable)
    if cls in _IMAGE_CLASSES:
        return 2

    # 12. Nodes with no text and no content-desc that are leaves -> exclude
    if _is_leaf(node) and not _has_text_content(node):
        return None

    # 13. Layout containers and non-leaf nodes with no text -> exclude
    if not _is_leaf(node) and not _has_text_content(node):
        return None

    # 14. Remaining labeled leaf nodes with text -> open (fallback)
    if _is_leaf(node) and _has_text_content(node):
        return 2

    return None


def label_graph(
    flattened: FlattenedHierarchy,
    mode: Literal["contextual", "local_only"],
) -> list[int]:
    """Return one label per node (-1 for unlabeled/excluded nodes)."""
    nodes = flattened.nodes
    ancestor_indices = flattened.ancestor_indices
    labels: list[int] = []
    label_counts = {0: 0, 1: 0, 2: 0, -1: 0}

    for i, node in enumerate(nodes):
        anc_idxs = ancestor_indices[i]
        ancestors = [nodes[a] for a in anc_idxs]
        result = assign_label(node, ancestors, mode)
        lbl = result if result is not None else -1
        labels.append(lbl)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    logger.debug(
        "label_graph mode=%s  canonical=%d  translatable=%d  open=%d  excluded=%d",
        mode,
        label_counts.get(0, 0),
        label_counts.get(1, 0),
        label_counts.get(2, 0),
        label_counts.get(-1, 0),
    )
    return labels
