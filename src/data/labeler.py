import re


_CURRENCY_RE = re.compile(r'[\$€£¥₹]\s*\d|\d+\.\d{2}')
_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
_CANONICAL_ID_TERMS = {"price", "amount", "total", "account", "uid", "balance", "order", "transaction"}
_AUTH_ANCESTOR_TERMS = {"login", "auth", "password", "signin", "sign_in", "credential"}
_CHECKOUT_ANCESTOR_TERMS = {"checkout", "payment", "cart", "order", "purchase", "billing"}
_TRANSLATABLE_ID_TERMS = {"title", "header", "label", "section", "heading", "subtitle"}
_NAV_ANCESTOR_TERMS = {"nav", "menu", "tab", "toolbar", "navigation", "bottom_bar", "actionbar"}
_OPEN_CLASS_TERMS = {"banner", "carousel", "ad", "promo", "promotion", "advertisement"}
_OPEN_ID_TERMS = {"banner", "ad", "promo", "carousel", "hero", "splash", "marketing"}


def _short_class(cls: str) -> str:
    return cls.split(".")[-1].lower() if cls else ""


def _resource_id_lower(node: dict) -> str:
    return (node.get("resource-id") or "").lower()


def _text_lower(node: dict) -> str:
    return (node.get("text") or "").lower()


def _content_desc_lower(node: dict) -> str:
    return (node.get("content-desc") or "").lower()


def _any_ancestor_matches(ancestors: list, terms: set) -> bool:
    for a in ancestors:
        cls = _short_class(a.get("class", ""))
        rid = (a.get("resource-id") or "").lower()
        if any(t in cls for t in terms) or any(t in rid for t in terms):
            return True
    return False


def assign_label(node: dict, ancestors: list) -> int:
    """
    First match wins.
    Returns 0 (canonical), 1 (translatable), 2 (open), or None if no usable info.
    """
    cls = _short_class(node.get("class", ""))
    rid = _resource_id_lower(node)
    text = _text_lower(node)
    desc = _content_desc_lower(node)
    bounds = node.get("bounds") or [0, 0, 0, 0]

    has_bounds = any(b != 0 for b in bounds)
    has_text_content = bool(text or node.get("text"))
    has_rid = bool(rid)

    if not has_bounds and not has_text_content and not has_rid:
        return None

    # --- CANONICAL ---

    # CAPTCHA
    if "captcha" in cls or "captcha" in rid:
        return 0

    # Currency/price text
    raw_text = node.get("text") or ""
    if _CURRENCY_RE.search(raw_text):
        # Promotional context overrides price → open; but default canonical
        if not _any_ancestor_matches(ancestors, {"banner", "promo", "marketing", "ad"}):
            return 0

    # EditText in auth context
    if "edittext" in cls and _any_ancestor_matches(ancestors, _AUTH_ANCESTOR_TERMS):
        return 0

    # Email address in text
    if _EMAIL_RE.search(raw_text):
        return 0

    # Resource-ID contains canonical financial/identity terms
    if any(t in rid for t in _CANONICAL_ID_TERMS):
        return 0

    # Content-desc references account/profile/order/transaction
    if any(t in desc for t in {"account", "profile", "order", "transaction"}):
        return 0

    # Price text inside checkout ancestor
    if _CURRENCY_RE.search(raw_text) and _any_ancestor_matches(ancestors, _CHECKOUT_ANCESTOR_TERMS):
        return 0

    # --- TRANSLATABLE ---

    depth = node.get("depth", 0)

    # TextView at shallow depth
    if cls == "textview" and depth <= 3:
        return 1

    # Button in nav context
    if "button" in cls and _any_ancestor_matches(ancestors, _NAV_ANCESTOR_TERMS):
        return 1

    # Resource-ID contains header/title/label terms
    if any(t in rid for t in _TRANSLATABLE_ID_TERMS):
        return 1

    # TextView with siblings at mid depth
    if cls == "textview" and 2 <= depth <= 5 and node.get("sibling_count", 0) > 0:
        return 1

    # --- OPEN ---

    # ImageView / ImageButton
    if cls in ("imageview", "imagebutton"):
        return 2

    # Banner/Carousel/Ad class
    if any(t in cls for t in _OPEN_CLASS_TERMS):
        return 2

    # Promo resource-id
    if any(t in rid for t in _OPEN_ID_TERMS):
        return 2

    # All remaining nodes
    return 2
