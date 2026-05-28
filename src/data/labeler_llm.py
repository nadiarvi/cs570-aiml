import os
import json
import hashlib
import time
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
_api_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
_current_key_idx = 0

CACHE_PATH = Path("data/llm_label_cache.json")
RATE_LIMIT_DELAY = 4.1  # seconds — stays under 15 RPM


def get_model():
    import google.generativeai as genai
    genai.configure(api_key=_api_keys[_current_key_idx])
    return genai.GenerativeModel("gemini-1.5-flash")


def rotate_key() -> bool:
    global _current_key_idx
    if _current_key_idx + 1 < len(_api_keys):
        _current_key_idx += 1
        print(f"Rotated to key {_current_key_idx + 1}/{len(_api_keys)}")
        return True
    return False


def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


SYSTEM_PROMPT = """You are a UI modifiability classifier for Android screens.
Classify a UI element as one of:
  0 = canonical   — Must NOT be modified. Identity-critical: prices, account IDs,
                    auth fields, legal text, CAPTCHAs, order numbers.
  1 = translatable — Form may change, meaning preserved. Nav labels, headers,
                     button text, section copy.
  2 = open         — Freely modifiable. Decorative images, banners, carousels,
                     promotional content.

Key rule: ancestor chain is your strongest signal. "$24.99" inside CheckoutFlow
is canonical; "$24.99" inside MarketingBanner is open. When uncertain, prefer canonical.
Respond ONLY with valid JSON: {"label": <0|1|2>, "reason": "<one sentence>"}"""


def build_node_prompt(node: dict, ancestors: list) -> str:
    chain = " → ".join(
        f"{a.get('class', '?').split('.')[-1]}[{a.get('resource-id', '')}]"
        for a in ancestors
    ) or "none"
    return (
        f"Class: {node.get('class', '?').split('.')[-1]}\n"
        f"Text: \"{node.get('text', '')}\"\n"
        f"Content-desc: \"{node.get('content-desc', '')}\"\n"
        f"Resource-ID: \"{node.get('resource-id', '')}\"\n"
        f"Depth: {node.get('depth', 0)} | Children: {node.get('child_count', 0)} "
        f"| Siblings: {node.get('sibling_count', 0)}\n"
        f"Ancestors: {chain}"
    )


def get_llm_label(node: dict, ancestors: list, cache: dict, retries: int = 3) -> int:
    prompt = build_node_prompt(node, ancestors)
    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            response = get_model().generate_content(
                [{"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + prompt]}],
                generation_config={"temperature": 0.0, "max_output_tokens": 64},
            )
            label = int(json.loads(response.text.strip())["label"])
            assert label in (0, 1, 2)
            cache[key] = label
            save_cache(cache)
            return label
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "resource_exhausted" in err or "429" in err:
                if rotate_key():
                    continue
                save_cache(cache)
                raise RuntimeError("All API keys exhausted. Resume tomorrow.")
            time.sleep(2 ** attempt * RATE_LIMIT_DELAY)
    return -1


def label_nodes_llm(nodes: list, cache: dict) -> list:
    return [
        get_llm_label(n, n.get("ancestors", []), cache)
        for n in tqdm(nodes, desc="LLM labeling", leave=False)
    ]
