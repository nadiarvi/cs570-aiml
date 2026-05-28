import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

CACHE_PATH = Path("data/llm_label_cache.json")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 32  # nodes per GPU batch

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    print(f"Loading {MODEL_NAME} (first call only) ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to("cuda")
    _model.eval()
    print("Model ready.")
    return _model, _tokenizer


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


def _parse_label(text: str) -> int:
    """Extract 0/1/2 from model output. Returns -1 if parsing fails."""
    text = text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            label = int(json.loads(text[start:end])["label"])
            if label in (0, 1, 2):
                return label
    except Exception:
        pass
    # Fallback: first digit found
    for ch in text:
        if ch in ("0", "1", "2"):
            return int(ch)
    return -1


def _run_batch(prompts: list, model, tokenizer) -> list:
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return [
        _parse_label(tokenizer.decode(out[input_len:], skip_special_tokens=True))
        for out in outputs
    ]


def label_nodes_llm(nodes: list, cache: dict) -> list:
    model, tokenizer = _load_model()

    # Split into cached and uncached
    all_prompts, all_keys = [], []
    cached_labels = {}
    uncached_positions = []

    for i, node in enumerate(nodes):
        prompt = build_node_prompt(node, node.get("ancestors", []))
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in cache:
            cached_labels[i] = cache[key]
        else:
            uncached_positions.append(i)
            all_prompts.append(prompt)
            all_keys.append(key)

    # Batched GPU inference for uncached nodes
    new_labels = []
    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch = all_prompts[i : i + BATCH_SIZE]
        new_labels.extend(_run_batch(batch, model, tokenizer))

    # Update cache
    for key, label in zip(all_keys, new_labels):
        cache[key] = label
    if new_labels:
        save_cache(cache)

    # Reassemble in original order
    new_iter = iter(new_labels)
    results = []
    for i in range(len(nodes)):
        if i in cached_labels:
            results.append(cached_labels[i])
        else:
            results.append(next(new_iter, -1))
    return results
