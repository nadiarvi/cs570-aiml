"""
Build LLM-generated gold labels for Rico UI nodes.

The output CSV is compatible with src.data.gold.load_gold_test_labels:
screen_id,node_id,label,annotator_id plus audit columns.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm

from src.data.labeler import label_graph
from src.data.rico_loader import flatten_hierarchy, get_app_id, load_hierarchy
from src.data.splits import load_splits

logger = logging.getLogger(__name__)

LABEL_NAMES = ["canonical", "translatable", "open"]
DEFAULT_MODEL = "gpt-5-nano"

SYSTEM_PROMPT = """You are labeling Android UI hierarchy nodes for a modifiability classifier.

Return JSON only. Label each sample with exactly one of:

canonical:
- Text, controls, or images that should usually stay fixed across UI variants because they represent user/account identity, credentials, payment, financial, legal, security, order, transaction, verification, or other critical business data.
- Examples: price, balance, account number, email, password, payment button, checkout total, order ID, verification code, profile/account identity, security warning.

translatable:
- Stable UI copy that should normally be translated/localized but not freely rewritten.
- Examples: navigation labels, headings, section titles, button text, form labels, captions, ordinary instructional text, content descriptions for standard controls.

open:
- Non-critical content or presentation elements that can be freely changed, personalized, reordered, promoted, or visually altered without breaking the core UI meaning.
- Examples: banners, promotional copy, recommendations, ads, hero images, decorative images, generic placeholders, feed/card content, non-critical thumbnails.

Decision rules:
- Use node text, content description, resource id, class, parent context, and siblings.
- If multiple labels seem possible, choose the stricter label in this order: canonical, then translatable, then open.
- Do not invent facts beyond the provided node/context fields.
- Confidence is a number from 0.0 to 1.0.
- Rationale must be short, under 20 words.
"""


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ if they are not already set."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass(frozen=True)
class NodeSample:
    sample_id: str
    screen_id: str
    app_id: str
    node_id: str
    heuristic_label: str
    fields: dict


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if v is not None).strip()
    return str(value).strip()


def _short(value, limit: int = 180) -> str:
    text = _as_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _partition_for_app(app_id: str, split: dict | None) -> str:
    if not split:
        return "all"
    if app_id in set(split.get("train_app_ids", [])):
        return "train"
    if app_id in set(split.get("val_app_ids", [])):
        return "val"
    return "other"


def _node_context(flattened, idx: int) -> dict:
    node = flattened.nodes[idx]
    ancestors = [flattened.nodes[a] for a in flattened.ancestor_indices[idx][-3:]]
    siblings = []
    parent_idx = flattened.parent_index[idx]
    if parent_idx is not None:
        for sib_idx, par in enumerate(flattened.parent_index):
            if par == parent_idx and sib_idx != idx:
                sib = flattened.nodes[sib_idx]
                sib_text = _short(sib.get("text") or sib.get("content-desc"), 80)
                if sib_text:
                    siblings.append(sib_text)
                if len(siblings) >= 6:
                    break

    return {
        "class": _short(node.get("class"), 120),
        "text": _short(node.get("text")),
        "content_desc": _short(node.get("content-desc")),
        "resource_id": _short(node.get("resource-id"), 160),
        "bounds": node.get("bounds") or node.get("visible-to-user-bounds") or "",
        "clickable": node.get("clickable", ""),
        "enabled": node.get("enabled", ""),
        "selected": node.get("selected", ""),
        "depth": node.get("depth", 0),
        "child_count": node.get("child_count", 0),
        "sibling_count": node.get("sibling_count", 0),
        "ancestors": [
            {
                "class": _short(a.get("class"), 80),
                "resource_id": _short(a.get("resource-id"), 100),
                "content_desc": _short(a.get("content-desc"), 100),
                "text": _short(a.get("text"), 100),
            }
            for a in ancestors
        ],
        "sibling_texts": siblings,
    }


def _reservoir_add(bucket: list[NodeSample], sample: NodeSample, seen: int, target: int, rng: random.Random) -> None:
    if len(bucket) < target:
        bucket.append(sample)
        return
    j = rng.randrange(seen)
    if j < target:
        bucket[j] = sample


def collect_balanced_samples(
    rico_dir: str,
    sample_size: int,
    split_path: str | None = None,
    partition: str = "val",
    label_mode: str = "contextual",
    seed: int = 42,
    max_screens: int | None = None,
) -> list[NodeSample]:
    """Collect an approximately class-balanced node sample from raw Rico JSON."""
    json_paths = sorted(glob.glob(os.path.join(rico_dir, "**", "*.json"), recursive=True))
    if max_screens:
        json_paths = json_paths[:max_screens]
    if not json_paths:
        raise FileNotFoundError(f"No Rico JSON files found under {rico_dir}")

    split = load_splits(split_path) if split_path and os.path.exists(split_path) else None
    if split is None and partition != "all":
        logger.warning("No split found; sampling from all apps instead of partition=%s", partition)
        partition = "all"

    rng = random.Random(seed)
    target_per_class = (sample_size + len(LABEL_NAMES) - 1) // len(LABEL_NAMES)
    buckets: dict[int, list[NodeSample]] = {0: [], 1: [], 2: []}
    seen_by_class: dict[int, int] = defaultdict(int)

    for json_path in tqdm(json_paths, desc="Scanning Rico nodes", unit="screen"):
        app_id = get_app_id(json_path)
        if partition != "all" and _partition_for_app(app_id, split) != partition:
            continue

        try:
            screen_id = os.path.splitext(os.path.basename(json_path))[0]
            flattened = flatten_hierarchy(load_hierarchy(json_path))
            labels = label_graph(flattened, mode=label_mode)
        except Exception as exc:
            logger.warning("Skipping %s: %s", json_path, exc)
            continue

        for idx, heuristic in enumerate(labels):
            if heuristic not in (0, 1, 2):
                continue
            node = flattened.nodes[idx]
            has_signal = any(_as_text(node.get(k)) for k in ("text", "content-desc", "resource-id", "class"))
            if not has_signal:
                continue

            seen_by_class[heuristic] += 1
            node_id = str(node.get("node_id", idx))
            sample = NodeSample(
                sample_id=f"{screen_id}:{node_id}",
                screen_id=screen_id,
                app_id=app_id,
                node_id=node_id,
                heuristic_label=LABEL_NAMES[heuristic],
                fields=_node_context(flattened, idx),
            )
            _reservoir_add(
                buckets[heuristic],
                sample,
                seen_by_class[heuristic],
                target_per_class,
                rng,
            )

    samples = [s for cls in (0, 1, 2) for s in buckets[cls]]
    rng.shuffle(samples)
    if len(samples) > sample_size:
        samples = samples[:sample_size]
    if len(samples) < sample_size:
        logger.warning("Collected only %d samples out of requested %d", len(samples), sample_size)
    return samples


def write_manifest(samples: Iterable[NodeSample], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample_to_record(sample), ensure_ascii=False) + "\n")


def load_manifest(path: str) -> list[NodeSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                NodeSample(
                    sample_id=row["sample_id"],
                    screen_id=str(row["screen_id"]),
                    app_id=str(row["app_id"]),
                    node_id=str(row["node_id"]),
                    heuristic_label=str(row.get("heuristic_label", "")),
                    fields=dict(row["fields"]),
                )
            )
    return samples


def sample_to_record(sample: NodeSample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "screen_id": sample.screen_id,
        "app_id": sample.app_id,
        "node_id": sample.node_id,
        "heuristic_label": sample.heuristic_label,
        "fields": sample.fields,
    }


def _prompt_for_batch(samples: list[NodeSample]) -> str:
    records = [
        {
            "sample_id": s.sample_id,
            "screen_id": s.screen_id,
            "app_id": s.app_id,
            "node_id": s.node_id,
            "node": s.fields,
        }
        for s in samples
    ]
    return (
        "Label these Android UI hierarchy node samples. Return one label object per input "
        "sample_id in the same order. JSON is required.\n\n"
        + json.dumps({"samples": records}, ensure_ascii=False, indent=2)
    )


def _response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sample_id", "label", "confidence", "rationale"],
                    "properties": {
                        "sample_id": {"type": "string"},
                        "label": {"type": "string", "enum": LABEL_NAMES},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def _response_body_for_batch(samples: list[NodeSample], model: str) -> dict:
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": _prompt_for_batch(samples),
        "max_output_tokens": max(800, 120 * len(samples)),
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "ui_gold_labels",
                "strict": True,
                "schema": _response_schema(),
            },
        },
    }
    if model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": "minimal"}
    return payload


def _extract_output_text(response: dict) -> str:
    if response.get("output_text"):
        return response["output_text"]
    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
            elif content.get("type") == "refusal":
                raise RuntimeError(f"Model refused: {content.get('refusal', '')}")
    if not chunks:
        raise RuntimeError(f"No text output in OpenAI response: {response}")
    return "".join(chunks)


def call_openai_labels(
    samples: list[NodeSample],
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    max_retries: int = 3,
) -> list[dict]:
    payload = _response_body_for_batch(samples, model)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(_extract_output_text(response))
            labels = parsed.get("labels", [])
            return _validate_batch_labels(samples, labels)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {detail}")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI labeling failed after {max_retries} attempts: {last_error}")


def _openai_json_request(
    api_key: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 120,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"https://api.openai.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload_batch_file(api_key: str, path: str, timeout: int = 120) -> dict:
    boundary = f"----codex{int(time.time() * 1000)}"
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        file_bytes = f.read()
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            "batch\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    req = urllib.request.Request(
        "https://api.openai.com/v1/files",
        data=b"".join(parts),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(api_key: str, file_id: str, out_path: str, timeout: int = 120) -> None:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/files/{file_id}/content",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)


def _chunked(samples: list[NodeSample], size: int) -> Iterable[list[NodeSample]]:
    for start in range(0, len(samples), size):
        yield samples[start : start + size]


def write_openai_batch_input(
    samples: list[NodeSample],
    input_path: str,
    metadata_path: str,
    batch_size: int,
    model: str,
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(input_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    metadata = {"model": model, "requests": {}}

    request_count = 0
    sample_count = 0
    with open(input_path, "w", encoding="utf-8") as f:
        for request_count, batch_samples in enumerate(_chunked(samples, batch_size), start=1):
            custom_id = f"llm-gold-{request_count:06d}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": _response_body_for_batch(batch_samples, model),
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            metadata["requests"][custom_id] = [sample_to_record(s) for s in batch_samples]
            sample_count += len(batch_samples)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "input_path": input_path,
        "metadata_path": metadata_path,
        "requests": request_count,
        "samples": sample_count,
        "model": model,
    }


def create_openai_batch(args: argparse.Namespace) -> dict:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api_key before creating a batch.")

    if os.path.exists(args.manifest_path) and not args.resample:
        samples = load_manifest(args.manifest_path)
    else:
        samples = collect_balanced_samples(
            rico_dir=args.rico_dir,
            sample_size=args.sample_size,
            split_path=args.split_path,
            partition=args.partition,
            label_mode=args.label_mode,
            seed=args.seed,
            max_screens=args.max_screens,
        )
        write_manifest(samples, args.manifest_path)

    existing = _existing_sample_ids(args.out_csv)
    remaining = [s for s in samples if s.sample_id not in existing]
    batch_info = write_openai_batch_input(
        remaining,
        input_path=args.batch_input_path,
        metadata_path=args.batch_metadata_path,
        batch_size=args.batch_size,
        model=args.model,
    )
    upload = _upload_batch_file(api_key, args.batch_input_path, timeout=args.timeout)
    batch = _openai_json_request(
        api_key,
        "POST",
        "/v1/batches",
        {
            "input_file_id": upload["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        },
        timeout=args.timeout,
    )
    result = {
        **batch_info,
        "input_file_id": upload["id"],
        "batch_id": batch["id"],
        "status": batch["status"],
        "batch": batch,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.batch_info_path)), exist_ok=True)
    with open(args.batch_info_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def retrieve_openai_batch(args: argparse.Namespace) -> dict:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api_key before checking a batch.")
    if not args.batch_id:
        raise RuntimeError("Pass --batch_id batch_... for --batch_status or --batch_collect.")
    return _openai_json_request(api_key, "GET", f"/v1/batches/{args.batch_id}", timeout=args.timeout)


def collect_openai_batch(args: argparse.Namespace) -> dict:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api_key before collecting a batch.")

    batch = retrieve_openai_batch(args)
    if batch.get("status") != "completed":
        raise RuntimeError(f"Batch is {batch.get('status')}; wait until status is completed.")
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"Completed batch has no output_file_id: {batch}")

    _download_file(api_key, output_file_id, args.batch_output_path, timeout=args.timeout)
    if batch.get("error_file_id"):
        error_path = os.path.splitext(args.batch_output_path)[0] + "_errors.jsonl"
        _download_file(api_key, batch["error_file_id"], error_path, timeout=args.timeout)
        logger.warning("Downloaded batch error file to %s", error_path)

    with open(args.batch_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows_to_write: list[dict] = []
    failures = []
    with open(args.batch_output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item["custom_id"]
            if item.get("error"):
                failures.append(item)
                continue
            response = item.get("response", {})
            if response.get("status_code") != 200:
                failures.append(item)
                continue

            sample_records = metadata["requests"].get(custom_id)
            if sample_records is None:
                raise ValueError(f"No local metadata for batch custom_id={custom_id}")
            samples = [
                NodeSample(
                    sample_id=row["sample_id"],
                    screen_id=str(row["screen_id"]),
                    app_id=str(row["app_id"]),
                    node_id=str(row["node_id"]),
                    heuristic_label=str(row.get("heuristic_label", "")),
                    fields=dict(row["fields"]),
                )
                for row in sample_records
            ]
            parsed = json.loads(_extract_output_text(response["body"]))
            rows = _validate_batch_labels(samples, parsed.get("labels", []))
            for row in rows:
                row["model"] = metadata.get("model", args.model)
            rows_to_write.extend(rows)

    existing = _existing_sample_ids(args.out_csv)
    rows_to_write = [row for row in rows_to_write if row["sample_id"] not in existing]
    _append_rows(args.out_csv, rows_to_write)
    _append_jsonl(args.raw_jsonl, rows_to_write)

    return {
        "batch_id": args.batch_id,
        "status": batch.get("status"),
        "downloaded_output": args.batch_output_path,
        "new_labels": len(rows_to_write),
        "failed_requests": len(failures),
        "out_csv": args.out_csv,
    }


def _validate_batch_labels(samples: list[NodeSample], labels: list[dict]) -> list[dict]:
    expected_ids = [s.sample_id for s in samples]
    by_id = {str(row.get("sample_id")): row for row in labels}
    missing = [sid for sid in expected_ids if sid not in by_id]
    if missing:
        raise ValueError(f"OpenAI response missing sample IDs: {missing[:5]}")

    validated = []
    for sample in samples:
        row = by_id[sample.sample_id]
        label = row.get("label")
        if label not in LABEL_NAMES:
            raise ValueError(f"Invalid label for {sample.sample_id}: {label!r}")
        confidence = float(row.get("confidence", 0.0))
        confidence = min(1.0, max(0.0, confidence))
        validated.append(
            {
                "sample_id": sample.sample_id,
                "screen_id": sample.screen_id,
                "app_id": sample.app_id,
                "node_id": sample.node_id,
                "label": label,
                "annotator_id": "openai_llm",
                "model": "",
                "confidence": confidence,
                "rationale": _short(row.get("rationale"), 240),
                "heuristic_label": sample.heuristic_label,
                "class": sample.fields.get("class", ""),
                "text": sample.fields.get("text", ""),
                "content_desc": sample.fields.get("content_desc", ""),
                "resource_id": sample.fields.get("resource_id", ""),
            }
        )
    return validated


CSV_FIELDS = [
    "screen_id",
    "node_id",
    "label",
    "annotator_id",
    "app_id",
    "sample_id",
    "model",
    "confidence",
    "rationale",
    "heuristic_label",
    "class",
    "text",
    "content_desc",
    "resource_id",
]


def _existing_sample_ids(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return {row.get("sample_id", "") for row in csv.DictReader(f) if row.get("sample_id")}


def _append_rows(csv_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_labeling(args: argparse.Namespace) -> dict:
    if os.path.exists(args.manifest_path) and not args.resample:
        samples = load_manifest(args.manifest_path)
        logger.info("Loaded %d samples from %s", len(samples), args.manifest_path)
    else:
        samples = collect_balanced_samples(
            rico_dir=args.rico_dir,
            sample_size=args.sample_size,
            split_path=args.split_path,
            partition=args.partition,
            label_mode=args.label_mode,
            seed=args.seed,
            max_screens=args.max_screens,
        )
        write_manifest(samples, args.manifest_path)
        logger.info("Wrote sample manifest to %s", args.manifest_path)

    if args.dry_run:
        preview = _prompt_for_batch(samples[: args.batch_size])
        os.makedirs(os.path.dirname(os.path.abspath(args.prompt_preview_path)), exist_ok=True)
        with open(args.prompt_preview_path, "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT)
            f.write("\n\n--- USER PROMPT PREVIEW ---\n\n")
            f.write(preview)
        logger.info("Dry run wrote prompt preview to %s", args.prompt_preview_path)
        return {"samples": len(samples), "labeled": 0, "remaining": len(samples), "dry_run": True}

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api_key before running LLM labeling.")

    existing = _existing_sample_ids(args.out_csv)
    remaining = [s for s in samples if s.sample_id not in existing]
    logger.info("%d samples already labeled; %d remaining", len(existing), len(remaining))

    for start in tqdm(range(0, len(remaining), args.batch_size), desc="LLM labeling", unit="batch"):
        batch = remaining[start : start + args.batch_size]
        rows = call_openai_labels(
            batch,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        for row in rows:
            row["model"] = args.model
        _append_rows(args.out_csv, rows)
        _append_jsonl(args.raw_jsonl, rows)

    labeled = len(_existing_sample_ids(args.out_csv))
    summary = {
        "samples": len(samples),
        "labeled": labeled,
        "remaining": max(0, len(samples) - labeled),
        "out_csv": args.out_csv,
        "manifest_path": args.manifest_path,
        "model": args.model,
    }
    summary_path = os.path.splitext(args.out_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate LLM gold labels for Rico UI nodes.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch_create",
        action="store_true",
        help="Write OpenAI Batch JSONL, upload it, and create a /v1/responses batch.",
    )
    mode.add_argument(
        "--batch_status",
        action="store_true",
        help="Retrieve status for --batch_id.",
    )
    mode.add_argument(
        "--batch_collect",
        action="store_true",
        help="Download completed --batch_id output and append labels to --out_csv.",
    )
    parser.add_argument("--rico_dir", default="data/raw")
    parser.add_argument("--split_path", default="data/splits/split_seed42.json")
    parser.add_argument("--partition", default="val", choices=["all", "train", "val", "other"])
    parser.add_argument("--label_mode", default="contextual", choices=["contextual", "local_only"])
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_screens", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--manifest_path", default="data/gold/llm_gold_sample_manifest.jsonl")
    parser.add_argument("--out_csv", default="data/gold/gold_test_labels.csv")
    parser.add_argument("--raw_jsonl", default="data/gold/llm_gold_raw.jsonl")
    parser.add_argument("--prompt_preview_path", default="data/gold/llm_prompt_preview.txt")
    parser.add_argument("--batch_id", default=None)
    parser.add_argument("--batch_input_path", default="data/gold/openai_batch_input.jsonl")
    parser.add_argument("--batch_metadata_path", default="data/gold/openai_batch_metadata.json")
    parser.add_argument("--batch_output_path", default="data/gold/openai_batch_output.jsonl")
    parser.add_argument("--batch_info_path", default="data/gold/openai_batch_info.json")
    parser.add_argument("--env_path", default=".env")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    load_dotenv(args.env_path)
    if args.batch_create:
        summary = create_openai_batch(args)
        logger.info(
            "Created OpenAI batch %s with %d requests for %d samples. Info: %s",
            summary["batch_id"],
            summary["requests"],
            summary["samples"],
            args.batch_info_path,
        )
    elif args.batch_status:
        summary = retrieve_openai_batch(args)
        logger.info("Batch %s status: %s", summary.get("id"), summary.get("status"))
        print(json.dumps(summary, indent=2))
    elif args.batch_collect:
        summary = collect_openai_batch(args)
        logger.info("Collected OpenAI batch results: %s", summary)
    else:
        summary = run_labeling(args)
        logger.info("LLM gold labeling complete: %s", summary)


if __name__ == "__main__":
    main()
