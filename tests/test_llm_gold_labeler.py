import json
from argparse import Namespace

import pytest

from src.data.llm_gold_labeler import (
    LABEL_NAMES,
    NodeSample,
    _prompt_for_batch,
    _validate_batch_labels,
    load_manifest,
    run_labeling,
    write_openai_batch_input,
    write_manifest,
)


def _sample(sample_id="screen1:0"):
    return NodeSample(
        sample_id=sample_id,
        screen_id="screen1",
        app_id="app.pkg",
        node_id="0",
        heuristic_label="translatable",
        fields={
            "class": "android.widget.TextView",
            "text": "Checkout",
            "content_desc": "",
            "resource_id": "title",
            "depth": 3,
            "child_count": 0,
        },
    )


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    samples = [_sample(), _sample("screen1:1")]

    write_manifest(samples, str(path))
    loaded = load_manifest(str(path))

    assert loaded == samples


def test_prompt_contains_samples_and_json_instruction():
    prompt = _prompt_for_batch([_sample()])

    assert "JSON is required" in prompt
    payload = json.loads(prompt.split("\n\n", 1)[1])
    assert payload["samples"][0]["sample_id"] == "screen1:0"
    assert payload["samples"][0]["node"]["text"] == "Checkout"


def test_validate_batch_labels_outputs_gold_csv_shape():
    rows = _validate_batch_labels(
        [_sample()],
        [
            {
                "sample_id": "screen1:0",
                "label": "translatable",
                "confidence": 0.87,
                "rationale": "Standard screen title.",
            }
        ],
    )

    assert rows[0]["screen_id"] == "screen1"
    assert rows[0]["node_id"] == "0"
    assert rows[0]["label"] in LABEL_NAMES
    assert rows[0]["annotator_id"] == "openai_llm"
    assert rows[0]["confidence"] == pytest.approx(0.87)


def test_validate_batch_labels_rejects_missing_sample_id():
    with pytest.raises(ValueError, match="missing sample IDs"):
        _validate_batch_labels([_sample()], [])


def test_write_openai_batch_input(tmp_path):
    input_path = tmp_path / "batch_input.jsonl"
    metadata_path = tmp_path / "batch_metadata.json"

    summary = write_openai_batch_input(
        [_sample(), _sample("screen1:1")],
        input_path=str(input_path),
        metadata_path=str(metadata_path),
        batch_size=1,
        model="gpt-5-nano",
    )

    lines = input_path.read_text().splitlines()
    assert summary["requests"] == 2
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["method"] == "POST"
    assert first["url"] == "/v1/responses"
    assert first["body"]["model"] == "gpt-5-nano"
    assert first["custom_id"] == "llm-gold-000001"

    metadata = json.loads(metadata_path.read_text())
    assert metadata["requests"]["llm-gold-000001"][0]["sample_id"] == "screen1:0"


def test_run_labeling_end_to_end_with_mocked_openai(tmp_path, monkeypatch):
    app_dir = tmp_path / "raw" / "app.pkg"
    app_dir.mkdir(parents=True)
    (app_dir / "screen1.json").write_text(
        json.dumps(
            {
                "activity": {
                    "root": {
                        "class": "android.widget.LinearLayout",
                        "resource-id": "main_container",
                        "children": [
                            {
                                "class": "android.widget.TextView",
                                "text": "$19.99",
                                "resource-id": "price_total",
                            },
                                {
                                    "class": "android.widget.Button",
                                    "text": "Continue",
                                    "resource-id": "next_button",
                                },
                            {
                                "class": "android.widget.ImageView",
                                "content-desc": "Promo banner",
                                "resource-id": "promo_banner",
                            },
                        ],
                    }
                }
            }
        )
    )

    def fake_call_openai_labels(samples, api_key, model, timeout, max_retries):
        response_rows = []
        for sample in samples:
            if "price" in sample.fields.get("resource_id", ""):
                label = "canonical"
            elif sample.fields.get("class") == "android.widget.Button":
                label = "translatable"
            else:
                label = "open"
            response_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "label": label,
                    "confidence": 0.9,
                    "rationale": "Mocked smoke label.",
                }
            )
        return _validate_batch_labels(samples, response_rows)

    monkeypatch.setattr(
        "src.data.llm_gold_labeler.call_openai_labels",
        fake_call_openai_labels,
    )

    out_csv = tmp_path / "gold_test_labels.csv"
    args = Namespace(
        rico_dir=str(tmp_path / "raw"),
        split_path=str(tmp_path / "missing_split.json"),
        partition="all",
        label_mode="contextual",
        sample_size=3,
        batch_size=2,
        seed=42,
        max_screens=None,
        model="gpt-5-nano",
        api_key="test-key",
        timeout=10,
        max_retries=1,
        resample=False,
        dry_run=False,
        manifest_path=str(tmp_path / "manifest.jsonl"),
        out_csv=str(out_csv),
        raw_jsonl=str(tmp_path / "raw_labels.jsonl"),
        prompt_preview_path=str(tmp_path / "prompt.txt"),
    )

    summary = run_labeling(args)

    assert summary["labeled"] == 3
    csv_text = out_csv.read_text()
    assert "screen_id,node_id,label,annotator_id" in csv_text
    assert "canonical" in csv_text
    assert "translatable" in csv_text
    assert "open" in csv_text
