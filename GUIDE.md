# Running Guide — UI Modifiability Prediction

CS570 Team 14 | Step-by-step from setup to final figures.

---

## Overview

The pipeline has two parallel tracks that run simultaneously after preprocessing:

```
Track A — heuristic labels (fast, start immediately)
Track B — LLM labels (slow, runs in background)

Step 1: Setup & download data
Step 2: Pilot run (500 screens, verify pipeline)
Step 3: Full preprocessing (72K screens)     ──→ Start Track B here (background)
Step 4: Train MLP / GCN / GAT (Track A)
Step 5: Run all ablations (Track A)
Step 6: Generate all figures
                                             ←── Track B finishes → re-run ablations
```

---

## Prerequisites

- GPU server with CUDA 12.1
- Python 3.10+ available on the server (`python3 --version` to check)
- Git access to clone this repo
- One or more free Gemini API keys (for Track B only): https://aistudio.google.com/app/apikey

---

## Step 1 — Clone, Install Conda & Setup

**On the GPU server:**

```bash
git clone https://github.com/<your-org>/ui-modifiability.git
cd ui-modifiability
```

If conda is not yet installed on the server, run the installer script first:

```bash
bash scripts/install_conda.sh
source ~/.bashrc
```

Expected output at the end:
```
Conda 24.x.x is ready.
```

Then create the environment and install all dependencies:

```bash
bash scripts/setup.sh
```

Expected output at the end:
```
GPU available: True
```

If you see `False`, check your CUDA driver (`nvidia-smi`) and that PyTorch was installed with the matching CUDA version.

---

## Step 2 — Download Rico Data

```bash
bash scripts/download_data.sh
```

This downloads the Rico dataset from HuggingFace. The raw JSON hierarchy files should end up in `data/raw/`, organized as:

```
data/raw/
  com.example.app1/
    0.json
    1.json
  com.example.app2/
    0.json
    ...
```

> If the HuggingFace download gives you a different structure, move the JSON files manually so that `data/raw/<app_package>/<screen_id>.json` is the layout. The app package directory name is what drives the app-level split.

---

## Step 3 — Pilot Run (verify pipeline on 500 screens)

Before committing to the full dataset, run the pilot to catch any issues early:

```bash
bash scripts/preprocess_pilot.sh
```

This preprocesses 500 screens and saves `.pt` graph files to `data/processed/`. It also creates `data/splits/split_seed42.json` (app-level 70/15/15 split).

**Verify the output:**
```bash
ls data/processed/ | wc -l          # should be ~500
ls data/splits/                     # should contain split_seed42.json
```

Check the label distribution in a Python session:
```python
import torch, glob
from collections import Counter

pts = glob.glob("data/processed/**/*.pt", recursive=True)[:100]
counts = Counter()
for p in pts:
    g = torch.load(p, weights_only=False)
    y = g["y_heuristic"]
    for lbl in y.tolist():
        if lbl >= 0:
            counts[lbl] += 1

total = sum(counts.values())
for lbl, name in [(0, "canonical"), (1, "translatable"), (2, "open")]:
    print(f"  {name}: {counts[lbl]/total*100:.1f}%")
```

Expected: ~10% canonical, ~20% translatable, ~70% open.

If the distribution looks wrong (e.g., everything is class 2), check `src/data/labeler.py`.

---

## Step 4 — Full Preprocessing

Once the pilot looks good, preprocess all screens:

```bash
bash scripts/preprocess_full.sh
```

This runs in a tmux session named `preprocess`. Attach to monitor progress:

```bash
tmux attach -t preprocess
# Detach: Ctrl+B then D
```

This produces ~72K `.pt` files and takes 1–3 hours depending on disk speed.

> **Start Track B now** — see Step 4B below. It runs in parallel.

---

## Step 4B — LLM Labeling (Track B, background)

**Do this immediately after starting Step 4**, in a new terminal.

First, create your `.env` file with Gemini API keys:

```bash
echo "GEMINI_API_KEYS=key1,key2,key3" > .env
```

Then start labeling (runs in tmux, safe to stop and resume):

```bash
bash scripts/add_llm_labels.sh
```

Monitor progress:
```bash
tmux attach -t llm_labels
```

This labels up to 5,000 screens at 15 requests/minute (~2.3 days with one key, faster with multiple keys). The cache at `data/llm_label_cache.json` saves every label, so it resumes safely if interrupted.

**If a key hits its daily quota**, the script rotates to the next key automatically. If all keys are exhausted, it will print a message and exit — re-run the next day and it will pick up from the cache.

---

## Step 5 — Train the Three Models (Track A)

Once full preprocessing is done, launch all three training runs in parallel tmux sessions:

```bash
bash scripts/train_mlp.sh   # tmux: train_mlp
bash scripts/train_gcn.sh   # tmux: train_gcn
bash scripts/train_gat.sh   # tmux: train_gat
```

Monitor any of them:
```bash
tmux attach -t train_gcn
```

Each run saves:
- Best checkpoint: `results/checkpoints/<run>/best_model.pt`
- Training log CSV: `results/logs/<run>_log.csv`
- Confusion matrix figure: `results/figures/<run>_confusion.png`

Default hyperparameters (all models):

| Param | MLP | GCN | GAT |
|-------|-----|-----|-----|
| hidden_dim | 256 | 256 | 256 |
| num_layers | 3 | 2 | 2 |
| num_heads | — | — | 4 |
| dropout | 0.3 | 0.3 | 0.3 |
| lr | 1e-3 | 1e-3 | 1e-3 |
| weight_decay | 1e-4 | 1e-4 | 1e-4 |
| epochs | 100 | 100 | 100 |
| patience | 15 | 15 | 15 |
| batch_size | 32 | 32 | 32 |

Training typically takes 2–8 hours per model on a GPU with 72K screens.

**Troubleshooting:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `macro_f1 ≈ 0.33` from epoch 1 | Model always predicts "open" | Check that class weights are non-zero; verify `data/splits/split_seed42.json` exists |
| CUDA OOM | Batch too large | Reduce `batch_size` in the config JSON to 16 |
| Val F1 oscillates badly | LR too high | Lower `lr` to `5e-4` in the config JSON |
| `edge_index` float error | Old .pt files | Re-run preprocess; ensure `dtype=torch.long` in graph builder |

---

## Step 6 — Run All Ablations (Track A)

Once the three baseline models finish training (or in parallel with them):

```bash
bash scripts/run_ablations.sh
```

Attach to monitor:
```bash
tmux attach -t ablations
```

This runs 7 ablation configs (Track A, heuristic labels):

| Config | What it tests |
|--------|--------------|
| `mlp_all` | MLP baseline (no edges) |
| `gcn_2l_all` | GCN with all edges |
| `gat_2l_all` | GAT with all edges |
| `gcn_2l_contain` | GCN with containment edges only (no sibling) |
| `gcn_2l_notext` | GCN without text embeddings (visual+structural+type only, d=36) |
| `gcn_1l_all` | GCN with 1 layer |
| `gcn_3l_all` | GCN with 3 layers |

Results are appended to `results/ablation_results.csv` as each run completes.

---

## Step 7 — Generate All Figures

```bash
bash scripts/generate_figures.sh
```

This produces:

| Figure | Path |
|--------|------|
| Model comparison bar | `results/figures/model_comparison.png` |
| MLP confusion matrix | `results/figures/mlp_all_confusion.png` |
| GCN confusion matrix | `results/figures/gcn_2l_all_confusion.png` |
| Ablation bar chart | `results/figures/ablation_comparison.png` |
| Training curves (per run) | `results/figures/<run>_curves.png` |

---

## Step 8 — Track B Ablations (after LLM labeling finishes)

Once `add_llm_labels.sh` completes, run the LLM-label ablations:

```bash
python src/ablation.py --names mlp_all_llm gcn_2l_all_llm gat_2l_all_llm \
    --output results/ablation_results.csv
```

Then regenerate figures:
```bash
bash scripts/generate_figures.sh
```

Compare the `*_llm` rows to the `*_heuristic` rows in the CSV to decide which label source is better for the report.

---

## Resuming After Interruption

| What was interrupted | How to resume |
|---------------------|--------------|
| Full preprocessing | Re-run `bash scripts/preprocess_full.sh` — already-saved `.pt` files are skipped |
| LLM labeling | Re-run `bash scripts/add_llm_labels.sh` — cache skips already-labeled nodes |
| Training | Re-run the training script — overwrite the checkpoint by training from scratch, or check the log CSV for where it stopped |
| Ablation runner | Use `--names` to specify only the configs that didn't finish |

---

## Running the Smoke Test (local, no dataset needed)

Before pushing changes to the GPU server, verify model code locally:

```bash
python src/tests/smoke_test.py
```

Expected output (under 5 seconds on CPU):
```
MLP ✓
GCN ✓
GAT ✓
All smoke tests passed.
```

This only requires PyTorch — no dataset, no SentenceTransformers.

---

## Key File Locations

```
data/splits/split_seed42.json     App-level split (created during preprocessing)
data/embed_cache.pkl              Text embedding cache (speeds up re-runs)
data/llm_label_cache.json         LLM label cache (resume-safe)
results/checkpoints/<run>/        Best model checkpoints
results/logs/<run>_log.csv        Epoch-by-epoch loss + val F1
results/figures/                  All PNG outputs
results/ablation_results.csv      All ablation metrics in one table
```

---

## What to Report

Report **only** the test-split numbers from `ablation_results.csv`. Never report val F1 as the final number.

The key metric is **Macro-F1** (weights all 3 classes equally). A large MLP → GCN gap confirms that neighborhood structure helps; a small gap is also a valid finding.

If `macro_f1 ≈ 0.33` for any model, the model is collapsing to the majority class ("open") — the weighted loss is either not applied or the weights are all equal. Check the class weight computation in `src/train.py`.
