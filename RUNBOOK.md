# Experiment Runbook

## Step 1 — Download the correct Rico data

Go to **http://interactionmining.org/rico** and find the
"UI Screenshots and View Hierarchies" download link. It is a ~3.5 GB tar file
of ~72K JSON files. Then on the server:

```bash
wget <URL from interactionmining.org/rico>
mkdir -p data/raw
tar -xzf <downloaded_filename> -C data/raw/
ls data/raw/    # note the actual subfolder name (e.g. "combined")
```

---

## Step 2 — Wipe all garbage outputs from the bad data run

```bash
rm -rf data/processed/
rm -f  data/splits/split_seed42.json
rm -f  data/llm_label_cache.json
rm -rf results/checkpoints/
rm -f  results/logs/*.csv
rm -f  results/ablation_results.csv
```

---

## Step 3 — Verify the raw data looks right

```bash
python inspect_raw.py
```

Expected output: `root class: android.widget.FrameLayout` (or similar)
and a **non-zero children count**.

**Do not continue past this step if it still shows 1 node or class='Unknown'.**

---

## Step 4 — Preprocess (heuristic labels)

First run the pilot (500 screens) and check the label distribution:

```bash
bash scripts/preprocess_pilot.sh
python diagnose_labels.py
```

Expected label distribution: roughly 10% class 0, 20% class 1, 70% class 2,
spread across multiple depths. If it looks right, run the full preprocessing:

```bash
bash scripts/preprocess_full.sh    # runs in tmux session 'preprocess'
```

Wait for it to finish before proceeding.

---

## Step 5 — Train all three models (heuristic track)

```bash
bash scripts/train_mlp.sh
bash scripts/train_gcn.sh
bash scripts/train_gat.sh
```

These run in parallel tmux sessions. Sanity check: in the log CSVs under
`results/logs/`, the `val_macro_f1` column should start around 0.3–0.5 at
epoch 1, not 1.0.

---

## Step 6 — Run all heuristic ablations

```bash
bash scripts/run_ablations.sh
```

This populates `results/ablation_results.csv` with the 7 heuristic rows.

---

## Step 7 — Generate figures

```bash
bash scripts/generate_figures.sh
```

---

## Step 8 (optional) — LLM labels

Only if time allows — takes 2–3 days at 15 RPM.

```bash
bash scripts/add_llm_labels.sh
```

After it finishes, run the LLM ablations into a **separate file** to avoid
overwriting heuristic results:

```bash
python src/ablation.py \
  --names mlp_all_llm gcn_2l_all_llm gat_2l_all_llm \
  --output results/ablation_results_llm.csv
```

Then manually merge `ablation_results_llm.csv` into `ablation_results.csv`
by appending the three LLM rows, and re-run:

```bash
bash scripts/generate_figures.sh
```
