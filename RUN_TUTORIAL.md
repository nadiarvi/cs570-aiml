# Running the Project

This guide explains how to run the Rico UI graph classification pipeline from a fresh checkout through GPU training.

## Table of Content
- [Running the Project](#running-the-project)
  - [Table of Content](#table-of-content)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Python Environment](#2-create-a-python-environment)
  - [3. Download the Rico Dataset](#3-download-the-rico-dataset)
  - [4. Preprocess Rico into Graphs](#4-preprocess-rico-into-graphs)
  - [5. Train a Model](#5-train-a-model)
    - [Resuming an Interrupted Run](#resuming-an-interrupted-run)
  - [6. Generate LLM Gold Labels](#6-generate-llm-gold-labels)
    - [Cheaper Batch API Run](#cheaper-batch-api-run)
  - [7. Run Hyperparameter Optimization](#7-run-hyperparameter-optimization)
    - [Inspect Search Results](#inspect-search-results)
    - [Retrain the Best Configurations](#retrain-the-best-configurations)
  - [8. Run the Ablation Matrix](#8-run-the-ablation-matrix)
  - [9. Recommended GPU Workflow](#9-recommended-gpu-workflow)
  - [10. Common Issues](#10-common-issues)
    - [`ModuleNotFoundError: No module named 'torch'`](#modulenotfounderror-no-module-named-torch)
    - [`torch.cuda.is_available()` prints `False`](#torchcudais_available-prints-false)
    - [No training graphs found](#no-training-graphs-found)
    - [Sentence-transformers download fails](#sentence-transformers-download-fails)
    - [`tensorboard: command not found`](#tensorboard-command-not-found)
    - [Out of memory during training](#out-of-memory-during-training)
  - [11. Useful Checks](#11-useful-checks)

## 1. Clone the Repository

```bash
git clone https://github.com/nadiarvi/cs570-aiml.git
cd cs570-aiml
```

If the repo is already cloned, update it:

```bash
git switch main
git pull origin main
```

## 2. Create a Python Environment

Use Python 3.10 or 3.11. PyTorch and `sentence-transformers` may not work smoothly on very new Python versions.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On a CUDA GPU machine, install PyTorch with CUDA first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The first line should print `True` on the GPU server.

## 3. Download the Rico Dataset

The project trains from Rico view hierarchy JSON files. Download and organize them with:

```bash
python -m src.data.download_rico --out_dir data/raw
```

This downloads the official Rico `unique_uis.tar.gz` archive into `data/downloads/` and extracts JSON files into:

```text
data/raw/<app_package>/<screen_id>.json
```

For a quick smoke test, download only a subset:

```bash
python -m src.data.download_rico --out_dir data/raw --max_screens 1000
```

If you already have the archive, place it at `data/downloads/unique_uis.tar.gz` and run:

```bash
python -m src.data.download_rico \
  --archive_path data/downloads/unique_uis.tar.gz \
  --out_dir data/raw \
  --skip_download
```

## 4. Preprocess Rico into Graphs

Run preprocessing for contextual heuristic labels:

```bash
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode contextual \
  --workers 8 \
  --embedding_cache_path data/processed/text_embedding_cache.json
```

The first preprocessing run may download the `all-MiniLM-L6-v2` text embedding model through `sentence-transformers`. If the GPU server has no internet, run once on a machine with internet or configure the model cache before preprocessing.

When `--embedding_cache_path` is provided, preprocessing bulk-encodes missing
unique text strings before graph construction. This is faster than encoding one
screen at a time and avoids rewriting the embedding cache after every screen.
The default embedding batch size is `256`; lower it if CUDA runs out of memory:

```bash
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode contextual \
  --workers 4 \
  --embedding_cache_path data/processed/text_embedding_cache.json \
  --embedding_batch_size 128
```

For the local-only label ablation, run preprocessing again with a different label mode:

```bash
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode local_only \
  --workers 8 \
  --embedding_cache_path data/processed/text_embedding_cache.json
```

## 5. Train a Model

Train the main GCN model:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json
```

Train the MLP baseline:

```bash
python -m src.train --config experiments/configs/mlp_all_contextual.json
```

Outputs are written under:

```text
results/checkpoints/<run_name>/
```

Each run saves:

- `best_model.pt`
- `run_metadata.json`
- `latest_checkpoint.pt` for resuming interrupted training
- `tensorboard/` event logs for browser-based metric monitoring

Training now shows `tqdm` progress bars in the terminal for class-weight
scanning, training batches, validation batches, and epoch-level metrics.

To monitor training curves outside the terminal, use TensorBoard. Leave the
training command running in one shell:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json
```

Then open a second shell on the same machine, activate the environment, and
start TensorBoard from the project root:

```bash
source .venv/bin/activate
tensorboard --logdir results/checkpoints
```

Open the URL printed by TensorBoard, usually:

```text
http://localhost:6006/
```

The dashboard will show per-run curves for:

- training loss
- validation Macro-F1
- validation accuracy
- learning rate
- early-stopping patience counter

If training is running on a remote GPU server, set up SSH port forwarding from
your laptop before opening TensorBoard in the browser:

```bash
ssh -L 6006:localhost:6006 ubuntu@<server>
```

Inside that SSH session, go to the repo, activate the environment, and start
TensorBoard:

```bash
cd cs570-aiml
source .venv/bin/activate
tensorboard --logdir results/checkpoints
```

Then open `http://localhost:6006/` in your laptop browser.

### Resuming an Interrupted Run

Training writes a full resume checkpoint after each completed epoch:

```text
results/checkpoints/<run_name>/latest_checkpoint.pt
```

If training stops or the SSH session disconnects, restart from the latest
completed epoch with:

```bash
python -m src.train \
  --config experiments/configs/gcn_2l_all_contextual.json \
  --resume
```

Resume restores:

- model weights
- optimizer state
- best validation Macro-F1
- early-stopping patience counter
- metric history
- random number generator state

Resume does not restart from the middle of a batch. If the process stops during
epoch 12, it resumes from the last fully saved epoch, usually epoch 11.

To resume from a specific checkpoint path:

```bash
python -m src.train \
  --config experiments/configs/gcn_2l_all_contextual.json \
  --resume_checkpoint_path results/checkpoints/gcn_2l_all_contextual/latest_checkpoint.pt
```

## 6. Generate LLM Gold Labels

The project no longer needs hand-labeled gold annotations. Generate a 5,000
node evaluation set from the validation partition with OpenAI:

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. The file is ignored by git.

```bash
python -m src.data.llm_gold_labeler \
  --rico_dir data/raw \
  --split_path data/splits/split_seed42.json \
  --partition val \
  --sample_size 5000 \
  --batch_size 20 \
  --model gpt-5-nano \
  --out_csv data/gold/gold_test_labels.csv
```

`gpt-5-nano` is the default because it is currently OpenAI's cheapest nano text
model. The job is resumable: it writes the sampled nodes to
`data/gold/llm_gold_sample_manifest.jsonl`, appends completed labels to
`data/gold/gold_test_labels.csv`, and writes an audit trail to
`data/gold/llm_gold_raw.jsonl`.

Preview the exact prompt and first batch without spending API credits:

```bash
python -m src.data.llm_gold_labeler \
  --rico_dir data/raw \
  --split_path data/splits/split_seed42.json \
  --sample_size 5000 \
  --dry_run
```

The generated CSV matches the existing gold-label loader:

```text
screen_id,node_id,label,annotator_id,app_id,sample_id,model,confidence,...
```

Use it directly in the ablation config via:

```json
"gold_labels_path": "data/gold/gold_test_labels.csv"
```

### Cheaper Batch API Run

If you can wait for asynchronous processing, use OpenAI Batch API. It runs the
same `/v1/responses` requests with a 24-hour completion window and lower token
cost.

Create and submit the batch:

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` if you have not already.

```bash
python -m src.data.llm_gold_labeler \
  --batch_create \
  --rico_dir data/raw \
  --split_path data/splits/split_seed42.json \
  --partition val \
  --sample_size 5000 \
  --batch_size 20 \
  --model gpt-5-nano \
  --out_csv data/gold/gold_test_labels.csv
```

The command prints and saves the `batch_id` in:

```text
data/gold/openai_batch_info.json
```

Check progress:

```bash
python -m src.data.llm_gold_labeler \
  --batch_status \
  --batch_id batch_...
```

When the status is `completed`, download and merge the labels:

```bash
python -m src.data.llm_gold_labeler \
  --batch_collect \
  --batch_id batch_... \
  --out_csv data/gold/gold_test_labels.csv
```

Batch files are kept locally for audit and reproducibility:

```text
data/gold/openai_batch_input.jsonl
data/gold/openai_batch_metadata.json
data/gold/openai_batch_output.jsonl
```

### After Labels Are Ready

Validate that the label CSV loads and has the expected class distribution:

```bash
python - <<'PY'
from src.data.gold import load_gold_test_labels
df = load_gold_test_labels("data/gold/gold_test_labels.csv")
print(df.head())
print(df["label"].value_counts())
print("rows:", len(df))
PY
```

These labels are for evaluation only. Do not train on
`data/gold/gold_test_labels.csv`; training still uses heuristic labels from
preprocessing. After validation, run the ablation matrix in Section 8 to
evaluate trained models against the LLM gold labels.

## 7. Run Hyperparameter Optimization

The first GCN and MLP runs are baseline configurations, not optimized models.
Run a small search before deciding whether graph structure helps.

The default search trains a curated set of MLP and GCN configurations:

```bash
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_results.csv
```

The search writes:

```text
results/hparam_search_results.csv
experiments/generated_configs/hparam/<run_name>.json
experiments/generated_configs/hparam/best_mlp.json
experiments/generated_configs/hparam/best_gcn.json
results/checkpoints/hparam/<run_name>/
```

If the search is interrupted, resume from saved per-run checkpoints:

```bash
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_results.csv \
  --resume \
  --skip_completed
```

For a very small smoke test of the search script:

```bash
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_smoke.csv \
  --max_runs 2
```

For a larger grid, use:

```bash
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_full.csv \
  --search_space full
```

The full grid is much slower. Use it only after the quick search runs
successfully.

### Inspect Search Results

Print the best runs by heuristic validation Macro-F1:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/hparam_search_results.csv")
cols = ["model_type", "name", "best_val_macro_f1", "best_epoch", "config_path"]
print(df.sort_values("best_val_macro_f1", ascending=False)[cols].head(10))
PY
```

Compare the best MLP against the best GCN. If the best tuned GCN still trails
the best tuned MLP, graph propagation is not helping on the heuristic
validation split.

### Retrain the Best Configurations

After the search finishes, retrain the best MLP and best GCN into clean
checkpoint directories:

```bash
python -m src.train --config experiments/generated_configs/hparam/best_mlp.json
python -m src.train --config experiments/generated_configs/hparam/best_gcn.json
```

If either retraining run is interrupted:

```bash
python -m src.train --config experiments/generated_configs/hparam/best_mlp.json --resume
python -m src.train --config experiments/generated_configs/hparam/best_gcn.json --resume
```

Use TensorBoard to compare all training curves:

```bash
tensorboard --logdir results/checkpoints
```

## 8. Run the Ablation Matrix

After preprocessing both `contextual` and `local_only` labels and generating
`data/gold/gold_test_labels.csv`, run:

```bash
python -m src.ablation \
  --config experiments/configs/ablation_base.json \
  --out_csv results/ablation_results.csv
```

This trains the configured model variants and writes summary metrics to:

```text
results/ablation_results.csv
```

Gold evaluation requires resolved gold labels at:

```text
data/gold/gold_test_labels.csv
```

If that file is missing, training can still run, but gold evaluation in the ablation script will fail.

## 9. Recommended GPU Workflow

For a first GPU run:

```bash
python -m src.data.download_rico --out_dir data/raw --max_screens 1000
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode contextual \
  --workers 8 \
  --embedding_cache_path data/processed/text_embedding_cache.json
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_results.csv \
  --max_runs 2
```

If that works, remove `--max_screens 1000`, clear the partial processed data if
needed, and run the full dataset plus the full quick hyperparameter search.

## 10. Common Issues

### `ModuleNotFoundError: No module named 'torch'`

Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### `torch.cuda.is_available()` prints `False`

Check that you are on the GPU machine and installed a CUDA PyTorch build:

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

### No training graphs found

Preprocessing did not create files in the expected directory. Check:

```bash
find data/processed -name '*.pt' | head
```

Then confirm that raw Rico JSON files exist:

```bash
find data/raw -name '*.json' | head
```

### Sentence-transformers download fails

The preprocessing step needs the `all-MiniLM-L6-v2` model for text embeddings. Use a machine with internet for the first run, or pre-populate the Hugging Face cache on the GPU server.

### `tensorboard: command not found`

Activate the virtual environment and reinstall dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Out of memory during training

Lower `batch_size` in the config file, for example:

```json
"batch_size": 8
```

Then rerun training.

## 11. Useful Checks

Check current Git version:

```bash
git status --short --branch
git log --oneline -3
```

Run tests:

```bash
pytest -q
```

Compile-check source files:

```bash
python -m compileall -q src tests
```
