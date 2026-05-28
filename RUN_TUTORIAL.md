# Running the Project

This guide explains how to run the Rico UI graph classification pipeline from a fresh checkout through GPU training.

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

## 6. Run Hyperparameter Optimization

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

## 7. Run the Ablation Matrix

After preprocessing both `contextual` and `local_only` labels, run:

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

## 8. Recommended GPU Workflow

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

## 9. Common Issues

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

## 10. Useful Checks

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
