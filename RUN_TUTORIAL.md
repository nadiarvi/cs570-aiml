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

## 6. Run the Ablation Matrix

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

## 7. Recommended GPU Workflow

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
```

If that works, remove `--max_screens 1000`, clear the partial processed data if needed, and run the full dataset.

## 8. Common Issues

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

### Out of memory during training

Lower `batch_size` in the config file, for example:

```json
"batch_size": 8
```

Then rerun training.

## 9. Useful Checks

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
