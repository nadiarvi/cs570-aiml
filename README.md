# CS570 AI/ML Project

Team 14 project repository for GCN-based modifiability prediction on UI hierarchy graphs.

## Layout

- `docs/` - proposal, feedback, and implementation notes
- `src/` - model, data pipeline, training, and evaluation code
- `data/` - raw Rico data, processed graph objects, and split files
- `experiments/` - experiment configuration files
- `results/` - checkpoints, logs, figures, and evaluation outputs

Large datasets and generated outputs are intentionally ignored by git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Rico locally

The training pipeline expects Rico view hierarchy JSON files under
`data/raw/<app_package>/<screen_id>.json`. Download and prepare them with:

```bash
python -m src.data.download_rico --out_dir data/raw
```

For a smaller smoke-test subset before a full run:

```bash
python -m src.data.download_rico --out_dir data/raw --max_screens 1000
```

Then preprocess:

```bash
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode contextual \
  --workers 8 \
  --embedding_cache_path data/processed/text_embedding_cache.json
```

When `--embedding_cache_path` is set, preprocessing first bulk-encodes missing
text embeddings in larger batches, then builds graph files. Tune the embedding
batch size if GPU memory allows:

```bash
python -m src.data.preprocess \
  --rico_dir data/raw \
  --out_dir data/processed \
  --split_path data/splits/split_seed42.json \
  --label_mode contextual \
  --workers 4 \
  --embedding_cache_path data/processed/text_embedding_cache.json \
  --embedding_batch_size 256
```

Train on GPU:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json
```

Training shows `tqdm` progress bars in the terminal and writes TensorBoard
metrics under the run checkpoint directory. To monitor loss, validation F1, and
validation accuracy in a browser:

```bash
tensorboard --logdir results/checkpoints
```

Then open the local TensorBoard URL printed by that command. On a remote GPU
server, forward port `6006` with SSH, for example:

```bash
ssh -L 6006:localhost:6006 ubuntu@<server>
```

Interrupted training can resume from the latest completed epoch:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json --resume
```
