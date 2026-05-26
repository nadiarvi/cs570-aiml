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

