# Project Progress and Remaining Work

Updated: May 28, 2026

## Completed

- Built the Rico preprocessing pipeline from raw hierarchy JSON files to saved
  graph `.pt` files.
- Added contextual and local-only heuristic labeling modes.
- Added feature extraction for visual, structural, widget type, and text
  embedding features.
- Added graph construction with containment edges and optional sibling edges.
- Added lazy graph loading and batching for disconnected graph minibatches.
- Implemented MLP, GCN, and GAT model code.
- Added deterministic training with early stopping on heuristic validation
  Macro-F1.
- Added class weighting for imbalanced node labels.
- Fixed graph loading so directories whose names end in `.pt` are ignored.
- Added `tqdm` progress bars for class-weight scanning, training batches,
  validation batches, and epoch progress.
- Added TensorBoard metric logging under each checkpoint directory.
- Added resume support through `latest_checkpoint.pt`.
- Added `src.hyperparameter_search` for controlled MLP and GCN
  hyperparameter optimization.
- Successfully trained initial contextual models:
  - GCN 2-layer contextual: validation Macro-F1 `0.5668`
  - MLP contextual baseline: validation Macro-F1 `0.6582`

## Current Interpretation

- The current MLP baseline outperforms the current GCN configuration on
  heuristic validation labels.
- This does not prove graph neural networks are ineffective for the task.
  It only shows that the first GCN hyperparameter setting is weaker than the
  first MLP setting.
- The likely next step is controlled hyperparameter optimization for both MLP
  and GCN, followed by a fair comparison of the best configuration from each
  model family.
- Final conclusions should wait for gold-label evaluation, because the current
  validation metric is based on heuristic labels.

## Remaining Work

- Run the quick hyperparameter search for MLP and GCN.
- Retrain the best MLP and best GCN configurations from the sweep.
- Complete or collect gold labels at `data/gold/gold_test_labels.csv`.
- Update ablation/evaluation results once gold labels are available.
- Compare heuristic validation results against gold-label test results.
- Decide whether graph structure helps after comparing tuned models, not just
  the initial default configurations.
- Consider additional GCN variants if tuned GCN remains weak:
  - fewer layers to reduce oversmoothing
  - containment-only edges
  - lower dropout
  - smaller hidden dimensions
  - lower learning rate
  - no-text feature ablation

## Useful Commands

Train the current GCN:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json
```

Train the current MLP:

```bash
python -m src.train --config experiments/configs/mlp_all_contextual.json
```

Resume interrupted training:

```bash
python -m src.train --config experiments/configs/gcn_2l_all_contextual.json --resume
```

Monitor training:

```bash
tensorboard --logdir results/checkpoints
```

Run quick hyperparameter search:

```bash
python -m src.hyperparameter_search \
  --base_config experiments/configs/ablation_base.json \
  --out_csv results/hparam_search_results.csv
```
