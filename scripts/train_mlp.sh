#!/usr/bin/env bash
set -e
tmux new -s train_mlp -d
tmux send-keys -t train_mlp "conda activate ui-gcn && python src/train.py --config experiments/configs/mlp_baseline.json" Enter
echo "Running in tmux 'train_mlp'."
