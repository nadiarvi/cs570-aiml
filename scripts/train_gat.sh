#!/usr/bin/env bash
set -e
tmux new -s train_gat -d
tmux send-keys -t train_gat "conda activate ui-gcn && python src/train.py --config experiments/configs/gat_baseline.json" Enter
echo "Running in tmux 'train_gat'."
