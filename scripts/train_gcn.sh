#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s train_gcn -d
tmux send-keys -t train_gcn "conda activate ui-gcn && python src/train.py --config experiments/configs/gcn_baseline.json" Enter
echo "Running in tmux 'train_gcn'."
