#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s preprocess -d
tmux send-keys -t preprocess "conda activate ui-gcn && python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 8" Enter
echo "Running in tmux 'preprocess'. Attach: tmux attach -t preprocess"
