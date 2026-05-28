#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s ablations -d
tmux send-keys -t ablations "conda activate ui-gcn && python src/ablation.py --output results/ablation_results.csv" Enter
echo "Running in tmux 'ablations'."
