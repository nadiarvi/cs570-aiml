#!/usr/bin/env bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ui-gcn
python src/evaluate.py --results_csv results/ablation_results.csv --logs_dir results/logs --figures_dir results/figures
ls results/figures/
