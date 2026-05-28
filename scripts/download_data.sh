#!/usr/bin/env bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ui-gcn
python scripts/download_rico.py
