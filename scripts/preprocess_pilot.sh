#!/usr/bin/env bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ui-gcn
python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 4 --max_screens 500
echo "Pilot done. Check data/processed/ and label distribution."
