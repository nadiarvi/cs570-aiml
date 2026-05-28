#!/usr/bin/env bash
set -e
# Run AFTER preprocess_full.sh. Patches y_llm into existing .pt files.
# Safe to stop and resume — cache skips already-labeled nodes.
[ ! -f .env ] && echo "Error: .env not found. Create: echo 'GEMINI_API_KEYS=key' > .env" && exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ui-gcn
python src/data/preprocess.py --processed_dir data/processed --add_llm --max_screens 5000
echo "LLM labeling done. Cache: data/llm_label_cache.json"
