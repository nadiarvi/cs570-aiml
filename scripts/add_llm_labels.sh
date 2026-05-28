#!/usr/bin/env bash
set -e
[ ! -f .env ] && echo "Error: .env not found. Create: echo 'GEMINI_API_KEYS=key' > .env" && exit 1
tmux new -s llm_labels -d
tmux send-keys -t llm_labels "conda activate ui-gcn && python src/data/preprocess.py --processed_dir data/processed --add_llm --max_screens 5000" Enter
echo "Running in tmux 'llm_labels'. Attach: tmux attach -t llm_labels"
