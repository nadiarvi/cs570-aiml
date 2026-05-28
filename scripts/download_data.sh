#!/usr/bin/env bash
set -e
conda activate ui-gcn
python -c "from datasets import load_dataset; load_dataset('creative-graphic-design/Rico', 'ui_layout_vectors'); print('Done.')"
