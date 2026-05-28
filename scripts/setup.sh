#!/usr/bin/env bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n ui-gcn python=3.10 -y
conda activate ui-gcn
# Install PyTorch via pip to avoid conda channel conflicts (jpeg/libjpeg-turbo clash)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers scikit-learn tqdm pandas matplotlib seaborn datasets Pillow google-generativeai python-dotenv
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
