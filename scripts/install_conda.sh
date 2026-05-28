#!/usr/bin/env bash
# Installs Miniconda to ~/miniconda3 and initialises it for bash/zsh.
# Run once on the GPU server, then open a new shell (or source ~/.bashrc).
set -e

INSTALL_DIR="$HOME/miniconda3"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER="/tmp/miniconda_install.sh"

if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/bin/conda" ]; then
    echo "Miniconda already installed at $INSTALL_DIR — skipping download."
else
    echo "Downloading Miniconda..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$INSTALLER_URL" -O "$INSTALLER"
    elif command -v curl &>/dev/null; then
        curl -# -L "$INSTALLER_URL" -o "$INSTALLER"
    else
        echo "Error: neither wget nor curl found. Install one and retry."
        exit 1
    fi

    echo "Installing Miniconda to $INSTALL_DIR ..."
    bash "$INSTALLER" -b -p "$INSTALL_DIR"
    rm -f "$INSTALLER"
    echo "Miniconda installed."
fi

CONDA_BIN="$INSTALL_DIR/bin/conda"

# Initialise for bash
"$CONDA_BIN" init bash

# Initialise for zsh if zsh is present
if command -v zsh &>/dev/null; then
    "$CONDA_BIN" init zsh
fi

# Make conda available in the current shell session without a restart
# shellcheck disable=SC1091
source "$INSTALL_DIR/etc/profile.d/conda.sh"

echo ""
echo "Conda $(conda --version) is ready."
echo ""
echo "Next steps:"
echo "  source ~/.bashrc          # or open a new terminal"
echo "  bash scripts/setup.sh     # create the ui-gcn environment"
