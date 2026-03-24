#!/usr/bin/env bash
set -euo pipefail

echo "=== Computer Vision 101 — Environment Setup ==="

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e ".[dev]" 2>/dev/null || pip install -e .

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo "Then launch JupyterLab with:"
echo "  jupyter lab"
