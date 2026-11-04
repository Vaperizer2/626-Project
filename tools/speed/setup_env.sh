#!/usr/bin/env bash
# Rebuilds the harness venv (already present on legion). torch/torchvision are CPU wheels: the harness only uses
# them for compclass's preprocessing and head code; inference runs in onnxruntime-gpu.
set -euo pipefail
VENV=/home/vaper/Work/Prototypes/626-artifacts/harness/venv
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    -r "$(dirname "$(readlink -f "$0")")/requirements.txt"
