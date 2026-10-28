#!/usr/bin/env bash
# Isolated overlay on an existing Python 3.11 / Isaac Sim 5.1 installation.
set -euo pipefail
BASE_PYTHON=${BASE_PYTHON:?set the existing Isaac Python 3.11 interpreter}
RUNTIME_DIR=${RUNTIME_DIR:?set a new task-owned runtime directory}
UV=${UV:-$HOME/.local/bin/uv}
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AA_DIR=$(cd "$PROJECT_DIR/../.." && pwd)
mkdir -p "$RUNTIME_DIR"
export UV_CACHE_DIR="$RUNTIME_DIR/uv-cache"
if [ ! -x "$RUNTIME_DIR/venv/bin/python" ]; then
  "$UV" venv --python "$BASE_PYTHON" --system-site-packages "$RUNTIME_DIR/venv"
fi
PYTHON="$RUNTIME_DIR/venv/bin/python"
"$UV" pip install --python "$PYTHON" torch==2.11.0 torchvision==0.26.0 numpy==1.26.4 \
  --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PYTHON" --no-deps \
  torchrl==0.13.1 tensordict==0.13.0 pyvers==0.2.2 \
  coacd==1.0.10 smplx==0.1.28 mujoco==3.10.0 warp-lang==1.10.1 \
  hydra-core==1.3.2 omegaconf==2.3.0 \
  setproctitle einops termcolor jaxtyping manifold3d
"$UV" pip install --python "$PYTHON" --no-deps -e "$AA_DIR" -e "$PROJECT_DIR"
"$UV" pip freeze --python "$PYTHON" > "$RUNTIME_DIR/environment-freeze.txt"
"$PYTHON" "$PROJECT_DIR/scripts/train.py" --cfg job > "$RUNTIME_DIR/train-config.yaml"
