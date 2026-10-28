#!/usr/bin/env bash
set -euo pipefail
RUNTIME_DIR=${RUNTIME_DIR:?set task runtime directory}
GPU=${GPU:-2}
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR="$PROJECT_DIR/third_party/FlashDexRetarget"
used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 500 ]; then
  echo "GPU $GPU is occupied ($used MiB); choose an idle GPU" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMNI_KIT_ACCEPT_EULA=1 MUJOCO_GL=disabled
cd "$SOURCE_DIR"
"$RUNTIME_DIR/venv/bin/python" src/simulator/isaacsim/convert_assets.py \
  --robot --robot-variant bimanual --robot-suffix _spec --headless
# The shared sync script deletes remote-only generated directories. Keep a
# durable asset copy outside its workspace; USD sublayers use relative paths.
mkdir -p "$RUNTIME_DIR/assets"
cp -a assets/robot/xhand/urdf/converted_spec "$RUNTIME_DIR/assets/"
echo "FDR_ROBOT_USD=$RUNTIME_DIR/assets/converted_spec/xhand_bimanual_spec.usd"
