#!/bin/bash
# Full pipeline for cloud pods (RunPod): setup → train → submit (single GPU, no DDP).
# Usage: bash train-runpod.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=2  lr=1e-4  wd=1e-4  workers=4  epochs=100
#
# Note: this was rewritten from a multi-GPU torchrun wrapper to a
# single-GPU pipeline script because DDP is not needed for this task.
set -e

BS=${1:-2}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-4}
EPOCHS=${5:-100}
RUN_NAME="bs${BS}_lr${LR}_wd${WD}"

echo "[runpod] Setup..."
python setup.py

echo "[runpod] Train ($RUN_NAME, single-GPU)..."
python train.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME"

echo "[runpod] Generate submission..."
python submission.py "./checkpoints/${RUN_NAME}_best.pth"

echo "[runpod] Cleaning *_last.pth..."
rm -f ./checkpoints/*_last.pth || true

echo "[runpod] Done."
