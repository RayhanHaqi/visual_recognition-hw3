#!/bin/bash
# Full pipeline: train_v2 → submit → git commit/push (single GPU, no DDP).
# Usage: bash train-v2.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=2  lr=1e-4  wd=1e-4  workers=4  epochs=100
#
# Note: this was rewritten from a multi-GPU torchrun wrapper to a
# single-GPU pipeline script because DDP is not needed for this task.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-2}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-2}
EPOCHS=${5:-100}
RUN_NAME="v2_bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}"

echo "=================================================="
echo "HW3 v2 PIPELINE: $RUN_NAME  (single-GPU)"
echo "=================================================="

# Step 1: train with plain python (no torchrun / DDP)
python train_v2.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME"

# Step 2: generate CodaBench submission ZIP
python submission.py "./checkpoints/${RUN_NAME}_best.pth"

# Step 3: auto-commit results and push to remote
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add -A
  git commit -m "pipeline: $RUN_NAME" || echo "(nothing to commit)"
  git pull --rebase || echo "(pull skipped)"
  git push || echo "(push skipped — no remote or permission)"
fi

echo "Pipeline done."
