#!/bin/bash
# Full pipeline: ConvNeXt-Base → submit → git commit/push (single GPU, no DDP).
# Usage: bash train-convnext.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=2  lr=1e-4  wd=1e-4  workers=8  epochs=200

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-1}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-8}
EPOCHS=${5:-150}
TRACKER="run_tracker_convnext.txt"
if [ -f "$TRACKER" ]; then RUN_NUM=$(($(cat "$TRACKER") + 1)); else RUN_NUM=1; fi
echo "$RUN_NUM" > "$TRACKER"
RUN_NAME="convnext_bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}_run${RUN_NUM}"

echo "=================================================="
echo "HW3 CONVNEXT PIPELINE: $RUN_NAME  (single-GPU)"
echo "=================================================="

# Step 1: train with plain python (no torchrun / DDP)
python train_v2.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME" \
  --backbone convnext_base

# Step 2: generate CodaBench submission ZIP
python submission.py \
    "./checkpoints/${RUN_NAME}_best.pth" \
    --min_size 800 --max_size 1333 \
    --anchor_sizes "8,16,32,64,128" \
    --box_detections_per_img 500

# Step 3: auto-commit results and push to remote
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add -A
  git commit -m "pipeline: $RUN_NAME" || echo "(nothing to commit)"
  git pull --rebase || echo "(pull skipped)"
  git push || echo "(push skipped — no remote or permission)"
fi

echo "Pipeline done."
