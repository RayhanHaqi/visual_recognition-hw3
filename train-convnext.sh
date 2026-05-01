#!/bin/bash
# Full pipeline: ConvNeXt-Base → submit → git commit/push (single GPU, no DDP).
# Usage: bash train-convnext.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=1  lr=2e-4  wd=1e-3  workers=8  epochs=150

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-1}
LR=${2:-2e-4}
WD=${3:-1e-3}
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

# Step 2: generate CodaBench submission ZIPs for periodic checkpoints
for EP in 050 075 100 125 150; do
    CKPT="./checkpoints/${RUN_NAME}_ep${EP}.pth"
    if [ -f "$CKPT" ]; then
        echo "  -> Submitting epoch $EP"
        python submission.py \
            "$CKPT" \
            --min_size 800 --max_size 1333 \
            --anchor_sizes "8,16,32,64,128" \
            --box_detections_per_img 500
        mv submission/submission.zip "submission/${RUN_NAME}_ep${EP}_HW3.zip" 2>/dev/null || true
    else
        echo "  -> Skip epoch $EP (checkpoint not found)"
    fi
done

# Step 3: auto-commit results and push to remote
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add -A
  git commit -m "pipeline: $RUN_NAME" || echo "(nothing to commit)"
  git pull --rebase || echo "(pull skipped)"
  git push || echo "(push skipped — no remote or permission)"
fi

echo "Pipeline done."
