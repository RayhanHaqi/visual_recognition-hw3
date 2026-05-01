#!/bin/bash
# Full RunPod pipeline: ConvNeXt-Base → submit → git push → kill pod.
# Usage: bash train-runpod-convnext.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=3  lr=2e-4  wd=2e-3  workers=8  epochs=150
...
BS=${1:-3}
LR=${2:-2e-4}
WD=${3:-2e-3}
WORKER=${4:-8}
EPOCHS=${5:-150}
TRACKER="run_tracker_convnext.txt"
if [ -f "$TRACKER" ]; then RUN_NUM=$(($(cat "$TRACKER") + 1)); else RUN_NUM=1; fi
echo "$RUN_NUM" > "$TRACKER"
RUN_NAME="convnext_bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}_run${RUN_NUM}_runpod"

echo "=================================================="
echo "  RUNPOD CONVNEXT PIPELINE: $RUN_NAME"
echo "=================================================="

if [ -z "$GH_TOKEN" ]; then
    read -rp "GitHub token: " GH_TOKEN
    echo
fi

if [ -n "$GH_TOKEN" ]; then
    echo "https://Rayhan:${GH_TOKEN}@github.com" > ~/.git-credentials
fi

echo "[1/4] Setup..."
python setup.py && \

echo "[2/4] Training (ConvNeXt-Base)..." && \
python train_v2.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME" --best_only \
  --backbone convnext_base && \

echo "[3/4] Generating submissions..." && \
for EP in 050 075 100 125 150; do
    CKPT="./checkpoints/${RUN_NAME}_ep${EP}.pth"
    if [ -f "$CKPT" ]; then
        python submission.py \
            "$CKPT" \
            --min_size 800 --max_size 1333 \
            --anchor_sizes "8,16,32,64,128" \
            --box_detections_per_img 500 \
            --backbone convnext_base
        mv submission/submission.zip "submission/${RUN_NAME}_ep${EP}_HW3.zip" 2>/dev/null || true
    fi
done
for PLAT_CKPT in ./checkpoints/${RUN_NAME}_plateau_ep*.pth; do
    if [ -f "$PLAT_CKPT" ]; then
        PLAT_EP=$(basename "$PLAT_CKPT" .pth | grep -oP 'ep\d+$')
        python submission.py \
            "$PLAT_CKPT" \
            --min_size 800 --max_size 1333 \
            --anchor_sizes "8,16,32,64,128" \
            --box_detections_per_img 500 \
            --backbone convnext_base
        mv submission/submission.zip "submission/${RUN_NAME}_${PLAT_EP}_HW3.zip" 2>/dev/null || true
    fi
done && \
rm -rf ./checkpoints/* && \

echo "[4/4] Saving to GitHub..." && \
git add -A && \
git commit -m "Auto-save: Done training $RUN_NAME" && \
git pull --rebase && \
git push && \

echo "Pipeline done!"

sleep 60

if [ -z "$RUNPOD_POD_ID" ]; then
    echo "RUNPOD_POD_ID not set — skipping pod deletion."
else
    echo "Deleting pod $RUNPOD_POD_ID..."
    runpodctl remove pod "$RUNPOD_POD_ID"
fi
