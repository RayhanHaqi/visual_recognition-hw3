#!/bin/bash
# Full RunPod pipeline: ConvNeXt-Base → submit → git push → kill pod.
# Usage: bash train-runpod-convnext.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=3  lr=1e-4  wd=1e-4  workers=8  epochs=100

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-3}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-8}
EPOCHS=${5:-100}
RUN_NAME="convnext_bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}_runpod"

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

echo "[3/4] Generating submission..." && \
python submission.py \
    "./checkpoints/${RUN_NAME}_best.pth" \
    --min_size 800 --max_size 1333 \
    --anchor_sizes "8,16,32,64,128" \
    --box_detections_per_img 500 \
    --backbone convnext_base && \
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
