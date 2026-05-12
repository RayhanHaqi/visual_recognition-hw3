#!/bin/bash
# Full RunPod pipeline: setup → train → submit → git push → kill pod.
# Usage: bash train-runpod.sh [bs] [lr] [wd] [workers] [epochs]
#   bs=6  lr=1e-4  wd=1e-4  workers=8  epochs=100

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-6}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-8}
EPOCHS=${5:-100}
RUN_NAME="bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}_runpod"

echo "=================================================="
echo "  RUNPOD PIPELINE: $RUN_NAME"
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

echo "[2/4] Training..." && \
python train.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME" --best_only && \

echo "[3/4] Generating submission..." && \
python submission.py "./checkpoints/${RUN_NAME}_best.pth" && \
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
