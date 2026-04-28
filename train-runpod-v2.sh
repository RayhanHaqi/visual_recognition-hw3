#!/bin/bash
# Full RunPod pipeline (v2): setup → train_v2 → submit → git push → kill pod.
# Usage: bash train-runpod-v2.sh [bs] [lr] [wd] [workers] [epochs] [github_token]
#   bs=2  lr=1e-4  wd=1e-4  workers=4  epochs=100  github_token=

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BS=${1:-2}
LR=${2:-1e-4}
WD=${3:-1e-4}
WORKER=${4:-4}
EPOCHS=${5:-100}
GH_TOKEN=${6:-}
SETUP_EXTRA=""
if [ -n "$GH_TOKEN" ]; then
    SETUP_EXTRA="--github-token $GH_TOKEN"
fi
RUN_NAME="v2_bs${BS}_lr${LR}_wd${WD}_ep${EPOCHS}_runpod"

echo "=================================================="
echo "  RUNPOD v2 PIPELINE: $RUN_NAME"
echo "=================================================="

echo "[1/4] Setup..."
python setup.py $SETUP_EXTRA && \

echo "[2/4] Training (v2)..." && \
python train_v2.py \
  --batch_size "$BS" --lr "$LR" --wd "$WD" --workers "$WORKER" \
  --epochs "$EPOCHS" --run_name "$RUN_NAME" && \

echo "[3/4] Generating submission..." && \
python submission.py "./checkpoints/${RUN_NAME}_best.pth" && \

echo "[4/4] Saving to GitHub..." && \
rm -f ./checkpoints/*_last.pth && \
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
