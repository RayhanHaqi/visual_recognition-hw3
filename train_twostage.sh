#!/bin/bash
set -euo pipefail

# HW3 Two-Stage Training Pipeline
# 1. Train 250 epochs with 80/20 split, extract best epoch by val AP50
# 2. Retrain on all 209 images for best_epoch+1 epochs
# 3. Generate submission
#
# Usage: bash train_twostage.sh [gpu=0]

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=42
EPOCHS=100
BACKBONE=convnext_base
BATCH_SIZE=2
LR=7e-4
WD=3e-3
PCT_START=0.5
VAL_FRAC=0.3
GPU=${1:-1}
DATA_PATH="datasets/train"

echo "=================================================="
echo "HW3 TWO-STAGE PIPELINE"
echo "GPU=${GPU}  BS=${BATCH_SIZE}  LR=${LR}  WD=${WD}  EPOCHS=${EPOCHS}  VAL=${VAL_FRAC}"
echo "=================================================="

# Stage 1: Train with validation split
echo ""
echo "[1/3] Training with val_frac=${VAL_FRAC} for ${EPOCHS} epochs..."
STAGE1_POSTFIX="bs${BATCH_SIZE}_lr${LR}_wd${WD}_ep${EPOCHS}_v${VAL_FRAC/./p}"
STAGE1_NAME="convnext_${STAGE1_POSTFIX}"

python train_v2.py \
    --run_name "${STAGE1_NAME}" \
    --backbone ${BACKBONE} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --epochs ${EPOCHS} \
    --pct_start ${PCT_START} \
    --gpu ${GPU} \
    --seed ${SEED} \
    --val_frac ${VAL_FRAC} \
    --patience 99999 \
    --best_only \
    --workers 16

STAGE1_CSV="log/${STAGE1_NAME}.csv"
if [ ! -f "${STAGE1_CSV}" ]; then
    echo "ERROR: Missing CSV log ${STAGE1_CSV}"
    exit 1
fi

BEST_ROW=$(tail -n +2 "${STAGE1_CSV}" | sort -t',' -k9 -nr | head -1)
BEST_AP50=$(echo "${BEST_ROW}" | cut -d',' -f9)
BEST_EPOCH=$(echo "${BEST_ROW}" | cut -d',' -f1)
echo "  Best val AP50=${BEST_AP50} at epoch ${BEST_EPOCH}"

TARGET_EPOCHS=$((BEST_EPOCH + 1))

# Stage 2: Train on all data
echo ""
echo "[2/3] Training full-data model for ${TARGET_EPOCHS} epochs (best_epoch=${BEST_EPOCH} + 1)..."
FINAL_POSTFIX="full_bestep${BEST_EPOCH}_bs${BATCH_SIZE}_lr${LR}_wd${WD}"
FINAL_NAME="convnext_${FINAL_POSTFIX}"

python train_v2.py \
    --run_name "${FINAL_NAME}" \
    --backbone ${BACKBONE} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --epochs ${TARGET_EPOCHS} \
    --pct_start ${PCT_START} \
    --gpu ${GPU} \
    --seed ${SEED} \
    --val_frac 0.0 \
    --workers 16

# Stage 3: Generate submission
echo ""
echo "[3/3] Generating submission..."
FINAL_CKPT="checkpoints/${FINAL_NAME}_ep$(printf "%03d" ${TARGET_EPOCHS}).pth"

if [ -f "${FINAL_CKPT}" ]; then
    echo "  Submitting: ${FINAL_CKPT}"
    python submission.py "${FINAL_CKPT}" --backbone ${BACKBONE} \
        --min_size 800 --max_size 1333 \
        --anchor_sizes "8,16,32,64,128" \
        --box_detections_per_img 500 \
        --tag "twostage"
else
    echo "  ERROR: Final checkpoint not found at ${FINAL_CKPT}"
    exit 1
fi

echo ""
echo "=================================================="
echo "Pipeline complete."
echo "Best val AP50=${BEST_AP50} at epoch ${BEST_EPOCH}"
echo "Full model trained for ${TARGET_EPOCHS} epochs"
echo "=================================================="
