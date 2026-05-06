#!/bin/bash
set -euo pipefail

# HW3 K-Fold Cross-Validation Pipeline
# 1. Generate stratified K-fold splits
# 2. Train K fold models (250 epochs each, ConvNeXt-Base)
# 3. Extract best fold and best epoch from CSV logs
# 4. Train final full-data model at the best epoch count
# 5. Generate submissions for best fold + final model
#
# Usage: bash train_kfold.sh [gpu=0]

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

K=5
SEED=42
EPOCHS=250
BACKBONE=convnext_base
BATCH_SIZE=2
LR=2e-4
WD=2e-3
PCT_START=0.5
GPU=${1:-1}
DATA_PATH="datasets/train"
KFOLD_SPLITS="log/kfold_splits.json"

echo "=================================================="
echo "HW3 K-FOLD (K=${K}) PIPELINE"
echo "GPU=${GPU}  BS=${BATCH_SIZE}  LR=${LR}  WD=${WD}  EPOCHS=${EPOCHS}"
echo "=================================================="

# Step 1: Generate stratified K-fold splits
echo "[1/5] Generating stratified ${K}-fold splits..."
python -c "
from data.dataset import generate_kfold_splits
splits = generate_kfold_splits('${DATA_PATH}', k=${K}, seed=${SEED}, output_path='${KFOLD_SPLITS}')
for k, v in splits.items():
    print(f'  {k}: train={len(v[\"train\"])}  val={len(v[\"val\"])}')
"

# Step 2: Train each fold
echo ""
echo "[2/5] Training ${K} fold models..."
BEST_AP50=0
BEST_FOLD=0
BEST_EPOCH=0

for FOLD in $(seq 0 $((K-1))); do
    echo ""
    echo "--- Fold ${FOLD}/${K} ---"
    POSTFIX="kf${K}_fold${FOLD}_bs${BATCH_SIZE}_lr${LR}_wd${WD}_ep${EPOCHS}"
    RUN_NAME="convnext_${POSTFIX}"

    python train_v2.py \
        --run_name "${RUN_NAME}" \
        --backbone ${BACKBONE} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --wd ${WD} \
        --epochs ${EPOCHS} \
        --pct_start ${PCT_START} \
        --gpu ${GPU} \
        --seed ${SEED} \
        --kfold_splits ${KFOLD_SPLITS} \
        --fold_idx ${FOLD}

    # Extract best AP50 and epoch from CSV
    FOLD_CSV="log/${RUN_NAME}.csv"
    if [ ! -f "${FOLD_CSV}" ]; then
        echo "ERROR: Missing CSV log ${FOLD_CSV}"
        continue
    fi

    # CSV columns: epoch, train_loss, loss_classifier, loss_box_reg, loss_mask,
    #              loss_objectness, loss_rpn_box_reg, val_AP, val_AP50, val_AP75,
    #              lr, grad_norm, secs
    # val_AP50 is column 9 (1-indexed)
    FOLD_BEST=$(tail -n +2 "${FOLD_CSV}" | sort -t',' -k9 -nr | head -1)
    FOLD_AP50=$(echo "${FOLD_BEST}" | cut -d',' -f9)
    FOLD_EPOCH=$(echo "${FOLD_BEST}" | cut -d',' -f1)
    echo "  Fold ${FOLD}: best AP50=${FOLD_AP50} at epoch ${FOLD_EPOCH}"

    if (( $(echo "${FOLD_AP50} > ${BEST_AP50}" | bc -l) )); then
        BEST_AP50=${FOLD_AP50}
        BEST_FOLD=${FOLD}
        BEST_EPOCH=${FOLD_EPOCH}
    fi
done

echo ""
echo "[3/5] Best fold: ${BEST_FOLD}  AP50=${BEST_AP50}  epoch=${BEST_EPOCH}"

# CSV epochs are 0-indexed; final model needs BEST_EPOCH+1 total epochs
TARGET_EPOCHS=$((BEST_EPOCH + 1))

# Step 4: Train final model on ALL data for TARGET_EPOCHS epochs
echo ""
echo "[4/5] Training final full-data model for ${TARGET_EPOCHS} epochs (best_epoch=${BEST_EPOCH} + 1)..."
FINAL_POSTFIX="kfold_full_bestep${BEST_EPOCH}_bs${BATCH_SIZE}_lr${LR}_wd${WD}"
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
    --val_frac 0.0

# Step 5: Generate submissions
echo ""
echo "[5/5] Generating submissions..."

# Best fold submission
PREFIX="convnext_kf${K}_fold${BEST_FOLD}_bs${BATCH_SIZE}_lr${LR}_wd${WD}_ep${EPOCHS}"
BEST_CKPT="checkpoints/${PREFIX}_best.pth"

if [ ! -f "${BEST_CKPT}" ]; then
    echo "  Best checkpoint not found at ${BEST_CKPT}, searching..."
    BEST_CKPT=$(ls checkpoints/${PREFIX}_top_ep*ap*.pth 2>/dev/null | sort | head -1 || echo "")
fi

if [ -n "${BEST_CKPT}" ] && [ -f "${BEST_CKPT}" ]; then
    echo "  Submitting best fold (fold ${BEST_FOLD}): ${BEST_CKPT}"
    python submission.py "${BEST_CKPT}" --backbone ${BACKBONE} \
        --min_size 800 --max_size 1333 \
        --anchor_sizes "8,16,32,64,128" \
        --box_detections_per_img 500 \
        --tag "kfold_best_fold${BEST_FOLD}"
else
    echo "  WARNING: Could not find best fold checkpoint for fold ${BEST_FOLD}"
fi

# Final full-data model submission
FINAL_CKPT="checkpoints/${FINAL_NAME}_ep${TARGET_EPOCHS}.pth"
if [ -f "${FINAL_CKPT}" ]; then
    echo "  Submitting final full-data model: ${FINAL_CKPT}"
    python submission.py "${FINAL_CKPT}" --backbone ${BACKBONE} \
        --min_size 800 --max_size 1333 \
        --anchor_sizes "8,16,32,64,128" \
        --box_detections_per_img 500 \
        --tag "kfold_full"
else
    echo "  WARNING: Final checkpoint not found at ${FINAL_CKPT}"
fi

echo ""
echo "=================================================="
echo "Pipeline complete."
echo "Best fold: ${BEST_FOLD}  AP50=${BEST_AP50}  epoch=${BEST_EPOCH}"
echo "=================================================="
