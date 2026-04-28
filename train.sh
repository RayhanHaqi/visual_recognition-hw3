#!/bin/bash
set -e

BS=${1:-2}
LR=${2:-1e-4}
WD=${3:-1e-4}
GPUS=${4:-2}
WORKER=${5:-4}
EPOCHS=${6:-100}

RUN_NAME="bs${BS}_lr${LR}_wd${WD}"

echo "=================================================="
echo "HW3 PIPELINE: $RUN_NAME  (gpus=$GPUS)"
echo "=================================================="

torchrun --standalone --nproc_per_node=$GPUS train.py \
  --batch_size $BS --lr $LR --wd $WD --workers $WORKER \
  --epochs $EPOCHS --run_name $RUN_NAME

python submission.py ./checkpoints/${RUN_NAME}_best.pth

echo "Pipeline done."
