# Cell Instance Segmentation - Mask R-CNN
**NYCU Visual Recognition using Deep Learning (Spring 2026) - Homework 3**

[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Mask%20R--CNN-blue?style=for-the-badge)](https://arxiv.org/abs/1703.06870)

This repository contains the implementation for the Cell Instance Segmentation task (Homework 3): segmenting four classes of cells in H&E-stained medical images, evaluated by **AP50** on a hidden CodaBench leaderboard.

## Introduction

The task ships 209 train images and 101 test images. Each training sample is a folder containing the source `image.tif` plus up to four mask files `class{1..4}.tif`, where every unique non-zero pixel value within a mask file represents one instance of that class.

Constraints (from the rubric):
- Must be Mask R-CNN-based.
- < 200M trainable params.
- Pure-vision only (no VLM / prompt models).
- ImageNet pretraining allowed.
- No external data.

This implementation uses `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` (~46M params) as the base, with replaced classification + mask heads for 4 cell classes + background.

## Environment Setup

```bash
cd HW3
python setup.py        # renames datesets→datasets, installs deps, sanity-checks dataset
```

Or manually:
```bash
pip install -r requirements.txt
```

Expected dataset layout (after setup):
```
HW3/datasets/
├── train/<UUID>/{image,class1,class2,class3,class4}.tif
├── test_release/<UUID>.tif
└── test_image_name_to_ids.json
```

## Usage

### Training

**Single-GPU:**
```bash
python train.py --run_name baseline --batch_size 2 --lr 1e-4 --wd 1e-4 --epochs 50
```

**Multi-GPU (DDP):**
```bash
bash train.sh <bs> <lr> <wd> <gpus> <workers> <epochs>
# e.g., bash train.sh 2 1e-4 1e-4 2 4 100
```

**Cloud pod (RunPod):**
```bash
bash train-runpod.sh <bs> <lr> <wd> <gpus> <workers> <epochs>
```

Per-epoch metrics (train loss, val AP / AP50 / AP75) are written to `log/<run_name>.csv`.
Checkpoints saved to `checkpoints/<run_name>_{best,last}.pth` (best-by-AP50).

### Inference & Submission

```bash
python submission.py checkpoints/<run_name>_best.pth --student_id <STUDENT_ID>
```

Produces `submission/test-results.json` (COCO RLE format, exact filename mandated by slides) and `submission/<STUDENT_ID>_HW3.zip` ready to upload to CodaBench.

## Performance Snapshot

| Run | Backbone | Val AP50 | CodaBench AP50 |
| :-- | :------: | :------: | :------------: |
| baseline | ResNet50-FPN-v2 | TBD | TBD |

Leaderboard screenshot:

`<insert when first submission lands>`

## Pre-Flight Reading

Before modifying the model:

1. **Mask R-CNN** (He et al. 2017, [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)) — RoIAlign + parallel mask head.
2. **`torchvision.models.detection.mask_rcnn`** source — trace `forward` to know where to swap components.
3. **`pycocotools.mask`** — RLE encode/decode (note: `counts` must be a `str`, not `bytes`, for submission).
4. **`pycocotools.COCOeval`** — `iouType="segm"`, AP50 is `stats[1]`.

## Future Experiments

- Backbone swap (ResNet101, ConvNeXt-tiny).
- Tiling pipeline for dense small instances.
- Anchor tuning (`anchor_sizes=((8,),(16,),(32,),(64,),(128,))`) — cells are smaller than COCO default scales.
- Dice loss for the mask head.
- 5-fold ensemble for the final submission.
- H&E-specific stain augmentation.

## Author

Muhammad Rayhan Athaillah (賴瑞涵)  
Student ID: 313540001  
Affiliation: National Yang Ming Chiao Tung University (NYCU)
