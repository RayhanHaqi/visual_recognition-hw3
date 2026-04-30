import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    from PIL import Image
    _HAS_TIFFFILE = False

import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_CLASSES = 5  # 4 cell classes + background
CLASS_FILES = ["class1.tif", "class2.tif", "class3.tif", "class4.tif"]


def _read_tif(path):
    if _HAS_TIFFFILE:
        return tifffile.imread(str(path))
    return np.array(Image.open(str(path)))


def make_splits(root, seed=42, val_frac=0.15):
    root = Path(root)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(folders)
    n_val = int(round(len(folders) * val_frac))
    val_ids = sorted(folders[:n_val])
    train_ids = sorted(folders[n_val:])
    return train_ids, val_ids


def build_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
            A.GaussNoise(p=0.2),
            ToTensorV2(),
        ]
    )


def build_val_transform():
    return A.Compose([ToTensorV2()])


def build_train_transform_v2():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.OneOf([A.MotionBlur(p=1.0), A.MedianBlur(blur_limit=5, p=1.0), A.GaussianBlur(p=1.0)], p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.2),
            A.GaussNoise(p=0.2),
            ToTensorV2(),
        ]
    )


class CellInstanceDataset(Dataset):
    def __init__(self, root, image_ids, transform=None):
        self.root = Path(root)
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def _load_instances(self, sample_dir):
        boxes, labels, masks = [], [], []
        for class_idx, fname in enumerate(CLASS_FILES, start=1):
            mask_path = sample_dir / fname
            if not mask_path.exists():
                continue
            mask_arr = _read_tif(mask_path)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            inst_ids = np.unique(mask_arr)
            inst_ids = inst_ids[inst_ids != 0]
            for inst_id in inst_ids:
                bin_mask = (mask_arr == inst_id).astype(np.uint8)
                ys, xs = np.where(bin_mask > 0)
                if ys.size == 0:
                    continue
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(class_idx)
                masks.append(bin_mask)
        return boxes, labels, masks

    def __getitem__(self, idx):
        sample_id = self.image_ids[idx]
        sample_dir = self.root / sample_id
        image = _read_tif(sample_dir / "image.tif")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.astype(np.uint8)

        boxes, labels, masks = self._load_instances(sample_dir)

        if self.transform is not None:
            if len(masks) == 0:
                out = self.transform(image=image)
                image_t = out["image"]
                masks_t = torch.zeros((0, image_t.shape[-2], image_t.shape[-1]), dtype=torch.uint8)
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
            else:
                out = self.transform(image=image, masks=masks)
                image_t = out["image"]
                masks_np = np.stack(out["masks"], axis=0).astype(np.uint8)
                masks_t = torch.from_numpy(masks_np)
                boxes_list, labels_list, mask_keep = [], [], []
                for i in range(masks_t.shape[0]):
                    ys, xs = torch.where(masks_t[i] > 0)
                    if ys.numel() == 0:
                        continue
                    x1, x2 = xs.min().item(), xs.max().item()
                    y1, y2 = ys.min().item(), ys.max().item()
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
                    labels_list.append(labels[i])
                    mask_keep.append(i)
                masks_t = masks_t[mask_keep] if mask_keep else torch.zeros((0, image_t.shape[-2], image_t.shape[-1]), dtype=torch.uint8)
                boxes_t = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.tensor(labels_list, dtype=torch.int64) if labels_list else torch.zeros((0,), dtype=torch.int64)
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.from_numpy(np.stack(masks, axis=0)) if masks else torch.zeros((0, image_t.shape[-2], image_t.shape[-1]), dtype=torch.uint8)

        if image_t.dtype != torch.float32:
            image_t = image_t.float()
        if image_t.max() > 1.5:
            image_t = image_t / 255.0

        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64),
        }
        return image_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
