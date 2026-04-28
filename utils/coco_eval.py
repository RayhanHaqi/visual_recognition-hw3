import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .rle import encode_binary_mask


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def build_coco_gt(dataset):
    images, annotations = [], []
    ann_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        masks = target["masks"].numpy().astype(np.uint8) if target["masks"].numel() else np.zeros((0,), dtype=np.uint8)
        boxes = target["boxes"].numpy() if target["boxes"].numel() else np.zeros((0, 4), dtype=np.float32)
        labels = target["labels"].numpy() if target["labels"].numel() else np.zeros((0,), dtype=np.int64)
        if masks.ndim == 3 and masks.shape[0] > 0:
            h, w = masks.shape[1], masks.shape[2]
        else:
            shape_ref = dataset[idx][0].shape
            h, w = int(shape_ref[-2]), int(shape_ref[-1])
        images.append({"id": int(idx), "height": int(h), "width": int(w)})
        for i in range(boxes.shape[0]):
            seg = encode_binary_mask(masks[i]) if masks.shape[0] else None
            annotations.append({
                "id": ann_id,
                "image_id": int(idx),
                "category_id": int(labels[i]),
                "bbox": _xyxy_to_xywh(boxes[i]),
                "area": float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])),
                "iscrowd": 0,
                "segmentation": seg,
            })
            ann_id += 1
    categories = [{"id": c, "name": f"class{c}"} for c in (1, 2, 3, 4)]
    coco_dict = {"images": images, "annotations": annotations, "categories": categories}
    coco = COCO()
    coco.dataset = coco_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def predictions_to_coco_results(predictions):
    results = []
    for image_id, pred in predictions.items():
        boxes = pred["boxes"].cpu().numpy() if hasattr(pred["boxes"], "cpu") else np.asarray(pred["boxes"])
        scores = pred["scores"].cpu().numpy() if hasattr(pred["scores"], "cpu") else np.asarray(pred["scores"])
        labels = pred["labels"].cpu().numpy() if hasattr(pred["labels"], "cpu") else np.asarray(pred["labels"])
        masks = pred["masks"].cpu().numpy() if hasattr(pred["masks"], "cpu") else np.asarray(pred["masks"])
        if masks.ndim == 4:
            masks = masks[:, 0]
        for i in range(boxes.shape[0]):
            results.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": _xyxy_to_xywh(boxes[i]),
                "score": float(scores[i]),
                "segmentation": encode_binary_mask((masks[i] >= 0.5).astype(np.uint8)),
            })
    return results


def evaluate_segm(coco_gt, coco_results):
    if len(coco_results) == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats
    return {"AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2])}
