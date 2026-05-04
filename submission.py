import argparse
import gc
import json
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import nms

from data.dataset import _read_tif
from model.build import build_maskrcnn
from utils.rle import encode_binary_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--test_path", type=str, default="datasets/test_release")
    p.add_argument("--ids_json", type=str, default="datasets/test_image_name_to_ids.json")
    p.add_argument("--out_dir", type=str, default="submission")
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--mask_thresh", type=float, default=0.5)
    p.add_argument("--min_size", type=int, default=512)
    p.add_argument("--max_size", type=int, default=1024)
    p.add_argument("--anchor_sizes", type=str, default=None,
                   help="Comma-separated sizes, e.g. '8,16,32,64,128'")
    p.add_argument("--box_detections_per_img", type=int, default=100)
    p.add_argument("--backbone", type=str, default=None, choices=["resnet50", "convnext_base"])
    p.add_argument("--tta", action="store_true", default=False, help="Horizontal flip test-time augmentation")
    p.add_argument("--ms_tta", action="store_true", default=False, help="Multi-scale TTA (640/800/960) min_size")
    p.add_argument("--tta_iou_thresh", type=float, default=0.5, help="NMS IoU threshold for TTA merge")
    p.add_argument("--tag", type=str, default="", help="Suffix for output ZIP filename")
    p.add_argument("--gpu", type=int, default=0)

    return p.parse_args()


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, x2 - x1, y2 - y1]


def _load_id_map(ids_json):
    with open(ids_json) as f:
        entries = json.load(f)
    return {e["file_name"]: int(e["id"]) for e in entries}


def hflip_predictions(preds, image_widths):
    out = []
    for pred, w in zip(preds, image_widths):
        boxes = pred["boxes"].clone()
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
        masks = torch.flip(pred["masks"], dims=[-1])
        out.append({
            "boxes": boxes,
            "scores": pred["scores"],
            "labels": pred["labels"],
            "masks": masks,
        })
    return out


def merge_tta(orig, flipped, iou_thresh=0.5):
    merged = []
    for o, f in zip(orig, flipped):
        boxes = torch.cat([o["boxes"], f["boxes"]], dim=0)
        scores = torch.cat([o["scores"], f["scores"]], dim=0)
        labels = torch.cat([o["labels"], f["labels"]], dim=0)
        masks = torch.cat([o["masks"], f["masks"]], dim=0)
        keep_all = []
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            keep = nms(boxes[idx], scores[idx], iou_thresh)
            keep_all.append(idx[keep])
        keep_all = torch.cat(keep_all) if keep_all else torch.zeros((0,), dtype=torch.long)
        merged.append({
            "boxes": boxes[keep_all],
            "scores": scores[keep_all],
            "labels": labels[keep_all],
            "masks": masks[keep_all],
        })
    return merged


def _merge_outputs(outputs, iou_thresh):
    boxes = torch.cat([o["boxes"] for o in outputs], dim=0)
    scores = torch.cat([o["scores"] for o in outputs], dim=0)
    labels = torch.cat([o["labels"] for o in outputs], dim=0)

    keep_all = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        keep = nms(boxes[idx], scores[idx], iou_thresh)
        keep_all.append(idx[keep])
    keep_all = torch.cat(keep_all) if keep_all else torch.zeros((0,), dtype=torch.long)

    masks_list = [o["masks"] for o in outputs]
    target_shape = masks_list[0].shape[-2:]
    offset = 0
    kept_masks = []
    for m in masks_list:
        n = m.shape[0]
        mask_keep = keep_all[(keep_all >= offset) & (keep_all < offset + n)] - offset
        if mask_keep.numel() > 0:
            m_sel = m[mask_keep]
            if m_sel.shape[-2:] != target_shape:
                m_sel = F.interpolate(m_sel, size=target_shape, mode="bilinear", align_corners=False)
            kept_masks.append(m_sel)
        offset += n
    masks = torch.cat(kept_masks, dim=0) if kept_masks else masks_list[0][:0]

    return {
        "boxes": boxes[keep_all],
        "scores": scores[keep_all],
        "labels": labels[keep_all],
        "masks": masks,
    }


def _to_cpu(out):
    return {k: v.cpu() for k, v in out.items()}


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    train_args = {}
    if isinstance(ckpt, dict) and "args" in ckpt:
        train_args = ckpt["args"]

    if isinstance(ckpt, dict) and "model_args" in ckpt:
        m = ckpt["model_args"]
        min_size = m.get("min_size", args.min_size)
        max_size = m.get("max_size", args.max_size)
        anchor_sizes = m.get("anchor_sizes", None)
        bdpi = m.get("box_detections_per_img", 100)
        backbone = m.get("backbone", "resnet50")
    else:
        min_size = train_args.get("min_size", args.min_size)
        max_size = train_args.get("max_size", args.max_size)
        anchor_sizes = train_args.get("anchor_sizes")
        bdpi = train_args.get("box_detections_per_img") or 100
        backbone = train_args.get("backbone", "resnet50")

    if args.anchor_sizes is not None:
        anchor_sizes = tuple((int(x),) for x in args.anchor_sizes.split(","))

    if args.backbone is not None:
        backbone = args.backbone

    model_kwargs = {
        "num_classes": 5, "pretrained": False,
        "min_size": min_size, "max_size": max_size,
        "box_detections_per_img": bdpi,
        "backbone": backbone,
    }
    if anchor_sizes is not None:
        model_kwargs["anchor_sizes"] = anchor_sizes

    print(f"Model config: backbone={backbone} min_size={min_size}, max_size={max_size}, anchors={'custom' if anchor_sizes else 'default'}, box_detections={bdpi}")

    model = build_maskrcnn(**model_kwargs)
    model.load_state_dict(state)
    model.to(device).eval()

    id_map = _load_id_map(args.ids_json)
    test_dir = Path(args.test_path)

    json_path = out_dir / "test-results.json"
    jsonl_path = out_dir / "test-results.jsonl"

    total = 0
    with jsonl_path.open("w") as fh:
        with torch.no_grad():
            for fpath in tqdm(sorted(test_dir.glob("*.tif")), desc="infer"):
                fname = fpath.name
                if fname not in id_map:
                    print(f"WARN: {fname} not in id map; skipping")
                    continue
                image_id = id_map[fname]
                arr = _read_tif(fpath)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                img_t = torch.from_numpy(arr.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
                img_t = img_t.to(device)
                W = img_t.shape[-1]

                cpu_outputs = [_to_cpu(model([img_t])[0])]

                if args.tta:
                    flipped = torch.flip(img_t, dims=[-1])
                    flipped_out = model([flipped])[0]
                    unflipped = hflip_predictions([flipped_out], [W])[0]
                    del flipped_out
                    cpu_outputs.append(_to_cpu(unflipped))
                    del unflipped

                if args.ms_tta:
                    orig_transform = model.transform
                    try:
                        for s in [640, 960]:
                            model.transform = GeneralizedRCNNTransform(
                                min_size=s, max_size=args.max_size or max_size,
                                image_mean=orig_transform.image_mean,
                                image_std=orig_transform.image_std,
                            )
                            out = model([img_t])[0]
                            cpu_outputs.append(_to_cpu(out))
                            del out
                            if args.tta:
                                flipped_out = model([flipped])[0]
                                unflipped = hflip_predictions([flipped_out], [W])[0]
                                del flipped_out
                                cpu_outputs.append(_to_cpu(unflipped))
                                del unflipped
                    finally:
                        model.transform = orig_transform

                if len(cpu_outputs) > 1:
                    output = _merge_outputs(cpu_outputs, iou_thresh=args.tta_iou_thresh)
                else:
                    output = cpu_outputs[0]
                del cpu_outputs

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()
                if masks.ndim == 4:
                    masks = masks[:, 0]
                keep = scores >= args.score_thresh
                boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
                for i in range(boxes.shape[0]):
                    bin_mask = (masks[i] >= args.mask_thresh).astype(np.uint8)
                    if bin_mask.sum() == 0:
                        continue
                    fh.write(json.dumps({
                        "image_id": image_id,
                        "bbox": _xyxy_to_xywh(boxes[i]),
                        "score": float(scores[i]),
                        "category_id": int(labels[i]),
                        "segmentation": encode_binary_mask(bin_mask),
                    }) + "\n")
                    total += 1
                del output, boxes, scores, labels, masks
                gc.collect()

    # Convert JSONL to JSON
    results = []
    with jsonl_path.open() as fh:
        for line in fh:
            results.append(json.loads(line))
    jsonl_path.unlink()

    with json_path.open("w") as f:
        json.dump(results, f)
    print(f"Wrote {total} predictions to {json_path}")

    stem = Path(args.checkpoint).stem
    suffix = ""
    if args.tta:
        suffix += "_tta"
    if args.ms_tta:
        suffix += "_ms"
    if args.tag:
        suffix += "_" + args.tag
    zip_path = out_dir / f"{stem}{suffix}_HW3.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="test-results.json")
    print(f"Zipped submission to {zip_path}")


if __name__ == "__main__":
    main()
