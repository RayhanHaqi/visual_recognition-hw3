import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

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
    p.add_argument("--gpu", type=int, default=0)

    return p.parse_args()


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, x2 - x1, y2 - y1]


def _load_id_map(ids_json):
    with open(ids_json) as f:
        entries = json.load(f)
    return {e["file_name"]: int(e["id"]) for e in entries}


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
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
    else:
        min_size = train_args.get("min_size", args.min_size)
        max_size = train_args.get("max_size", args.max_size)
        anchor_sizes = train_args.get("anchor_sizes")
        bdpi = train_args.get("box_detections_per_img") or 100

    if args.anchor_sizes is not None:
        anchor_sizes = tuple((int(x),) for x in args.anchor_sizes.split(","))

    model_kwargs = {
        "num_classes": 5, "pretrained": False,
        "min_size": min_size, "max_size": max_size,
        "box_detections_per_img": bdpi,
    }
    if anchor_sizes is not None:
        model_kwargs["anchor_sizes"] = anchor_sizes

    print(f"Model config: min_size={min_size}, max_size={max_size}, anchors={'custom' if anchor_sizes else 'default'}, box_detections={bdpi}")

    model = build_maskrcnn(**model_kwargs)
    model.load_state_dict(state)
    model.to(device).eval()

    id_map = _load_id_map(args.ids_json)
    test_dir = Path(args.test_path)

    results = []
    with torch.no_grad():
        for fpath in sorted(test_dir.glob("*.tif")):
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
            outputs = model([img_t.to(device)])[0]
            boxes = outputs["boxes"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()
            masks = outputs["masks"].cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            keep = scores >= args.score_thresh
            boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
            for i in range(boxes.shape[0]):
                bin_mask = (masks[i] >= args.mask_thresh).astype(np.uint8)
                if bin_mask.sum() == 0:
                    continue
                results.append({
                    "image_id": image_id,
                    "bbox": _xyxy_to_xywh(boxes[i]),
                    "score": float(scores[i]),
                    "category_id": int(labels[i]),
                    "segmentation": encode_binary_mask(bin_mask),
                })

    json_path = out_dir / "test-results.json"
    with json_path.open("w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} predictions to {json_path}")

    stem = Path(args.checkpoint).stem
    zip_path = out_dir / f"{stem}_HW3.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="test-results.json")
    print(f"Zipped submission to {zip_path}")


if __name__ == "__main__":
    main()
