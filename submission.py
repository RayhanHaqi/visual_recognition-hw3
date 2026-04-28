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
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--student_id", type=str, default="313540001")
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

    model = build_maskrcnn(num_classes=5, pretrained=False, min_size=args.min_size, max_size=args.max_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
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
                    "category_id": int(labels[i]),
                    "bbox": _xyxy_to_xywh(boxes[i]),
                    "score": float(scores[i]),
                    "segmentation": encode_binary_mask(bin_mask),
                })

    json_path = out_dir / "test-results.json"
    with json_path.open("w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} predictions to {json_path}")

    zip_path = out_dir / f"{args.student_id}_HW3.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="test-results.json")
    print(f"Zipped submission to {zip_path}")


if __name__ == "__main__":
    main()
