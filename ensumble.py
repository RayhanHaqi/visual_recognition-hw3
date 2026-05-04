import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from torchvision.ops import nms
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=str, help="JSON files or ZIP files to ensemble")
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--out", type=str, default="submission/ensemble.zip")
    p.add_argument("--score_thresh", type=float, default=0.05)
    return p.parse_args()


def _load_predictions(path):
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            json_name = next((n for n in names if n.endswith(".json")), names[0])
            with z.open(json_name) as f:
                return json.load(f)
    else:
        with p.open() as f:
            return json.load(f)


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def main():
    args = parse_args()

    all_predictions = []
    for path in args.inputs:
        preds = _load_predictions(path)
        all_predictions.append(preds)
        print(f"  Loaded {path}: {len(preds)} preds")

    by_image = defaultdict(list)
    for preds in all_predictions:
        for p in preds:
            by_image[p["image_id"]].append(p)

    merged = []
    for image_id, preds in by_image.items():
        boxes = torch.tensor([_xywh_to_xyxy(p["bbox"]) for p in preds], dtype=torch.float32)
        scores = torch.tensor([p["score"] for p in preds])
        labels = torch.tensor([p["category_id"] for p in preds])

        keep_all = []
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            keep = nms(boxes[idx], scores[idx], args.iou_thresh)
            keep_all.append(idx[keep])
        keep_all = torch.cat(keep_all) if keep_all else torch.zeros((0,), dtype=torch.long)

        for i in keep_all.tolist():
            p = preds[i]
            if p["score"] >= args.score_thresh:
                merged.append(p)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(merged, f)
    print(f"Wrote {len(merged)} ensemble predictions to {json_path}")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="test-results.json")
    json_path.unlink()
    print(f"Zipped to {out_path}")


if __name__ == "__main__":
    main()
