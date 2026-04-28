import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops import nms
from tqdm import tqdm

from data.dataset import (
    CellInstanceDataset,
    build_train_transform_v2,
    build_val_transform,
    collate_fn,
    make_splits,
)
from model.build import build_maskrcnn, count_trainable_params
from utils.coco_eval import build_coco_gt, evaluate_segm, predictions_to_coco_results
from utils.ddp import cleanup_distributed, init_distributed, is_main


LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="v2")
    p.add_argument("--data_path", type=str, default="datasets/train")
    p.add_argument("--save_path", type=str, default="checkpoints")
    p.add_argument("--log_path", type=str, default="log")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--min_size", type=int, default=800)
    p.add_argument("--max_size", type=int, default=1333)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--save_top_k", type=int, default=3)
    p.add_argument("--tta", action="store_true", default=False)
    p.add_argument("--multi_scale", action="store_true", default=False)
    p.add_argument("--best_only", action="store_true", default=False, help="Only save best checkpoint, no _last or top-k")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = {}
        for n, p in model.state_dict().items():
            if torch.is_floating_point(p):
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        for n in self.shadow:
            self.shadow[n].mul_(self.decay).add_(sd[n].detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {}
        sd = model.state_dict()
        for n in self.shadow:
            backup[n] = sd[n].detach().clone()
            sd[n].copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        sd = model.state_dict()
        for n, p in backup.items():
            sd[n].copy_(p)


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


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = make_splits(args.data_path, seed=args.seed, val_frac=args.val_frac)
    train_ds = CellInstanceDataset(args.data_path, train_ids, transform=build_train_transform_v2())
    val_ds = CellInstanceDataset(args.data_path, val_ids, transform=build_val_transform())

    if is_main():
        print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  World size: {world_size}")

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fn, pin_memory=True)

    min_size_arg = (640, 800, 960) if args.multi_scale else args.min_size
    model = build_maskrcnn(
        num_classes=5, pretrained=True,
        min_size=min_size_arg, max_size=args.max_size,
        anchor_sizes=((8,), (16,), (32,), (64,), (128,)),
        box_detections_per_img=500,
    )
    model.to(device)
    if is_main():
        print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M")
        print(f"Multi-scale: {args.multi_scale} | TTA: {args.tta} | EMA decay: {args.ema_decay} | save_top_k: {args.save_top_k}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    target_module = model.module if distributed else model
    ema = EMA(target_module, decay=args.ema_decay) if is_main() else None

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_ap50 = -1.0
    epochs_since_improve = 0
    top_k_ckpts = []

    import pickle
    coco_gt = None
    if is_main():
        cache_path = Path(args.data_path).parent / f"coco_gt_s{args.seed}_f{args.val_frac}.pkl"
        if cache_path.exists():
            coco_gt = pickle.loads(cache_path.read_bytes())
            print("Loaded cached COCO ground truth.")
        else:
            coco_gt = build_coco_gt(val_ds)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(coco_gt))
            tmp.rename(cache_path)

    log_file = Path(args.log_path) / f"{args.run_name}.csv"
    if is_main() and not log_file.exists():
        with log_file.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss",
                *LOSS_KEYS,
                "val_AP", "val_AP50", "val_AP75", "lr", "secs",
            ])

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_components = {k: 0.0 for k in LOSS_KEYS}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[ep {epoch:03d}] train", leave=False, disable=not is_main())
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(target_module)
            running_loss += float(loss.detach())
            for k in LOSS_KEYS:
                if k in loss_dict:
                    running_components[k] += float(loss_dict[k].detach())
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss/max(1,n_batches):.4f}")
        train_loss = running_loss / max(1, n_batches)
        comp_means = {k: running_components[k] / max(1, n_batches) for k in LOSS_KEYS}

        ap_metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        if is_main():
            ap_metrics = run_validation(target_module, val_loader, device, coco_gt, tta=args.tta)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[ep {epoch:03d}] loss={train_loss:.4f}  AP={ap_metrics['AP']:.4f}  AP50={ap_metrics['AP50']:.4f}  AP75={ap_metrics['AP75']:.4f}  lr={current_lr:.2e}  ({elapsed:.1f}s)")
            with log_file.open("a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, train_loss,
                    *[comp_means[k] for k in LOSS_KEYS],
                    ap_metrics["AP"], ap_metrics["AP50"], ap_metrics["AP75"],
                    current_lr, elapsed,
                ])

            last_ckpt = {
                "model": target_module.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_ap50": best_ap50,
                "args": vars(args),
            }
            if not args.best_only:
                torch.save(last_ckpt, Path(args.save_path) / f"{args.run_name}_last.pth")

            if ap_metrics["AP50"] > best_ap50:
                best_ap50 = ap_metrics["AP50"]
                if ema is not None:
                    backup = ema.apply_to(target_module)
                    best_ckpt = dict(last_ckpt)
                    best_ckpt["model"] = {k: v.cpu().clone() for k, v in target_module.state_dict().items()}
                    best_ckpt["best_ap50"] = best_ap50
                    torch.save(best_ckpt, Path(args.save_path) / f"{args.run_name}_best.pth")
                    ema.restore(target_module, backup)
                else:
                    last_ckpt["best_ap50"] = best_ap50
                    torch.save(last_ckpt, Path(args.save_path) / f"{args.run_name}_best.pth")
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if not args.best_only:
                cand_path = Path(args.save_path) / f"{args.run_name}_top_ep{epoch:03d}_ap{ap_metrics['AP50']:.4f}.pth"
                top_k_ckpts.append((ap_metrics["AP50"], cand_path))
                top_k_ckpts.sort(key=lambda x: x[0], reverse=True)
                if (ap_metrics["AP50"], cand_path) in top_k_ckpts[:args.save_top_k]:
                    torch.save(last_ckpt, cand_path)
                for _, evicted_path in top_k_ckpts[args.save_top_k:]:
                    if evicted_path.exists():
                        evicted_path.unlink()
                top_k_ckpts = top_k_ckpts[:args.save_top_k]

        if distributed:
            stop_signal = torch.tensor([1 if epochs_since_improve >= args.patience else 0], device=device)
            torch.distributed.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                break
        else:
            if epochs_since_improve >= args.patience:
                if is_main():
                    print(f"Early stop after {args.patience} epochs without AP50 improvement.")
                break

    cleanup_distributed()


@torch.no_grad()
def run_validation(model, loader, device, coco_gt, tta=False):
    model.eval()
    predictions = {}
    pbar = tqdm(loader, desc="validate", leave=False, disable=not is_main())
    for idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        if tta:
            flipped_imgs = [torch.flip(img, dims=[-1]) for img in images]
            flipped_out = model(flipped_imgs)
            widths = [img.shape[-1] for img in images]
            unflipped = hflip_predictions(flipped_out, widths)
            outputs = merge_tta(outputs, unflipped, iou_thresh=0.5)
        for t, out in zip(targets, outputs):
            image_id = int(t["image_id"].item())
            predictions[image_id] = {
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
                "masks": out["masks"].detach().cpu(),
            }
    coco_results = predictions_to_coco_results(predictions)
    metrics = evaluate_segm(coco_gt, coco_results)
    return metrics


if __name__ == "__main__":
    main()
