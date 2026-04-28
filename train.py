import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data.dataset import (
    CellInstanceDataset,
    build_train_transform,
    build_val_transform,
    collate_fn,
    make_splits,
)
from model.build import build_maskrcnn, count_trainable_params
from utils.coco_eval import build_coco_gt, evaluate_segm, predictions_to_coco_results
from utils.ddp import cleanup_distributed, init_distributed, is_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="baseline")
    p.add_argument("--data_path", type=str, default="datasets/train")
    p.add_argument("--save_path", type=str, default="checkpoints")
    p.add_argument("--log_path", type=str, default="log")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--min_size", type=int, default=384)
    p.add_argument("--max_size", type=int, default=768)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--best_only", action="store_true", default=False)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = make_splits(args.data_path, seed=args.seed, val_frac=args.val_frac)
    train_ds = CellInstanceDataset(args.data_path, train_ids, transform=build_train_transform())
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

    model = build_maskrcnn(num_classes=5, pretrained=True, min_size=args.min_size, max_size=args.max_size)
    model.to(device)
    if is_main():
        print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

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

    import pickle
    coco_gt = None
    if is_main():
        cache_path = Path(args.data_path).parent / f"coco_gt_s{args.seed}_f{args.val_frac}.pkl"
        if cache_path.exists():
            coco_gt = pickle.loads(cache_path.read_bytes())
            print("Loaded cached COCO ground truth.")
        else:
            coco_gt = build_coco_gt(val_ds)
            cache_path.write_bytes(pickle.dumps(coco_gt))

    log_file = Path(args.log_path) / f"{args.run_name}.csv"
    if is_main() and not log_file.exists():
        with log_file.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_AP", "val_AP50", "val_AP75", "lr", "secs"])

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running_loss = 0.0
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
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss/max(1,n_batches):.4f}")
        train_loss = running_loss / max(1, n_batches)

        ap_metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        if is_main():
            ap_metrics = run_validation(model.module if distributed else model, val_loader, device, coco_gt)
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[ep {epoch:03d}] loss={train_loss:.4f}  AP={ap_metrics['AP']:.4f}  AP50={ap_metrics['AP50']:.4f}  AP75={ap_metrics['AP75']:.4f}  lr={current_lr:.2e}  ({elapsed:.1f}s total)")
            with log_file.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, ap_metrics["AP"], ap_metrics["AP50"], ap_metrics["AP75"], current_lr, elapsed])

            target = model.module if distributed else model
            ckpt = {
                "model": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_ap50": best_ap50,
                "args": vars(args),
            }
            if not args.best_only:
                torch.save(ckpt, Path(args.save_path) / f"{args.run_name}_last.pth")
            if ap_metrics["AP50"] > best_ap50:
                best_ap50 = ap_metrics["AP50"]
                ckpt["best_ap50"] = best_ap50
                torch.save(ckpt, Path(args.save_path) / f"{args.run_name}_best.pth")
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

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
def run_validation(model, loader, device, coco_gt):
    model.eval()
    predictions = {}
    pbar = tqdm(loader, desc="validate", leave=False, disable=not is_main())
    for idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
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
