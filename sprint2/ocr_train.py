from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import OCRConfig
from .ctc_decode import decode_with_plate_constraints
from .data import MultiFrameTrackDataset
from .ocr_model import MultiFrameCRNN
from .paths import get_paths
from .utils import ensure_dir, seed_everything, strip_chars


def _accuracy(preds, gts):
    correct = 0
    for p, g in zip(preds, gts):
        if p == g:
            correct += 1
    return correct / max(1, len(gts))


def train_ocr(
    cfg: OCRConfig,
    out_dir: Optional[Path] = None,
    hr_plus_root: str = "",
    hr_plus_scale: int = 2,
    split_ratio: float = 0.9,
    prefer_template: Optional[str] = None,
) -> Dict[str, str]:
    """Train OCR model with CTC; decode with template constraints."""

    paths = get_paths()
    if out_dir is None:
        out_dir = paths.out_root / "ocr"
    ensure_dir(out_dir)

    seed_everything(cfg.seed)

    device = cfg.device
    model = MultiFrameCRNN(num_classes=cfg.num_classes, frames=cfg.frames_per_sample).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    split_file = str(out_dir / "val_tracks.json")

    train_ds = MultiFrameTrackDataset(
        cfg.data_root,
        char2idx=cfg.char2idx,
        mode="train",
        split_ratio=split_ratio,
        seed=cfg.seed,
        frames_per_sample=cfg.frames_per_sample,
        img_hw=(cfg.img_height, cfg.img_width),
        eval_strip_chars=cfg.eval_strip_chars,
        hr_plus_root=hr_plus_root,
        hr_plus_scale=hr_plus_scale,
        split_file=split_file,
    )
    val_ds = MultiFrameTrackDataset(
        cfg.data_root,
        char2idx=cfg.char2idx,
        mode="val",
        split_ratio=split_ratio,
        seed=cfg.seed,
        frames_per_sample=cfg.frames_per_sample,
        img_hw=(cfg.img_height, cfg.img_width),
        eval_strip_chars=cfg.eval_strip_chars,
        hr_plus_root=hr_plus_root,
        hr_plus_scale=hr_plus_scale,
        split_file=split_file,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=MultiFrameTrackDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=MultiFrameTrackDataset.collate_fn,
    )

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cfg": asdict(cfg),
        "data_root": os.path.abspath(cfg.data_root),
        "hr_plus_root": hr_plus_root,
        "hr_plus_scale": int(hr_plus_scale),
        "split_ratio": float(split_ratio),
        "prefer_template": prefer_template,
    }
    with open(out_dir / "ocr_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_acc_constrained"])
        w.writeheader()

    best_acc = 0.0
    best_ckpt = out_dir / "ocr_best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"[OCR] epoch {epoch+1}/{cfg.epochs}")
        for images, targets, target_lengths, _labels in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                log_probs = model(images)  # [B, T, C]
                log_probs_t = log_probs.permute(1, 0, 2)  # [T, B, C]
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=log_probs.size(1),
                    dtype=torch.long,
                    device=targets.device,
                )
                loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            pbar.set_postfix({"ctc": float(loss.item())})

        train_loss = running / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        all_gt = []
        all_pred_greedy = []
        all_pred_constrained = []

        with torch.no_grad():
            for images, targets, target_lengths, labels in tqdm(val_loader, desc="[OCR] validating"):
                images = images.to(device)
                targets = targets.to(device)
                with autocast(device_type="cuda", enabled=use_amp):
                    log_probs = model(images)
                    log_probs_t = log_probs.permute(1, 0, 2)
                    input_lengths = torch.full(
                        size=(images.size(0),),
                        fill_value=log_probs.size(1),
                        dtype=torch.long,
                        device=targets.device,
                    )
                    loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
                val_loss += float(loss.item())

                # Greedy decode (fast)
                greedy_ids = torch.argmax(log_probs, dim=2).cpu().tolist()
                for seq in greedy_ids:
                    # collapse repeats and remove blank
                    pred = []
                    last = None
                    for s in seq:
                        if s != 0 and s != last:
                            pred.append(cfg.idx2char.get(s, ""))
                        last = s
                    all_pred_greedy.append(strip_chars("".join(pred), cfg.eval_strip_chars))

                constrained = decode_with_plate_constraints(
                    log_probs.cpu(),
                    idx2char=cfg.idx2char,
                    beam_size=10,
                    prefer_template=prefer_template,
                    eval_strip=cfg.eval_strip_chars,
                )
                all_pred_constrained.extend(constrained)

                for gt in labels:
                    all_gt.append(strip_chars(gt, cfg.eval_strip_chars))

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = _accuracy(all_pred_greedy, all_gt)
        val_acc_c = _accuracy(all_pred_constrained, all_gt)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_acc_constrained"])
            w.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_acc_constrained": val_acc_c,
                }
            )

        # Save best by constrained accuracy
        if val_acc_c > best_acc:
            best_acc = val_acc_c
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, best_ckpt)

        # Save a small debug sample
        sample_path = out_dir / "val_samples.json"
        samples = []
        for i in range(min(25, len(all_gt))):
            samples.append(
                {
                    "gt": all_gt[i],
                    "pred_greedy": all_pred_greedy[i],
                    "pred_constrained": all_pred_constrained[i],
                }
            )
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

    return {
        "best_ckpt": str(best_ckpt),
        "metrics_csv": str(metrics_path),
        "out_dir": str(out_dir),
    }
