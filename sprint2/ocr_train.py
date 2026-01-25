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


def _char_accuracy(preds, gts) -> float:
    # Match baseline-style: compare up to max length of each pair.
    total = 0
    correct = 0
    for p, g in zip(preds, gts):
        m = max(len(p), len(g))
        if m == 0:
            continue
        total += m
        for i in range(m):
            if i < len(p) and i < len(g) and p[i] == g[i]:
                correct += 1
    return correct / max(1, total)


def _edit_distance(a: str, b: str) -> int:
    # Levenshtein distance (iterative, O(min(n,m)) memory)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure a is the shorter string
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        cur = [j]
        for i, ai in enumerate(a, start=1):
            ins = cur[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ai == bj else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _summarize(preds, gts) -> Dict[str, float]:
    exact = _accuracy(preds, gts)
    char_acc = _char_accuracy(preds, gts)
    dists = [_edit_distance(p, g) for p, g in zip(preds, gts)]
    avg_ed = float(sum(dists) / max(1, len(dists)))
    norm = []
    for (p, g), d in zip(zip(preds, gts), dists):
        denom = max(1, max(len(p), len(g)))
        norm.append(d / denom)
    avg_norm_ed = float(sum(norm) / max(1, len(norm)))
    return {
        "exact_match_acc": float(exact),
        "char_acc": float(char_acc),
        "avg_edit_distance": float(avg_ed),
        "avg_norm_edit_distance": float(avg_norm_ed),
        "n": float(len(gts)),
    }


def train_ocr(
    cfg: OCRConfig,
    out_dir: Optional[Path] = None,
    hr_plus_root: str = "",
    hr_plus_scale: int = 2,
    split_ratio: float = 0.9,
    test_ratio: float = 0.0,
    prefer_template: Optional[str] = None,
    beam_size: int = 10,
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

    split_file = str(out_dir / "splits.json")

    train_ds = MultiFrameTrackDataset(
        cfg.data_root,
        char2idx=cfg.char2idx,
        mode="train",
        split_ratio=split_ratio,
        test_ratio=test_ratio,
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
        test_ratio=test_ratio,
        seed=cfg.seed,
        frames_per_sample=cfg.frames_per_sample,
        img_hw=(cfg.img_height, cfg.img_width),
        eval_strip_chars=cfg.eval_strip_chars,
        hr_plus_root=hr_plus_root,
        hr_plus_scale=hr_plus_scale,
        split_file=split_file,
    )

    test_loader = None
    if float(test_ratio) > 0:
        test_ds = MultiFrameTrackDataset(
            cfg.data_root,
            char2idx=cfg.char2idx,
            mode="test",
            split_ratio=split_ratio,
            test_ratio=test_ratio,
            seed=cfg.seed,
            frames_per_sample=cfg.frames_per_sample,
            img_hw=(cfg.img_height, cfg.img_width),
            eval_strip_chars=cfg.eval_strip_chars,
            hr_plus_root=hr_plus_root,
            hr_plus_scale=hr_plus_scale,
            split_file=split_file,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=MultiFrameTrackDataset.collate_fn,
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
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_acc_constrained",
                "val_char_acc",
                "val_char_acc_constrained",
            ],
        )
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
                    beam_size=int(beam_size),
                    prefer_template=prefer_template,
                    eval_strip=cfg.eval_strip_chars,
                )
                all_pred_constrained.extend(constrained)

                for gt in labels:
                    all_gt.append(strip_chars(gt, cfg.eval_strip_chars))

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = _accuracy(all_pred_greedy, all_gt)
        val_acc_c = _accuracy(all_pred_constrained, all_gt)
        val_char_acc = _char_accuracy(all_pred_greedy, all_gt)
        val_char_acc_c = _char_accuracy(all_pred_constrained, all_gt)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_acc_constrained",
                    "val_char_acc",
                    "val_char_acc_constrained",
                ],
            )
            w.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_acc_constrained": val_acc_c,
                    "val_char_acc": val_char_acc,
                    "val_char_acc_constrained": val_char_acc_c,
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

    # Final evaluation summary: best checkpoint on val/test with greedy vs constrained
    eval_summary = {
        "best_ckpt": str(best_ckpt),
        "best_val_acc_constrained": float(best_acc),
        "split_file": str(split_file),
        "hr_plus_root": str(hr_plus_root),
        "hr_plus_scale": int(hr_plus_scale),
        "beam_size": int(beam_size),
        "prefer_template": prefer_template,
        "val": {},
        "test": {},
    }

    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        model.eval()

        def _run_eval(loader, split_name: str) -> Tuple[dict, list]:
            all_gt2 = []
            pred_g = []
            pred_c = []
            with torch.no_grad():
                for images, _targets, _tlen, labels in tqdm(loader, desc=f"[OCR] eval {split_name}"):
                    images = images.to(device)
                    with autocast(device_type="cuda", enabled=use_amp):
                        log_probs = model(images)

                    greedy_ids = torch.argmax(log_probs, dim=2).cpu().tolist()
                    for seq in greedy_ids:
                        pred = []
                        last = None
                        for s in seq:
                            if s != 0 and s != last:
                                pred.append(cfg.idx2char.get(s, ""))
                            last = s
                        pred_g.append(strip_chars("".join(pred), cfg.eval_strip_chars))

                    constrained2 = decode_with_plate_constraints(
                        log_probs.cpu(),
                        idx2char=cfg.idx2char,
                        beam_size=int(beam_size),
                        prefer_template=prefer_template,
                        eval_strip=cfg.eval_strip_chars,
                    )
                    pred_c.extend(constrained2)

                    for gt in labels:
                        all_gt2.append(strip_chars(gt, cfg.eval_strip_chars))

            summary = {
                "greedy": _summarize(pred_g, all_gt2),
                "constrained": _summarize(pred_c, all_gt2),
            }
            samples2 = []
            for i in range(min(50, len(all_gt2))):
                samples2.append({"gt": all_gt2[i], "pred_greedy": pred_g[i], "pred_constrained": pred_c[i]})
            return summary, samples2

        val_summary, _ = _run_eval(val_loader, "val")
        eval_summary["val"] = val_summary

        if test_loader is not None:
            test_summary, test_samples = _run_eval(test_loader, "test")
            eval_summary["test"] = test_summary
            with open(out_dir / "test_samples.json", "w", encoding="utf-8") as f:
                json.dump(test_samples, f, indent=2)

        with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

    return {
        "best_ckpt": str(best_ckpt),
        "metrics_csv": str(metrics_path),
        "eval_summary": str(out_dir / "eval_summary.json"),
        "out_dir": str(out_dir),
    }
