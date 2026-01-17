"""baseline_icpr_2026.train

Training script for Multi-Frame CRNN License Plate Recognition.

Default usage (backward compatible):
    python train.py

Reproducible/recorded run usage:
    python train.py --data-root ..\\data\\train --output-dir ..\\runs\\sprint1\\baseline
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Support both running as module and direct script execution
try:
    from .config import Config
    from .dataset import AdvancedMultiFrameDataset
    from .models import MultiFrameCRNN
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import AdvancedMultiFrameDataset
    from models import MultiFrameCRNN
    from utils import seed_everything, decode_predictions


def train_pipeline():
    """Main training pipeline."""
    args = _parse_args()

    # Allow overriding config from CLI (useful for sprint reproducibility)
    if args.data_root:
        Config.DATA_ROOT = args.data_root
    if args.epochs is not None:
        Config.EPOCHS = int(args.epochs)
    if args.batch_size is not None:
        Config.BATCH_SIZE = int(args.batch_size)
    if args.num_workers is not None:
        Config.NUM_WORKERS = int(args.num_workers)
    if args.eval_strip_chars is not None:
        Config.EVAL_STRIP_CHARS = str(args.eval_strip_chars)
    split_ratio = float(args.split_ratio)
    test_ratio = float(args.test_ratio)
    hr_plus_root = str(args.hr_plus_root) if args.hr_plus_root else ""
    hr_plus_method = str(args.hr_plus_method)
    hr_plus_scale = int(args.hr_plus_scale)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep split file per-run to avoid cross-run leakage
    Config.VAL_SPLIT_FILE = str(output_dir / "val_tracks.json")
    Config.TEST_SPLIT_FILE = str(output_dir / "test_tracks.json")

    seed_everything(Config.SEED)
    print(f"🚀 TRAINING START | Device: {Config.DEVICE} | Output: {output_dir}")
    
    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        return

    # Persist run config
    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": os.path.abspath(Config.DATA_ROOT),
        "epochs": int(Config.EPOCHS),
        "batch_size": int(Config.BATCH_SIZE),
        "learning_rate": float(Config.LEARNING_RATE),
        "num_workers": int(Config.NUM_WORKERS),
        "seed": int(Config.SEED),
        "device": str(Config.DEVICE),
        "split_ratio": split_ratio,
        "test_ratio": test_ratio,
        "val_split_file": os.path.abspath(Config.VAL_SPLIT_FILE),
        "test_split_file": os.path.abspath(Config.TEST_SPLIT_FILE),
        "eval_strip_chars": str(getattr(Config, "EVAL_STRIP_CHARS", "")),
        "hr_plus_root": hr_plus_root,
        "hr_plus_method": hr_plus_method,
        "hr_plus_scale": hr_plus_scale,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Eval-only: run evaluation from a checkpoint without training
    if args.eval_only:
        ckpt_path = args.checkpoint
        if not ckpt_path:
            ckpt_path = str(output_dir / "best_model.pth")
        _run_eval_only(
            checkpoint_path=ckpt_path,
            data_root=Config.DATA_ROOT,
            output_dir=output_dir,
            split_ratio=split_ratio,
            test_ratio=test_ratio,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            eval_on=args.eval_on,
            hr_plus_root=hr_plus_root,
            hr_plus_method=hr_plus_method,
            hr_plus_scale=hr_plus_scale,
            eval_results_json=args.eval_results_json,
        )
        return

    # Create datasets
    train_ds = AdvancedMultiFrameDataset(
        Config.DATA_ROOT,
        mode='train',
        split_ratio=split_ratio,
        test_ratio=test_ratio,
        hr_plus_root=hr_plus_root,
        hr_plus_method=hr_plus_method,
        hr_plus_scale=hr_plus_scale,
    )
    val_ds = AdvancedMultiFrameDataset(
        Config.DATA_ROOT,
        mode='val',
        split_ratio=split_ratio,
        test_ratio=test_ratio,
        hr_plus_root=hr_plus_root,
        hr_plus_method=hr_plus_method,
        hr_plus_scale=hr_plus_scale,
    )
    test_ds = (
        AdvancedMultiFrameDataset(
            Config.DATA_ROOT,
            mode='test',
            split_ratio=split_ratio,
            test_ratio=test_ratio,
            hr_plus_root=hr_plus_root,
            hr_plus_method=hr_plus_method,
            hr_plus_scale=hr_plus_scale,
        )
        if test_ratio > 0
        else None
    )
    
    if len(train_ds) == 0: 
        print("❌ Dataset Train rỗng!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=AdvancedMultiFrameDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=AdvancedMultiFrameDataset.collate_fn, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True
        )
    else:
        print("⚠️ CẢNH BÁO: Validation Set rỗng. Sẽ bỏ qua bước validate.")
        val_loader = None

    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=AdvancedMultiFrameDataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # Initialize model, loss, optimizer
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.EPOCHS
    )
    use_cuda_amp = Config.DEVICE.type == "cuda"
    scaler = GradScaler(enabled=use_cuda_amp)

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "lr", "best_val_acc"],
        )
        writer.writeheader()

    best_acc = 0.0
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS}")
        for images, targets, target_lengths, _ in pbar:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type="cuda", enabled=use_cuda_amp):
                preds = model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),), 
                    fill_value=preds.size(1), 
                    dtype=torch.long
                )
                loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

            if use_cuda_amp:
                scaler_scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Only step scheduler if optimizer actually stepped
                if scaler.get_scale() >= scaler_scale_before:
                    scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_acc = 0
        avg_val_loss = 0
        
        if val_loader:
            model.eval()
            val_loss = 0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for images, targets, target_lengths, labels_text in val_loader:
                    images = images.to(Config.DEVICE)
                    targets = targets.to(Config.DEVICE)
                    preds = model(images)
                    
                    loss = criterion(
                        preds.permute(1, 0, 2), 
                        targets, 
                        torch.full((images.size(0),), preds.size(1), dtype=torch.long), 
                        target_lengths
                    )
                    val_loss += loss.item()
                    
                    decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
                    strip_chars = str(getattr(Config, "EVAL_STRIP_CHARS", ""))
                    for i in range(len(labels_text)):
                        pred_text = decoded[i]
                        gt_text = labels_text[i]

                        if strip_chars:
                            for ch in strip_chars:
                                pred_text = pred_text.replace(ch, "")
                                gt_text = gt_text.replace(ch, "")

                        if pred_text == gt_text:
                            total_correct += 1
                    total_samples += len(labels_text)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else float(Config.LEARNING_RATE)
        print(f"Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(output_dir / "best_model.pth"))
            print(f" -> ⭐ Saved Best Model! ({val_acc:.2f}%)")

        # Append metrics
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "lr", "best_val_acc"],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val_loss),
                    "val_acc": float(val_acc),
                    "lr": float(lr_now),
                    "best_val_acc": float(best_acc),
                }
            )

    # Optional: Evaluate best checkpoint on test split
    if args.run_test:
        if test_loader is None:
            print("⚠️ Test loader is empty. Set --test-ratio > 0 to create a test split.")
        else:
            best_path = output_dir / "best_model.pth"
            if not best_path.exists():
                print("⚠️ best_model.pth not found; skipping test evaluation.")
            else:
                _evaluate_and_save(
                    model=model,
                    checkpoint_path=str(best_path),
                    loader=test_loader,
                    output_path=str(output_dir / "test_results.json"),
                    title="TEST",
                )


def _apply_eval_strip(text: str) -> str:
    strip_chars = str(getattr(Config, "EVAL_STRIP_CHARS", ""))
    if not strip_chars:
        return text
    for ch in strip_chars:
        text = text.replace(ch, "")
    return text


def _evaluate_and_save(model, checkpoint_path: str, loader, output_path: str, title: str):
    try:
        state = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    exact_correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    sample_errors = []

    with torch.no_grad():
        for images, _, _, labels_text in tqdm(loader, desc=f"Evaluating {title}"):
            images = images.to(Config.DEVICE)
            preds = model(images)
            decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)

            for i in range(len(labels_text)):
                gt = _apply_eval_strip(labels_text[i])
                pred = _apply_eval_strip(decoded[i])

                if pred == gt:
                    exact_correct += 1
                total += 1

                # Character-level accuracy (position-wise, up to max length)
                m = max(len(gt), len(pred))
                for j in range(m):
                    char_total += 1
                    if j < len(gt) and j < len(pred) and gt[j] == pred[j]:
                        char_correct += 1

                if pred != gt and len(sample_errors) < 20:
                    sample_errors.append({"gt": gt, "pred": pred})

    exact_acc = (exact_correct / total) * 100 if total else 0.0
    char_acc = (char_correct / char_total) * 100 if char_total else 0.0

    result = {
        "split": title,
        "checkpoint": os.path.abspath(checkpoint_path),
        "exact_match_acc": exact_acc,
        "char_acc": char_acc,
        "total_samples": total,
        "correct_samples": exact_correct,
        "eval_strip_chars": str(getattr(Config, "EVAL_STRIP_CHARS", "")),
        "sample_errors": sample_errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n📊 {title} RESULTS:")
    print(f"   • Exact Match Accuracy: {exact_acc:.2f}% ({exact_correct}/{total})")
    print(f"   • Character Accuracy:   {char_acc:.2f}%")
    print(f"💾 Saved: {output_path}")


def _run_eval_only(
    checkpoint_path: str,
    data_root: str,
    output_dir: Path,
    split_ratio: float,
    test_ratio: float,
    batch_size: int,
    num_workers: int,
    eval_on: str,
    hr_plus_root: str,
    hr_plus_method: str,
    hr_plus_scale: int,
    eval_results_json: str,
):
    if not os.path.exists(data_root):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {data_root}")
        return

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    ds = AdvancedMultiFrameDataset(
        data_root,
        mode=eval_on,
        split_ratio=split_ratio,
        test_ratio=test_ratio,
        hr_plus_root=hr_plus_root,
        hr_plus_method=hr_plus_method,
        hr_plus_scale=hr_plus_scale,
    )
    if len(ds) == 0:
        print(f"❌ {eval_on} dataset is empty. If you want test evaluation, set --test-ratio > 0.")
        return

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AdvancedMultiFrameDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    if eval_results_json:
        out_path = eval_results_json
    else:
        out_name = "test_results.json" if eval_on == "test" else "val_results.json"
        out_path = str(output_dir / out_name)
    _evaluate_and_save(
        model=model,
        checkpoint_path=checkpoint_path,
        loader=loader,
        output_path=out_path,
        title=eval_on.upper(),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Multi-Frame CRNN baseline (CTC).")
    p.add_argument("--data-root", type=str, default="", help="Path to dataset root (track_* under it).")
    p.add_argument("--output-dir", type=str, default="", help="Directory to write run artifacts (models/metrics).")
    p.add_argument("--epochs", type=int, default=None, help="Override Config.EPOCHS")
    p.add_argument("--batch-size", type=int, default=None, help="Override Config.BATCH_SIZE")
    p.add_argument("--num-workers", type=int, default=None, help="Override Config.NUM_WORKERS")
    p.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio (val = 1 - ratio).")
    p.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio (0 disables test split).")
    p.add_argument("--run-test", action="store_true", help="After training, evaluate best_model.pth on test split.")
    p.add_argument(
        "--eval-strip-chars",
        type=str,
        default=None,
        help="Characters to strip from pred/gt strings when computing val_acc (evaluation only).",
    )
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for eval-only.")
    p.add_argument("--eval-only", action="store_true", help="Skip training; run evaluation from checkpoint.")
    p.add_argument(
        "--eval-on",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate in eval-only mode.",
    )
    p.add_argument(
        "--eval-results-json",
        type=str,
        default="",
        help="Path to write evaluation JSON (eval-only). Defaults to <output-dir>/(test|val)_results.json",
    )
    p.add_argument(
        "--hr-plus-root",
        type=str,
        default="",
        help="Root folder that contains HR_plus outputs (expects x<scale>/<method>/... structure).",
    )
    p.add_argument(
        "--hr-plus-method",
        type=str,
        default="lanczos2",
        help="HR_plus method subfolder name (e.g. lanczos2, denoise_clahe_sharpen).",
    )
    p.add_argument(
        "--hr-plus-scale",
        type=int,
        default=4,
        help="HR_plus scale factor used in folder naming (expects x<scale>).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train_pipeline()
