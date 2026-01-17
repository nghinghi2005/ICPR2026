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
    split_ratio = float(args.split_ratio)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep split file per-run to avoid cross-run leakage
    Config.VAL_SPLIT_FILE = str(output_dir / "val_tracks.json")

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
        "val_split_file": os.path.abspath(Config.VAL_SPLIT_FILE),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Create datasets
    train_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train', split_ratio=split_ratio)
    val_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val', split_ratio=split_ratio)
    
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
                    for i in range(len(labels_text)):
                        if decoded[i] == labels_text[i]:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Multi-Frame CRNN baseline (CTC).")
    p.add_argument("--data-root", type=str, default="", help="Path to dataset root (track_* under it).")
    p.add_argument("--output-dir", type=str, default="", help="Directory to write run artifacts (models/metrics).")
    p.add_argument("--epochs", type=int, default=None, help="Override Config.EPOCHS")
    p.add_argument("--batch-size", type=int, default=None, help="Override Config.BATCH_SIZE")
    p.add_argument("--num-workers", type=int, default=None, help="Override Config.NUM_WORKERS")
    p.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio (val = 1 - ratio).")
    return p.parse_args()


if __name__ == "__main__":
    train_pipeline()
