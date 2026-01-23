from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import SRConfig
from .data import PairedLRHRPatchDataset, SRPatchDataset
from .paths import get_paths
from .sr_model import EDSRLite, RRDBNetLite
from .utils import ensure_dir, seed_everything


def _make_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = k // 2
    xs = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    gauss = torch.exp(-(xs**2) / (2.0 * float(sigma) ** 2 + 1e-8))
    gauss = gauss / gauss.sum()
    kernel2d = gauss[:, None] * gauss[None, :]
    return kernel2d


def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.2) -> torch.Tensor:
    # x: [B,C,H,W]
    b, c, _, _ = x.shape
    device = x.device
    k2d = _make_gaussian_kernel(kernel_size, sigma, device=device)
    w = k2d.view(1, 1, k2d.shape[0], k2d.shape[1]).repeat(c, 1, 1, 1)
    pad = k2d.shape[0] // 2
    return torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=pad, groups=c)


def _degrade_to_lr(sr_hr: torch.Tensor, lr_shape_hw: tuple[int, int], blur_sigma: float = 1.2) -> torch.Tensor:
    # sr_hr: [B,C,H,W] in [-1, 1]
    # output: [B,C,lr_h,lr_w] in [-1, 1]
    lr_h, lr_w = int(lr_shape_hw[0]), int(lr_shape_hw[1])
    x = _gaussian_blur(sr_hr, kernel_size=5, sigma=blur_sigma)
    x = torch.nn.functional.interpolate(x, size=(lr_h, lr_w), mode="bicubic", align_corners=False)
    return x


def _denorm_to_u8(x: torch.Tensor) -> np.ndarray:
    # x: CHW in [-1, 1]
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return x


def _to_u8_from_01(x: torch.Tensor) -> np.ndarray:
    # x: CHW in [0, 1]
    x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return x


def _pad_to_multiple(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad BCHW tensor on bottom/right so H and W become multiples of mult.

    Returns (padded, (pad_h, pad_w)).
    """
    mult = int(mult)
    if mult <= 1:
        return x, (0, 0)
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # pad format: (left, right, top, bottom)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


def train_sr(
    data_root: str,
    cfg: SRConfig,
    out_dir: Optional[Path] = None,
    max_steps_per_epoch: Optional[int] = None,
) -> Path:
    """Train SR model on synthetic pairs and save best checkpoint."""
    paths = get_paths()
    if out_dir is None:
        out_dir = paths.out_root / "sr"
    ensure_dir(out_dir)

    seed_everything(cfg.seed)

    # If requested, try to use a pretrained SR model (available model) instead
    # of training from scratch. We still emit a checkpoint in our format so the
    # rest of the pipeline stays unchanged.
    if bool(getattr(cfg, "use_pretrained", False)):
        return _prepare_pretrained_sr_ckpt(cfg, out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = getattr(cfg, "model", "rrdb_lite")
    model_name = str(model_name).lower().strip()
    if model_name in ("rrdb", "rrdb_lite", "esrgan", "esrgan_lite"):
        model = RRDBNetLite(scale=cfg.scale, num_rrdb=getattr(cfg, "rrdb_blocks", 6)).to(device)
    elif model_name in ("edsr", "edsr_lite"):
        model = EDSRLite(scale=cfg.scale).to(device)
    else:
        raise ValueError("SRConfig.model must be one of: rrdb_lite, edsr_lite")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")
    l1 = nn.L1Loss()

    # Prefer real paired LR/HR frames when available; fall back to synthetic patches
    use_paired = bool(getattr(cfg, "use_paired", True))
    ds: torch.utils.data.Dataset
    paired_ok = False
    if use_paired:
        try:
            ds = PairedLRHRPatchDataset(
                data_root=data_root,
                patch_size_hr=cfg.patch_size_hr,
                scale=cfg.scale,
                max_tracks=int(getattr(cfg, "paired_max_tracks", 4000)),
                seed=cfg.seed,
            )
            paired_ok = True
        except Exception:
            paired_ok = False

    if not paired_ok:
        ds = SRPatchDataset(
            data_root=data_root,
            patch_size_hr=cfg.patch_size_hr,
            scale=cfg.scale,
            down_min=cfg.lr_downscale_min,
            down_max=cfg.lr_downscale_max,
            seed=cfg.seed,
        )

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    meta = {
        "data_root": os.path.abspath(data_root),
        "cfg": asdict(cfg),
    }
    with open(out_dir / "sr_run_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    best_loss = float("inf")
    best_path = out_dir / "sr_best.pt"

    lam_cycle = float(getattr(cfg, "lambda_cycle", 0.25))
    blur_sigma = float(getattr(cfg, "cycle_blur_sigma", 1.2))

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(dl, desc=f"[SR] epoch {epoch+1}/{cfg.epochs} ({'paired' if paired_ok else 'synthetic'})")
        for step, batch in enumerate(pbar, start=1):
            if max_steps_per_epoch and step > max_steps_per_epoch:
                break

            if paired_ok:
                lr, hr, _s = batch
            else:
                lr, hr = batch

            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda")):
                pred = model(lr)

                # Supervised SR: LR -> HR
                loss_sr = l1(pred, hr)

                # Degradation-consistency: degrade(pred) -> LR (to match LR distribution)
                if paired_ok and lam_cycle > 0:
                    lr_hat = _degrade_to_lr(pred, lr_shape_hw=(lr.shape[-2], lr.shape[-1]), blur_sigma=blur_sigma)
                    loss_cycle = l1(lr_hat, lr)
                else:
                    loss_cycle = pred.new_tensor(0.0)

                loss = loss_sr + lam_cycle * loss_cycle

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running += float(loss.item())
            pbar.set_postfix({"sr_l1": float(loss_sr.item()), "cycle_l1": float(loss_cycle.item()), "loss": float(loss.item())})

        avg = running / max(1, (step if not max_steps_per_epoch else min(step, max_steps_per_epoch)))
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "model_name": model_name,
                    "paired": paired_ok,
                    "lambda_cycle": lam_cycle,
                    "cycle_blur_sigma": blur_sigma,
                },
                best_path,
            )

    return best_path


def _prepare_pretrained_sr_ckpt(cfg: SRConfig, out_dir: Path) -> Path:
    """Prepare a sprint2-compatible checkpoint from an external pretrained SR model.

    Current best supported option: Real-ESRGAN RRDBNet weights via `basicsr`.
    """
    ensure_dir(out_dir)

    name = str(getattr(cfg, "pretrained_name", "")).lower().strip() or "realesrgan_x2plus"
    scale = int(getattr(cfg, "scale", 2))
    weights_path = str(getattr(cfg, "pretrained_path", "")).strip()
    weights_url = str(getattr(cfg, "pretrained_url", "")).strip()
    allow_download = bool(getattr(cfg, "allow_download", False))

    # Common defaults (can override via cfg.pretrained_url or cfg.pretrained_path)
    default_urls = {
        # Official Real-ESRGAN release weights (if internet is enabled)
        "realesrgan_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "realesrgan_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth",
    }

    if not weights_path and not weights_url:
        weights_url = default_urls.get(name, "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Pretrained SR requested but 'basicsr' is not available. "
            "In Kaggle, install it (if internet allowed): pip install basicsr realesrgan. "
            "Or disable use_pretrained to train the internal model."
        ) from e

    # Pick a reasonable RRDBNet config for Real-ESRGAN weights
    if "x4" in name:
        scale = 4
    elif "x2" in name:
        scale = 2

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale).to(device)

    state = None
    if weights_path:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"pretrained_path not found: {p}")
        state = torch.load(str(p), map_location=device)
    elif weights_url and allow_download:
        state = torch.hub.load_state_dict_from_url(weights_url, map_location=device, progress=True)
    else:
        raise RuntimeError(
            "Pretrained SR requested but no weights provided. "
            "Set SRConfig.pretrained_path to a .pth, or set SRConfig.allow_download=True (internet on) "
            "and optionally SRConfig.pretrained_url."
        )

    # Real-ESRGAN weights often store under 'params_ema' or 'params'
    if isinstance(state, dict) and "params_ema" in state:
        sd = state["params_ema"]
    elif isinstance(state, dict) and "params" in state:
        sd = state["params"]
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        sd = state
    else:
        raise RuntimeError("Unrecognized pretrained weight format")

    model.load_state_dict(sd, strict=True)
    model.eval()

    out_path = out_dir / "sr_pretrained.pt"
    torch.save(
        {
            "external": "basicsr_rrdbnet",
            "arch": {"scale": int(scale), "num_feat": 64, "num_block": 23, "num_grow_ch": 32},
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "model_name": name,
        },
        out_path,
    )
    return out_path


@torch.no_grad()
def export_hr_plus(
    data_root: str,
    sr_ckpt: Path,
    hr_plus_root: Optional[Path] = None,
    scale: int = 2,
    source: str = "hr",
    limit_tracks: Optional[int] = None,
) -> Path:
    """Export SR outputs (HR+) by upscaling frames under data_root.

        Output layout:
            <hr_plus_root>/x<scale>/<relative_path_to_frame_parent>/<frame_stem>_x<scale>.png

        Args:
                source: "hr" (default) to upscale hr-* frames (HR→HR+),
                                "lr" to upscale lr-* frames (LR→HR+),
                                "auto" to use hr-* if present for a track, otherwise lr-*.
    """
    paths = get_paths()
    if hr_plus_root is None:
        hr_plus_root = paths.out_root / "hr_plus"
    out_root = ensure_dir(hr_plus_root / f"x{int(scale)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(sr_ckpt, map_location=device)

    # External pretrained ckpt (e.g., BasicSR RRDBNet)
    is_external_basicsr = ckpt.get("external") == "basicsr_rrdbnet"

    if is_external_basicsr:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "This SR checkpoint requires 'basicsr' to export images. "
                "Install basicsr (and optionally realesrgan) in Kaggle, or re-train using internal model."
            ) from e
        arch = ckpt.get("arch", {})
        m = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=int(arch.get("num_feat", 64)),
            num_block=int(arch.get("num_block", 23)),
            num_grow_ch=int(arch.get("num_grow_ch", 32)),
            scale=int(arch.get("scale", scale)),
        ).to(device)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        model = m
        model.eval()
    else:
        model_name = str(ckpt.get("model_name", "edsr_lite")).lower().strip()
        if model_name in ("rrdb", "rrdb_lite", "esrgan", "esrgan_lite"):
            model = RRDBNetLite(scale=scale, num_rrdb=int(ckpt.get("cfg", {}).get("rrdb_blocks", 6))).to(device)
        else:
            model = EDSRLite(scale=scale).to(device)

        sd = ckpt.get("model") or ckpt.get("state_dict")
        if sd is None:
            raise RuntimeError("SR checkpoint missing state dict")
        model.load_state_dict(sd, strict=True)
        model.eval()

    source = str(source).lower().strip()
    if source not in ("hr", "lr", "auto"):
        raise ValueError("source must be one of: 'hr', 'lr', 'auto'")

    frame_paths = []

    if source in ("hr", "lr"):
        prefix = "hr" if source == "hr" else "lr"
        for ext in ("png", "jpg", "jpeg"):
            frame_paths.extend(list(Path(data_root).rglob(f"{prefix}-*.{ext}")))
    else:
        # auto: decide per-track
        track_dirs = sorted(Path(data_root).rglob("track_*"))
        for tdir in track_dirs:
            hr = sorted(list(tdir.glob("hr-*.png")) + list(tdir.glob("hr-*.jpg")) + list(tdir.glob("hr-*.jpeg")))
            lr = sorted(list(tdir.glob("lr-*.png")) + list(tdir.glob("lr-*.jpg")) + list(tdir.glob("lr-*.jpeg")))
            frame_paths.extend(hr if hr else lr)

    # Optional speed limiter by tracks
    if limit_tracks is not None:
        seen = set()
        filtered = []
        for p in frame_paths:
            tdir = str(p.parent)
            if tdir not in seen:
                if len(seen) >= int(limit_tracks):
                    continue
                seen.add(tdir)
            filtered.append(p)
        frame_paths = filtered

    for p in tqdm(frame_paths, desc=f"[SR] exporting HR+ ({source})"):
        rel_parent = p.parent.resolve().relative_to(Path(data_root).resolve())
        out_dir = ensure_dir(out_root / rel_parent)
        out_path = out_dir / f"{p.stem}_x{int(scale)}.png"
        if out_path.exists():
            continue

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Prepare input tensor
        if is_external_basicsr:
            # BasicSR RRDBNet expects [0, 1]
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        else:
            # Internal models expect [-1, 1]
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            x = (x - 0.5) / 0.5

        x = x.unsqueeze(0).to(device)

        # Real-ESRGAN RRDBNet uses pixel-unshuffle and requires H/W divisible by scale.
        # Pad (bottom/right) to nearest multiple, then crop the output back.
        in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
        x_pad, (pad_h, pad_w) = _pad_to_multiple(x, mult=int(scale) if is_external_basicsr else 1)

        y = model(x_pad)[0]

        # Crop SR output if we padded the input
        if is_external_basicsr and (pad_h or pad_w):
            out_h = (in_h) * int(scale)
            out_w = (in_w) * int(scale)
            y = y[:, :out_h, :out_w]

        if is_external_basicsr:
            out_rgb = _to_u8_from_01(y)
        else:
            out_rgb = _denorm_to_u8(y)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    return out_root
