from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import strip_chars


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _resize(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    # img: HWC RGB uint8
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = (x - 0.5) / 0.5
    return x


@dataclass(frozen=True)
class TrackSample:
    track_dir: str
    lr_paths: List[str]
    hr_paths: List[str]
    label: str


class MultiFrameTrackDataset(Dataset):
    """Loads track_* folders containing lr-* frames and annotations.

    Supports swapping LR frames to pre-exported HR+ frames if `hr_plus_root` is provided.

    HR+ expected layout (created by sprint2 SR exporter):
      <hr_plus_root>/x<scale>/<rel_track_dir>/<lr_stem>_x<scale>.png

    where rel_track_dir is relative to DATA_ROOT.
    """

    def __init__(
        self,
        data_root: str,
        char2idx: dict,
        mode: str = "train",
        split_ratio: float = 0.9,
        test_ratio: float = 0.0,
        seed: int = 42,
        frames_per_sample: int = 5,
        img_hw: Tuple[int, int] = (32, 128),
        eval_strip_chars: str = "-",
        hr_plus_root: str = "",
        hr_plus_scale: int = 2,
        use_hr_if_available: bool = True,
        split_file: Optional[str] = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.mode = mode
        self.char2idx = char2idx
        self.frames_per_sample = int(frames_per_sample)
        self.img_hw = img_hw
        self.eval_strip_chars = str(eval_strip_chars or "")
        self.hr_plus_root = str(hr_plus_root or "")
        self.hr_plus_scale = int(hr_plus_scale)
        self.use_hr_if_available = bool(use_hr_if_available)

        self.samples: List[TrackSample] = []

        tracks = sorted(glob.glob(os.path.join(self.data_root, "**", "track_*"), recursive=True))
        if not tracks:
            raise RuntimeError(f"No track_* found under: {self.data_root}")

        train_tracks, val_tracks, test_tracks = self._split(
            tracks,
            split_ratio=split_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_file=split_file,
        )

        if mode == "train":
            selected = train_tracks
        elif mode == "val":
            selected = val_tracks
        elif mode == "test":
            selected = test_tracks
        else:
            raise ValueError("mode must be one of: train, val, test")

        for track_dir in selected:
            ann = os.path.join(track_dir, "annotations.json")
            if not os.path.exists(ann):
                continue
            try:
                with open(ann, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                raw_label = data.get("plate_text", data.get("license_plate", data.get("text", "")))
                label = self._normalize_label(str(raw_label or ""))
                if not label:
                    continue

                lr_paths = sorted(glob.glob(os.path.join(track_dir, "lr-*.png")) + glob.glob(os.path.join(track_dir, "lr-*.jpg")))
                hr_paths = sorted(glob.glob(os.path.join(track_dir, "hr-*.png")) + glob.glob(os.path.join(track_dir, "hr-*.jpg")))
                if not lr_paths:
                    continue

                self.samples.append(TrackSample(track_dir=track_dir, lr_paths=lr_paths, hr_paths=hr_paths, label=label))
            except Exception:
                continue

        if not self.samples:
            raise RuntimeError(f"No usable samples for mode={mode}")

    def _split(self, tracks: Sequence[str], split_ratio: float, test_ratio: float, seed: int, split_file: Optional[str]):
        tracks = list(tracks)
        rng = random.Random(int(seed))

        train_ratio = float(split_ratio)
        test_ratio = float(test_ratio)
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError("split_ratio (train ratio) must be in (0, 1)")
        if test_ratio < 0 or test_ratio >= 1:
            raise ValueError("test_ratio must be in [0, 1)")
        val_ratio = 1.0 - train_ratio - test_ratio
        if val_ratio <= 0:
            raise ValueError("val_ratio must be > 0; adjust split_ratio/test_ratio")

        if split_file and os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Backward compatible:
            # - list: val tracks only
            # - dict: {"val": [...], "test": [...]} (train is the remainder)
            if isinstance(data, list):
                val_names = set(data)
                test_names: set[str] = set()
            elif isinstance(data, dict):
                val_names = set(data.get("val", []) or [])
                test_names = set(data.get("test", []) or [])
            else:
                val_names = set()
                test_names = set()

            train, val, test = [], [], []
            for t in tracks:
                if os.path.basename(t) in val_names:
                    val.append(t)
                elif os.path.basename(t) in test_names:
                    test.append(t)
                else:
                    train.append(t)

            # Only accept persisted splits when non-empty for requested structure
            if train and val and (test_ratio <= 0 or test):
                return train, val, test

        rng.shuffle(tracks)
        n = len(tracks)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = tracks[:n_train]
        val = tracks[n_train : n_train + n_val]
        test = tracks[n_train + n_val :]

        # If test_ratio == 0, collapse into val
        if test_ratio <= 0:
            val = val + test
            test = []

        if split_file:
            with open(split_file, "w", encoding="utf-8") as f:
                if test:
                    json.dump(
                        {
                            "val": [os.path.basename(t) for t in val],
                            "test": [os.path.basename(t) for t in test],
                            "seed": int(seed),
                            "train_ratio": train_ratio,
                            "val_ratio": val_ratio,
                            "test_ratio": test_ratio,
                        },
                        f,
                        indent=2,
                    )
                else:
                    json.dump([os.path.basename(t) for t in val], f, indent=2)

        return train, val, test

    def _normalize_label(self, s: str) -> str:
        s = "".join(s.split()).upper()
        s = strip_chars(s, self.eval_strip_chars)
        s = "".join([c for c in s if c in self.char2idx])

        # Training-time robustness: optionally allow hyphen even if not in GT
        # (we already stripped it above; keep consistent canonical labels)
        return s

    def _pad_or_sample(self, paths: Sequence[str]) -> List[str]:
        paths = list(paths)
        if len(paths) >= self.frames_per_sample:
            # random contiguous chunk for train, deterministic head for val
            if self.mode == "train":
                start = random.randint(0, len(paths) - self.frames_per_sample)
                return paths[start : start + self.frames_per_sample]
            return paths[: self.frames_per_sample]
        # pad with last frame
        return paths + [paths[-1]] * (self.frames_per_sample - len(paths))

    def _map_to_hr_plus(self, lr_path: str) -> str:
        if not self.hr_plus_root:
            return lr_path
        try:
            rel = Path(lr_path).resolve().relative_to(Path(self.data_root).resolve())
        except Exception:
            return lr_path

        out = (
            Path(self.hr_plus_root)
            / f"x{self.hr_plus_scale}"
            / rel.parent
            / f"{Path(lr_path).stem}_x{self.hr_plus_scale}.png"
        )
        return str(out) if out.exists() else lr_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        base_paths = item.hr_paths if (self.use_hr_if_available and len(item.hr_paths) > 0) else item.lr_paths
        frame_paths = self._pad_or_sample(base_paths)
        frame_paths = [self._map_to_hr_plus(p) for p in frame_paths]

        frames = []
        for p in frame_paths:
            img = _read_rgb(p)
            img = _resize(img, self.img_hw)
            frames.append(_to_tensor(img))
        x = torch.stack(frames, dim=0)  # [T, C, H, W]

        target = [self.char2idx[c] for c in item.label if c in self.char2idx]
        if not target:
            target = [0]
        target_t = torch.tensor(target, dtype=torch.long)

        return x, target_t, len(target), item.label

    @staticmethod
    def collate_fn(batch):
        xs, targets, target_lengths, labels = zip(*batch)
        xs = torch.stack(xs, dim=0)
        targets = torch.cat(targets, dim=0)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return xs, targets, target_lengths, labels


class SRPatchDataset(Dataset):
    """Build synthetic LR->HR pairs from HR frames.

    - Sample random HR patches.
    - Downscale by a random factor in [down_min, down_max].
    - Resize back to (patch_size_hr/scale) then SR model outputs HR.

    This is kept simple to run quickly on Kaggle.
    """

    def __init__(
        self,
        data_root: str,
        patch_size_hr: int = 128,
        scale: int = 2,
        down_min: float = 0.35,
        down_max: float = 0.6,
        max_frames: int = 20000,
        seed: int = 42,
    ):
        self.data_root = os.path.abspath(data_root)
        self.patch_size_hr = int(patch_size_hr)
        self.scale = int(scale)
        self.down_min = float(down_min)
        self.down_max = float(down_max)

        rng = random.Random(int(seed))

        # Prefer hr-* frames if present, else fall back to lr-* (still trains a denoise-ish SR)
        hr_paths = sorted(glob.glob(os.path.join(self.data_root, "**", "hr-*.png"), recursive=True))
        if not hr_paths:
            hr_paths = sorted(glob.glob(os.path.join(self.data_root, "**", "lr-*.png"), recursive=True))
        if not hr_paths:
            raise RuntimeError(f"No hr-* or lr-* pngs found under: {self.data_root}")

        if len(hr_paths) > max_frames:
            rng.shuffle(hr_paths)
            hr_paths = hr_paths[:max_frames]

        self.hr_paths = hr_paths

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        path = self.hr_paths[idx]
        img = _read_rgb(path)

        h, w = img.shape[:2]
        ps = self.patch_size_hr
        if h < ps or w < ps:
            img = cv2.resize(img, (max(w, ps), max(h, ps)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        y0 = random.randint(0, h - ps)
        x0 = random.randint(0, w - ps)
        hr = img[y0 : y0 + ps, x0 : x0 + ps]

        # synthetic degradation: downscale then bicubic back to LR size
        factor = random.uniform(self.down_min, self.down_max)
        down_w = max(1, int(ps * factor))
        down_h = max(1, int(ps * factor))
        small = cv2.resize(hr, (down_w, down_h), interpolation=cv2.INTER_AREA)

        lr_h = ps // self.scale
        lr_w = ps // self.scale
        lr = cv2.resize(small, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        lr_t = _to_tensor(lr)
        hr_t = _to_tensor(hr)
        return lr_t, hr_t


class PairedLRHRPatchDataset(Dataset):
    """Aligned LR->HR patch dataset using real paired frames.

    Expects tracks containing both lr-*.{png/jpg} and hr-*.{png/jpg}.
    Uses frame index pairing by sorting filenames.

    This enables training losses that match the *real* LR distribution.
    """

    def __init__(
        self,
        data_root: str,
        patch_size_hr: int = 128,
        scale: Optional[int] = 2,
        max_tracks: int = 4000,
        seed: int = 42,
    ):
        self.data_root = os.path.abspath(data_root)
        self.patch_size_hr = int(patch_size_hr)
        self.scale = None if scale is None else int(scale)

        rng = random.Random(int(seed))

        track_dirs = sorted(glob.glob(os.path.join(self.data_root, "**", "track_*"), recursive=True))
        if not track_dirs:
            raise RuntimeError(f"No track_* found under: {self.data_root}")

        if len(track_dirs) > max_tracks:
            rng.shuffle(track_dirs)
            track_dirs = track_dirs[:max_tracks]

        pairs: list[tuple[str, str]] = []
        for tdir in track_dirs:
            lr_files = sorted(
                glob.glob(os.path.join(tdir, "lr-*.png"))
                + glob.glob(os.path.join(tdir, "lr-*.jpg"))
                + glob.glob(os.path.join(tdir, "lr-*.jpeg"))
            )
            hr_files = sorted(
                glob.glob(os.path.join(tdir, "hr-*.png"))
                + glob.glob(os.path.join(tdir, "hr-*.jpg"))
                + glob.glob(os.path.join(tdir, "hr-*.jpeg"))
            )
            if not lr_files or not hr_files:
                continue
            n = min(len(lr_files), len(hr_files))
            for i in range(n):
                pairs.append((lr_files[i], hr_files[i]))

        if not pairs:
            raise RuntimeError(f"No paired lr/hr frames found under: {self.data_root}")

        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _infer_scale(lr_hw: tuple[int, int], hr_hw: tuple[int, int]) -> int:
        lr_h, lr_w = lr_hw
        hr_h, hr_w = hr_hw
        if lr_h <= 0 or lr_w <= 0:
            raise ValueError("Invalid LR shape")
        sh = hr_h / lr_h
        sw = hr_w / lr_w
        # Expect near-integer ratios
        s = int(round((sh + sw) * 0.5))
        if s < 1:
            raise ValueError("Invalid inferred scale")
        return s

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr_img = _read_rgb(lr_path)
        hr_img = _read_rgb(hr_path)

        lr_h, lr_w = lr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]

        inferred = self._infer_scale((lr_h, lr_w), (hr_h, hr_w))
        if self.scale is not None and inferred != self.scale:
            # Fallback to inferred scale to keep things consistent rather than crashing
            s = inferred
        else:
            s = inferred

        ps_hr = self.patch_size_hr
        ps_lr = max(1, ps_hr // s)

        # Ensure images are large enough; if not, resize up (rare)
        if hr_h < ps_hr or hr_w < ps_hr:
            hr_img = cv2.resize(hr_img, (max(hr_w, ps_hr), max(hr_h, ps_hr)), interpolation=cv2.INTER_CUBIC)
            hr_h, hr_w = hr_img.shape[:2]
            # keep LR aligned by resizing with same ratio
            new_lr_w = max(1, int(round(hr_w / s)))
            new_lr_h = max(1, int(round(hr_h / s)))
            lr_img = cv2.resize(lr_img, (new_lr_w, new_lr_h), interpolation=cv2.INTER_CUBIC)
            lr_h, lr_w = lr_img.shape[:2]

        # Sample HR crop and map to LR crop coordinates
        y0_hr = random.randint(0, hr_h - ps_hr)
        x0_hr = random.randint(0, hr_w - ps_hr)
        y0_lr = int(round(y0_hr / s))
        x0_lr = int(round(x0_hr / s))

        # Clamp LR crop window
        y0_lr = min(max(0, y0_lr), max(0, lr_h - ps_lr))
        x0_lr = min(max(0, x0_lr), max(0, lr_w - ps_lr))

        hr = hr_img[y0_hr : y0_hr + ps_hr, x0_hr : x0_hr + ps_hr]
        lr = lr_img[y0_lr : y0_lr + ps_lr, x0_lr : x0_lr + ps_lr]

        # If sizes are off by 1 due to rounding, enforce exact ps_lr
        if lr.shape[0] != ps_lr or lr.shape[1] != ps_lr:
            lr = cv2.resize(lr, (ps_lr, ps_lr), interpolation=cv2.INTER_CUBIC)

        lr_t = _to_tensor(lr)
        hr_t = _to_tensor(hr)
        return lr_t, hr_t, s
