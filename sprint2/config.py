from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # allows importing config on minimal environments
    torch = None  # type: ignore


@dataclass
class OCRConfig:
    # Data
    data_root: str = "/kaggle/input/icpr2026/train"  # override in notebook
    frames_per_sample: int = 5

    # Input size for OCR model (CRNN expects fixed height)
    img_height: int = 32
    img_width: int = 128

    # Vocabulary (CTC: 0 is blank)
    chars: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

    # Training
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 30
    num_workers: int = 2

    # Mixed precision
    amp: bool = True

    # Evaluation
    # Strip '-' at eval for robustness across plate formats
    eval_strip_chars: str = "-"

    @property
    def device(self):
        if torch is None:
            return "cpu"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def char2idx(self):
        return {c: i + 1 for i, c in enumerate(self.chars)}

    @property
    def idx2char(self):
        return {i + 1: c for i, c in enumerate(self.chars)}

    @property
    def num_classes(self):
        return len(self.chars) + 1


@dataclass
class SRConfig:
    # SR upscaling factor (LR -> HR+). Use 2 or 4.
    scale: int = 2

    # Model
    # - 'rrdb_lite' is stronger (ESRGAN-style) and usually a better default.
    # - 'edsr_lite' is faster but less expressive.
    model: str = "rrdb_lite"
    rrdb_blocks: int = 6

    # Optional: use an available pretrained SR model (recommended when you want
    # better results quickly and consistent LR distribution via degrade-back loss).
    # This requires extra deps/weights; the notebook will fall back to training
    # our internal model if not available.
    use_pretrained: bool = False
    pretrained_name: str = "realesrgan_x2plus"  # common choice: 'realesrgan_x2plus' or 'realesrgan_x4plus'
    pretrained_path: str = ""  # local path to weights file (.pth)
    pretrained_url: str = ""  # optional URL to weights
    allow_download: bool = False  # set True only if Kaggle internet is enabled

    # Train patches
    patch_size_hr: int = 128

    # Training
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 2e-4
    epochs: int = 3  # keep small for Kaggle
    num_workers: int = 2

    # Prefer real paired lr/hr patches when hr-* exists
    use_paired: bool = True
    paired_max_tracks: int = 4000

    # Synthetic degradation
    lr_downscale_min: float = 0.35
    lr_downscale_max: float = 0.6

    # Degradation-consistency loss (only active when paired lr/hr is available)
    # Encourages degrade(SR(LR)) to match the *real* LR distribution.
    lambda_cycle: float = 0.25
    cycle_blur_sigma: float = 1.2

    # Mixed precision
    amp: bool = True
