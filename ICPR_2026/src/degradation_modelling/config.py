from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class LossWeights:
    """Weights for LR matching loss."""

    l1: float = 1.0
    grad_l1: float = 0.25


@dataclass(frozen=True)
class LightingSearchSpace:
    """Discrete search space for lighting degradation."""

    alphas: List[float] = field(default_factory=lambda: [0.85, 0.95, 1.0, 1.05, 1.15])
    betas: List[int] = field(default_factory=lambda: [-30, -15, 0, 15, 30])
    gammas: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.25])


@dataclass(frozen=True)
class MotionBlurSearchSpace:
    """Discrete search space for motion blur."""

    lengths: List[int] = field(default_factory=lambda: [0, 3, 5, 7, 9, 11])
    angles_deg: List[float] = field(
        default_factory=lambda: [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    )


@dataclass(frozen=True)
class GaussianBlurSearchSpace:
    """Discrete search space for defocus/out-of-focus blur."""

    sigmas: List[float] = field(default_factory=lambda: [0.0, 0.6, 1.0, 1.4, 1.8])


@dataclass(frozen=True)
class GaussianNoiseSearchSpace:
    """Discrete search space for sensor noise."""

    stds: List[float] = field(default_factory=lambda: [0.0, 3.0, 6.0, 10.0, 15.0])


@dataclass(frozen=True)
class ScaleDownSearchSpace:
    """
    Discrete search space for scale-down operator (before final bicubic resize).

    This follows the idea from `my_models/degradation.py`:
      interpolation_methods = {"bicubic": INTER_CUBIC, "bilinear": INTER_LINEAR, "nearest": INTER_NEAREST}
    """

    scales: List[float] = field(default_factory=lambda: [0.25, 0.35, 0.5, 0.65, 0.8])
    interpolations: List[str] = field(default_factory=lambda: ["bicubic", "bilinear", "nearest"])


@dataclass(frozen=True)
class GreedySearchConfig:
    """
    Config for greedy degradation modelling.

    Notes:
      - Downsampling is ALWAYS bicubic and ALWAYS matches LR_real spatial size.
      - Search is greedy over a fixed operator order:
          lighting -> motion_blur -> gaussian_blur -> gaussian_noise -> bicubic_downsample
    """

    data_root: str = "dataset/train"
    overwrite: bool = False
    max_tracks: int = 0  # 0 means no limit
    num_frames: int = 0  # 0 means all paired frames

    # Global fitting:
    #   - If True: fit ONE set of params on pairs from ALL tracks, then apply globally.
    #   - If False: fit params per-frame (slower, not requested by user).
    global_fit: bool = True
    max_pairs_for_fit: int = 2000  # 0 means all pairs (can be very slow)
    fit_seed: int = 42

    # Enforce that degradation is applied (not all params are "no-op").
    # In practice, scale_down (scale < 1) already guarantees some degradation.
    require_degradation: bool = True

    output_suffix: str = "-downsample.png"
    params_filename: str = "degradation_modelling_params.json"
    global_params_filename: str = "degradation_modelling_global_params.json"

    loss_weights: LossWeights = field(default_factory=LossWeights)
    lighting: LightingSearchSpace = field(default_factory=LightingSearchSpace)
    motion_blur: MotionBlurSearchSpace = field(default_factory=MotionBlurSearchSpace)
    gaussian_blur: GaussianBlurSearchSpace = field(default_factory=GaussianBlurSearchSpace)
    scale_down: ScaleDownSearchSpace = field(default_factory=ScaleDownSearchSpace)
    gaussian_noise: GaussianNoiseSearchSpace = field(default_factory=GaussianNoiseSearchSpace)

    to_gray_for_grad: bool = True
    downsample_interpolation: int = 2  # cv2.INTER_CUBIC; stored as int to avoid importing cv2 here

    def validate(self) -> Tuple[bool, str]:
        if not self.data_root:
            return False, "data_root is empty"
        if not self.output_suffix.endswith(".png"):
            return False, "output_suffix must end with .png"
        if self.max_tracks < 0 or self.num_frames < 0:
            return False, "max_tracks/num_frames must be >= 0"
        if self.max_pairs_for_fit < 0:
            return False, "max_pairs_for_fit must be >= 0"
        if any(s <= 0.0 or s >= 1.0 for s in self.scale_down.scales):
            return False, "scale_down.scales must be in (0, 1)"
        if not self.scale_down.interpolations:
            return False, "scale_down.interpolations is empty"
        return True, "ok"


