from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class LightingParams:
    alpha: float = 1.0
    beta: int = 0
    gamma: float = 1.0


@dataclass(frozen=True)
class MotionBlurParams:
    length: int = 0
    angle_deg: float = 0.0


@dataclass(frozen=True)
class GaussianBlurParams:
    sigma: float = 0.0


@dataclass(frozen=True)
class GaussianNoiseParams:
    std: float = 0.0


@dataclass(frozen=True)
class ScaleDownParams:
    """
    Scale-down operator params (before the final bicubic resize to LR size).

    interpolation follows `my_models/degradation.py`:
      - bicubic / bilinear / nearest
    """

    scale: float = 0.35
    interpolation: str = "bicubic"


@dataclass(frozen=True)
class DegradationParams:
    lighting: LightingParams = LightingParams()
    motion_blur: MotionBlurParams = MotionBlurParams()
    gaussian_blur: GaussianBlurParams = GaussianBlurParams()
    scale_down: ScaleDownParams = ScaleDownParams()
    gaussian_noise: GaussianNoiseParams = GaussianNoiseParams()


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'opencv-python'. Install it (e.g. `pip install opencv-python`)."
        ) from exc
    return cv2


def apply_lighting(img_bgr_u8: np.ndarray, params: LightingParams) -> np.ndarray:
    cv2 = _require_cv2()
    x = img_bgr_u8.astype(np.float32)
    x = x * float(params.alpha) + float(params.beta)
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)

    if abs(params.gamma - 1.0) < 1e-6:
        return x
    inv_gamma = 1.0 / float(params.gamma)
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
    lut = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(x, lut)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    cv2 = _require_cv2()
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    center = (length / 2.0, length / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    s = float(k.sum())
    return k / s if s > 1e-6 else np.array([[1.0]], dtype=np.float32)


def apply_motion_blur(img_bgr_u8: np.ndarray, params: MotionBlurParams) -> np.ndarray:
    cv2 = _require_cv2()
    if params.length <= 1:
        return img_bgr_u8
    kernel = _motion_kernel(params.length, params.angle_deg)
    return cv2.filter2D(img_bgr_u8, ddepth=-1, kernel=kernel)


def _gaussian_ksize(sigma: float) -> int:
    if sigma <= 0:
        return 0
    k = int(np.ceil(6.0 * sigma + 1.0))
    if k % 2 == 0:
        k += 1
    return max(k, 3)


def apply_gaussian_blur(img_bgr_u8: np.ndarray, params: GaussianBlurParams) -> np.ndarray:
    cv2 = _require_cv2()
    if params.sigma <= 0:
        return img_bgr_u8
    k = _gaussian_ksize(params.sigma)
    return cv2.GaussianBlur(img_bgr_u8, (k, k), sigmaX=float(params.sigma), sigmaY=float(params.sigma))


def apply_gaussian_noise(img_bgr_u8: np.ndarray, params: GaussianNoiseParams, rng: np.random.Generator) -> np.ndarray:
    if params.std <= 0:
        return img_bgr_u8
    noise = rng.normal(loc=0.0, scale=float(params.std), size=img_bgr_u8.shape).astype(np.float32)
    x = img_bgr_u8.astype(np.float32) + noise
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def apply_scale_down(img_bgr_u8: np.ndarray, params: ScaleDownParams) -> np.ndarray:
    """
    Scale down by factor (0, 1) using selected interpolation.

    NOTE: this is NOT the final resize to LR size. The final resize is always
    bicubic to match LR_real size exactly.
    """
    cv2 = _require_cv2()
    if params.scale <= 0.0 or params.scale >= 1.0:
        raise ValueError("scale_down.scale must be in (0, 1)")

    interpolation_methods = {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    if params.interpolation not in interpolation_methods:
        raise ValueError(f"Unsupported interpolation: {params.interpolation}")

    new_w = max(1, int(img_bgr_u8.shape[1] * float(params.scale)))
    new_h = max(1, int(img_bgr_u8.shape[0] * float(params.scale)))
    return cv2.resize(img_bgr_u8, (new_w, new_h), interpolation=interpolation_methods[params.interpolation])


def bicubic_downsample(img_bgr_u8: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    cv2 = _require_cv2()
    h, w = int(target_hw[0]), int(target_hw[1])
    return cv2.resize(img_bgr_u8, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_degradation(img_bgr_u8: np.ndarray, params: DegradationParams, rng: np.random.Generator) -> np.ndarray:
    x = apply_lighting(img_bgr_u8, params.lighting)
    x = apply_motion_blur(x, params.motion_blur)
    x = apply_gaussian_blur(x, params.gaussian_blur)
    x = apply_scale_down(x, params.scale_down)
    x = apply_gaussian_noise(x, params.gaussian_noise, rng=rng)
    return x


