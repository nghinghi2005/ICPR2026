from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LossWeights


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'opencv-python'. Install it (e.g. `pip install opencv-python`)."
        ) from exc
    return cv2


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    l1: float
    grad_l1: float


def _to_float01(img_bgr_u8: np.ndarray) -> np.ndarray:
    return img_bgr_u8.astype(np.float32) / 255.0


def _grad_mag(img_bgr_u8: np.ndarray, to_gray: bool) -> np.ndarray:
    cv2 = _require_cv2()
    if to_gray:
        x = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2GRAY)
    else:
        x = img_bgr_u8.mean(axis=2).astype(np.uint8)
    x_f = x.astype(np.float32) / 255.0
    gx = cv2.Sobel(x_f, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(x_f, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def match_loss(
    lr_fake_bgr_u8: np.ndarray,
    lr_real_bgr_u8: np.ndarray,
    weights: LossWeights,
    to_gray_for_grad: bool,
) -> LossBreakdown:
    """
    LR matching loss for degradation modelling.

    We use:
      - L1 on pixel values (robust to noise)
      - L1 on gradient magnitudes (sensitive to blur/motion)
    """
    x = _to_float01(lr_fake_bgr_u8)
    y = _to_float01(lr_real_bgr_u8)
    l1 = float(np.mean(np.abs(x - y)))

    gx = _grad_mag(lr_fake_bgr_u8, to_gray=to_gray_for_grad)
    gy = _grad_mag(lr_real_bgr_u8, to_gray=to_gray_for_grad)
    grad_l1 = float(np.mean(np.abs(gx - gy)))

    total = weights.l1 * l1 + weights.grad_l1 * grad_l1
    return LossBreakdown(total=total, l1=l1, grad_l1=grad_l1)


