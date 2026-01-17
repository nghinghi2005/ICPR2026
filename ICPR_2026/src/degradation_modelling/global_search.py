from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .config import GreedySearchConfig
from .degradations import (
    DegradationParams,
    GaussianBlurParams,
    GaussianNoiseParams,
    LightingParams,
    MotionBlurParams,
    ScaleDownParams,
    apply_degradation,
    bicubic_downsample,
)
from .losses import LossBreakdown, match_loss


@dataclass(frozen=True)
class GlobalFitResult:
    params: DegradationParams
    avg_loss: LossBreakdown
    num_pairs: int


def _safe_tqdm(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _iter_lighting(space) -> Iterator[LightingParams]:
    for alpha in space.alphas:
        for beta in space.betas:
            for gamma in space.gammas:
                yield LightingParams(alpha=float(alpha), beta=int(beta), gamma=float(gamma))


def _iter_motion(space) -> Iterator[MotionBlurParams]:
    for length in space.lengths:
        for angle in space.angles_deg:
            yield MotionBlurParams(length=int(length), angle_deg=float(angle))


def _iter_gaussian_blur(space) -> Iterator[GaussianBlurParams]:
    for sigma in space.sigmas:
        yield GaussianBlurParams(sigma=float(sigma))


def _iter_gaussian_noise(space) -> Iterator[GaussianNoiseParams]:
    for std in space.stds:
        yield GaussianNoiseParams(std=float(std))


def _iter_scale_down(space) -> Iterator[ScaleDownParams]:
    for s in space.scales:
        for interp in space.interpolations:
            yield ScaleDownParams(scale=float(s), interpolation=str(interp))


def _avg_loss_over_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    params: DegradationParams,
    cfg: GreedySearchConfig,
    desc: str,
) -> LossBreakdown:
    """
    Compute average match loss over pairs.

    Args:
        pairs: list of (hr_bgr_u8, lr_real_bgr_u8, seed_int)
    """
    total_l1 = 0.0
    total_grad = 0.0
    total = 0.0
    n = max(len(pairs), 1)

    for hr, lr_real, seed in _safe_tqdm(pairs, total=len(pairs), desc=desc):
        rng = np.random.default_rng(seed)
        target_hw = (int(lr_real.shape[0]), int(lr_real.shape[1]))
        degraded = apply_degradation(hr, params=params, rng=rng)
        lr_fake = bicubic_downsample(degraded, target_hw=target_hw)
        loss = match_loss(
            lr_fake_bgr_u8=lr_fake,
            lr_real_bgr_u8=lr_real,
            weights=cfg.loss_weights,
            to_gray_for_grad=cfg.to_gray_for_grad,
        )
        total += loss.total
        total_l1 += loss.l1
        total_grad += loss.grad_l1

    return LossBreakdown(total=total / n, l1=total_l1 / n, grad_l1=total_grad / n)


def fit_global_params_greedy(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    cfg: GreedySearchConfig,
) -> GlobalFitResult:
    """
    Fit ONE set of degradation params over ALL pairs using greedy search.
    """
    best = DegradationParams()
    best_loss = _avg_loss_over_pairs(pairs=pairs, params=best, cfg=cfg, desc="fit: baseline loss")

    # 1) lighting
    best_lighting = best.lighting
    for cand in _safe_tqdm(
        list(_iter_lighting(cfg.lighting)),
        total=len(cfg.lighting.alphas) * len(cfg.lighting.betas) * len(cfg.lighting.gammas),
        desc="fit: lighting",
    ):
        cand_params = DegradationParams(
            lighting=cand,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            gaussian_noise=best.gaussian_noise,
        )
        loss = _avg_loss_over_pairs(pairs=pairs, params=cand_params, cfg=cfg, desc="eval lighting")
        if loss.total < best_loss.total:
            best_loss = loss
            best_lighting = cand
    best = DegradationParams(
        lighting=best_lighting,
        motion_blur=best.motion_blur,
        gaussian_blur=best.gaussian_blur,
        gaussian_noise=best.gaussian_noise,
    )

    # 2) motion blur
    best_motion = best.motion_blur
    for cand in _safe_tqdm(
        list(_iter_motion(cfg.motion_blur)),
        total=len(cfg.motion_blur.lengths) * len(cfg.motion_blur.angles_deg),
        desc="fit: motion_blur",
    ):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=cand,
            gaussian_blur=best.gaussian_blur,
            gaussian_noise=best.gaussian_noise,
        )
        loss = _avg_loss_over_pairs(pairs=pairs, params=cand_params, cfg=cfg, desc="eval motion")
        if loss.total < best_loss.total:
            best_loss = loss
            best_motion = cand
    best = DegradationParams(
        lighting=best.lighting,
        motion_blur=best_motion,
        gaussian_blur=best.gaussian_blur,
        gaussian_noise=best.gaussian_noise,
    )

    # 3) gaussian blur
    best_gb = best.gaussian_blur
    for cand in _safe_tqdm(
        list(_iter_gaussian_blur(cfg.gaussian_blur)),
        total=len(cfg.gaussian_blur.sigmas),
        desc="fit: gaussian_blur",
    ):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=cand,
            gaussian_noise=best.gaussian_noise,
        )
        loss = _avg_loss_over_pairs(pairs=pairs, params=cand_params, cfg=cfg, desc="eval gaussian_blur")
        if loss.total < best_loss.total:
            best_loss = loss
            best_gb = cand
    best = DegradationParams(
        lighting=best.lighting,
        motion_blur=best.motion_blur,
        gaussian_blur=best_gb,
        scale_down=best.scale_down,
        gaussian_noise=best.gaussian_noise,
    )

    # 4) scale down
    best_sd = best.scale_down
    for cand in _safe_tqdm(
        list(_iter_scale_down(cfg.scale_down)),
        total=len(cfg.scale_down.scales) * len(cfg.scale_down.interpolations),
        desc="fit: scale_down",
    ):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            scale_down=cand,
            gaussian_noise=best.gaussian_noise,
        )
        loss = _avg_loss_over_pairs(pairs=pairs, params=cand_params, cfg=cfg, desc="eval scale_down")
        if loss.total < best_loss.total:
            best_loss = loss
            best_sd = cand
    best = DegradationParams(
        lighting=best.lighting,
        motion_blur=best.motion_blur,
        gaussian_blur=best.gaussian_blur,
        scale_down=best_sd,
        gaussian_noise=best.gaussian_noise,
    )

    # 5) gaussian noise
    best_gn = best.gaussian_noise
    for cand in _safe_tqdm(
        list(_iter_gaussian_noise(cfg.gaussian_noise)),
        total=len(cfg.gaussian_noise.stds),
        desc="fit: gaussian_noise",
    ):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            scale_down=best.scale_down,
            gaussian_noise=cand,
        )
        loss = _avg_loss_over_pairs(pairs=pairs, params=cand_params, cfg=cfg, desc="eval gaussian_noise")
        if loss.total < best_loss.total:
            best_loss = loss
            best_gn = cand
    best = DegradationParams(
        lighting=best.lighting,
        motion_blur=best.motion_blur,
        gaussian_blur=best.gaussian_blur,
        scale_down=best.scale_down,
        gaussian_noise=best_gn,
    )

    return GlobalFitResult(params=best, avg_loss=best_loss, num_pairs=len(pairs))


