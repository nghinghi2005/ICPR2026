from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Iterator

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
class SearchResult:
    params: DegradationParams
    loss: LossBreakdown


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


def _is_noop(params: DegradationParams) -> bool:
    return (
        abs(params.lighting.alpha - 1.0) < 1e-6
        and params.lighting.beta == 0
        and abs(params.lighting.gamma - 1.0) < 1e-6
        and params.motion_blur.length <= 1
        and params.gaussian_blur.sigma <= 0.0
        and params.gaussian_noise.std <= 0.0
        and (params.scale_down.scale >= 1.0)
    )

def greedy_search_params(
    hr_bgr_u8: np.ndarray,
    lr_real_bgr_u8: np.ndarray,
    cfg: GreedySearchConfig,
    rng: np.random.Generator,
) -> SearchResult:
    """
    Greedy search to fit domain-specific degradation parameters for a (HR, LR) pair.

    Constraint:
      - Downsample step is fixed to bicubic and matches LR_real size.
    """
    target_hw = (int(lr_real_bgr_u8.shape[0]), int(lr_real_bgr_u8.shape[1]))

    best = DegradationParams()

    def evaluate(params: DegradationParams) -> LossBreakdown:
        degraded = apply_degradation(hr_bgr_u8, params=params, rng=rng)
        lr_fake = bicubic_downsample(degraded, target_hw=target_hw)
        return match_loss(
            lr_fake_bgr_u8=lr_fake,
            lr_real_bgr_u8=lr_real_bgr_u8,
            weights=cfg.loss_weights,
            to_gray_for_grad=cfg.to_gray_for_grad,
        )

    best_loss = evaluate(best)

    # 1) lighting
    best_lighting = best.lighting
    for cand in _iter_lighting(cfg.lighting):
        cand_params = DegradationParams(
            lighting=cand,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            gaussian_noise=best.gaussian_noise,
        )
        loss = evaluate(cand_params)
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
    for cand in _iter_motion(cfg.motion_blur):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=cand,
            gaussian_blur=best.gaussian_blur,
            gaussian_noise=best.gaussian_noise,
        )
        loss = evaluate(cand_params)
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
    for cand in _iter_gaussian_blur(cfg.gaussian_blur):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=cand,
            gaussian_noise=best.gaussian_noise,
        )
        loss = evaluate(cand_params)
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

    # 4) scale down (interpolation + scale factor)
    best_sd = best.scale_down
    for cand in _iter_scale_down(cfg.scale_down):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            scale_down=cand,
            gaussian_noise=best.gaussian_noise,
        )
        loss = evaluate(cand_params)
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
    for cand in _iter_gaussian_noise(cfg.gaussian_noise):
        cand_params = DegradationParams(
            lighting=best.lighting,
            motion_blur=best.motion_blur,
            gaussian_blur=best.gaussian_blur,
            scale_down=best.scale_down,
            gaussian_noise=cand,
        )
        loss = evaluate(cand_params)
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

    # Enforce that we actually apply some degradation (no identity solution).
    # Since scale_down.scales are in (0,1), this should already hold; this is a safety net.
    if cfg.require_degradation and _is_noop(best):
        forced_best = best
        forced_loss = best_loss
        # Try "single-operator" non-noop candidates and pick the best.
        for cand in _iter_motion(cfg.motion_blur):
            if cand.length <= 1:
                continue
            cand_params = DegradationParams(motion_blur=cand)
            loss = evaluate(cand_params)
            if loss.total < forced_loss.total:
                forced_loss = loss
                forced_best = cand_params
        for cand in _iter_gaussian_blur(cfg.gaussian_blur):
            if cand.sigma <= 0:
                continue
            cand_params = DegradationParams(gaussian_blur=cand)
            loss = evaluate(cand_params)
            if loss.total < forced_loss.total:
                forced_loss = loss
                forced_best = cand_params
        for cand in _iter_gaussian_noise(cfg.gaussian_noise):
            if cand.std <= 0:
                continue
            cand_params = DegradationParams(gaussian_noise=cand)
            loss = evaluate(cand_params)
            if loss.total < forced_loss.total:
                forced_loss = loss
                forced_best = cand_params
        for cand in _iter_scale_down(cfg.scale_down):
            cand_params = DegradationParams(scale_down=cand)
            loss = evaluate(cand_params)
            if loss.total < forced_loss.total:
                forced_loss = loss
                forced_best = cand_params
        best = forced_best
        best_loss = forced_loss

    return SearchResult(params=best, loss=best_loss)


def build_lr_fake(
    hr_bgr_u8: np.ndarray,
    lr_real_bgr_u8: np.ndarray,
    params: DegradationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    target_hw = (int(lr_real_bgr_u8.shape[0]), int(lr_real_bgr_u8.shape[1]))
    degraded = apply_degradation(hr_bgr_u8, params=params, rng=rng)
    return bicubic_downsample(degraded, target_hw=target_hw)


