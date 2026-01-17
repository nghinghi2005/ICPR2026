from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .config import GreedySearchConfig
from .degradations import DegradationParams
from .global_search import fit_global_params_greedy
from .losses import match_loss
from .search import build_lr_fake, greedy_search_params

LOGGER = logging.getLogger(__name__)

# NOTE: raw-string needs a single backslash for \d and \.
_FRAME_RE = re.compile(r"^(?P<prefix>hr|lr)-(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'opencv-python'. Install it (e.g. `pip install opencv-python`)."
        ) from exc
    return cv2


def _safe_tqdm(iterable: Iterable, total: Optional[int], desc: str):
    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _iter_tracks(data_root: str) -> Iterator[str]:
    for root, _, files in os.walk(data_root):
        if "annotations.json" in files and os.path.basename(root).startswith("track_"):
            yield root


def _index_frames(track_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    hr_map: Dict[str, str] = {}
    lr_map: Dict[str, str] = {}
    for name in os.listdir(track_dir):
        m = _FRAME_RE.match(name)
        if not m:
            continue
        idx = m.group("idx")
        prefix = m.group("prefix").lower()
        path = os.path.join(track_dir, name)
        if prefix == "hr":
            hr_map[idx] = path
        else:
            lr_map[idx] = path
    return hr_map, lr_map


def _pair_frames(track_dir: str) -> List[Tuple[str, str, str]]:
    hr_map, lr_map = _index_frames(track_dir)
    common = sorted(set(hr_map.keys()) & set(lr_map.keys()))
    return [(idx, hr_map[idx], lr_map[idx]) for idx in common]


def _output_path_for_lr(lr_path: str, output_suffix: str) -> str:
    base = os.path.splitext(os.path.basename(lr_path))[0]  # lr-001
    return os.path.join(os.path.dirname(lr_path), f"{base}{output_suffix}")


def _read_bgr_u8(path: str) -> np.ndarray:
    cv2 = _require_cv2()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _write_png(path: str, img_bgr_u8: np.ndarray) -> None:
    cv2 = _require_cv2()
    ok = cv2.imwrite(path, img_bgr_u8)
    if not ok:
        raise OSError(f"Could not write image: {path}")


def run_degradation_modelling(cfg: GreedySearchConfig) -> None:
    ok, msg = cfg.validate()
    if not ok:
        raise ValueError(f"Invalid config: {msg}")

    data_root = os.path.abspath(cfg.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    tracks = sorted(list(_iter_tracks(data_root)))
    if cfg.max_tracks > 0:
        tracks = tracks[: cfg.max_tracks]

    LOGGER.info("Found %d tracks under %s", len(tracks), data_root)

    # ------------------------------------
    # Step 1: Fit ONE global params set
    # ------------------------------------
    global_params = DegradationParams()
    if cfg.global_fit:
        all_pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
        fit_rng = np.random.default_rng(int(cfg.fit_seed))

        for track_dir in _safe_tqdm(tracks, total=len(tracks), desc="collect pairs"):
            pairs = _pair_frames(track_dir)
            if not pairs:
                continue
            if cfg.num_frames > 0:
                pairs = pairs[: cfg.num_frames]

            for idx, hr_path, lr_path in pairs:
                hr = _read_bgr_u8(hr_path)
                lr_real = _read_bgr_u8(lr_path)
                seed = (hash(os.path.relpath(track_dir, data_root)) ^ int(idx)) & 0xFFFFFFFF
                all_pairs.append((hr, lr_real, seed))
                if cfg.max_pairs_for_fit > 0 and len(all_pairs) >= cfg.max_pairs_for_fit:
                    break
            if cfg.max_pairs_for_fit > 0 and len(all_pairs) >= cfg.max_pairs_for_fit:
                break

        if not all_pairs:
            raise RuntimeError("No (HR, LR) pairs found for global fit. Check track naming and data_root.")

        # Optional: shuffle/sample deterministically
        if cfg.max_pairs_for_fit > 0 and len(all_pairs) > cfg.max_pairs_for_fit:
            idxs = fit_rng.choice(len(all_pairs), size=cfg.max_pairs_for_fit, replace=False)
            all_pairs = [all_pairs[int(i)] for i in idxs]

        LOGGER.info("Global greedy fit on %d (HR, LR) pairs", len(all_pairs))
        fit = fit_global_params_greedy(pairs=all_pairs, cfg=cfg)
        global_params = fit.params

        global_path = os.path.join(data_root, cfg.global_params_filename)
        with open(global_path, "w") as f:
            json.dump(
                {
                    "num_pairs": fit.num_pairs,
                    "avg_loss": asdict(fit.avg_loss),
                    "params": asdict(fit.params),
                },
                f,
                indent=2,
            )
        LOGGER.info("Saved global params: %s", global_path)

    for track_dir in _safe_tqdm(tracks, total=len(tracks), desc="degradation_modelling"):
        pairs = _pair_frames(track_dir)
        if not pairs:
            continue
        if cfg.num_frames > 0:
            pairs = pairs[: cfg.num_frames]

        params_path = os.path.join(track_dir, cfg.params_filename)
        persisted: Dict[str, dict] = {}
        if os.path.exists(params_path) and not cfg.overwrite:
            try:
                with open(params_path, "r") as f:
                    persisted = json.load(f)
            except Exception:
                persisted = {}

        for idx, hr_path, lr_path in pairs:
            out_path = _output_path_for_lr(lr_path, output_suffix=cfg.output_suffix)
            if os.path.exists(out_path) and not cfg.overwrite:
                continue

            hr = _read_bgr_u8(hr_path)
            lr_real = _read_bgr_u8(lr_path)

            seed = (hash(os.path.relpath(track_dir, data_root)) ^ int(idx)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

            if cfg.global_fit:
                used_params = global_params
                lr_fake = build_lr_fake(hr_bgr_u8=hr, lr_real_bgr_u8=lr_real, params=used_params, rng=rng)
                # Per-frame loss for monitoring (params are fixed global)
                loss = match_loss(
                    lr_fake_bgr_u8=lr_fake,
                    lr_real_bgr_u8=lr_real,
                    weights=cfg.loss_weights,
                    to_gray_for_grad=cfg.to_gray_for_grad,
                )
                result_loss = loss
            else:
                result = greedy_search_params(hr_bgr_u8=hr, lr_real_bgr_u8=lr_real, cfg=cfg, rng=rng)
                used_params = result.params
                lr_fake = build_lr_fake(hr_bgr_u8=hr, lr_real_bgr_u8=lr_real, params=used_params, rng=rng)
                result_loss = result.loss
            _write_png(out_path, lr_fake)

            persisted[f"frame_{idx}"] = {
                "hr_path": os.path.basename(hr_path),
                "lr_path": os.path.basename(lr_path),
                "output_path": os.path.basename(out_path),
                "loss": asdict(result_loss),
                "params": asdict(used_params),
            }

        if persisted:
            with open(params_path, "w") as f:
                json.dump(persisted, f, indent=2)


