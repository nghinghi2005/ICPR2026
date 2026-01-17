from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

try:
    # Preferred when executed as a module: `python -m src.handcrafted_modelling.export_hr_plus ...`
    from .base import HandCrafter
    from .model import LicensePlateEnhancer
except ImportError:
    # Fallback when executed as a script
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.handcrafted_modelling.base import HandCrafter  # type: ignore
    from src.handcrafted_modelling.model import LicensePlateEnhancer  # type: ignore


_FRAME_RE = re.compile(r"^(?P<prefix>hr|lr)-(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'opencv-python'. Install it (e.g. `pip install opencv-python`)."
        ) from exc
    return cv2


def _safe_tqdm(it, total: Optional[int], desc: str):
    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        return it
    return tqdm(it, total=total, desc=desc)


def _iter_tracks(data_root: Path) -> Iterator[Path]:
    for root, _, files in os.walk(data_root):
        if "annotations.json" in files and os.path.basename(root).startswith("track_"):
            yield Path(root)


def _index_frames(track_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    hr: Dict[str, Path] = {}
    lr: Dict[str, Path] = {}
    for name in os.listdir(track_dir):
        m = _FRAME_RE.match(name)
        if not m:
            continue
        idx = m.group("idx")
        prefix = m.group("prefix").lower()
        p = track_dir / name
        if prefix == "hr":
            hr[idx] = p
        else:
            lr[idx] = p
    return hr, lr


def _read_annotations(track_dir: Path) -> dict:
    p = track_dir / "annotations.json"
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            data = data[0]
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _psnr_db(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate HR_plus (x4) images from LR frames using handcrafted upscalers.")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root that contains track_* folders (or contains train/...).")
    p.add_argument("--output-root", type=str, required=True, help="Where to write generated HR_plus images and manifest.")
    p.add_argument(
        "--method",
        type=str,
        default="lanczos2",
        choices=sorted(list(HandCrafter.METHOD_MAPPING.keys())),
        help="Upscaling/enhancement method.",
    )
    p.add_argument("--scale", type=float, default=4.0, help="Upscaling factor (Sprint 1 uses 4).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing generated images.")
    p.add_argument("--max-tracks", type=int, default=0, help="Limit tracks processed (0 = all).")
    return p.parse_args()


def main() -> None:
    cv2 = _require_cv2()
    args = _parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    method = str(args.method)
    scale = float(args.scale)

    enhancer = LicensePlateEnhancer(scale=scale, use_visualize=False)

    tracks = sorted(list(_iter_tracks(data_root)))
    if args.max_tracks and int(args.max_tracks) > 0:
        tracks = tracks[: int(args.max_tracks)]

    manifest_path = output_root / f"manifest_hr_plus_x{int(scale)}_{method}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_dir",
                "lr_frame",
                "out_path",
                "plate_text",
                "plate_layout",
                "method",
                "scale",
                "has_hr_pair",
                "psnr_to_hr_resized_db",
            ],
        )
        writer.writeheader()

        for track_dir in _safe_tqdm(tracks, total=len(tracks), desc=f"HR_plus x{scale:g} ({method})"):
            ann = _read_annotations(track_dir)
            plate_text = ann.get("plate_text", ann.get("license_plate", ann.get("text", "")))
            plate_layout = ann.get("plate_layout", "")

            hr_map, lr_map = _index_frames(track_dir)
            for idx, lr_path in sorted(lr_map.items(), key=lambda kv: int(kv[0])):
                rel = lr_path.relative_to(data_root)
                out_dir = output_root / f"x{int(scale)}" / method / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{lr_path.stem}_x{int(scale)}.png"

                if out_path.exists() and not bool(args.overwrite):
                    # Still record the row (useful for resumes)
                    has_hr = idx in hr_map
                    psnr = ""
                    if has_hr:
                        hr_img = cv2.imread(str(hr_map[idx]), cv2.IMREAD_COLOR)
                        if hr_img is not None:
                            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                            out_img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
                            if out_img is not None:
                                out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                                hr_rs = cv2.resize(hr_img, (out_img.shape[1], out_img.shape[0]), interpolation=cv2.INTER_CUBIC)
                                psnr = f"{_psnr_db(out_img, hr_rs):.4f}"
                    writer.writerow(
                        {
                            "track_dir": str(track_dir),
                            "lr_frame": str(lr_path.name),
                            "out_path": str(out_path),
                            "plate_text": plate_text,
                            "plate_layout": plate_layout,
                            "method": method,
                            "scale": scale,
                            "has_hr_pair": bool(has_hr),
                            "psnr_to_hr_resized_db": psnr,
                        }
                    )
                    continue

                out = enhancer.enhance(name=method, img_path=str(lr_path))
                cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

                has_hr = idx in hr_map
                psnr = ""
                if has_hr:
                    hr_img = cv2.imread(str(hr_map[idx]), cv2.IMREAD_COLOR)
                    if hr_img is not None:
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                        hr_rs = cv2.resize(hr_img, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
                        psnr = f"{_psnr_db(out, hr_rs):.4f}"

                writer.writerow(
                    {
                        "track_dir": str(track_dir),
                        "lr_frame": str(lr_path.name),
                        "out_path": str(out_path),
                        "plate_text": plate_text,
                        "plate_layout": plate_layout,
                        "method": method,
                        "scale": scale,
                        "has_hr_pair": bool(has_hr),
                        "psnr_to_hr_resized_db": psnr,
                    }
                )

    meta = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "method": method,
        "scale": scale,
        "manifest": str(manifest_path),
        "methods_available": sorted(list(HandCrafter.METHOD_MAPPING.keys())),
    }
    with open(output_root / f"run_hr_plus_x{int(scale)}_{method}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
