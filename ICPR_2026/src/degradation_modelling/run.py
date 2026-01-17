from __future__ import annotations

import argparse
import logging
import os
import sys

try:
    # Preferred when executed as a module: `python -m src.degradation_modelling.run ...`
    from .config import GreedySearchConfig
    from .runner import run_degradation_modelling
except ImportError:
    # Fallback when executed as a script: `python src/degradation_modelling/run.py ...`
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.degradation_modelling.config import GreedySearchConfig  # type: ignore
    from src.degradation_modelling.runner import run_degradation_modelling  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage0: domain-specific degradation modelling (greedy search).")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root containing track_* folders.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs and params JSON.")
    p.add_argument("--max-tracks", type=int, default=0, help="Limit number of tracks (0 = all).")
    p.add_argument("--num-frames", type=int, default=0, help="Limit frames per track (0 = all).")
    p.add_argument("--global-fit", action="store_true", help="Fit ONE global params set on all tracks, then apply.")
    p.add_argument("--no-global-fit", action="store_true", help="Disable global fit and search per-frame (slow).")
    p.add_argument("--max-pairs-for-fit", type=int, default=2000, help="Max (HR,LR) pairs used to fit global params.")
    p.add_argument("--fit-seed", type=int, default=42, help="Random seed for global fit sampling.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args()
    cfg = GreedySearchConfig(
        data_root=os.path.abspath(args.data_root),
        overwrite=bool(args.overwrite),
        max_tracks=int(args.max_tracks),
        num_frames=int(args.num_frames),
        global_fit=bool(args.global_fit) and (not bool(args.no_global_fit)),
        max_pairs_for_fit=int(args.max_pairs_for_fit),
        fit_seed=int(args.fit_seed),
    )
    run_degradation_modelling(cfg)


if __name__ == "__main__":
    main()


