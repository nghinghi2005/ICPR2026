from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KagglePaths:
    input_root: Path
    work_root: Path
    out_root: Path


def in_kaggle() -> bool:
    # Kaggle sets KAGGLE_URL_BASE and provides /kaggle paths
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def get_paths(project_name: str = "sprint2_outputs") -> KagglePaths:
    if in_kaggle():
        input_root = Path("/kaggle/input")
        work_root = Path("/kaggle/working")
    else:
        input_root = Path.cwd() / "data"
        work_root = Path.cwd() / "outputs"

    out_root = work_root / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    return KagglePaths(input_root=input_root, work_root=work_root, out_root=out_root)
