from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(d: Any) -> Any:
    if hasattr(d, "__dict__"):
        try:
            return asdict(d)
        except Exception:
            return d.__dict__
    if isinstance(d, Path):
        return str(d)
    return d


def strip_chars(s: str, chars: str) -> str:
    if not chars:
        return s
    for ch in chars:
        s = s.replace(ch, "")
    return s


def batch_list(it: Iterable[Any], batch_size: int) -> List[List[Any]]:
    buf: List[Any] = []
    batches: List[List[Any]] = []
    for x in it:
        buf.append(x)
        if len(buf) >= batch_size:
            batches.append(buf)
            buf = []
    if buf:
        batches.append(buf)
    return batches


@torch.no_grad()
def ctc_greedy_decode(log_probs: torch.Tensor) -> torch.Tensor:
    # log_probs: [B, T, C]
    return torch.argmax(log_probs, dim=2)


def collapse_repeats_and_remove_blank(seq: List[int], blank: int = 0) -> List[int]:
    out: List[int] = []
    prev = None
    for s in seq:
        if s == blank:
            prev = s
            continue
        if prev is None or s != prev:
            out.append(s)
        prev = s
    return out
