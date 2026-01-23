from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .utils import collapse_repeats_and_remove_blank, strip_chars


BRA_OLD = re.compile(r"^[A-Z]{3}-?[0-9]{4}$")
BR_MERCOSUR = re.compile(r"^[A-Z]{3}-?[0-9][A-Z][0-9]{2}$")


@dataclass(frozen=True)
class PlateTemplate:
    name: str
    regex: re.Pattern
    # canonical pattern without hyphen
    pattern: str


TEMPLATES: List[PlateTemplate] = [
    PlateTemplate("brazil_old", BRA_OLD, "LLLDDDD"),
    PlateTemplate("mercosur", BR_MERCOSUR, "LLLDLDD"),
]


LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_SET = set("0123456789")

# Common OCR confusions for license plates
DIGIT_LIKE: Dict[str, str] = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

LETTER_LIKE: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _normalize_raw_text(s: str) -> str:
    if not s:
        return ""
    s = "".join(s.split()).upper()
    # keep only A-Z, 0-9, '-'
    s = "".join([c for c in s if (c in LETTER_SET or c in DIGIT_SET or c == "-")])
    return s


def _apply_position_map(s: str, pattern: str) -> str:
    # pattern uses L/D, hyphen removed prior
    s = strip_chars(_normalize_raw_text(s), "-")
    if len(s) != len(pattern):
        return s
    out = []
    for ch, p in zip(s, pattern):
        if p == "D":
            out.append(DIGIT_LIKE.get(ch, ch))
        else:
            out.append(LETTER_LIKE.get(ch, ch))
    return "".join(out)


def _best_template_fit(text: str) -> Optional[PlateTemplate]:
    t = _normalize_raw_text(text)
    for tpl in TEMPLATES:
        if tpl.regex.match(t):
            return tpl
    return None


def normalize_plate(text: str, prefer: Optional[str] = None) -> str:
    """Template-aware normalization for Brazil/Mercosur.

    - Uppercase, strip spaces
    - Keep only [A-Z0-9-]
    - If matches a template: return canonical WITHOUT hyphen
    - Else: try position-aware confusion mapping for each template length
    """
    t = _normalize_raw_text(text)
    if not t:
        return t

    # If already matches a template, canonicalize
    tpl = _best_template_fit(t)
    if tpl:
        return strip_chars(t, "-")

    candidates: List[Tuple[int, str]] = []

    # Try to force to each template by length and confusion maps
    for tpl in TEMPLATES:
        raw = strip_chars(t, "-")
        if len(raw) != len(tpl.pattern):
            continue
        mapped = _apply_position_map(raw, tpl.pattern)
        # score: count invalid chars for positions
        score = 0
        for ch, p in zip(mapped, tpl.pattern):
            if p == "D" and ch not in DIGIT_SET:
                score += 1
            if p == "L" and ch not in LETTER_SET:
                score += 1
        candidates.append((score, mapped))

    if not candidates:
        return strip_chars(t, "-")

    candidates.sort(key=lambda x: x[0])
    best = candidates[0][1]

    if prefer:
        for tpl in TEMPLATES:
            if tpl.name == prefer and len(best) == len(tpl.pattern):
                # if preferred template can be matched after mapping
                mapped = _apply_position_map(strip_chars(t, "-"), tpl.pattern)
                if tpl.regex.match(mapped) or tpl.regex.match(mapped[:3] + "-" + mapped[3:]):
                    return mapped

    return best


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    log_probs_1: torch.Tensor,
    beam_size: int = 10,
    blank: int = 0,
) -> List[Tuple[List[int], float]]:
    """CTC prefix beam search for ONE sample.

    Args:
        log_probs_1: [T, C] log-probabilities
    Returns:
        List of (token_ids, log_score) sorted best-first.
    """
    T, C = log_probs_1.shape

    # beams: prefix -> (p_blank, p_nonblank)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}

    for t in range(T):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (pb, pnb) in beams.items():
            for c in range(C):
                p = float(log_probs_1[t, c].item())
                if c == blank:
                    nb_pb, nb_pnb = next_beams.get(prefix, (-math.inf, -math.inf))
                    nb_pb = _logsumexp(nb_pb, pb + p)
                    nb_pb = _logsumexp(nb_pb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)
                    continue

                end = prefix[-1] if prefix else None
                new_prefix = prefix + (c,)

                # If same as last char, only extend from blank
                if end == c:
                    nb_pb, nb_pnb = next_beams.get(prefix, (-math.inf, -math.inf))
                    nb_pnb = _logsumexp(nb_pnb, pnb + p)  # stay on same prefix
                    next_beams[prefix] = (nb_pb, nb_pnb)

                    nb_pb2, nb_pnb2 = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    nb_pnb2 = _logsumexp(nb_pnb2, pb + p)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)
                else:
                    nb_pb2, nb_pnb2 = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    nb_pnb2 = _logsumexp(nb_pnb2, pb + p)
                    nb_pnb2 = _logsumexp(nb_pnb2, pnb + p)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)

        # prune
        beams_scored = []
        for prefix, (pb, pnb) in next_beams.items():
            beams_scored.append((prefix, _logsumexp(pb, pnb)))
        beams_scored.sort(key=lambda x: x[1], reverse=True)
        beams = {p: next_beams[p] for p, _ in beams_scored[:beam_size]}

    out = []
    for prefix, (pb, pnb) in beams.items():
        score = _logsumexp(pb, pnb)
        out.append((list(prefix), score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def tokens_to_text(token_ids: Sequence[int], idx2char: Dict[int, str], blank: int = 0) -> str:
    collapsed = collapse_repeats_and_remove_blank(list(token_ids), blank=blank)
    chars = [idx2char[i] for i in collapsed if i in idx2char]
    return "".join(chars)


def decode_with_plate_constraints(
    log_probs: torch.Tensor,
    idx2char: Dict[int, str],
    beam_size: int = 10,
    prefer_template: Optional[str] = None,
    eval_strip: str = "-",
) -> List[str]:
    """Decode a batch with beam search then normalize to Brazil/Mercosur templates."""
    # log_probs: [B, T, C]
    B = log_probs.shape[0]
    out: List[str] = []
    for b in range(B):
        beams = ctc_prefix_beam_search(log_probs[b], beam_size=beam_size, blank=0)
        texts: List[Tuple[str, float]] = []
        for token_ids, score in beams:
            raw = tokens_to_text(token_ids, idx2char, blank=0)
            raw = strip_chars(raw, eval_strip)
            norm = normalize_plate(raw, prefer=prefer_template)
            texts.append((norm, score))

        # pick highest score among valid template matches if possible
        best = texts[0][0] if texts else ""
        best_score = -math.inf
        best_any = texts[0][0] if texts else ""
        best_any_score = texts[0][1] if texts else -math.inf

        for s, sc in texts:
            if sc > best_any_score:
                best_any, best_any_score = s, sc
            if BRA_OLD.match(s) or BR_MERCOSUR.match(s):
                if sc > best_score:
                    best, best_score = s, sc

        out.append(best if best_score > -math.inf else best_any)
    return out
