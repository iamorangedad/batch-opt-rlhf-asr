"""ASR evaluation metrics: WER / CER with light text normalization."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    """Normalize transcript text for fair WER/CER comparison."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Classic Levenshtein distance over token/char sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * m
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Word Error Rate. Returns 0.0 for empty reference and empty hypothesis."""
    ref = normalize_text(reference) if normalize else reference.strip()
    hyp = normalize_text(hypothesis) if normalize else hypothesis.strip()
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Character Error Rate (spaces removed for CJK-friendly scoring)."""
    ref = normalize_text(reference) if normalize else reference.strip()
    hyp = normalize_text(hypothesis) if normalize else hypothesis.strip()
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def aggregate_error_rates(
    pairs: Iterable[Tuple[str, str]], metric: str = "wer"
) -> float:
    """Micro-average WER/CER over (reference, hypothesis) pairs."""
    refs: List[str] = []
    hyps: List[str] = []
    for ref, hyp in pairs:
        refs.append(ref)
        hyps.append(hyp)

    if metric == "wer":
        total_edits = 0
        total_tokens = 0
        for ref, hyp in zip(refs, hyps):
            r = normalize_text(ref).split()
            h = normalize_text(hyp).split()
            if not r:
                continue
            total_edits += _edit_distance(r, h)
            total_tokens += len(r)
        return total_edits / max(total_tokens, 1)

    if metric == "cer":
        total_edits = 0
        total_chars = 0
        for ref, hyp in zip(refs, hyps):
            r = list(normalize_text(ref).replace(" ", ""))
            h = list(normalize_text(hyp).replace(" ", ""))
            if not r:
                continue
            total_edits += _edit_distance(r, h)
            total_chars += len(r)
        return total_edits / max(total_chars, 1)

    raise ValueError(f"Unsupported metric: {metric}")
