"""Rule-based ASR rewards for RL fine-tuning (GRPO / PPO).

Reward is derived from verifiable transcription quality (WER/CER),
optionally with length and repetition penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .metrics import compute_cer, compute_wer, normalize_text


@dataclass
class RewardConfig:
    """Weights for the blended ASR reward."""

    wer_weight: float = 0.7
    cer_weight: float = 0.3
    length_penalty: float = 0.05
    repetition_penalty: float = 0.1
    # Clamp reward to [-1, 1] for stable advantages.
    clip: bool = True


def _length_ratio_penalty(reference: str, hypothesis: str) -> float:
    ref_len = max(len(normalize_text(reference).split()), 1)
    hyp_len = len(normalize_text(hypothesis).split())
    ratio = hyp_len / ref_len
    # Penalize strong under/over generation.
    return abs(1.0 - ratio)


def _repetition_score(hypothesis: str) -> float:
    tokens = normalize_text(hypothesis).split()
    if len(tokens) < 2:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


def compute_asr_reward(
    reference: str,
    hypothesis: str,
    config: Optional[RewardConfig] = None,
) -> float:
    """Higher is better. Perfect transcript -> reward near +1."""
    cfg = config or RewardConfig()
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    length_pen = _length_ratio_penalty(reference, hypothesis)
    rep_pen = _repetition_score(hypothesis)

    # Map error rates to reward: 0 error -> 1.0, high error -> negative.
    quality = 1.0 - (cfg.wer_weight * wer + cfg.cer_weight * cer)
    reward = quality - cfg.length_penalty * length_pen - cfg.repetition_penalty * rep_pen
    if cfg.clip:
        reward = max(-1.0, min(1.0, reward))
    return float(reward)


def compute_group_rewards(
    reference: str,
    hypotheses: Sequence[str],
    config: Optional[RewardConfig] = None,
) -> List[float]:
    """Score a GRPO generation group for one audio clip."""
    return [compute_asr_reward(reference, h, config) for h in hypotheses]


def advantage_normalize(rewards: Sequence[float], eps: float = 1e-8) -> List[float]:
    """Group-relative advantage: (r - mean) / (std + eps)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
    std = var**0.5
    return [(r - mean) / (std + eps) for r in rewards]
