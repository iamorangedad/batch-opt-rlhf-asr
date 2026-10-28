#!/usr/bin/env python3
"""CPU-only unit tests for shared metrics/rewards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.common.metrics import compute_cer, compute_wer, normalize_text
from benchmark.common.rewards import (
    RewardConfig,
    advantage_normalize,
    compute_asr_reward,
    compute_group_rewards,
)


class MetricsTest(unittest.TestCase):
    def test_perfect_wer(self):
        self.assertEqual(compute_wer("hello world", "hello world"), 0.0)

    def test_substitution_wer(self):
        self.assertAlmostEqual(compute_wer("hello world", "hello there"), 0.5)

    def test_normalize(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_cer(self):
        self.assertEqual(compute_cer("abc", "abc"), 0.0)
        self.assertGreater(compute_cer("abc", "axc"), 0.0)


class RewardTest(unittest.TestCase):
    def test_reward_ranking(self):
        ref = "reinforcement learning for speech"
        good = ref
        bad = "totally unrelated text here"
        r_good = compute_asr_reward(ref, good)
        r_bad = compute_asr_reward(ref, bad)
        self.assertGreater(r_good, r_bad)

    def test_group_advantages(self):
        ref = "one two three"
        hyps = [ref, "one two", "zzz"]
        rewards = compute_group_rewards(ref, hyps, RewardConfig())
        adv = advantage_normalize(rewards)
        self.assertEqual(len(adv), 3)
        self.assertAlmostEqual(sum(adv), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
