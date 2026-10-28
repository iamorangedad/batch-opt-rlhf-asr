#!/usr/bin/env python3
"""Generate a tiny local ASR demo dataset for RL / preference benchmarks.

Creates short sine-wave WAVs with known reference transcripts. The audio is
not speech — it only validates the end-to-end data and training pipelines.
For real benchmarks, replace with LibriSpeech / Common Voice via prepare_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_sine_wav(path: Path, freq: float, duration: float, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            # Quiet tone; enough energy for feature extraction.
            val = int(0.2 * 32767 * math.sin(2 * math.pi * freq * (i / sr)))
            frames += int(val).to_bytes(2, byteorder="little", signed=True)
        wf.writeframes(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "benchmark/data/demo"),
    )
    p.add_argument("--num_train", type=int, default=8)
    p.add_argument("--num_eval", type=int, default=4)
    args = p.parse_args()

    out = Path(args.out_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    phrases = [
        "the quick brown fox jumps over the lazy dog",
        "open the window and listen to the rain",
        "reinforcement learning improves speech recognition",
        "whisper models can be fine tuned with rewards",
        "group relative policy optimization is sample efficient",
        "word error rate is a common evaluation metric",
        "please transcribe this short audio clip carefully",
        "benchmark frameworks include trl openrlhf and verl",
        "preference data needs chosen and rejected transcripts",
        "character error rate helps for tonal languages",
        "mixed precision training reduces memory usage",
        "low rank adapters keep most weights frozen",
    ]

    def make_split(n: int, split: str, start_freq: float):
        rl_rows = []
        pref_rows = []
        for i in range(n):
            text = phrases[i % len(phrases)]
            duration = 1.0 + (i % 3) * 0.25
            wav = audio_dir / f"{split}_{i:03d}.wav"
            write_sine_wav(wav, freq=start_freq + i * 17.0, duration=duration)
            # Mild corruption for rejected preference text.
            toks = text.split()
            rejected = " ".join(toks[::-1] if len(toks) > 2 else toks + ["noise"])
            row_rl = {
                "audio": {"path": str(wav.relative_to(REPO_ROOT))},
                "duration": duration,
                "reference": text,
                "language": "en",
                "id": f"{split}-{i}",
            }
            row_pref = {
                **row_rl,
                "chosen": text,
                "rejected": rejected,
            }
            rl_rows.append(row_rl)
            pref_rows.append(row_pref)
        return rl_rows, pref_rows

    train_rl, train_pref = make_split(args.num_train, "train", 220.0)
    eval_rl, eval_pref = make_split(args.num_eval, "eval", 330.0)

    def dump(path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out / "train_rl.jsonl", train_rl)
    dump(out / "eval_rl.jsonl", eval_rl)
    dump(out / "train_pref.jsonl", train_pref)
    dump(out / "eval_pref.jsonl", eval_pref)

    meta = {
        "note": "Synthetic tones for pipeline validation only — not real speech.",
        "num_train": args.num_train,
        "num_eval": args.num_eval,
        "files": [
            "train_rl.jsonl",
            "eval_rl.jsonl",
            "train_pref.jsonl",
            "eval_pref.jsonl",
            "audio/",
        ],
    }
    (out / "README.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote demo dataset to {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
