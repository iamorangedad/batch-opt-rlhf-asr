#!/usr/bin/env python3
"""Prepare real ASR datasets (LibriSpeech / Common Voice style) for the benchmark.

Examples:
  # From a local JSON/JSONL already in repo format
  python benchmark/scripts/prepare_data.py --input /data/train.jsonl --out_dir benchmark/data/librispeech

  # From HuggingFace datasets (requires network + datasets)
  python benchmark/scripts/prepare_data.py --hf_dataset librispeech_asr --hf_config clean \
      --hf_split train.100 --out_dir benchmark/data/librispeech --max_samples 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common.data import (
    ASRSample,
    dump_preference_jsonl,
    dump_rl_jsonl,
    load_asr_jsonl,
)


def from_hf(args) -> list[ASRSample]:
    from datasets import load_dataset
    import soundfile as sf

    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, streaming=False)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    out_audio = Path(args.out_dir) / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    samples: list[ASRSample] = []
    for i, row in enumerate(ds):
        text = row.get("text") or row.get("sentence") or row.get("transcription") or ""
        audio = row["audio"]
        array = audio["array"]
        sr = int(audio["sampling_rate"])
        wav_path = out_audio / f"{args.hf_split.replace('.', '_')}_{i:06d}.wav"
        sf.write(str(wav_path), array, sr)
        duration = float(len(array) / sr)
        # Build a simple rejected hypothesis by token shuffle for ORPO demos.
        toks = text.strip().split()
        rejected = " ".join(reversed(toks)) if len(toks) > 1 else (text + " <unk>")
        samples.append(
            ASRSample(
                audio_path=str(wav_path),
                reference=text.strip(),
                language=args.language,
                duration=duration,
                chosen=text.strip(),
                rejected=rejected,
                sample_id=f"{args.hf_split}-{i}",
            )
        )
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Existing JSON/JSONL path")
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.input:
        samples = load_asr_jsonl(args.input, default_language=args.language)
    elif args.hf_dataset:
        samples = from_hf(args)
    else:
        raise SystemExit("Provide --input or --hf_dataset")

    n_eval = max(1, int(len(samples) * args.eval_ratio)) if len(samples) > 1 else 0
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:] if n_eval else samples

    dump_rl_jsonl(train_samples, out / "train_rl.jsonl")
    dump_rl_jsonl(eval_samples or train_samples[:1], out / "eval_rl.jsonl")
    dump_preference_jsonl(train_samples, out / "train_pref.jsonl")
    dump_preference_jsonl(eval_samples or train_samples[:1], out / "eval_pref.jsonl")

    meta = {
        "num_train": len(train_samples),
        "num_eval": len(eval_samples) or 1,
        "language": args.language,
        "source": args.input or f"{args.hf_dataset}:{args.hf_config}:{args.hf_split}",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
