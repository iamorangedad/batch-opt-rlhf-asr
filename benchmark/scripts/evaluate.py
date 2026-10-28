#!/usr/bin/env python3
"""Evaluate a Whisper checkpoint (or base model) on a JSONL eval set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common.data import load_asr_jsonl, load_audio_array
from benchmark.common.gpu_memory import GpuMemoryTracker
from benchmark.common.metrics import aggregate_error_rates, compute_cer, compute_wer


def load_model(model_path: str, base_model: str | None = None):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    processor_src = base_model or model_path
    processor = WhisperProcessor.from_pretrained(processor_src)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    except Exception:
        if not base_model:
            raise
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None, help="Required when model_path is a LoRA adapter")
    p.add_argument("--eval_path", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--output", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = GpuMemoryTracker(device)
    model, processor = load_model(args.model_path, args.base_model)
    model.to(device)
    mem.sample()

    samples = load_asr_jsonl(args.eval_path, args.language)
    if args.max_samples:
        samples = samples[: args.max_samples]

    rows = []
    for s in samples:
        audio = load_audio_array(s.audio_path)
        feats = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=128)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        rows.append(
            {
                "id": s.sample_id,
                "reference": s.reference,
                "hypothesis": hyp,
                "wer": compute_wer(s.reference, hyp),
                "cer": compute_cer(s.reference, hyp),
            }
        )

    summary = {
        "model_path": args.model_path,
        "num_samples": len(rows),
        "wer": aggregate_error_rates(((r["reference"], r["hypothesis"]) for r in rows), "wer"),
        "cer": aggregate_error_rates(((r["reference"], r["hypothesis"]) for r in rows), "cer"),
        "device": str(device),
    }
    summary.update(mem.summary())
    out = Path(args.output) if args.output else Path(args.model_path) / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "predictions": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
