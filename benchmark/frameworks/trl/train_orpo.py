#!/usr/bin/env python3
"""TRL ORPO preference fine-tuning adapter for Whisper (offline RLHF).

Uses preference pairs (chosen / rejected) and an ORPO-style odds-ratio loss,
aligned with HuggingFace TRL's ORPO objective but implemented for Whisper's
encoder-decoder interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common.config import load_yaml
from benchmark.common.data import load_asr_jsonl, load_audio_array
from benchmark.common.metrics import compute_cer, compute_wer


class PrefDataset(Dataset):
    def __init__(self, samples, sample_rate=16000):
        self.samples = [s for s in samples if (s.chosen or s.reference) and s.rejected]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "audio": load_audio_array(s.audio_path, self.sample_rate),
            "chosen": s.chosen or s.reference,
            "rejected": s.rejected,
            "reference": s.reference,
            "language": s.language,
            "id": s.sample_id or str(idx),
        }


def collate(batch):
    return batch


def build_model_and_processor(cfg):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model

    name = cfg["model"]["name_or_path"]
    processor = WhisperProcessor.from_pretrained(
        name, language=cfg["model"]["language"], task=cfg["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if cfg["model"].get("use_lora", True):
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(cfg["model"].get("lora_r", 16)),
                lora_alpha=int(cfg["model"].get("lora_alpha", 32)),
                lora_dropout=float(cfg["model"].get("lora_dropout", 0.05)),
                target_modules=list(
                    cfg["model"].get("target_modules", ["q_proj", "v_proj"])
                ),
                bias="none",
            ),
        )
    return model, processor


def tokenize(processor, texts, device):
    tok = processor.tokenizer(
        list(texts), return_tensors="pt", padding=True, truncation=True, max_length=448
    )
    labels = tok["input_ids"].to(device)
    labels = labels.masked_fill(tok["attention_mask"].to(device).ne(1), -100)
    return labels


def avg_logp(logits, labels, pad=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe = shift_labels.clone()
    safe[safe == pad] = 0
    token_lp = log_probs.gather(2, safe.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != pad
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def orpo_loss(model, features, chosen_labels, rejected_labels, beta: float):
    out_pos = model(input_features=features, labels=chosen_labels)
    out_neg = model(input_features=features, labels=rejected_labels)
    pos_lp = avg_logp(out_pos.logits, chosen_labels)
    neg_lp = avg_logp(out_neg.logits, rejected_labels)
    # Odds-ratio term (same spirit as ORPO / TRL ORPOTrainer).
    log_odds = (pos_lp - neg_lp) - (
        torch.log1p(-torch.exp(pos_lp.clamp(max=-1e-6)))
        - torch.log1p(-torch.exp(neg_lp.clamp(max=-1e-6)))
    )
    ratio = torch.log(torch.sigmoid(log_odds))
    loss = (out_pos.loss - beta * ratio).mean()
    return loss, out_pos.loss.detach()


def train(cfg, smoke=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pref_train = cfg["data"].get("preference_train_path") or cfg["data"]["train_path"]
    pref_eval = cfg["data"].get("preference_eval_path") or cfg["data"]["eval_path"]
    samples = load_asr_jsonl(pref_train, cfg["model"]["language"])
    eval_samples = load_asr_jsonl(pref_eval, cfg["model"]["language"])
    if smoke:
        samples = samples[:4]
        eval_samples = eval_samples[:2]
        cfg["training"]["max_steps"] = min(int(cfg["training"].get("max_steps", 10)), 3)

    ds = PrefDataset(samples, int(cfg["data"]["sample_rate"]))
    if len(ds) == 0:
        raise RuntimeError(
            f"No preference pairs in {pref_train}. "
            "Run: python benchmark/scripts/generate_demo_data.py"
        )

    model, processor = build_model_and_processor(cfg)
    model.to(device)
    if cfg["training"].get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
    )
    beta = float(cfg.get("orpo", {}).get("beta", 0.1))
    max_steps = int(cfg["training"].get("max_steps", 50))
    step = 0
    t0 = time.time()
    metrics_path = out_dir / "train_metrics.jsonl"

    pbar = tqdm(total=max_steps, desc="trl-orpo")
    model.train()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            audios = [b["audio"] for b in batch]
            feats = processor.feature_extractor(
                audios,
                sampling_rate=int(cfg["data"]["sample_rate"]),
                return_tensors="pt",
            ).input_features.to(device)
            chosen = tokenize(processor, [b["chosen"] for b in batch], device)
            rejected = tokenize(processor, [b["rejected"] for b in batch], device)
            loss, nll = orpo_loss(model, feats, chosen, rejected, beta)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            pbar.update(1)
            if step % int(cfg["training"].get("logging_steps", 5)) == 0 or step == 1:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "nll": float(nll.cpu()),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                pbar.set_postfix(loss=row["loss"])
    pbar.close()

    eval_rows = []
    model.eval()
    for s in eval_samples:
        audio = load_audio_array(s.audio_path, int(cfg["data"]["sample_rate"]))
        feats = processor.feature_extractor(
            audio,
            sampling_rate=int(cfg["data"]["sample_rate"]),
            return_tensors="pt",
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=128)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        eval_rows.append(
            {
                "id": s.sample_id,
                "reference": s.reference,
                "hypothesis": hyp,
                "wer": compute_wer(s.reference, hyp),
                "cer": compute_cer(s.reference, hyp),
            }
        )
    summary = {
        "framework": "trl",
        "algorithm": "orpo",
        "steps": step,
        "eval_wer": sum(r["wer"] for r in eval_rows) / max(len(eval_rows), 1),
        "eval_cer": sum(r["cer"] for r in eval_rows) / max(len(eval_rows), 1),
        "device": str(device),
        "smoke": smoke,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "eval_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8",
    )
    save_path = out_dir / "checkpoint-final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "benchmark/configs/frameworks/trl_orpo.yaml"),
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    os.chdir(REPO_ROOT)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
