#!/usr/bin/env python3
"""verl / HybridFlow inspired single-controller GRPO for Whisper.

Control flow mirrors HybridFlow's colocated rollout -> advantage -> update
phases on one controller, without requiring Megatron/vLLM (Whisper is
encoder-decoder and does not fit those engines out of the box).
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
from benchmark.common.rewards import (
    RewardConfig,
    advantage_normalize,
    compute_asr_reward,
)


class RLDataset(Dataset):
    def __init__(self, samples, sr=16000):
        self.samples = samples
        self.sr = sr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "audio": load_audio_array(s.audio_path, self.sr),
            "reference": s.reference,
            "language": s.language,
            "id": s.sample_id or str(idx),
        }


def collate(batch):
    return batch


def build_model(cfg):
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


def freeze_copy(model):
    import copy

    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def tokenize(processor, texts, device):
    tok = processor.tokenizer(
        list(texts), return_tensors="pt", padding=True, truncation=True, max_length=448
    )
    labels = tok["input_ids"].to(device)
    labels = labels.masked_fill(tok["attention_mask"].to(device).ne(1), -100)
    return labels


def seq_logprob(model, features, labels, pad=-100):
    out = model(input_features=features, labels=labels)
    logits = out.logits[:, :-1]
    lab = labels[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    safe = lab.clone()
    safe[safe == pad] = 0
    token = logp.gather(2, safe.unsqueeze(-1)).squeeze(-1)
    mask = lab != pad
    return (token * mask).sum(-1) / mask.sum(-1).clamp(min=1)


class SingleController:
    """HybridFlow-like phases: rollout -> reward/advantage -> update."""

    def __init__(self, model, ref, processor, cfg, device):
        self.model = model
        self.ref = ref
        self.processor = processor
        self.cfg = cfg
        self.device = device
        self.vcfg = cfg.get("verl", {})
        self.reward_cfg = RewardConfig(**cfg.get("reward", {}))
        self.optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["learning_rate"]),
        )

    @torch.no_grad()
    def rollout_phase(self, item):
        n = int(self.vcfg.get("rollout_n", 4))
        temperature = float(self.vcfg.get("temperature", 0.7))
        feats = self.processor.feature_extractor(
            item["audio"],
            sampling_rate=int(self.cfg["data"]["sample_rate"]),
            return_tensors="pt",
        ).input_features.to(self.device)
        self.model.eval()
        kwargs = {
            "max_new_tokens": 128,
            "num_return_sequences": n,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if hasattr(self.processor, "get_decoder_prompt_ids"):
            kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=item["language"], task="transcribe"
            )
        ids = self.model.generate(feats, **kwargs)
        hyps = [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return feats, hyps

    def reward_phase(self, reference, hyps):
        rewards = [compute_asr_reward(reference, h, self.reward_cfg) for h in hyps]
        advantages = advantage_normalize(rewards)
        return rewards, advantages

    def update_phase(self, feats, hyps, advantages):
        n = len(hyps)
        labels = tokenize(self.processor, hyps, self.device)
        feats_g = feats.repeat(n, 1, 1)
        adv = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        kl_coef = float(self.vcfg.get("kl_coef", 0.04))

        self.model.train()
        logp = seq_logprob(self.model, feats_g, labels)
        with torch.no_grad():
            ref_logp = seq_logprob(self.ref, feats_g, labels)
        # Optional PPO-style clip on importance ratio vs a detached old policy
        # approximated by current forward before step (single-step GRPO).
        loss = -(adv.detach() * logp).mean() + kl_coef * (logp - ref_logp).mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return float(loss.detach().cpu())


def train(cfg, smoke=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_asr_jsonl(cfg["data"]["train_path"], cfg["model"]["language"])
    eval_samples = load_asr_jsonl(cfg["data"]["eval_path"], cfg["model"]["language"])
    if smoke:
        samples = samples[:4]
        eval_samples = eval_samples[:2]
        cfg["training"]["max_steps"] = min(int(cfg["training"].get("max_steps", 10)), 3)
        cfg.setdefault("verl", {})["rollout_n"] = min(
            int(cfg.get("verl", {}).get("rollout_n", 4)), 2
        )
    if not samples:
        raise RuntimeError("Empty train set. Run benchmark/scripts/generate_demo_data.py")

    model, processor = build_model(cfg)
    model.to(device)
    ref = freeze_copy(model).to(device)
    if cfg["training"].get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    controller = SingleController(model, ref, processor, cfg, device)
    loader = DataLoader(
        RLDataset(samples, int(cfg["data"]["sample_rate"])),
        batch_size=int(cfg.get("verl", {}).get("micro_batch_size", 1)),
        shuffle=True,
        collate_fn=collate,
    )

    max_steps = int(cfg["training"].get("max_steps", 30))
    step = 0
    t0 = time.time()
    metrics_path = out_dir / "train_metrics.jsonl"
    pbar = tqdm(total=max_steps, desc="verl-grpo")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            losses, reward_means = [], []
            for item in batch:
                feats, hyps = controller.rollout_phase(item)
                rewards, advantages = controller.reward_phase(item["reference"], hyps)
                loss = controller.update_phase(feats, hyps, advantages)
                losses.append(loss)
                reward_means.append(sum(rewards) / len(rewards))
            step += 1
            pbar.update(1)
            row = {
                "step": step,
                "loss": sum(losses) / max(len(losses), 1),
                "reward_mean": sum(reward_means) / max(len(reward_means), 1),
                "elapsed_sec": round(time.time() - t0, 2),
                "phase": "rollout->reward->update",
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            pbar.set_postfix(loss=row["loss"], reward=row["reward_mean"])
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
        "framework": "verl",
        "algorithm": "grpo",
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
    save = out_dir / "checkpoint-final"
    model.save_pretrained(save)
    processor.save_pretrained(save)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "benchmark/configs/frameworks/verl.yaml"),
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
