#!/usr/bin/env python3
"""TRL-style GRPO fine-tuning for Whisper with WER/CER rewards.

Whisper is encoder-decoder, so we implement a self-contained GRPO loop that
follows the same group-relative advantage recipe used by HuggingFace TRL,
without depending on decoder-only GRPOTrainer APIs.
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

# Allow `from benchmark.common ...` when run from repo root.
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


class WhisperRLDataset(Dataset):
    def __init__(self, samples, sample_rate: int = 16000):
        self.samples = samples
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = load_audio_array(s.audio_path, self.sample_rate)
        return {
            "audio": audio,
            "reference": s.reference,
            "language": s.language,
            "id": s.sample_id or str(idx),
        }


def collate_keep_list(batch):
    return batch


def build_model_and_processor(cfg):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model

    model_name = cfg["model"]["name_or_path"]
    language = cfg["model"]["language"]
    task = cfg["model"]["task"]

    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if cfg["model"].get("use_lora", True):
        # Do not set task_type=SEQ_2_SEQ_LM: that wraps Whisper in
        # PeftModelForSeq2SeqLM whose generate() expects text input_ids.
        lora = LoraConfig(
            r=int(cfg["model"].get("lora_r", 16)),
            lora_alpha=int(cfg["model"].get("lora_alpha", 32)),
            lora_dropout=float(cfg["model"].get("lora_dropout", 0.05)),
            target_modules=list(cfg["model"].get("target_modules", ["q_proj", "v_proj"])),
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    return model, processor


def clone_reference(model):
    """Frozen reference policy for KL penalty."""
    import copy

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_model


def sequence_logprob(model, input_features, labels, pad_token_id: int = -100):
    """Average token log-prob of `labels` under the model."""
    outputs = model(input_features=input_features, labels=labels)
    logits = outputs.logits  # [B, T, V]
    # Whisper shifts labels internally for loss; align manually for logprob.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == pad_token_id] = 0
    token_logp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != pad_token_id
    seq_logp = (token_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return seq_logp, outputs.loss


def tokenize_transcripts(processor, texts, device):
    tok = processor.tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=448,
    )
    labels = tok["input_ids"].to(device)
    labels = labels.masked_fill(tok["attention_mask"].to(device).ne(1), -100)
    return labels


@torch.no_grad()
def sample_group(model, processor, features, language, n, temperature, max_new_tokens):
    model.eval()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": n,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if hasattr(processor, "get_decoder_prompt_ids"):
        gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
    ids = model.generate(features, **gen_kwargs)
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def train(cfg: dict, smoke: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_asr_jsonl(
        cfg["data"]["train_path"],
        default_language=cfg["model"]["language"],
    )
    eval_samples = load_asr_jsonl(
        cfg["data"]["eval_path"],
        default_language=cfg["model"]["language"],
    )
    if smoke:
        samples = samples[: max(2, min(4, len(samples)))]
        eval_samples = eval_samples[: max(1, min(2, len(eval_samples)))]
        cfg["training"]["max_steps"] = min(int(cfg["training"].get("max_steps", 10)), 3)
        cfg.setdefault("grpo", {})["num_generations"] = min(
            int(cfg.get("grpo", {}).get("num_generations", 4)), 2
        )

    if not samples:
        raise RuntimeError(
            f"No training samples in {cfg['data']['train_path']}. "
            "Run: python benchmark/scripts/generate_demo_data.py"
        )

    model, processor = build_model_and_processor(cfg)
    model.to(device)
    ref_model = clone_reference(model)
    ref_model.to(device)

    if cfg["training"].get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    reward_cfg = RewardConfig(**cfg.get("reward", {}))
    grpo = cfg.get("grpo", {})
    num_generations = int(grpo.get("num_generations", 4))
    temperature = float(grpo.get("temperature", 0.7))
    beta = float(grpo.get("beta", 0.04))
    max_new_tokens = int(grpo.get("max_completion_length", 128))

    ds = WhisperRLDataset(samples, sample_rate=int(cfg["data"]["sample_rate"]))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collate_keep_list,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
    )

    max_steps = int(cfg["training"].get("max_steps", 50))
    log_every = int(cfg["training"].get("logging_steps", 5))
    metrics_path = out_dir / "train_metrics.jsonl"
    step = 0
    t0 = time.time()

    model.train()
    pbar = tqdm(total=max_steps, desc="trl-grpo")
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            # Process one example at a time for clear GRPO groups.
            losses = []
            reward_means = []
            for item in batch:
                audio = item["audio"]
                reference = item["reference"]
                language = item["language"]
                feats = processor.feature_extractor(
                    audio,
                    sampling_rate=int(cfg["data"]["sample_rate"]),
                    return_tensors="pt",
                ).input_features.to(device)

                hyps = sample_group(
                    model,
                    processor,
                    feats,
                    language=language,
                    n=num_generations,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                rewards = [compute_asr_reward(reference, h, reward_cfg) for h in hyps]
                advantages = torch.tensor(
                    advantage_normalize(rewards), device=device, dtype=torch.float32
                )
                reward_means.append(sum(rewards) / len(rewards))

                labels = tokenize_transcripts(processor, hyps, device)
                # Repeat features for the group.
                feats_g = feats.repeat(num_generations, 1, 1)

                model.train()
                logp, nll = sequence_logprob(model, feats_g, labels)
                with torch.no_grad():
                    ref_logp, _ = sequence_logprob(ref_model, feats_g, labels)
                kl = logp - ref_logp
                # GRPO-style surrogate: -A * logπ + β * KL
                loss = -(advantages.detach() * logp).mean() + beta * kl.mean()
                losses.append(loss)

            if not losses:
                continue
            total_loss = torch.stack(losses).mean()
            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            step += 1
            pbar.update(1)
            if step % log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "loss": float(total_loss.detach().cpu()),
                    "reward_mean": float(sum(reward_means) / max(len(reward_means), 1)),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                pbar.set_postfix(loss=row["loss"], reward=row["reward_mean"])

    pbar.close()

    # Quick eval
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
            ids = model.generate(feats, max_new_tokens=max_new_tokens)
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

    mean_wer = sum(r["wer"] for r in eval_rows) / max(len(eval_rows), 1)
    mean_cer = sum(r["cer"] for r in eval_rows) / max(len(eval_rows), 1)
    summary = {
        "framework": "trl",
        "algorithm": "grpo",
        "steps": step,
        "eval_wer": mean_wer,
        "eval_cer": mean_cer,
        "device": str(device),
        "smoke": smoke,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "eval_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8",
    )

    # Save adapter / model
    save_path = out_dir / "checkpoint-final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="TRL-style GRPO for Whisper ASR")
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "benchmark/configs/frameworks/trl_grpo.yaml"),
    )
    p.add_argument("--smoke", action="store_true", help="Tiny run for CI / env check")
    p.add_argument("--train_path", default=None)
    p.add_argument("--eval_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.train_path:
        cfg["data"]["train_path"] = args.train_path
    if args.eval_path:
        cfg["data"]["eval_path"] = args.eval_path
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    # Resolve relative data paths from repo root.
    os.chdir(REPO_ROOT)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
