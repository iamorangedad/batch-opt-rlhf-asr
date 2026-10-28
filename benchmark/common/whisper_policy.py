"""Thin Whisper generation helpers used by RL adapters."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


@torch.inference_mode()
def generate_transcriptions(
    model,
    processor,
    input_features: torch.Tensor,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    language: Optional[str] = "en",
    task: str = "transcribe",
) -> List[str]:
    """Sample one or more transcriptions for a single (or batched) feature tensor.

    When ``num_return_sequences > 1``, the first batch item is expanded and
    returned as a flat list of decoded strings (GRPO group generation).
    """
    device = input_features.device
    model.eval()

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5) if temperature > 0 else None,
        "num_return_sequences": num_return_sequences,
    }
    # Drop None values for transformers compatibility.
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # Pin language / task when the processor supports forced decoder ids.
    if hasattr(processor, "get_decoder_prompt_ids") and language is not None:
        forced = processor.get_decoder_prompt_ids(language=language, task=task)
        gen_kwargs["forced_decoder_ids"] = forced

    if num_return_sequences > 1 and input_features.shape[0] != 1:
        # Expand each item separately to keep group membership clear.
        all_texts: List[str] = []
        for i in range(input_features.shape[0]):
            feats = input_features[i : i + 1]
            ids = model.generate(feats, **gen_kwargs)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            all_texts.extend([t.strip() for t in texts])
        return all_texts

    ids = model.generate(input_features, **gen_kwargs)
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def greedy_transcribe(
    model,
    processor,
    input_features: torch.Tensor,
    language: Optional[str] = "en",
    task: str = "transcribe",
    max_new_tokens: int = 128,
) -> List[str]:
    """Deterministic decode used for validation WER/CER."""
    return generate_transcriptions(
        model,
        processor,
        input_features,
        num_return_sequences=1,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        language=language,
        task=task,
    )
