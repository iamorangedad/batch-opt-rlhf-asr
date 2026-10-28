"""Unified ASR dataset helpers for preference and RL (prompt/ref) formats."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


@dataclass
class ASRSample:
    """Canonical sample used across all framework adapters."""

    audio_path: str
    reference: str
    language: str = "en"
    duration: Optional[float] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    sample_id: Optional[str] = None

    def to_preference_dict(self) -> Dict[str, Any]:
        """JSONL row compatible with the repo's native ORPO trainer."""
        return {
            "audio": {"path": self.audio_path},
            "duration": self.duration if self.duration is not None else 0.0,
            "chosen": self.chosen or self.reference,
            "rejected": self.rejected or "",
            "language": self.language,
            "reference": self.reference,
            "id": self.sample_id,
        }

    def to_rl_dict(self) -> Dict[str, Any]:
        """JSONL row for online RL (GRPO/PPO): audio + reference only."""
        return {
            "audio": {"path": self.audio_path},
            "duration": self.duration if self.duration is not None else 0.0,
            "reference": self.reference,
            "language": self.language,
            "id": self.sample_id,
        }


def _read_json_or_jsonl(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON array
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return data
    # JSONL / one JSON object per line
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
    return rows


def _extract_audio_path(row: dict) -> str:
    audio = row.get("audio")
    if isinstance(audio, str):
        return audio
    if isinstance(audio, dict) and "path" in audio:
        return audio["path"]
    if "audio_path" in row:
        return row["audio_path"]
    raise KeyError(f"Cannot find audio path in row keys={list(row.keys())}")


def load_asr_jsonl(
    path: Union[str, Path],
    default_language: str = "en",
) -> List[ASRSample]:
    """Load preference or RL ASR samples from JSON / JSONL."""
    rows = _read_json_or_jsonl(path)
    samples: List[ASRSample] = []
    for idx, row in enumerate(rows):
        audio_path = _extract_audio_path(row)
        reference = (
            row.get("reference")
            or row.get("chosen")
            or row.get("text")
            or row.get("sentence")
            or ""
        )
        samples.append(
            ASRSample(
                audio_path=audio_path,
                reference=reference,
                language=row.get("language", default_language),
                duration=row.get("duration"),
                chosen=row.get("chosen"),
                rejected=row.get("rejected"),
                sample_id=str(row.get("id", idx)),
            )
        )
    return samples


def write_jsonl(path: Union[str, Path], rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_preference_jsonl(samples: Sequence[ASRSample], path: Union[str, Path]) -> None:
    write_jsonl(path, (s.to_preference_dict() for s in samples))


def dump_rl_jsonl(samples: Sequence[ASRSample], path: Union[str, Path]) -> None:
    write_jsonl(path, (s.to_rl_dict() for s in samples))


def load_audio_array(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load mono float32 audio at the target sample rate."""
    try:
        import soundfile as sf
        import librosa
    except ImportError as exc:
        raise ImportError(
            "soundfile and librosa are required to load audio. "
            "Install via: pip install soundfile librosa"
        ) from exc

    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)


def prepare_whisper_batch(
    processor,
    audio_arrays: Sequence[np.ndarray],
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Feature-extract a list of waveforms for Whisper."""
    features = processor.feature_extractor(
        list(audio_arrays),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    return dict(features)


def samples_to_hf_dataset(samples: Sequence[ASRSample]):
    """Convert samples to a HuggingFace datasets.Dataset for TRL adapters."""
    from datasets import Dataset

    rows = [s.to_rl_dict() for s in samples]
    # Flatten audio path for simpler column access.
    for r in rows:
        r["audio_path"] = r["audio"]["path"]
    return Dataset.from_list(rows)
