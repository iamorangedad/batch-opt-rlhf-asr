# Batch-Opt-RLHF-ASR

A PyTorch implementation of Reinforcement Learning from Human Feedback (RLHF) for Automatic Speech Recognition (ASR) using the Whisper model. This project implements the ORPO (Odds Ratio Preference Optimization) algorithm to fine-tune Whisper models with preference data.
By optimizing data batch processing, this project reduces memory consumption by more than 50%.

## Project Overview

This project enables fine-tuning of Whisper ASR models using preference-based learning, where the model learns to prefer high-quality transcriptions over low-quality ones. The implementation uses the ORPO algorithm, which is an efficient alternative to traditional RLHF methods.
By optimizing data batch processing, this project reduces memory consumption by more than 50%.

## Architecture

### Core Components

- **CustomWhisperModel**: Extended Whisper model with custom forward pass for preference learning
- **ORPOTrainer**: Custom trainer implementing the ORPO loss function
- **CustomDataset**: Dataset loader for audio-text preference pairs
- **DataCollatorSpeechSeq2SeqWithPadding**: Custom data collator for batch processing

### Key Features

- **Preference Learning**: Uses positive/negative transcription pairs for training
- **ORPO Algorithm**: Implements odds ratio preference optimization
- **Multi-GPU Support**: Distributed training with PyTorch DDP
- **PEFT Integration**: Support for LoRA and AdaLoRA fine-tuning
- **Audio Processing**: Built-in audio preprocessing and augmentation
- **Flexible Configuration**: Extensive hyperparameter tuning options

## Project Structure

```
batch-opt-rlhf-asr/
├── finetune.py              # Main ORPO training entrypoint
├── scripts/
│   ├── setup_uv.sh          # Bootstrap Python env with uv
│   └── run_rlhf_asr.sh      # Multi-GPU training launcher
├── src/
│   ├── model.py             # Custom Whisper model
│   ├── orpo_trainer.py      # ORPO trainer / loss
│   ├── data_reader.py       # Preference dataset loader
│   ├── data_utils.py        # Collation and text utils
│   ├── log.py               # Logging helpers
│   └── utils.py             # CLI / download helpers
├── pyproject.toml           # uv dependency declaration
├── uv.lock                  # Locked dependency graph
└── benchmark/               # Optional comparison suite (see below)
```

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended)
- Python 3.10+ (installed by `.python-version` / uv; default 3.12)

### Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
bash scripts/setup_uv.sh base
# equivalent: uv sync
```

### Data Format

Your training data should be in JSON / JSONL format with the following structure:

```json
{
  "audio": {"path": "path/to/audio.wav"},
  "duration": 10.5,
  "chosen": "high quality transcription",
  "rejected": "low quality transcription",
  "language": "Chinese"
}
```

### Training

1. **Single GPU Training**:
```bash
uv run python finetune.py \
  --base_model openai/whisper-tiny \
  --train_data train.json \
  --test_data test.json \
  --output_dir ./output \
  --num_train_epochs 3
```

2. **Multi-GPU Training**:
```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 finetune.py \
  --base_model openai/whisper-tiny \
  --train_data train.json \
  --test_data test.json \
  --output_dir ./output
```

## Configuration Options

### Model Configuration
- `--base_model`: Base Whisper model (default: "openai/whisper-tiny")
- `--language`: Target language for transcription (default: "Chinese")
- `--task`: Task type - "transcribe" or "translate" (default: "transcribe")

### Training Configuration
- `--learning_rate`: Learning rate (default: 1e-3)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--warmup_steps`: Warmup steps (default: 50)

### ORPO Configuration
- `--alpha`: Hyperparameter for weighting L_OR (default: 1.0)
- `--disable_prompt_loss`: Disable prompt loss calculation (default: False)

### Audio Processing
- `--min_audio_len`: Minimum audio duration in seconds (default: 0.5)
- `--max_audio_len`: Maximum audio duration in seconds (default: 30)
- `--timestamps`: Use timestamp data during training (default: False)

### Optimization
- `--use_adalora`: Use AdaLoRA instead of LoRA (default: True)
- `--fp16`: Use FP16 training (default: True)
- `--use_8bit`: Use 8-bit quantization (default: False)
- `--use_compile`: Use PyTorch 2.0 compiler (default: False)

## Technical Details

### ORPO Algorithm

The ORPO algorithm optimizes the odds ratio between preferred and rejected outputs:

```
L_OR = L_NLL - α * log(σ(log_odds))
```

Where:
- `L_NLL`: Negative log-likelihood loss for preferred outputs
- `log_odds`: Log odds ratio between preferred and rejected outputs
- `α`: Weighting hyperparameter
- `σ`: Sigmoid function

### Custom Model Architecture

The `CustomWhisperModel` extends the standard Whisper model with:
- Custom forward pass for preference learning
- Enhanced encoder-decoder attention mechanism
- Support for both positive and negative label processing

### Data Processing Pipeline

1. **Audio Loading**: Load and preprocess audio files
2. **Feature Extraction**: Extract log-Mel spectrogram features
3. **Tokenization**: Tokenize positive and negative transcriptions
4. **Batch Collation**: Pad sequences for batch processing

## Performance Optimization

### Memory Efficiency
- Gradient checkpointing
- Mixed precision training (FP16)
- 8-bit quantization support
- Efficient data loading with multiple workers

### Training Speed
- PyTorch 2.0 compilation
- Distributed training support
- Optimized data collation
- Batch processing optimizations

## Advanced Usage

### Custom Loss Functions

You can extend the ORPO trainer to implement custom loss functions:

```python
class CustomORPOTrainer(ORPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss implementation
        pass
```

### Data Augmentation

Configure audio augmentation through the `augment_config_path` parameter:

```json
{
  "noise": {"type": "gaussian", "std": 0.01},
  "speed": {"factor": 0.9},
  "pitch": {"steps": 2}
}
```

### Model Checkpointing

Resume training from checkpoints:

```bash
uv run python finetune.py \
  --resume_from_checkpoint ./checkpoint-1000 \
  --train_data train.json \
  --output_dir ./output
```

## Optional: Mainstream Framework Comparison

The core of this repository is the **in-house ORPO Whisper trainer** above.
The `benchmark/` directory is an **optional** suite for comparing that baseline
against mainstream RL stacks (TRL / OpenRLHF / verl) under a shared data and
reward protocol.

See **[`benchmark/README.md`](benchmark/README.md)** for setup, workflows, and
comparison docs. It is not required to train with `finetune.py`.

## Optional: Jetson Orin GPU test environment (Docker / Kubernetes)

For on-device smoke tests and short fine-tunes on **NVIDIA Jetson Orin**
(Orin Nano / NX), see **[`deploy/jetson/README.md`](deploy/jetson/README.md)**.

That tree provides:

- Host prep + GPU check scripts
- L4T-matched Dockerfile and Compose (NVIDIA runtime)
- k3s manifests (RuntimeClass, PVC, smoke Job)
- Low-memory config overlay for Orin Nano–class boards

```bash
# On the Jetson board
bash deploy/jetson/scripts/prepare_host.sh
bash deploy/jetson/scripts/build_and_run.sh smoke
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- The RLHF research community for preference learning algorithms

## Support

For questions and support, please open an issue on the GitHub repository.
