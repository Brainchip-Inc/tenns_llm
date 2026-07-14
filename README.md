# TENNs LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Built on TENNs-Core](https://img.shields.io/badge/built%20on-TENNs--Core-6f42c1.svg)](https://github.com/Brainchip-Inc/tenns-core)

A ~1B parameter autoregressive language model built on [TENNs-Core](https://github.com/Brainchip-Inc/tenns-core) State Space Model (SSM) layers. Uses 24 TENNsBlock layers in gate mode for efficient text generation with constant memory during inference.

## Model

- **Architecture**: 24-layer TENNsBlock backbone (gate-mode SSM)
- **Parameters**: ~1B
- **Tokenizer**: Mistral-7B (32k vocabulary)
- **Weights**: [BrainChip-AI/tenns-llm-1b](https://huggingface.co/BrainChip-AI/tenns-llm-1b) on Hugging Face

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Brainchip-Inc/tenns_llm.git
cd tenns_llm
uv sync
source .venv/bin/activate
```

## Model Weights

Pretrained weights are hosted on Hugging Face at
[BrainChip-AI/tenns-llm-1b](https://huggingface.co/BrainChip-AI/tenns-llm-1b):

```bash
huggingface-cli download BrainChip-AI/tenns-llm-1b model.safetensors --local-dir .
```

The inference script looks for `model.safetensors` in the current directory by
default (override with `--ckpt`).

**Note:** the code in this repository is MIT-licensed, but the model weights are
released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial). The Hugging Face repo also provides a `transformers`-compatible
loading path (`trust_remote_code=True`) — see the
[model card](https://huggingface.co/BrainChip-AI/tenns-llm-1b) for details.

## Usage

```bash
# Default prompt
python tenns_llm.py

# Custom prompt
python tenns_llm.py --prompt "What is the meaning of life?"

# With sampling parameters
python tenns_llm.py --prompt "Once upon a time" --max-tokens 100 --temperature 0.8 --top-k 50

# Specify checkpoint path
python tenns_llm.py --prompt "Hello" --ckpt path/to/model.safetensors
```

## Files

| File | Purpose |
|------|---------|
| `tenns_llm.py` | Model definition and inference script |
| `pyproject.toml` | Project dependencies |

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Improvements to the underlying SSM layers belong in [TENNs-Core](https://github.com/Brainchip-Inc/tenns-core).

## License

MIT
