# TENNs LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Built on TENNs-Core](https://img.shields.io/badge/built%20on-TENNs--Core-6f42c1.svg)](https://github.com/Brainchip-Inc/tenns-core)

A ~1B parameter autoregressive language model built on [TENNs-Core](https://github.com/Brainchip-Inc/tenns-core) State Space Model (SSM) layers. Uses 24 TENNsBlock layers in gate mode for efficient text generation with constant memory during inference.

## Model

- **Architecture**: 24-layer TENNsBlock backbone (gate-mode SSM)
- **Parameters**: ~1B
- **Tokenizer**: Mistral-7B (32k vocabulary)

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Brainchip-Inc/tenns_llm.git
cd tenns_llm
uv sync
source .venv/bin/activate
```

## Usage

```bash
# Default prompt
python tenns_llm.py

# Custom prompt
python tenns_llm.py --prompt "What is the meaning of life?"

# With sampling parameters
python tenns_llm.py --prompt "Once upon a time" --max-tokens 100 --temperature 0.8 --top-k 50

# Specify checkpoint path
python tenns_llm.py --prompt "Hello" --ckpt path/to/checkpoint.ckpt
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
