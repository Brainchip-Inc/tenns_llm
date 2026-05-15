# TENNs LLM

A ~1B parameter autoregressive language model built on [TENNs-Core](https://github.com/Brainchip-Inc/tenns-core) State Space Model (SSM) layers. Uses 24 TENNsBlock layers in gate mode for efficient text generation with constant memory during inference.

## Model

- **Architecture**: 24-layer TENNsBlock backbone (gate-mode SSM)
- **Parameters**: ~1B
- **Tokenizer**: Mistral-7B (32k vocabulary)
- **Weights**: [BrainChipInc/tenns-llm-1b](https://huggingface.co/BrainChipInc/tenns-llm-1b) on HuggingFace Hub

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Brainchip-Inc/tenns_llm.git
cd tenns_llm
./setup.sh
source venv/bin/activate
```

## Usage

### Via HuggingFace (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BrainChipInc/tenns-llm-1b')
model = AutoModelForCausalLM.from_pretrained(
    'BrainChipInc/tenns-llm-1b',
    trust_remote_code=True,
)

output = model.generate_text(
    'The history of artificial intelligence',
    tokenizer,
    max_new_tokens=100,
)
print(output)
```

### Direct inference script

```bash
# Default prompt and checkpoint
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
| `tenns_llm.py` | Core model definition and inference script |
| `modeling_tenns_llm.py` | HuggingFace `PreTrainedModel` wrapper |
| `configuration_tenns_llm.py` | HuggingFace `PretrainedConfig` subclass |
| `config.json` | Architecture hyperparameters |
| `convert_to_safetensors.py` | Convert a raw training checkpoint to safetensors |

## Converting a checkpoint

If you have a raw training checkpoint (`.ckpt`), convert it to safetensors with:

```bash
python convert_to_safetensors.py --ckpt path/to/checkpoint.ckpt --out path/to/output/
```

## License

MIT
