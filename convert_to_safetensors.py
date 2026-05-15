"""
Convert TENNs LLM checkpoint to safetensors for HuggingFace deployment.

Merges LoRA adapters into base weights and strips Lightning checkpoint
wrapper, producing a clean state dict ready for from_pretrained().

Usage:
    python convert_to_safetensors.py
    python convert_to_safetensors.py --ckpt /path/to/checkpoint.ckpt --out /path/to/output/
"""

import argparse
import sys
import torch
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from tenns_llm import TENNsLLM, load_weights

DEFAULT_CKPT = "checkpoint.ckpt"
DEFAULT_OUT  = "output"


def convert(ckpt_path: str, out_dir: str):
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    model = TENNsLLM()
    model.eval()

    load_weights(model, ckpt_path, strict=False)

    state_dict = model.state_dict()

    # Validate: no LoRA keys should remain
    lora_keys = [k for k in state_dict if 'lora_in' in k or 'lora_out' in k]
    if lora_keys:
        print(f"ERROR: LoRA keys still present after merge: {lora_keys}")
        sys.exit(1)

    # Validate: parameter count
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Validate: no NaNs
    nan_keys = [k for k, v in state_dict.items() if v.is_floating_point() and v.isnan().any()]
    if nan_keys:
        print(f"ERROR: NaN values found in: {nan_keys}")
        sys.exit(1)

    print("Validation passed. Saving safetensors...")

    # safetensors requires contiguous tensors
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    out_file = out_path / "model.safetensors"
    save_file(state_dict, str(out_file))
    size_gb = out_file.stat().st_size / 1e9
    print(f"Saved: {out_file} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out",  default=DEFAULT_OUT)
    args = parser.parse_args()

    convert(args.ckpt, args.out)
