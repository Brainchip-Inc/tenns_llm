"""
TENNs LLM - Recurrent Mode
===========================

PyTorch implementation of a TENNs LLM using gate-mode SSM layers from tenns-core.

Architecture (1B model):
- Embedding: 32000 vocab → 2048 dim
- Backbone: 24 TENNsBlock layers (gate mode, channels=2048, d_inner=4096)
- Head: RMSNorm → Linear(2048, 32000)

Each TENNsBlock:
  RMSNorm → in_proj(2048→8192) → split(x, res) → causal_conv(4) → SiLU
  → SSM(A, B, C, log_dt, state) + D*x_conv → gate(res) → out_proj(4096→2048) → residual

Usage:
    python tenns_llm.py --prompt "Hello, world!" --ckpt model.safetensors
    python tenns_llm.py --prompt "What is the meaning of life?" --max-tokens 50

Weights: https://huggingface.co/BrainChip-AI/tenns-llm-1b
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn import RMSNorm
from transformers import AutoTokenizer

from tenns_core import SSMLayer
from tenns_core.recurrent_ops import recurrent_gate


# ============================================================================
# Model Components
# ============================================================================


class CausalConvDwFast(nn.Module):
    """Holds depthwise causal convolution weights for TENNs blocks."""
    def __init__(self, coeffs, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size, coeffs))


class PassthroughConv(nn.Module):
    """Applies causal convolution via FIFO buffer for streaming inference."""
    def __init__(self, causal_conv, d_inner):
        super().__init__()
        self.causal_conv = causal_conv
        self.d_inner = d_inner
        self.fifo = None

    def apply_conv(self, x):
        """Apply causal convolution. x: (B, T, C) -> (B, T, C)"""
        B, T, C = x.shape

        if self.fifo is None or self.fifo.shape[0] != B:
            self.fifo = torch.zeros(B, C, 4, device=x.device, dtype=x.dtype)

        conv_weight = self.causal_conv.weight.squeeze().T  # (C, 4)

        x_conv = []
        for t in range(T):
            self.fifo = self.fifo.roll(-1, dims=-1)
            self.fifo[:, :, -1] = x[:, t, :]
            x_t = (self.fifo * conv_weight).sum(-1)
            x_conv.append(x_t)

        x_conv = torch.stack(x_conv, dim=1)
        x_conv = F.silu(x_conv)
        return x_conv

    def reset_states(self):
        if self.fifo is not None:
            self.fifo.zero_()


class TENNsBlock(nn.Module):
    """TENNs block with gate-mode SSM for LLM inference.

    Uses recurrent_gate from tenns_core for streaming state updates.
    Supports state_lora as a learned initial hidden state.
    """
    def __init__(self, channels, num_coeffs, repeat, mode='gate'):
        super().__init__()
        d_inner = channels * 2
        self.d_inner = d_inner

        self.pre_norm = RMSNorm(channels, elementwise_affine=True)
        self.pre_conv = CausalConvDwFast(d_inner, 4)
        self.in_proj = nn.Linear(channels, d_inner * 2, bias=True)
        self.out_proj = nn.Linear(d_inner, channels, bias=True)

        self.ssm_layer = SSMLayer(num_coeffs, d_inner, d_inner,
                                  repeat=repeat, mode=mode, transposed=True)

        # Register state_lora as a buffer so load_state_dict picks it up
        self.ssm_layer.register_buffer('state_lora', torch.zeros(d_inner))

        self.D = nn.Parameter(torch.ones(d_inner, dtype=torch.float))

        self._conv_handler = None
        self.state = None

    def forward(self, input):
        x = self.pre_norm(input)
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], -1)

        if self._conv_handler is None:
            self._conv_handler = PassthroughConv(self.pre_conv, self.d_inner)

        x_conv = self._conv_handler.apply_conv(x)

        # Use state_lora as initial state when no state exists yet
        state = self.state
        if state is None:
            state = self.ssm_layer.state_lora

        y, self.state = recurrent_gate(
            x_conv,
            self.ssm_layer.A,
            self.ssm_layer.B,
            self.ssm_layer.C,
            self.ssm_layer.log_dt,
            state
        )

        # y is (B, D, T), need (B, T, D)
        y = y.transpose(1, 2)
        y = y + self.D * x_conv
        output = self.out_proj(y * F.silu(res))

        return input + output

    def reset_states(self):
        if self._conv_handler is not None:
            self._conv_handler.reset_states()
        self.state = None


class TENNsLLM(nn.Module):
    """TENNs-based language model for autoregressive text generation."""
    def __init__(self,
                 vocab_size=32000,
                 channels=2048,
                 num_blocks=24,
                 num_coeffs=16,
                 repeat=256):
        super().__init__()
        self.channels = channels

        self.embedding = nn.Embedding(vocab_size, channels)

        self.backbone = nn.Sequential(
            *[TENNsBlock(channels, num_coeffs, repeat, mode='gate')
              for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            RMSNorm(channels, elementwise_affine=False),
            nn.Linear(channels, vocab_size, bias=False),
        )

    def forward(self, tokens):
        """Forward pass for a single token or sequence.

        Args:
            tokens: Token indices of shape (B, T) or (B, 1) for streaming

        Returns:
            logits: Shape (B, T, vocab_size)
        """
        x = self.embedding(tokens)  # (B, T, C)
        x = self.backbone(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def reset_states(self):
        for module in self.modules():
            if isinstance(module, TENNsBlock):
                module.reset_states()


# ============================================================================
# Checkpoint Loading
# ============================================================================


def load_weights(model, ckpt_path, strict=False):
    """Load weights from a safetensors file or a training checkpoint.

    Safetensors files (e.g. from https://huggingface.co/BrainChip-AI/tenns-llm-1b)
    contain final merged weights and load directly.

    Training checkpoints (.ckpt) contain LoRA adapters (lora_in/lora_out) for
    in_proj, out_proj, and ssm_layer.C. These are merged into the base weights:
        weight_merged = weight + lora_out @ lora_in
    """
    if str(ckpt_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
            for k in missing_keys[:10]:
                print(f"    {k}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            for k in unexpected_keys[:10]:
                print(f"    {k}")
        return model

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    # Strip model._orig_mod. prefix
    state_dict = {k.replace('model._orig_mod.', ''): v
                  for k, v in state_dict.items()}

    # Merge LoRA weights into base weights
    lora_in_keys = [k for k in state_dict if k.endswith('.lora_in')]
    for lora_in_key in lora_in_keys:
        lora_out_key = lora_in_key.replace('.lora_in', '.lora_out')
        base_key = lora_in_key.replace('.lora_in', '.weight')

        if lora_out_key in state_dict and base_key in state_dict:
            lora_in = state_dict[lora_in_key]
            lora_out = state_dict[lora_out_key]
            state_dict[base_key] = state_dict[base_key] + lora_out @ lora_in

    # Remove LoRA keys
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith('.lora_in') and not k.endswith('.lora_out')}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
        for k in missing_keys[:10]:
            print(f"    {k}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:10]:
            print(f"    {k}")

    return model


# ============================================================================
# Text Generation
# ============================================================================


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    """Autoregressive text generation with greedy or top-k sampling.

    Args:
        model: TENNsLLM model
        tokenizer: HuggingFace tokenizer
        prompt: Input text string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = greedy when top_k is None)
        top_k: If set, sample from top-k tokens instead of greedy argmax
    """
    model.eval()
    model.reset_states()

    input_ids = tokenizer(prompt, return_tensors='pt',
                          add_special_tokens=False)['input_ids'].squeeze()

    print(f"Input tokens: {input_ids.tolist()}")

    # Ingest prompt tokens (print predicted next-token at each step)
    print(f"\n--- Prompt ingestion ({len(input_ids)} tokens) ---")
    for token in input_ids:
        logits = model(token.view(1, 1))
        probs = F.softmax(logits[0, -1], dim=-1)
        next_token = torch.argmax(probs).item()
        print(next_token)

    # Autoregressive generation
    print(f"\n--- Generation (max {max_new_tokens} tokens) ---")
    output_ids = []
    token = next_token
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([[token]]))
        next_logits = logits[0, -1]  # (vocab_size,)

        if temperature != 1.0:
            next_logits = next_logits / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)

        if top_k is not None:
            token = torch.multinomial(probs, 1).item()
        else:
            token = torch.argmax(probs).item()

        print(token)

        if token == tokenizer.eos_token_id:
            break

        output_ids.append(token)

    return tokenizer.decode(output_ids)


# ============================================================================
# CLI
# ============================================================================

# Download from https://huggingface.co/BrainChip-AI/tenns-llm-1b:
#   huggingface-cli download BrainChip-AI/tenns-llm-1b model.safetensors --local-dir .
DEFAULT_CHECKPOINT_PATH = "model.safetensors"



if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='TENNs LLM - Autoregressive text generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tenns-llm.py --prompt "Hello, world!"
  python tenns-llm.py --prompt "What is AI?" --max-tokens 100
  python tenns-llm.py --prompt "Once upon a time" --temperature 0.8 --top-k 50
        """
    )
    parser.add_argument('--prompt', type=str, default="Hello, world!",
                        help='Input prompt for text generation')
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help=f'Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling (None = greedy argmax)')

    args = parser.parse_args()

    print("=" * 70)
    print("TENNs LLM - Recurrent Mode Generation")
    print("=" * 70)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        exit(1)

    # Load tokenizer
    print("\nLoading Mistral tokenizer...")
    # Mistral-7B tokenizer, bundled with the weights repo (the upstream
    # mistralai/Mistral-7B-v0.1 repo is gated and would require access approval)
    tokenizer = AutoTokenizer.from_pretrained("BrainChip-AI/tenns-llm-1b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create and load model
    print("Creating model...")
    model = TENNsLLM()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"Loading weights from {ckpt_path}...")
    load_weights(model, str(ckpt_path), strict=False)

    # Generate
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("-" * 70)

    start_time = time.perf_counter()
    output = generate(model, tokenizer, args.prompt,
                      max_new_tokens=args.max_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    elapsed = time.perf_counter() - start_time

    print(f"\n{args.prompt}{output}")
    print("-" * 70)
    print(f"Generated {len(output.split())} words in {elapsed:.2f}s")
    print("=" * 70)
