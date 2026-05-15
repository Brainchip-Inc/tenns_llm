import importlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_tenns_llm import TennsLLMConfig

def _get_tenns_core_path():
    """Return a directory that contains tenns_core/.

    HF's from_pretrained only downloads the .py files listed in auto_map —
    it does not download subdirectories like tenns_core/. We use
    snapshot_download (with local cache) to ensure tenns_core/ is present.
    The first call downloads it; subsequent calls are instant cache hits.
    """
    # Derive the repo_id from __file__ path in the HF modules cache:
    # .../modules/transformers_modules/ORG/REPO_SLUG/HASH/modeling_tenns_llm.py
    here = os.path.dirname(os.path.abspath(__file__))
    parts = here.replace("\\", "/").split("/")
    try:
        idx = next(i for i, p in enumerate(parts) if p == "transformers_modules")
        org_id  = parts[idx + 1].replace("_hyphen_", "-")
        repo_id = parts[idx + 2].replace("_hyphen_", "-")
    except (StopIteration, IndexError):
        return here  # not in HF cache — assume tenns_core/ is next to this file

    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        f"{org_id}/{repo_id}",
        allow_patterns=["tenns_core/**"],
    )
    return snapshot


_tenns_core_dir = _get_tenns_core_path()
if _tenns_core_dir not in sys.path:
    sys.path.insert(0, _tenns_core_dir)

_tc = importlib.import_module("tenns_core")
_rc = importlib.import_module("tenns_core.recurrent_ops")
SSMLayer = _tc.SSMLayer
recurrent_gate = _rc.recurrent_gate


# ============================================================================
# Model Components (from tenns_llm.py)
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
    """TENNs block with gate-mode SSM for LLM inference."""
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
    def __init__(self, vocab_size=32000, channels=2048, num_blocks=24,
                 num_coeffs=16, repeat=256):
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
        x = self.embedding(tokens)
        x = self.backbone(x)
        return self.head(x)

    def reset_states(self):
        for module in self.modules():
            if isinstance(module, TENNsBlock):
                module.reset_states()


# ============================================================================
# HuggingFace wrapper
# ============================================================================


class TennsLLMForCausalLM(PreTrainedModel):
    """HuggingFace PreTrainedModel wrapper for TENNsLLM.

    Load with:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "aliborji/tenns-llm-1b", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("aliborji/tenns-llm-1b")

    Generate with:
        output = model.generate_text("Hello, world!", tokenizer, max_new_tokens=50)
        print(output)

    Note: This model uses recurrent SSM states. Use generate_text() rather than
    model.generate(), which is designed for attention-based KV-cache models.
    """
    config_class = TennsLLMConfig
    # Weights are saved without a 'model.' prefix — flatten components directly
    # onto this class so state dict keys match the safetensors file exactly.
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: TennsLLMConfig):
        super().__init__(config)
        # Assign TENNsLLM components directly (not as self.model) so that
        # state dict keys match the safetensors: embedding.weight, backbone.0...
        _backbone = TENNsLLM(
            vocab_size=config.vocab_size,
            channels=config.channels,
            num_blocks=config.num_blocks,
            num_coeffs=config.num_coeffs,
            repeat=config.repeat,
        )
        self.embedding = _backbone.embedding
        self.backbone  = _backbone.backbone
        self.head      = _backbone.head

    def _reset_states(self):
        for module in self.modules():
            if isinstance(module, TENNsBlock):
                module.reset_states()

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        x = self.backbone(x)
        logits = self.head(x)
        return CausalLMOutputWithPast(logits=logits)

    @torch.no_grad()
    def generate_text(self, prompt, tokenizer, max_new_tokens=50,
                      temperature=1.0, top_k=None):
        """Autoregressive text generation.

        Args:
            prompt: Input text string
            tokenizer: HuggingFace tokenizer
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, sample from top-k tokens; otherwise greedy argmax

        Returns:
            Generated text string (not including the prompt)
        """
        self.eval()
        self._reset_states()

        input_ids = tokenizer(prompt, return_tensors='pt',
                              add_special_tokens=False)['input_ids'].squeeze()
        input_ids = input_ids.to(self.device)

        # Ingest prompt tokens
        for token in input_ids:
            logits = self.forward(token.view(1, 1)).logits
            probs = F.softmax(logits[0, -1], dim=-1)
            next_token = torch.argmax(probs).item()

        # Autoregressive generation
        output_ids = []
        token = next_token
        for _ in range(max_new_tokens):
            logits = self.forward(torch.tensor([[token]], device=self.device)).logits
            next_logits = logits[0, -1]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            token = (torch.multinomial(probs, 1).item() if top_k is not None
                     else torch.argmax(probs).item())

            if token == tokenizer.eos_token_id:
                break

            output_ids.append(token)

        return tokenizer.decode(output_ids)
