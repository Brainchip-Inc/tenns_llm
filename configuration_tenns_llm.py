import os
import sys

from transformers import PretrainedConfig

# Inject the repo directory into sys.path so the bundled tenns_core/ is
# importable without a pip install, both locally and when loaded from HF hub.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class TennsLLMConfig(PretrainedConfig):
    model_type = "tenns_llm"

    def __init__(
        self,
        vocab_size=32000,
        channels=2048,
        num_blocks=24,
        num_coeffs=16,
        repeat=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_coeffs = num_coeffs
        self.repeat = repeat
