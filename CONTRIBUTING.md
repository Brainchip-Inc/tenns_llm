# Contributing to TENNs LLM

Thanks for your interest in contributing! We welcome bug reports, feature requests, and pull requests.

## Getting Set Up

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Brainchip-Inc/tenns_llm.git
cd tenns_llm
uv sync
source .venv/bin/activate
```

## Where Changes Belong

This repository contains the model definition and inference script for the ~1B parameter TENNs language model. The underlying SSM layers live in [TENNs-Core](https://github.com/Brainchip-Inc/tenns-core) — improvements to the core layers (new modes, streaming inference, ONNX export) should be contributed there instead.

## Reporting Bugs

Please open a [GitHub issue](https://github.com/Brainchip-Inc/tenns_llm/issues) and include:

- The exact command you ran (prompt, sampling parameters, checkpoint)
- What you expected and what actually happened, including any traceback
- Your Python, PyTorch, and OS versions

## Proposing Changes

- For small fixes (typos, docs, obvious bugs), feel free to open a pull request directly.
- For larger changes (architecture modifications, new sampling strategies, quantization), please open an issue first so we can discuss the approach.

## Pull Request Guidelines

1. Fork the repository and create a branch from `main`.
2. Keep pull requests focused — one logical change per PR.
3. Describe *what* the change does and *why* in the PR description.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
