#!/bin/bash
# Setup TENNs LLM environment
set -e

UV="${UV:-${HOME}/.local/bin/uv}"

echo "Setting up TENNs LLM environment..."
"$UV" sync

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
