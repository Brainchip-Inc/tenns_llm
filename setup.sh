#!/bin/bash
# Setup virtual environment for TENNs LLM
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up TENNs LLM environment..."

UV="${UV:-${HOME}/.local/bin/uv}"

cd "$SCRIPT_DIR"
"$UV" venv venv --python 3.12
source venv/bin/activate

# Install tenns-core
"$UV" pip install git+https://github.com/Brainchip-Inc/tenns-core.git

# Install LLM dependencies
"$UV" pip install transformers

echo ""
echo "Done. Activate with:"
echo "  source $SCRIPT_DIR/venv/bin/activate"
