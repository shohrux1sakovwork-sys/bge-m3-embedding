#!/usr/bin/env bash
# Sets up an isolated virtual environment for the BGE-M3 server using uv.
# Keeps deps separate from the TTS project so the two can't interfere.
set -euo pipefail

cd "$(dirname "$0")"

# Create a virtual environment using uv
uv venv venv
# shellcheck disable=SC1091
source venv/bin/activate

# PyTorch build must match your CUDA. Uncomment ONE of these lines for your
# setup. The default `pip install torch` ships CUDA 12.x wheels which work
# with the RTX 5070 (Blackwell) on recent drivers.
#
# CUDA 12.8 (recommended for RTX 50-series / Blackwell, matches TTS Dockerfile):
# uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#
# CUDA 12.1:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install requirements using uv
uv pip install -r requirements.txt

echo ""
echo "Done. To start the server:"
echo "  source venv/bin/activate"
echo "  python server.py --port 8060"
