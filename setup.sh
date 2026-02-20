#!/bin/bash
set -e

echo "=== ACE-Step Music Generation API — Setup ==="
echo ""

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment..."
uv venv .venv 2>/dev/null || true
source .venv/bin/activate

echo "Installing PyTorch with CUDA 12.1..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing project dependencies..."
uv pip install -e .

echo "Installing ACE-Step 1.5 from GitHub..."
uv pip install "ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git"

mkdir -p generated_audio lora_models

echo ""
echo "Downloading ACE-Step models from HuggingFace..."
python -c "
from acestep.model_downloader import download_all_models
from pathlib import Path
import os

checkpoints_dir = Path(os.getcwd()) / 'checkpoints'
checkpoints_dir.mkdir(exist_ok=True)

hf_token = os.environ.get('HF_TOKEN')
success, messages = download_all_models(checkpoints_dir, token=hf_token)
for msg in messages:
    print(msg)
if success:
    print('All models downloaded successfully.')
else:
    print('WARNING: Some models failed to download. They will be retried on first run.')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the service:"
echo "  source .venv/bin/activate"
echo "  ./start.sh"
echo ""
echo "Or with Docker:"
echo "  ./start_docker.sh"
