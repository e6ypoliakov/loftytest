#!/bin/bash
set -e

echo "=== ACE-Step Music Generation API Setup ==="

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies..."
uv pip install -e . 2>/dev/null || uv pip install \
    fastapi "uvicorn[standard]" "celery[redis]" redis pydantic pydantic-settings python-multipart

echo "Installing ACE-Step from GitHub..."
uv pip install "ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git" || \
    echo "WARNING: ACE-Step installation failed. The API will run in mock mode."

mkdir -p generated_audio lora_models

echo ""
echo "=== Setup Complete ==="
echo "To start the service:"
echo "  1. Start Redis:         redis-server --daemonize yes"
echo "  2. Start Celery worker: celery -A core.celery_app worker --loglevel=info"
echo "  3. Start API server:    uvicorn api.main:app --host 0.0.0.0 --port 5000"
echo "  4. Open Swagger UI:     http://localhost:5000/docs"
